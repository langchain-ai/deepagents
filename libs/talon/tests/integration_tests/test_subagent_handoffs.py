from __future__ import annotations

import json
import os

import pytest
from deepagents.backends.state import StateBackend
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents_talon.subagents import TaskTools, _compile_fresh

_OBSERVED_AT = "2026-06-15T12:00:00Z"
_REDACTION_MARKER = "synthetic-private-credential-do-not-repeat"
_ITEMS = [
    {
        "id": "CASE-101",
        "url": "https://example.com/cases/101",
        "state": "ready",
        "evidence": "All required approvals recorded",
        "observed_at": _OBSERVED_AT,
    },
    {
        "id": "CASE-102",
        "url": "https://example.com/cases/102",
        "state": "blocked",
        "evidence": "Required approval missing",
        "observed_at": _OBSERVED_AT,
    },
]
_TASK = """Determine which open cases are ready for action and whether the inventory
covers every candidate. Research only; do not take action. Scope: all open cases in
this fixture. Available source: read_candidates if attached. Freshness: evidence
observed on 2026-06-15 is acceptable. Stop after this source's response; there are
no other sources or retries available.
Return only a JSON object with fields:
items: assessed candidates, each with id, url, state, evidence, observed_at;
total: integer or null when unknown;
assessed: integer;
unassessed: list of exact known candidate IDs, or "unknown" for an unidentified remainder;
error_kind: one of none, zero_matches, unavailable_tools, permission, authentication,
timeout, truncation, partial_coverage;
attempted, succeeded, unknown: lists of short factual descriptions;
uncertainties: list of remaining limitations or conflicts.
Succeeded describes useful source evidence obtained, not merely a tool invocation.
"""


@pytest.fixture
def handoff_model():
    model = os.environ.get("TALON_HANDOFF_EVAL_MODEL")
    if not model:
        pytest.skip("Set TALON_HANDOFF_EVAL_MODEL to run real-model handoff evals")
    return init_chat_model(model, temperature=0)


def _parse_handoff(content):
    text = (
        content
        if isinstance(content, str)
        else "".join(block.get("text", "") for block in content if isinstance(block, dict))
    )
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text), text


@pytest.mark.parametrize(
    ("source", "expected_kind", "coverage"),
    [
        pytest.param(
            {"items": _ITEMS, "total": 2, "complete": True},
            "none",
            (2, 2, []),
            id="complete",
        ),
        pytest.param(
            {"items": _ITEMS, "total": 3, "complete": False, "unassessed": ["CASE-103"]},
            "partial_coverage",
            (3, 2, ["CASE-103"]),
            id="known-partial",
        ),
        pytest.param(
            {"items": _ITEMS, "total": None, "complete": False, "truncated": True},
            "truncation",
            (None, 2, "unknown"),
            id="unknown-total-truncated",
        ),
        pytest.param(
            {"items": [], "total": 0, "complete": True},
            "zero_matches",
            (0, 0, []),
            id="zero-matches",
        ),
        pytest.param(None, "unavailable_tools", (None, 0, "unknown"), id="missing-tools"),
        *[
            pytest.param(
                {
                    "error": kind,
                    "diagnostic": f"{detail}; Authorization: Bearer {_REDACTION_MARKER}",
                    "items": [],
                    "total": None,
                },
                kind,
                (None, 0, "unknown"),
                id=kind,
            )
            for kind, detail in [
                ("permission", "HTTP 403: source access denied"),
                ("authentication", "HTTP 401: credentials rejected"),
                ("timeout", "Source did not respond within 30 seconds"),
            ]
        ],
    ],
)
async def test_subagent_handoff_semantics(
    handoff_model,
    source,
    expected_kind,
    coverage,
) -> None:
    total, assessed, remainder = coverage
    calls = []

    @tool
    def read_candidates() -> str:
        """Read the sole open-case inventory source without changing anything."""
        calls.append("read_candidates")
        return json.dumps(source)

    graph = _compile_fresh(
        {
            "name": "case-research",
            "description": "Research case readiness",
            "system_prompt": "You investigate delegated questions using the attached sources.",
            "tools": [read_candidates] if source is not None else [],
        },
        handoff_model,
        None,
    )["runnable"]
    result = await graph.ainvoke(
        {"messages": [HumanMessage(_TASK)]},
        {"recursion_limit": 10},
    )
    handoff, text = _parse_handoff(result["messages"][-1].content)
    assert _REDACTION_MARKER not in text
    assert calls == (["read_candidates"] if source is not None else [])
    assert handoff["error_kind"] == expected_kind
    assert handoff["total"] == total
    assert handoff["assessed"] == assessed
    assert handoff["unassessed"] == remainder
    assert len(handoff["items"]) == assessed
    expected_items = {item["id"]: item for item in _ITEMS[:assessed]}
    assert {item["id"] for item in handoff["items"]} == set(expected_items)
    for item in handoff["items"]:
        expected = expected_items[item["id"]]
        for field in ("url", "state", "observed_at"):
            assert item[field] == expected[field]
        assert item["evidence"]
    for field in ("attempted", "succeeded", "unknown", "uncertainties"):
        assert isinstance(handoff[field], list)
    if source is not None:
        assert handoff["attempted"]
    if assessed or expected_kind == "zero_matches":
        assert handoff["succeeded"]
    else:
        assert not handoff["succeeded"]
    if remainder:
        assert handoff["unknown"]
        assert handoff["uncertainties"]
    else:
        assert not handoff["unknown"]


def _parent_graph(model, *, retrieval, handoff, calls):
    @tool
    def get_agent_tools() -> str:
        """Inspect effective configured and selectable research tools."""
        calls.append("get_agent_tools")
        return json.dumps(
            {
                "agents": [
                    {
                        "name": "external-research",
                        "mode": "fresh",
                        "tools": ["read_candidates"] if retrieval else [],
                        "selectable_tools": ["read_candidates"] if retrieval else [],
                    }
                ]
            }
        )

    @tool
    def task(description: str, subagent_type: str) -> str:
        """Delegate research to external-research and receive its evidence handoff."""
        del description, subagent_type
        calls.append("task")
        return json.dumps(handoff)

    @tool
    def read_candidates() -> str:
        """Broadly search every open public case."""
        calls.append("read_candidates")
        return json.dumps({"items": _ITEMS, "total": 2, "complete": True})

    catalog = {item.name: item for item in (get_agent_tools, task)}
    if retrieval:
        catalog[read_candidates.name] = read_candidates
    middleware = TaskTools(model, None, backend=StateBackend())
    middleware.bind(catalog)
    return create_agent(
        model,
        tools=list(catalog.values()),
        middleware=[middleware],
        system_prompt="You coordinate research and make evidence-based decisions.",
    )


async def test_parent_preflights_missing_public_tools(handoff_model) -> None:
    calls = []
    graph = _parent_graph(handoff_model, retrieval=False, handoff={}, calls=calls)
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    "Delegate public research to external-research to find all open public "
                    "cases and decide which are ready. Require current source evidence, IDs, "
                    "links, readiness state and complete coverage. Return only JSON with "
                    "error_kind (none, unavailable_tools, or zero_matches) and unknown "
                    "(a list of facts you cannot determine)."
                )
            ]
        },
        {"recursion_limit": 10},
    )
    report, _ = _parse_handoff(result["messages"][-1].content)
    assert "get_agent_tools" in calls
    assert "task" not in calls
    assert report["error_kind"] == "unavailable_tools"
    assert report["unknown"]


async def test_parent_reuses_complete_handoff(handoff_model) -> None:
    calls = []
    handoff = {
        "items": _ITEMS,
        "total": 2,
        "assessed": 2,
        "unassessed": [],
        "coverage": "complete",
        "uncertainties": [],
        "error_kind": "none",
    }
    graph = _parent_graph(handoff_model, retrieval=True, handoff=handoff, calls=calls)
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    "Delegate to external-research to assess all open public cases against "
                    "the required-approval criterion. Request IDs, links, readiness state, "
                    "evidence and coverage. Evidence observed on 2026-06-15 is fresh enough; "
                    "stop once every candidate is assessed. Decide which are ready without "
                    "taking action. Return only JSON with ready (a list of exact case IDs), "
                    "blocked (a list of exact case IDs), total, assessed and unassessed."
                )
            ]
        },
        {"recursion_limit": 10},
    )
    decision, _ = _parse_handoff(result["messages"][-1].content)
    assert "get_agent_tools" in calls
    assert calls.count("task") == 1
    assert "read_candidates" not in calls
    assert decision["ready"] == ["CASE-101"]
    assert decision["blocked"] == ["CASE-102"]
    assert decision["total"] == decision["assessed"] == 2
    assert decision["unassessed"] == []
