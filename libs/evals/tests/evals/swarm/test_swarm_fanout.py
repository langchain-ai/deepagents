"""Swarm-skill fan-out smoke test.

Confirms the REPL can load the ``swarm`` skill and that calling
``runSwarm`` actually dispatches parallel subagent tasks which produce
usable results. One pytest case — five items, one agent invocation —
assertions cover correctness plus two guards that the skill did the
work (not the root agent solving it sequentially).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

import pytest
from langsmith import testing as t
from langsmith.run_helpers import get_current_run_tree

from tests.evals.swarm.runner import build_swarm_agent
from tests.evals.utils import run_agent_async

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.utils import AgentTrajectory

logger = logging.getLogger(__name__)

_LANGSMITH_CONFIGURED = bool(os.environ.get("LANGSMITH_API_KEY"))
_langsmith_mark = pytest.mark.langsmith if _LANGSMITH_CONFIGURED else lambda f: f


# Five items with unambiguous categories. Chosen so model knowledge
# isn't the failure mode — if the test fails on correctness, the skill
# wiring or prompt is the problem, not the model's grasp of what a
# granite is.
_ITEMS: dict[str, tuple[str, str]] = {
    "01.txt": ("elephant", "animal"),
    "02.txt": ("granite", "mineral"),
    "03.txt": ("oak", "plant"),
    "04.txt": ("salmon", "animal"),
    "05.txt": ("quartz", "mineral"),
}

_INITIAL_FILES: dict[str, str] = {
    f"/items/{name}": word for name, (word, _) in _ITEMS.items()
}
_EXPECTED: dict[str, str] = {name: category for name, (_, category) in _ITEMS.items()}

_QUERY = (
    "There are five files under /items/ named 01.txt through 05.txt. "
    "Each contains a single word. For every file, read its contents and "
    "classify the word as exactly one of: animal, plant, or mineral. "
    "Use the `swarm` skill so the classifications run in parallel — "
    "import it with `const { runSwarm } = await import(\"@/skills/swarm\")` "
    "inside an `eval` call, and dispatch one task per file. "
    "Return ONLY a JSON object mapping filename (e.g. \"01.txt\") to "
    "its category. No extra commentary, no code fences."
)


def _extract_json_object(text: str) -> dict[str, str] | None:
    """Pull the first JSON object out of the model's final text.

    The prompt asks for a bare object, but models sometimes wrap it in
    ```json``` fences or preamble. Scan for the first ``{...}`` span
    and try to parse it; return ``None`` on any failure so the caller
    can fail the test with a helpful message.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(k): str(v).strip().lower() for k, v in parsed.items()}


def _count_run_swarm_invocations(trajectory: AgentTrajectory) -> int:
    """Count ``eval`` tool calls whose code body references ``runSwarm(``.
    Proves the skill was loaded and invoked.

    We match on the source string (not the tool-call name) because the
    skill is JS code run via ``REPLMiddleware``'s ``eval`` tool — there
    is no ``runSwarm`` tool to introspect directly. The REPL middleware
    exposes the tool as ``eval`` by default (see
    ``deepagents_repl.middleware._DEFAULT_TOOL_NAME``); if a caller ever
    renames it via ``REPLMiddleware(tool_name=...)`` this check would
    need to follow.
    """
    count = 0
    for step in trajectory.steps:
        for tc in step.action.tool_calls:
            if tc.get("name") != "eval":
                continue
            args = tc.get("args") or {}
            code = str(args.get("code") or "")
            if "runSwarm(" in code:
                count += 1
    return count


_STDOUT_RE = re.compile(r"<stdout>\n?(.*?)\n?</stdout>", re.DOTALL)


def _extract_swarm_summary(trajectory: AgentTrajectory) -> dict | None:
    """Find the ``SwarmSummary`` emitted by ``runSwarm`` in the ``eval``
    observation.

    The root agent only ever sees one tool call — the ``eval`` that
    invokes ``runSwarm`` — so we can't count ``task`` tool calls on
    the root trajectory. The subagent fan-out happens inside the REPL
    via PTC, not as sibling tool calls on the root. Instead, we rely
    on the skill's ``console.log(JSON.stringify(summary))`` pattern
    (documented in SKILL.md) — that stdout shows up in the ``eval``
    ``ToolMessage`` wrapped in ``<stdout>…</stdout>``, which we parse
    out and JSON-load.

    Returns the parsed ``SwarmSummary`` dict on success, ``None`` if
    no ``eval`` observation contained a parseable summary. ``None``
    means either the skill wasn't invoked, the model didn't log its
    result, or the JSON in stdout wasn't well-formed.
    """
    for step in trajectory.steps:
        for obs in step.observations:
            content = obs.content if isinstance(obs.content, str) else ""
            match = _STDOUT_RE.search(content)
            if match is None:
                continue
            try:
                parsed = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and {"total", "completed", "results"} <= parsed.keys():
                return parsed
    return None


@_langsmith_mark
async def test_swarm_fanout(model: BaseChatModel) -> None:
    """Classify five items via the swarm skill, verify fan-out happened."""
    agent = build_swarm_agent(model=model)
    # Skill source comes from the ``/skills/`` route of the agent's
    # ``CompositeBackend`` (pointing at a real filesystem dir), so we
    # only need to seed the item files into state via ``initial_files``
    # — which the composite's default route (``StateBackend``) picks up.
    trajectory = await run_agent_async(
        agent,
        model=model,
        query=_QUERY,
        initial_files=_INITIAL_FILES,
    )

    final_text = trajectory.answer
    parsed = _extract_json_object(final_text)
    run_swarm_calls = _count_run_swarm_invocations(trajectory)
    summary = _extract_swarm_summary(trajectory)
    summary_total = summary.get("total") if summary else None
    summary_completed = summary.get("completed") if summary else None

    clean_inputs = {
        "items": _INITIAL_FILES,
        "query": _QUERY,
    }
    t.log_inputs(clean_inputs)
    t.log_reference_outputs({"answer": _EXPECTED})
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.inputs = clean_inputs

    t.log_outputs(
        {
            "prediction": parsed,
            "final_text": final_text,
            "run_swarm_invocations": run_swarm_calls,
            "swarm_summary": summary,
        }
    )

    correct = parsed == _EXPECTED
    skill_invoked = run_swarm_calls >= 1
    # ``completed`` is the count of subagents that returned a result
    # without throwing — that's what we actually care about proving
    # happened. ``total`` just reflects the input; a five-task call
    # where every subagent raised would still have ``total=5``.
    fanout_ok = (
        summary is not None
        and summary_completed is not None
        and summary_completed >= len(_EXPECTED)
    )

    t.log_feedback(key="correct", score=1 if correct else 0)
    t.log_feedback(key="skill_invoked", score=1 if skill_invoked else 0)
    t.log_feedback(key="fanout_ok", score=1 if fanout_ok else 0)
    t.log_feedback(key="run_swarm_invocations", score=run_swarm_calls)
    if summary_total is not None:
        t.log_feedback(key="swarm_total", score=summary_total)
    if summary_completed is not None:
        t.log_feedback(key="swarm_completed", score=summary_completed)

    # Order the assertions so the most informative failure fires first:
    # if the skill wasn't invoked, correctness and fan-out are meaningless.
    if not skill_invoked:
        pytest.fail(
            f"swarm skill was never invoked (expected at least 1 `runSwarm(` "
            f"in an `eval` call, got {run_swarm_calls}).\n\n"
            f"trajectory:\n{trajectory.pretty()}"
        )
    if not fanout_ok:
        pytest.fail(
            f"swarm fan-out did not complete the expected number of subagent tasks "
            f"(summary={summary!r}, expected completed >= {len(_EXPECTED)}). "
            f"If no summary was captured, the skill's "
            f"``console.log(JSON.stringify(summary))`` output did not land in the "
            f"eval observation.\n\n"
            f"trajectory:\n{trajectory.pretty()}"
        )
    if not correct:
        pytest.fail(
            f"classification mismatch.\n"
            f"  expected: {_EXPECTED}\n"
            f"  got:      {parsed}\n"
            f"  final_text: {final_text!r}"
        )
