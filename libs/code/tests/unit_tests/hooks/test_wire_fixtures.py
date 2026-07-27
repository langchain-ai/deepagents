"""Differential golden fixtures for the Hooks v2 external wire contract.

Committed JSON under `fixtures/wire/` pins stdin payloads, registry policy
rows, and handler-exit → domain-decision interpretation. Exhaustiveness is
enforced by iterating `HookEvent` / the capability registry — a new event
without a fixture fails the suite.
"""

from __future__ import annotations

import json
from operator import itemgetter
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_code.hooks.models.adapters import HOOK_WIRE_INPUT_ADAPTER
from deepagents_code.hooks.models.domain import HookEvent
from deepagents_code.hooks.projection import project_hook_input
from deepagents_code.hooks.reducer import reduce_hook_results
from unit_tests.hooks.wire_fixture_helpers import (
    FIXED_TRANSCRIPT_PATH,
    INPUTS_DIR,
    REDUCTION_CASES_PATH,
    ReductionCaseFixture,
    agent_transcript_path_for,
    assert_exact_mapping,
    decision_snapshot,
    handler_result_from_exit,
    load_reduction_cases,
    load_registry_policies,
    load_wire_input_fixture,
    normalize_wire_payload,
    registry_policy_row,
    representative_invocation,
)

if TYPE_CHECKING:
    from deepagents_code.json_types import JsonObject


def test_wire_input_fixtures_cover_every_registry_event() -> None:
    fixture_events = {path.stem for path in INPUTS_DIR.glob("*.json")}
    registry_events = {event.value for event in HookEvent}
    assert fixture_events == registry_events, (
        "Wire input fixtures must cover every HookEvent exactly: "
        f"missing={sorted(registry_events - fixture_events)!r} "
        f"extra={sorted(fixture_events - registry_events)!r}"
    )


@pytest.mark.parametrize("event", list(HookEvent), ids=lambda event: event.value)
def test_projected_wire_input_matches_fixture(event: HookEvent) -> None:
    invocation = representative_invocation(event)
    projected = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        project_hook_input(
            invocation,
            transcript_path=FIXED_TRANSCRIPT_PATH,
            agent_transcript_path=agent_transcript_path_for(event),
        ),
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    actual = normalize_wire_payload(cast("JsonObject", projected))
    expected = normalize_wire_payload(load_wire_input_fixture(event))
    assert_exact_mapping(actual, expected, label=f"{event.value} wire input")


def test_registry_policy_fixture_covers_every_event() -> None:
    policies = load_registry_policies()
    registry_events = {event.value for event in HookEvent}
    assert set(policies) == registry_events, (
        "registry_policies.json must cover every HookEvent exactly: "
        f"missing={sorted(registry_events - set(policies))!r} "
        f"extra={sorted(set(policies) - registry_events)!r}"
    )


@pytest.mark.parametrize("event", list(HookEvent), ids=lambda event: event.value)
def test_registry_policy_matches_fixture(event: HookEvent) -> None:
    policies = load_registry_policies()
    actual = cast("JsonObject", dict(registry_policy_row(event)))
    expected = cast("JsonObject", dict(policies[event.value]))
    assert_exact_mapping(actual, expected, label=f"{event.value} registry policy")


def test_reduction_cases_fixture_is_nonempty() -> None:
    cases = load_reduction_cases()
    assert cases, f"Expected reduction cases in {REDUCTION_CASES_PATH}"
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), f"Duplicate reduction case ids: {ids!r}"


@pytest.mark.parametrize(
    "case",
    load_reduction_cases(),
    ids=itemgetter("id"),
)
async def test_handler_output_reduction_matches_fixture(
    case: ReductionCaseFixture,
) -> None:
    event = HookEvent(case["event"])
    invocation = representative_invocation(event)
    result = await handler_result_from_exit(
        event=event,
        exit_code=case["exit_code"],
        stdout=case["stdout"],
        stderr=case["stderr"],
    )
    decision = reduce_hook_results(invocation, [result])
    actual = decision_snapshot(decision)
    assert_exact_mapping(
        actual,
        case["expected"],
        label=f"{case['id']} reduced decision",
    )


def test_reduction_cases_cover_required_exit_shapes() -> None:
    """Pin the verification matrix: exit 0 JSON, exit 2, other nonzero, plain."""
    cases = load_reduction_cases()
    by_event: dict[str, set[str]] = {}
    for case in cases:
        shapes = by_event.setdefault(case["event"], set())
        if case["exit_code"] == 0 and case["stdout"].lstrip().startswith("{"):
            shapes.add("exit0_json")
        elif case["exit_code"] == 2:
            shapes.add("exit2")
        elif case["exit_code"] != 0:
            shapes.add("nonzero")
        elif case["exit_code"] == 0:
            shapes.add("plain")
    required = {"exit0_json", "exit2", "nonzero", "plain"}
    assert by_event, "No reduction cases loaded"
    for event, shapes in by_event.items():
        assert shapes == required, (
            f"{event} reduction cases missing shapes: "
            f"have={sorted(shapes)!r} need={sorted(required)!r}"
        )


def test_wire_input_fixture_files_are_objects() -> None:
    for path in sorted(INPUTS_DIR.glob("*.json")):
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), f"{path.name} must be a JSON object"
