"""Contract tests for the coding-agent-v1 trace-metadata standard (LSEN-277).

These load the machine-readable contract (`validator.json`, vendored under
`data/`) and assert that the metadata Deep Agents Code stamps onto its
LangGraph stream config satisfies the contract — required keys, types,
allowed values, and the run-type `appliesTo` rules — for every run type the
trace-wide metadata block lands on.

The vendored validator is a copy of the shared
`coding-agent-v1/validator.json`; keep it in sync when the contract changes.
End-to-end acceptance is a live trace validated with `validate-thread.mjs`
(the `deepagents-code` profile), not these hermetic unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code import config as config_module
from deepagents_code._version import __version__
from deepagents_code.config import build_coding_agent_metadata, build_stream_config

if TYPE_CHECKING:
    from collections.abc import Iterator

_VALIDATOR_PATH = Path(__file__).parent / "data" / "coding_agent_v1_validator.json"

# Run types the trace-wide stream-config metadata propagates to.
_TRACE_WIDE_RUN_TYPES = ("root", "llm", "tool", "subagent", "interrupted")

# Scope-restricted keys not emitted (would leak trace-wide; see helper docstring).
_OMITTED_SCOPE_RESTRICTED_KEYS = frozenset(
    {"approval_policy", "ls_subagent_id", "ls_subagent_type"}
)


@pytest.fixture(scope="module")
def contract() -> dict:
    """Load the vendored coding-agent-v1 validator contract."""
    return json.loads(_VALIDATOR_PATH.read_text(encoding="utf-8"))


def _type_ok(value: object, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def _validate(
    metadata: dict, run_type: str, contract: dict
) -> tuple[list[str], list[str]]:
    """Mirror validate-thread.mjs's per-run rules for one metadata dict.

    Returns:
        `(errors, missing_where_known)` for `metadata` classified as `run_type`.
    """
    errors: list[str] = []
    missing_where_known: list[str] = []

    for spec in contract["keys"]:
        key = spec["key"]
        applies = run_type in spec["appliesTo"]
        present = key in metadata

        if applies and not present:
            if spec["requirement"] == "always":
                errors.append(f"missing required key {key!r}")
            elif spec["requirement"] == "where_known" and spec.get(
                "requiredWhereKnown"
            ):
                missing_where_known.append(key)
            continue

        if applies and present:
            value = metadata[key]
            if not _type_ok(value, spec["type"]):
                errors.append(f"{key!r} wrong type: {value!r}")
            allowed = spec.get("allowedValues")
            if allowed and value not in allowed:
                errors.append(f"{key!r}={value!r} not in {allowed}")

        # Leakage: a contract key present on a run type outside its appliesTo.
        if not applies and present:
            errors.append(
                f"{key!r} leaked onto {run_type!r} (only {spec['appliesTo']})"
            )

    return errors, missing_where_known


@pytest.fixture
def known_env() -> Iterator[None]:
    """Patch git/user lookups so every where-known contract key is resolvable."""
    repo = (
        "https://github.com/langchain-ai/deepagents",
        "github",
        "langchain-ai/deepagents",
    )
    sha = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
    with (
        patch.object(config_module, "_get_git_branch", return_value="main"),
        patch.object(config_module, "_get_git_commit_sha", return_value=sha),
        patch.object(config_module, "_get_repository_metadata", return_value=repo),
        patch.dict("os.environ", {"DEEPAGENTS_CODE_USER_ID": "u_test"}),
    ):
        yield


@pytest.mark.usefixtures("known_env")
class TestContractCompliance:
    """The trace-wide metadata block satisfies the contract on every run type."""

    def test_no_hard_errors_on_any_run_type(self, contract: dict) -> None:
        config = build_stream_config(
            "thread-123", assistant_id="agent", turn_id="turn-abc", turn_number=2
        )
        metadata = config["metadata"]
        for run_type in _TRACE_WIDE_RUN_TYPES:
            errors, _ = _validate(metadata, run_type, contract)
            assert errors == [], f"{run_type}: {errors}"

    def test_all_where_known_keys_present_when_known(self, contract: dict) -> None:
        config = build_stream_config(
            "thread-123", assistant_id="agent", turn_id="turn-abc", turn_number=2
        )
        metadata = config["metadata"]
        # With git + user fully resolvable, no where_known key should be missing.
        for run_type in _TRACE_WIDE_RUN_TYPES:
            _, missing = _validate(metadata, run_type, contract)
            assert missing == [], f"{run_type}: missing where-known {missing}"

    def test_scope_restricted_keys_are_omitted(self) -> None:
        # These would leak trace-wide and fail validation, so they are not
        # emitted by design (documented limitation).
        config = build_stream_config(
            "thread-123", assistant_id="agent", turn_id="turn-abc", turn_number=2
        )
        metadata = config["metadata"]
        for key in _OMITTED_SCOPE_RESTRICTED_KEYS:
            assert key not in metadata


@pytest.mark.usefixtures("known_env")
class TestContractValueSemantics:
    """Exact values of the identity block, versions, and derived keys."""

    def test_identity_block(self) -> None:
        metadata = build_coding_agent_metadata(
            thread_id="t1",
            turn_id="turn-1",
            turn_number=1,
            cwd="/work",
            git_branch="main",
            sandbox_type=None,
            user_id=None,
        )
        assert metadata["ls_agent_kind"] == "coding_agent"
        assert metadata["ls_integration"] == "deepagents-code"
        assert metadata["ls_agent_runtime"] == "Deep Agents Code"
        assert metadata["ls_trace_schema_version"] == "coding-agent-v1"
        assert metadata["thread_id"] == "t1"

    def test_versions_coincide_with_package_version(self) -> None:
        metadata = build_coding_agent_metadata(
            thread_id="t1",
            turn_id=None,
            turn_number=None,
            cwd="",
            git_branch=None,
            sandbox_type=None,
            user_id=None,
        )
        assert metadata["ls_integration_version"] == __version__
        assert metadata["ls_agent_runtime_version"] == __version__

    def test_repository_keys_from_metadata(self) -> None:
        metadata = build_coding_agent_metadata(
            thread_id="t1",
            turn_id=None,
            turn_number=None,
            cwd="",
            git_branch=None,
            sandbox_type=None,
            user_id=None,
        )
        assert (
            metadata["repository_url"] == "https://github.com/langchain-ai/deepagents"
        )
        assert metadata["repository_provider"] == "github"
        assert metadata["repository_name"] == "langchain-ai/deepagents"

    def test_turn_markers_present_and_typed(self) -> None:
        metadata = build_coding_agent_metadata(
            thread_id="t1",
            turn_id="turn-xyz",
            turn_number=3,
            cwd="",
            git_branch=None,
            sandbox_type=None,
            user_id=None,
        )
        assert metadata["turn_id"] == "turn-xyz"
        assert metadata["turn_number"] == 3
        assert isinstance(metadata["turn_number"], int)


class TestUnknownKeysOmitted:
    """Keys with unknown values are omitted regardless of environment."""

    def test_unknown_keys_omitted(self) -> None:
        with (
            patch.object(config_module, "_get_git_commit_sha", return_value=None),
            patch.object(config_module, "_get_repository_metadata", return_value=None),
        ):
            metadata = build_coding_agent_metadata(
                thread_id="t1",
                turn_id=None,
                turn_number=None,
                cwd="",
                git_branch=None,
                sandbox_type="none",
                user_id=None,
            )
        for absent in (
            "turn_id",
            "turn_number",
            "repository_url",
            "git_branch",
            "git_commit_sha",
            "cwd",
            "user_id",
            "sandbox_type",
        ):
            assert absent not in metadata
        assert metadata["ls_agent_kind"] == "coding_agent"
        assert metadata["thread_id"] == "t1"
