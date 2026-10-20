"""Tests for resolving `dcode -r <id> -n ...` into a resumable thread."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from deepagents_code.client.headless_resume import (
    ResumeError,
    ResumeTarget,
    cwd_conflict,
    persisted_model,
    resolve_resume_target,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Replace the session-store lookups with an existing, policy-clean thread."""
    mocks = {
        "thread_exists": AsyncMock(return_value=True),
        "find_similar_threads": AsyncMock(return_value=[]),
        "get_thread_agent": AsyncMock(return_value="coder"),
        "get_thread_cwd": AsyncMock(return_value="/work"),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(f"deepagents_code.sessions.{name}", mock)
    monkeypatch.setattr("deepagents_code.resume_policy.resume_cutoff", lambda: None)
    return mocks


class TestResolveResumeTarget:
    @pytest.mark.usefixtures("sessions")
    async def test_existing_thread(self) -> None:
        target = await resolve_resume_target("t-1")
        assert target == ResumeTarget("t-1", agent_name="coder", cwd="/work")

    async def test_missing_thread_suggests_similar_ids(
        self, sessions: dict[str, AsyncMock]
    ) -> None:
        sessions["thread_exists"].return_value = False
        sessions["find_similar_threads"].return_value = ["t-10", "t-11"]
        with pytest.raises(ResumeError, match=r"'t-1' not found\. Did you mean: t-10"):
            await resolve_resume_target("t-1")

    @pytest.mark.usefixtures("sessions")
    async def test_resume_age_policy_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.resume_policy.thread_resume_block",
            AsyncMock(return_value="Thread t-1 cannot be resumed because it is old."),
        )
        with pytest.raises(ResumeError, match="cannot be resumed"):
            await resolve_resume_target("t-1")


class TestCwdConflict:
    def test_same_directory(self, tmp_path: Path) -> None:
        target = ResumeTarget("t-1", agent_name="coder", cwd=str(tmp_path))
        assert cwd_conflict(target, str(tmp_path)) is None

    def test_other_directory(self, tmp_path: Path) -> None:
        origin = tmp_path / "origin"
        origin.mkdir()
        target = ResumeTarget("t-1", agent_name=None, cwd=str(origin))
        message = cwd_conflict(target, str(tmp_path))
        assert message is not None
        assert str(origin.resolve()) in message

    def test_missing_directory(self, tmp_path: Path) -> None:
        target = ResumeTarget("t-1", agent_name=None, cwd=str(tmp_path / "gone"))
        message = cwd_conflict(target, str(tmp_path))
        assert message is not None
        assert "no longer available" in message


def test_persisted_model() -> None:
    assert persisted_model(
        {"_model_spec": " openai:x ", "_model_params": {"temperature": 0}}
    ) == ("openai:x", {"temperature": 0})
    assert persisted_model({"_model_spec": "openai:x", "_model_params": "bad"}) == (
        "openai:x",
        None,
    )
    assert persisted_model({"_model_spec": "  "}) is None
