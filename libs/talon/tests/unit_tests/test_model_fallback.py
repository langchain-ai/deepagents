"""Model fallback chain behavior through real, network-free runtime graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

from deepagents_talon.interfaces import AgentRequest, SendResult
from deepagents_talon.model_fallback import FALLBACKS_ENV_KEY, ModelFallbackExhaustedError
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.tool_approvals import ToolApprovalStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.outputs import ChatResult


_ANTHROPIC_OVERLOADED = (
    "Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', "
    "'message': 'Overloaded'}}"
)
"""What `anthropic.OverloadedError` stringifies to; it carries no retry wording."""


class StatusError(Exception):
    def __init__(self, status_code: int, message: str = "request failed") -> None:
        super().__init__(message)
        self.status_code = status_code


class FlakyModel(FakeMessagesListChatModel):
    """Fake model that fails with `error` for its first `failures` calls."""

    model: str = "flaky"
    error: Exception | None = None
    failures: int = 1_000
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    settings: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, _tools: object, **kwargs: Any) -> FlakyModel:
        self.settings.append(kwargs)
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        if self.error is not None and len(self.seen) <= self.failures:
            raise self.error
        return super()._generate(messages, *args, **kwargs)


class FlakyAnthropic(ChatAnthropic):
    """Anthropic model whose calls always fail, so prompt caching tags its requests."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)
    settings: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, _tools: object, **kwargs: Any) -> FlakyAnthropic:  # type: ignore[override]  # records settings only
        self.settings.append(kwargs)
        return self

    async def _agenerate(
        self, messages: list[BaseMessage], *_args: Any, **_kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        raise StatusError(529, _ANTHROPIC_OVERLOADED)


def answer(name: str, *texts: str) -> FlakyModel:
    return FlakyModel(
        model=name, responses=[AIMessage(content=text) for text in texts] or [AIMessage("ok")]
    )


def failing(name: str, error: Exception, *, failures: int = 1_000) -> FlakyModel:
    return FlakyModel(
        model=name, responses=[AIMessage("recovered")], error=error, failures=failures
    )


def task_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "task", "id": "t1", "args": {"description": "go", "subagent_type": "helper"}}
        ],
    )


def todo_call() -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "write_todos", "id": "w1", "args": {"todos": []}}]
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents_talon.model_fallback._backoff", lambda _attempt: 0)


def make_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: dict[str, Any],
    *,
    fallbacks: str = "test:backup",
    max_retries: int = 3,
    **kwargs: Any,
) -> DeepAgentRuntime:
    builds: list[str] = []

    def resolve(spec: str, *_args: object, **_kwargs: object) -> object:
        builds.append(spec)
        model = models[spec]
        if isinstance(model, Exception):
            raise model
        return model

    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", resolve)
    runtime = DeepAgentRuntime(
        model="test:primary",
        assistant_dir=tmp_path,
        approval_store=ToolApprovalStore(tmp_path / "tools.json"),
        include_web_tools=False,
        skills=(),
        memory=(),
        max_retries=max_retries,
        env={FALLBACKS_ENV_KEY: fallbacks},
        **kwargs,
    )
    runtime.builds = builds  # type: ignore[attr-defined]  # test-only build log
    return runtime


async def ask(runtime: DeepAgentRuntime, text: str = "hi", **kwargs: Any) -> str:
    return (await runtime.invoke(AgentRequest(conversation_id="chat", text=text, **kwargs))).text


@pytest.mark.parametrize(
    "error",
    [StatusError(503), StatusError(529, _ANTHROPIC_OVERLOADED), RuntimeError("Overloaded")],
    ids=["503", "anthropic-529", "statusless-overloaded"],
)
async def test_retryable_error_falls_back_after_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    primary = failing("primary", error)
    backup = answer("backup", "from backup")
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()

    assert await ask(runtime) == "from backup"
    assert len(primary.seen) == 3
    assert len(backup.seen) == 1


async def test_primary_that_recovers_within_retries_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(429), failures=2)
    backup = answer("backup")
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()

    assert await ask(runtime) == "recovered"
    assert backup.seen == []


@pytest.mark.parametrize(
    "error",
    [StatusError(400, "invalid request body"), StatusError(401, "invalid api key")],
)
async def test_non_retryable_error_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    primary = failing("primary", error)
    backup = answer("backup")
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()

    with pytest.raises(StatusError) as raised:
        await ask(runtime)

    assert raised.value is error
    assert len(primary.seen) == 1
    assert backup.seen == []


async def test_context_overflow_is_left_to_summarization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(400, "maximum context length exceeded"))
    backup = answer("backup")
    runtime = make_runtime(
        tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup}, max_retries=1
    )
    await runtime.start()

    # Summarization recovery saw the overflow and gave up with its own terminal error.
    with pytest.raises(ContextOverflowError):
        await ask(runtime)

    assert backup.seen == []


async def test_fallback_is_sticky_within_a_turn_and_resets_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(503))
    backup = FlakyModel(
        model="backup", responses=[todo_call(), AIMessage("first"), AIMessage("second")]
    )
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()

    assert await ask(runtime) == "first"
    assert len(primary.seen) == 3
    assert len(backup.seen) == 2

    assert await ask(runtime) == "second"
    assert len(primary.seen) == 6


async def test_exhausted_chain_is_not_rerun_by_the_turn_retry_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(503))
    backup = failing("backup", StatusError(502))
    runtime = make_runtime(
        tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup}, max_retries=2
    )
    await runtime.start()

    with pytest.raises(ModelFallbackExhaustedError) as raised:
        await ask(runtime)

    assert isinstance(raised.value.__cause__, StatusError)
    assert len(primary.seen) == 2
    assert len(backup.seen) == 2


async def test_chat_is_told_once_per_turn_when_a_fallback_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(503))
    backup = FlakyModel(model="backup", responses=[todo_call(), AIMessage("done")])
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()
    notices: list[str] = []

    async def message_handler(text: str) -> SendResult:
        notices.append(text)
        return SendResult(success=True)

    assert await ask(runtime, message_handler=message_handler) == "done"
    assert notices == ["Primary model unavailable; answering with `backup`."]


async def test_fallback_models_are_built_on_first_use_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(503))
    backup = answer("backup", "one", "two")
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup})
    await runtime.start()
    builds: Callable[[], int] = lambda: runtime.builds.count("test:backup")  # type: ignore[attr-defined]  # noqa: E731

    assert builds() == 0
    await ask(runtime)
    await ask(runtime)
    assert builds() == 1


async def test_unbuildable_fallback_moves_on_to_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(503))
    backup = answer("backup", "from backup")
    models = {
        "test:primary": primary,
        "test:missing": ImportError("provider package not installed"),
        "test:backup": backup,
    }
    runtime = make_runtime(tmp_path, monkeypatch, models, fallbacks="test:missing,test:backup")
    await runtime.start()

    assert await ask(runtime) == "from backup"


async def test_local_subagent_uses_the_fallback_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = FlakyModel(model="main", responses=[task_call(), AIMessage("delegated")])
    sub = failing("sub", StatusError(503))
    backup = answer("backup", "subagent answer")
    spec = {
        "name": "helper",
        "description": "Helps.",
        "system_prompt": "Help.",
        "model": "test:sub",
    }
    runtime = make_runtime(
        tmp_path,
        monkeypatch,
        {"test:primary": main, "test:sub": sub, "test:backup": backup},
        subagents=[spec],
    )
    await runtime.start()

    assert await ask(runtime) == "delegated"
    assert len(sub.seen) == 3
    assert len(backup.seen) == 1


async def test_fallback_from_anthropic_carries_no_cache_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = FlakyAnthropic(model="claude-test", api_key="test-key")
    backup = answer("backup", "from backup")
    runtime = make_runtime(
        tmp_path, monkeypatch, {"test:primary": primary, "test:backup": backup}, max_retries=1
    )
    await runtime.start()

    assert await ask(runtime) == "from backup"
    assert "cache_control" in primary.settings[0]
    assert all("cache_control" not in settings for settings in backup.settings)
    assert "cache_control" not in repr(backup.seen)


def test_malformed_fallback_spec_fails_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=FALLBACKS_ENV_KEY):
        make_runtime(tmp_path, monkeypatch, {}, fallbacks="gpt-5")


async def test_no_fallbacks_configured_keeps_the_middleware_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = failing("primary", StatusError(401))
    runtime = make_runtime(tmp_path, monkeypatch, {"test:primary": primary}, fallbacks="")
    await runtime.start()

    with pytest.raises(StatusError):
        await ask(runtime)

    assert len(primary.seen) == 1
