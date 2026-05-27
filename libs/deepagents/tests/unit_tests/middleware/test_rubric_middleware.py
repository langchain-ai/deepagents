"""Unit tests for `RubricMiddleware`.

These tests cover edge cases and pure-function behavior: construction
validation, `before_agent` rubric-change detection, grader-plumbing
internals, transcript building, and rubric-tracking across multi-turn
invocations. The grader is stubbed via `monkeypatch` on
`_grade`/`_agrade` so no real model calls fire.

End-to-end coverage of the happy path, the revision loop, the iteration
cap, the no-rubric no-op, and `KeyboardInterrupt` propagation lives in
`TestRubricMiddlewareEndToEnd` in
`tests/unit_tests/test_end_to_end.py`. That suite uses
`create_deep_agent` with a fake chat model for both the main agent and
the grader sub-agent, so it survives internal refactors that this file's
direct-hook unit tests could not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.outcomes import (
    GraderResponse,
    RubricEvaluation,
    RubricMiddleware,
    _build_grader_transcript,
)
from deepagents.middleware.skills import SkillsMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

# Placeholder model identifier used wherever the grader sub-agent is stubbed
# via `monkeypatch` and the value would never reach a real provider client.
_STUB_MODEL = "stub:test"


def _stub_spec(**overrides: Any) -> dict[str, Any]:
    """Minimal grader SubAgent spec for tests; only `model` is required."""
    spec: dict[str, Any] = {"model": _STUB_MODEL}
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _runtime(events: list[dict[str, Any]] | None = None) -> Any:  # noqa: ANN401
    """Build a minimal stub of the LangGraph runtime.

    `RubricMiddleware` only touches `runtime.stream_writer`, so a
    `SimpleNamespace` is plenty.
    """
    sink = events if events is not None else []
    return SimpleNamespace(stream_writer=sink.append)


def _stub_grader(
    middleware: RubricMiddleware,
    monkeypatch: pytest.MonkeyPatch,
    *responses: GraderResponse,
    exc: BaseException | None = None,
) -> list[int]:
    """Wire `_grade` (and `_agrade`) to return canned responses in order.

    Returns a counter list whose length grows by one each time the grader
    is invoked. Useful for asserting iteration count.
    """
    call_log: list[int] = []
    iterator = iter(responses)

    def _grade(state: dict[str, Any], iteration: int) -> GraderResponse:  # noqa: ARG001
        if exc is not None:
            raise exc
        call_log.append(iteration)
        return next(iterator)

    async def _agrade(state: dict[str, Any], iteration: int) -> GraderResponse:  # noqa: ARG001
        if exc is not None:
            raise exc
        call_log.append(iteration)
        return next(iterator)

    monkeypatch.setattr(middleware, "_grade", _grade)
    monkeypatch.setattr(middleware, "_agrade", _agrade)
    return call_log


# ---------------------------------------------------------------------- #
# Construction / validation
# ---------------------------------------------------------------------- #


class TestConstruction:
    def test_defaults(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        assert mw.max_iterations == 3
        assert mw._grader_spec["model"] == _STUB_MODEL
        assert mw._backend is None
        assert mw._fallback_model is None
        assert mw._fallback_backend is None

    def test_grader_omitted_uses_empty_spec(self) -> None:
        # `grader` is optional; omitting it means "use all defaults"
        # (main agent's model captured at runtime, no tools, default
        # grader prompt). Construction succeeds; the spec is an empty
        # dict internally so later `spec.get(...)` calls return None.
        mw = RubricMiddleware()
        assert mw._grader_spec == {}

    def test_grader_none_uses_empty_spec(self) -> None:
        # Same as omitting -- `None` is normalized to an empty spec.
        mw = RubricMiddleware(grader=None)
        assert mw._grader_spec == {}

    def test_grader_empty_dict_accepted(self) -> None:
        # An empty dict at construction is fine -- model and backend
        # resolution is deferred to `_ensure_grader`.
        mw = RubricMiddleware(grader={})  # type: ignore[typeddict-item]
        assert mw._grader_spec == {}

    def test_grader_spec_missing_model_construction_accepted(self) -> None:
        # No construction-time error -- the fallback model from the main
        # agent might be available at grader-construction time.
        mw = RubricMiddleware(grader={"name": "g"})  # type: ignore[typeddict-item]
        assert mw._grader_spec.get("name") == "g"

    def test_grader_skills_without_backend_construction_accepted(self) -> None:
        # No construction-time error -- the fallback backend from
        # create_deep_agent might be injected before `_ensure_grader` runs.
        mw = RubricMiddleware(grader=_stub_spec(skills=["/skills/grading/"]))
        assert mw._grader_spec.get("skills") == ["/skills/grading/"]

    def test_grader_permissions_without_backend_construction_accepted(self) -> None:
        # Same: no construction-time error for permissions either.
        rules = [FilesystemPermission(operations=["read"], paths=["/**"], mode="allow")]
        mw = RubricMiddleware(grader=_stub_spec(permissions=rules))
        assert mw._grader_spec.get("permissions") == rules

    def test_max_iterations_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            RubricMiddleware(grader=_stub_spec(), max_iterations=0)

    def test_max_iterations_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            RubricMiddleware(grader=_stub_spec(), max_iterations=21)

    def test_max_iterations_bool_rejected(self) -> None:
        # bool is a subclass of int; reject explicitly so True/False can't
        # silently configure the cap.
        with pytest.raises(TypeError):
            RubricMiddleware(grader=_stub_spec(), max_iterations=True)  # type: ignore[arg-type]

    def test_max_iterations_non_int_rejected(self) -> None:
        with pytest.raises(TypeError):
            RubricMiddleware(grader=_stub_spec(), max_iterations="3")  # type: ignore[arg-type]

    def test_grader_tools_propagated(self) -> None:
        @tool
        def my_tool(query: str) -> str:
            """A tool."""
            return query

        mw = RubricMiddleware(grader=_stub_spec(tools=[my_tool]))
        assert mw._grader_spec.get("tools") == [my_tool]


# ---------------------------------------------------------------------- #
# before_agent semantics
# ---------------------------------------------------------------------- #


class TestBeforeAgent:
    def test_no_rubric_is_noop(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        result = mw.before_agent({"messages": []}, _runtime())
        assert result is None

    def test_new_rubric_mints_attempt(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        result = mw.before_agent({"messages": [], "rubric": "- ship it"}, _runtime())
        assert result is not None
        assert result["_rubric_iterations"] == 0
        assert result["_rubric_status"] is None
        assert result["_active_rubric"] == "- ship it"
        assert isinstance(result["_current_rubric_id"], str)
        assert result["_current_rubric_id"]  # non-empty

    def test_sticky_rubric_is_noop(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        state = {
            "messages": [],
            "rubric": "- ship it",
            "_active_rubric": "- ship it",
            "_current_rubric_id": "rubric-1",
            "_rubric_iterations": 2,
        }
        assert mw.before_agent(state, _runtime()) is None

    def test_new_rubric_resets_existing_attempt(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        state = {
            "messages": [],
            "rubric": "- write a limerick",
            "_active_rubric": "- write a haiku",
            "_current_rubric_id": "rubric-prev",
            "_rubric_iterations": 5,
            "_rubric_status": "satisfied",
        }
        result = mw.before_agent(state, _runtime())
        assert result is not None
        assert result["_rubric_iterations"] == 0
        assert result["_rubric_status"] is None
        assert result["_active_rubric"] == "- write a limerick"
        assert result["_current_rubric_id"] != "rubric-prev"

    @pytest.mark.asyncio
    async def test_abefore_agent_matches_sync(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        result = await mw.abefore_agent({"messages": [], "rubric": "- be terse"}, _runtime())
        assert result is not None
        assert result["_active_rubric"] == "- be terse"


# ---------------------------------------------------------------------- #
# after_agent semantics — direct hook invocation
# ---------------------------------------------------------------------- #


class TestAfterAgentDirect:
    def _state(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "messages": [
                HumanMessage(content="Build a thing"),
                AIMessage(content="Done."),
            ],
            "rubric": "- The thing is built",
            "_active_rubric": "- The thing is built",
            "_current_rubric_id": "rubric-direct",
            "_rubric_iterations": 0,
        }
        base.update(overrides)
        return base

    def test_grader_failed_status_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(
            mw,
            monkeypatch,
            GraderResponse(
                result="failed",
                explanation="Rubric is contradictory.",
                criteria=[],
            ),
        )
        update = mw.after_agent(self._state(), _runtime())
        assert update is not None
        assert update["_rubric_status"] == "failed"
        assert "jump_to" not in update

    def test_grader_exception_becomes_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(mw, monkeypatch, exc=RuntimeError("grader exploded"))
        update = mw.after_agent(self._state(), _runtime())
        assert update is not None
        assert update["_rubric_status"] == "failed"
        assert "jump_to" not in update
        evals = update["_rubric_evaluations"]
        assert len(evals) == 1
        assert evals[0]["result"] == "failed"
        assert "grader exploded" in evals[0]["explanation"]

    def test_keyboard_interrupt_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `KeyboardInterrupt` (and `asyncio.CancelledError`) are
        # `BaseException` subclasses, not `Exception`. They must propagate
        # out of `after_agent` so Ctrl+C / task cancellation actually stop
        # execution instead of being swallowed into an evaluation record.
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(mw, monkeypatch, exc=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            mw.after_agent(self._state(), _runtime())

    def test_on_evaluation_callback_fires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[RubricEvaluation] = []
        mw = RubricMiddleware(
            grader=_stub_spec(),
            max_iterations=3,
            on_evaluation=seen.append,
        )
        _stub_grader(
            mw,
            monkeypatch,
            GraderResponse(result="satisfied", explanation="ok", criteria=[]),
        )
        mw.after_agent(self._state(), _runtime())
        assert len(seen) == 1
        assert seen[0]["result"] == "satisfied"

    def test_stream_events_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[dict[str, Any]] = []
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(
            mw,
            monkeypatch,
            GraderResponse(result="satisfied", explanation="ok", criteria=[]),
        )
        mw.after_agent(self._state(), _runtime(events))
        types = [e["type"] for e in events]
        assert types == ["rubric_evaluation_start", "rubric_evaluation_end"]
        assert events[0]["rubric_id"] == "rubric-direct"
        assert events[0]["iteration"] == 0
        assert events[1]["result"] == "satisfied"


# ---------------------------------------------------------------------- #
# Grader plumbing
# ---------------------------------------------------------------------- #


class TestGraderPlumbing:
    def test_pure_llm_grader_constructed_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A grader with no tools is built only when first needed."""
        built: list[dict[str, Any]] = []

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]
            built.append(
                {
                    "model": model,
                    "system_prompt": system_prompt,
                    "tools": list(tools),
                    "middleware": list(middleware),
                    "name": name,
                    "response_format": response_format,
                }
            )
            return SimpleNamespace(
                invoke=lambda _payload: {
                    "messages": [],
                    "structured_response": GraderResponse(result="satisfied", explanation="ok", criteria=[]),
                },
                ainvoke=None,
            )

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        # `resolve_model` is imported lazily inside `_ensure_grader`; patch
        # at its source so the stub model string never hits init_chat_model.
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)
        mw = RubricMiddleware(grader=_stub_spec())
        assert not built  # nothing constructed yet
        mw._ensure_grader()
        assert len(built) == 1
        assert built[0]["tools"] == []
        assert built[0]["middleware"] == []
        assert built[0]["name"] == "rubric_grader"
        assert built[0]["response_format"] is GraderResponse
        # Trust-boundary language is preserved in the grader prompt so
        # adversarial transcript content can't redirect grading.
        prompt = built[0]["system_prompt"]
        assert "adversarial" in prompt
        assert "Trust only `<rubric>`" in prompt
        # idempotent
        mw._ensure_grader()
        assert len(built) == 1

    def test_grader_tools_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        @tool
        def shell(cmd: str) -> str:
            """Run a shell command."""
            return f"$ {cmd}\n(no-op)"

        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["tools"] = list(tools)
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)
        mw = RubricMiddleware(grader=_stub_spec(tools=[shell]))
        mw._ensure_grader()
        assert seen["tools"] == [shell]

    def test_model_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["model"] = model
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)
        mw = RubricMiddleware(grader={"model": "custom-grader-model"})
        mw._ensure_grader()
        assert seen["model"] == "custom-grader-model"

    def test_custom_name_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A user-supplied `name` on the grader spec replaces the default trace name."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["name"] = name
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)
        mw = RubricMiddleware(grader=_stub_spec(name="code-review-grader"))
        mw._ensure_grader()
        assert seen["name"] == "code-review-grader"

    def test_custom_system_prompt_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A user-supplied `system_prompt` replaces the default grader prompt."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["system_prompt"] = system_prompt
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)
        mw = RubricMiddleware(
            grader=_stub_spec(system_prompt="OVERRIDE_MARKER: be strict."),
        )
        mw._ensure_grader()
        assert seen["system_prompt"] == "OVERRIDE_MARKER: be strict."

    def test_user_middleware_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Middleware on the grader spec lands in the create_agent call."""

        class _DummyMiddleware(AgentMiddleware):
            pass

        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["middleware"] = list(middleware)
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        custom = _DummyMiddleware()
        mw = RubricMiddleware(grader=_stub_spec(middleware=[custom]))
        mw._ensure_grader()
        assert custom in seen["middleware"]

    def test_skills_auto_wires_skills_middleware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`skills` on the grader spec auto-appends `SkillsMiddleware(backend=...)`."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["middleware"] = list(middleware)
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        backend = StateBackend()
        mw = RubricMiddleware(
            grader=_stub_spec(skills=["/skills/grading/"]),
            backend=backend,
        )
        mw._ensure_grader()
        # SkillsMiddleware should be in the middleware list.
        skills_mw = [m for m in seen["middleware"] if isinstance(m, SkillsMiddleware)]
        assert len(skills_mw) == 1

    def test_permissions_auto_wires_filesystem_middleware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`permissions` on the grader spec auto-appends `FilesystemMiddleware(_permissions=...)`."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["middleware"] = list(middleware)
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        backend = StateBackend()
        rules = [FilesystemPermission(operations=["read"], paths=["/grading/**"], mode="allow")]
        mw = RubricMiddleware(
            grader=_stub_spec(permissions=rules),
            backend=backend,
        )
        mw._ensure_grader()
        fs_mw = [m for m in seen["middleware"] if isinstance(m, FilesystemMiddleware)]
        assert len(fs_mw) == 1

    def test_no_model_anywhere_raises_at_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When neither the spec nor the captured main model has a model, `_ensure_grader` raises."""
        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", lambda **_: SimpleNamespace())
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        mw = RubricMiddleware()  # no grader spec; no main-agent call has happened
        with pytest.raises(RuntimeError, match="no grader model available"):
            mw._ensure_grader()

    def test_model_falls_back_to_captured_main_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the spec omits `model`, the grader uses the model captured by wrap_model_call."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["model"] = model
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        sentinel_main_model = SimpleNamespace(name="captured-main-model")
        mw = RubricMiddleware()  # no grader spec at all

        # Simulate the main agent calling its model -- wrap_model_call
        # captures the model reference into `_fallback_model`.
        request = SimpleNamespace(model=sentinel_main_model)
        called: list[bool] = []

        def handler(_req: object) -> object:
            called.append(True)
            return SimpleNamespace()

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        assert called == [True]
        assert mw._fallback_model is sentinel_main_model

        mw._ensure_grader()
        assert seen["model"] is sentinel_main_model

    def test_explicit_model_wins_over_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the spec specifies `model`, it takes precedence over the captured main model."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["model"] = model
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        mw = RubricMiddleware(grader={"model": "explicit-grader-model"})
        mw.wrap_model_call(SimpleNamespace(model="main-model"), lambda _req: SimpleNamespace())  # type: ignore[arg-type]

        mw._ensure_grader()
        assert seen["model"] == "explicit-grader-model"

    def test_skills_without_any_backend_raises_at_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skills configured but no backend (explicit or fallback) -> RuntimeError at grader-construction."""
        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", lambda **_: SimpleNamespace())
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        mw = RubricMiddleware(grader=_stub_spec(skills=["/skills/grading/"]))
        # No explicit backend, no fallback injected by create_deep_agent.
        with pytest.raises(RuntimeError, match="require a backend"):
            mw._ensure_grader()

    def test_permissions_without_any_backend_raises_at_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Permissions configured but no backend -> RuntimeError at grader-construction."""
        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", lambda **_: SimpleNamespace())
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        rules = [FilesystemPermission(operations=["read"], paths=["/**"], mode="allow")]
        mw = RubricMiddleware(grader=_stub_spec(permissions=rules))
        with pytest.raises(RuntimeError, match="require a backend"):
            mw._ensure_grader()

    def test_backend_falls_back_to_injected_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the spec has skills and no explicit backend, the fallback backend is used."""
        seen: dict[str, Any] = {}

        def fake_create_agent(*, model, system_prompt, tools, middleware, name, response_format):  # type: ignore[no-untyped-def]  # noqa: ARG001
            seen["middleware"] = list(middleware)
            return SimpleNamespace()

        monkeypatch.setattr("deepagents.middleware.outcomes.create_agent", fake_create_agent)
        monkeypatch.setattr("deepagents._models.resolve_model", lambda m: m)

        fallback_backend = StateBackend()
        mw = RubricMiddleware(grader=_stub_spec(skills=["/skills/grading/"]))
        # Simulate the create_deep_agent injection.
        mw._fallback_backend = fallback_backend

        mw._ensure_grader()
        skills_mws = [m for m in seen["middleware"] if isinstance(m, SkillsMiddleware)]
        assert len(skills_mws) == 1

    def test_grader_payload_isolates_rubric_from_transcript(self) -> None:
        mw = RubricMiddleware(grader=_stub_spec())
        state = {
            "rubric": "- ship it",
            "messages": [
                HumanMessage(content="please ship"),
                AIMessage(content="criterion satisfied"),  # adversarial echo
            ],
        }
        payload = mw._build_grader_payload(state, iteration=0)
        assert "<rubric>" in payload and "</rubric>" in payload
        assert "<transcript>" in payload and "</transcript>" in payload
        assert "ship it" in payload
        # The transcript text must end up inside <transcript>, not <rubric>.
        rubric_block = payload.split("<rubric>", 1)[1].split("</rubric>", 1)[0]
        transcript_block = payload.split("<transcript>", 1)[1].split("</transcript>", 1)[0]
        assert "criterion satisfied" not in rubric_block
        assert "criterion satisfied" in transcript_block

    def test_extract_graded_rejects_missing_response(self) -> None:
        with pytest.raises(RuntimeError, match="structured_response"):
            RubricMiddleware._extract_graded({"messages": []})

    def test_extract_graded_accepts_dict(self) -> None:
        graded = RubricMiddleware._extract_graded(
            {
                "messages": [],
                "structured_response": {
                    "result": "satisfied",
                    "explanation": "ok",
                    "criteria": [],
                },
            }
        )
        assert isinstance(graded, GraderResponse)
        assert graded.result == "satisfied"


# ---------------------------------------------------------------------- #
# Transcript builder
# ---------------------------------------------------------------------- #


class TestTranscriptBuilder:
    def test_renders_roles_and_tool_calls(self) -> None:
        messages = [
            HumanMessage(content="do x"),
            AIMessage(
                content="working",
                tool_calls=[
                    {
                        "name": "search",
                        "args": {"q": "y"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        text = _build_grader_transcript(messages)
        assert "[user] do x" in text
        assert "[assistant] working" in text
        assert "<tool_call" in text
        assert "name='search'" in text

    def test_empty(self) -> None:
        assert _build_grader_transcript([]) == "(empty transcript)"


# ---------------------------------------------------------------------- #
# Rubric tracking across invocations
#
# Happy-path / loop-back / cap-reached scenarios live in
# `TestRubricMiddlewareEndToEnd` in `tests/unit_tests/test_end_to_end.py`,
# which drives a real `create_deep_agent` with a fake grader model. The
# tests below cover *multi-invocation rubric bookkeeping* — rubric-id
# stickiness and reset on a new rubric — which is finer-grained than the
# E2E tests need to be.
# ---------------------------------------------------------------------- #


class TestRubricTracking:
    """Rubric stickiness and rubric-id minting across multiple `agent.invoke` calls.

    The grader is stubbed via `_stub_grader` so these tests stay focused on
    `before_agent`'s rubric-change detection, not on grader plumbing
    (covered by `TestGraderPlumbing` and the E2E suite).
    """

    def test_sticky_rubric_across_invocations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="first"),
                    AIMessage(content="second"),
                ]
            )
        )
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(
            mw,
            monkeypatch,
            GraderResponse(result="satisfied", explanation="ok", criteria=[]),
            GraderResponse(result="satisfied", explanation="still ok", criteria=[]),
        )
        agent = create_agent(
            model=agent_model,
            tools=[],
            middleware=[mw],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "session-stick"}}

        # First invocation supplies the rubric.
        agent.invoke(
            {"messages": [HumanMessage("do it")], "rubric": "- be terse"},
            config=config,
        )
        first_evals = agent.get_state(config).values["_rubric_evaluations"]
        first_id = first_evals[0]["rubric_id"]

        # Second invocation omits the rubric — sticky from the prior call.
        agent.invoke({"messages": [HumanMessage("again")]}, config=config)
        second_evals = agent.get_state(config).values["_rubric_evaluations"]
        assert len(second_evals) == 2
        assert second_evals[1]["rubric_id"] == first_id

    def test_new_rubric_mints_new_rubric_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="haiku"),
                    AIMessage(content="limerick"),
                ]
            )
        )
        mw = RubricMiddleware(grader=_stub_spec(), max_iterations=3)
        _stub_grader(
            mw,
            monkeypatch,
            GraderResponse(result="satisfied", explanation="ok", criteria=[]),
            GraderResponse(result="satisfied", explanation="ok", criteria=[]),
        )
        agent = create_agent(
            model=agent_model,
            tools=[],
            middleware=[mw],
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "session-new"}}

        agent.invoke(
            {
                "messages": [HumanMessage("haiku please")],
                "rubric": "- haiku format",
            },
            config=config,
        )
        first_evals = agent.get_state(config).values["_rubric_evaluations"]
        first_id = first_evals[0]["rubric_id"]

        agent.invoke(
            {
                "messages": [HumanMessage("now a limerick")],
                "rubric": "- limerick format",
            },
            config=config,
        )
        second_evals = agent.get_state(config).values["_rubric_evaluations"]
        second_id = second_evals[-1]["rubric_id"]
        assert first_id != second_id
        # Both evaluations are retained across the rubric change.
        assert len(second_evals) == 2
