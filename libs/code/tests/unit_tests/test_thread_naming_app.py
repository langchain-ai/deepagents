"""Thread-name command, background generation, and confirmation behavior."""

from __future__ import annotations

import asyncio
import contextvars
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.widgets import Input, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.model_config import ThreadConfig
from deepagents_code.tui.modals.thread_name import ThreadNameScreen


@pytest.fixture
def naming_app(monkeypatch: pytest.MonkeyPatch) -> DeepAgentsApp:
    app = DeepAgentsApp(thread_id="original")
    monkeypatch.setattr(app, "notify", MagicMock())
    monkeypatch.setattr(app, "_effective_model_spec", lambda: "provider:chat")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(
            return_value={
                "messages": [HumanMessage("Fix caching"), AIMessage("Here is the fix")]
            }
        ),
    )
    monkeypatch.setattr(
        "deepagents_code.sessions.get_thread_name", AsyncMock(return_value=None)
    )
    return app


async def test_manual_rename_updates_active_name(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    await naming_app._handle_command("/rename Cache invalidation")
    rename.assert_awaited_once_with("original", "Cache invalidation")
    assert naming_app._thread_name == "Cache invalidation"


async def test_invalid_manual_name_is_not_saved(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = AsyncMock()
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    await naming_app._handle_command("/rename " + "x" * 51)
    rename.assert_not_awaited()


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("rename_model", ["", "openai:test-titles", "openai:test-chat"])
async def test_naming_uses_the_selected_models_endpoint(
    naming_app: DeepAgentsApp,
    monkeypatch: pytest.MonkeyPatch,
    rename_model: str,
    *,
    automatic: bool,
) -> None:
    """Inherit the chat endpoint only when no dedicated naming model is set."""
    requests: list[tuple[str, str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), json.loads(request.content)["model"]))
        return httpx.Response(
            200,
            json={
                "id": "title-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Cache repair"},
                    }
                ],
            },
        )

    monkeypatch.setattr(naming_app, "_effective_model_spec", lambda: "openai:test-chat")
    naming_app._model_params_override = {"base_url": "https://chat.example/v1"}
    monkeypatch.setattr(
        "deepagents_code.model_config.load_thread_config",
        lambda: ThreadConfig(
            {}, True, "updated_at", "cwd", auto_rename=True, rename_model=rename_model
        ),
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.apply_stored_credentials", lambda _: None
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.resolve_provider_credential", lambda _: None
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.has_provider_credentials", lambda _: True
    )
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "deepagents_code.config._get_provider_kwargs",
            lambda *_args, **_kwargs: {
                "api_key": "test-only-placeholder",
                "base_url": "https://configured.example/v1",
                "http_async_client": client,
                "use_responses_api": False,
            },
        )
        async with naming_app.run_test() as pilot:
            if automatic:
                naming_app._maybe_auto_name_thread(
                    "original",
                    "openai:test-chat",
                    model_params=naming_app._model_params_override,
                )
            else:
                await naming_app._handle_command("/rename")
            # A later model switch must not redirect the queued naming request.
            naming_app._model_params_override["base_url"] = "https://later.example/v1"
            await asyncio.gather(*naming_app._thread_name_tasks.values())
            await pilot.pause()
            if not automatic:
                assert isinstance(naming_app.screen, ThreadNameScreen)
                await pilot.press("enter")
                await pilot.pause()
            assert rename.await_args is not None
            assert rename.await_args.args == ("original", "Cache repair")

    endpoint = "configured" if rename_model else "chat"
    model = rename_model.removeprefix("openai:") if rename_model else "test-chat"
    assert requests == [(f"https://{endpoint}.example/v1/chat/completions", model)]


async def test_generated_name_can_be_edited_and_confirmed(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    monkeypatch.setattr(
        "deepagents_code.thread_titles.generate_thread_name",
        AsyncMock(return_value="Cache repair"),
    )
    async with naming_app.run_test() as pilot:
        await naming_app._handle_command("/rename")
        await asyncio.gather(*naming_app._thread_name_tasks.values())
        await pilot.pause()
        assert isinstance(naming_app.screen, ThreadNameScreen)
        rename.assert_not_awaited()
        field = naming_app.screen.query_one(Input)
        field.value = "Cache invalidation"
        await pilot.press("shift+tab")
        assert naming_app.screen.focused is field
        await pilot.press("enter")
        await pilot.pause()
        rename.assert_awaited_once_with("original", "Cache invalidation")
        assert naming_app._thread_name == "Cache invalidation"


async def test_generated_name_cancel_does_not_save(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    rename = AsyncMock()
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        naming_app._offer_thread_name("original", "Cache repair")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        rename.assert_not_awaited()
        assert not isinstance(naming_app.screen, ThreadNameScreen)


async def test_generation_does_not_offer_on_switched_thread(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def generate(*_args: object, **_kwargs: object) -> str:
        started.set()
        await release.wait()
        return "Old conversation"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    offer = MagicMock()
    monkeypatch.setattr(naming_app, "_offer_thread_name", offer)
    naming_app._start_thread_name_generation(
        "original", "provider:chat", automatic=False
    )
    task = naming_app._thread_name_tasks["original"]
    await started.wait()
    naming_app._lc_thread_id = "new"
    release.set()
    await task
    offer.assert_not_called()


async def test_auto_name_is_conditional_and_context_isolated(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = contextvars.ContextVar("stream", default="isolated")
    context.set("chat-stream")
    contexts: list[str] = []
    specs: list[object] = []

    async def generate(*args: object, **_kwargs: object) -> str:
        await asyncio.sleep(0)
        contexts.append(context.get())
        specs.append(args[0])
        return "Cache repair"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(
        "deepagents_code.model_config.load_thread_config",
        lambda: ThreadConfig(
            {},
            True,
            "updated_at",
            "cwd",
            auto_rename=True,
            rename_model="provider:titles",
        ),
    )
    rename = AsyncMock(return_value=False)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    task = naming_app._thread_name_tasks["original"]
    await task
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    assert contexts == ["isolated"]
    assert specs == ["provider:titles"]
    rename.assert_awaited_once_with("original", "Cache repair", only_if_unnamed=True)


async def test_auto_name_skips_resumed_conversation(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        naming_app,
        "_get_thread_state_values",
        AsyncMock(
            return_value={
                "messages": [
                    HumanMessage("First"),
                    AIMessage("Reply"),
                    HumanMessage("Second"),
                    AIMessage("Reply"),
                ]
            }
        ),
    )
    generate = AsyncMock()
    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    await naming_app._generate_thread_name("original", "provider:chat", automatic=True)
    generate.assert_not_awaited()


def test_auto_name_disabled_by_default(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.load_thread_config",
        lambda: ThreadConfig({}, True, "updated_at", "cwd"),
    )
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    assert not naming_app._thread_name_tasks


async def test_load_name_does_not_overwrite_new_thread(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def get_name(_thread_id: str) -> str:
        started.set()
        await release.wait()
        return "Old name"

    monkeypatch.setattr("deepagents_code.sessions.get_thread_name", get_name)
    task = asyncio.create_task(naming_app._load_thread_name())
    await started.wait()
    naming_app._lc_thread_id = "new"
    naming_app._thread_name = "New name"
    release.set()
    await task
    assert naming_app._thread_name == "New name"


@pytest.mark.parametrize("completed", [False, True])
async def test_auto_naming_waits_for_completed_response(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, *, completed: bool
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    adapter = MagicMock(stream_completed=False)
    model_params = {"base_url": "https://chat.example/v1"}
    naming_app._model_params_override = model_params

    async def execute(*_args: object, **_kwargs: object) -> None:
        started.set()
        await release.wait()
        adapter.stream_completed = completed
        model_params["base_url"] = "https://later.example/v1"

    monkeypatch.setattr(naming_app, "_ui_adapter", adapter)
    monkeypatch.setattr(naming_app, "_agent", MagicMock())
    monkeypatch.setattr(naming_app, "_cleanup_agent_task", AsyncMock())
    monkeypatch.setattr(naming_app, "_sync_session_cost_from_checkpoint", AsyncMock())
    monkeypatch.setattr(naming_app, "_refresh_cache_timing", AsyncMock())
    monkeypatch.setattr(
        "deepagents_code.tui.textual_adapter.execute_task_textual", execute
    )
    schedule = MagicMock()
    monkeypatch.setattr(naming_app, "_maybe_auto_name_thread", schedule)
    async with naming_app.run_test():
        task = asyncio.create_task(naming_app._run_agent_task("Fix caching"))
        await asyncio.wait_for(started.wait(), timeout=5)
        schedule.assert_not_called()
        release.set()
        await task
        if completed:
            schedule.assert_called_once_with(
                "original",
                "provider:chat",
                model_params={"base_url": "https://chat.example/v1"},
            )
        else:
            schedule.assert_not_called()


async def test_manual_name_cancels_pending_proposal(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()

    async def generate(*_args: object, **_kwargs: object) -> str:
        started.set()
        await asyncio.Event().wait()
        return "Stale proposal"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(
        "deepagents_code.sessions.rename_thread", AsyncMock(return_value=True)
    )
    naming_app._start_thread_name_generation(
        "original", "provider:chat", automatic=False
    )
    task = naming_app._thread_name_tasks["original"]
    await asyncio.wait_for(started.wait(), timeout=5)
    await naming_app._rename_current_thread("My choice")
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert naming_app._thread_name == "My choice"


async def test_auto_name_ignores_internal_human_messages(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    visible = [HumanMessage("Fix caching"), AIMessage("Here is the fix")]
    monkeypatch.setattr(
        naming_app,
        "_get_thread_state_values",
        AsyncMock(
            return_value={
                "messages": [
                    visible[0],
                    HumanMessage(
                        "Goal changed", additional_kwargs={"lc_source": "goal_state"}
                    ),
                    HumanMessage(
                        "Hidden context",
                        additional_kwargs={"lc_source": "local_context"},
                    ),
                    visible[1],
                ]
            }
        ),
    )
    generate = AsyncMock(return_value="Cache repair")
    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(
        "deepagents_code.sessions.rename_thread", AsyncMock(return_value=True)
    )
    await naming_app._generate_thread_name("original", "provider:chat", automatic=True)
    assert generate.await_count == 1
    assert generate.await_args is not None
    assert generate.await_args.args[1] == visible


async def test_stale_load_cannot_overwrite_manual_name(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def get_name(_thread_id: str) -> str:
        started.set()
        await release.wait()
        return "Old name"

    monkeypatch.setattr("deepagents_code.sessions.get_thread_name", get_name)
    monkeypatch.setattr(
        "deepagents_code.sessions.rename_thread", AsyncMock(return_value=True)
    )
    task = asyncio.create_task(naming_app._load_thread_name())
    await started.wait()
    await naming_app._save_thread_name("original", "Manual name")
    release.set()
    await task
    assert naming_app._thread_name == "Manual name"


async def test_auto_name_refreshes_open_thread_selector(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.tui.widgets.thread_selector import ThreadSelectorScreen

    thread = {"thread_id": "original", "updated_at": None, "thread_name": None}
    monkeypatch.setattr(
        "deepagents_code.sessions.list_threads",
        AsyncMock(side_effect=lambda **_: [dict(thread)]),
    )
    monkeypatch.setattr(
        ThreadSelectorScreen, "_load_available_agent_names", AsyncMock()
    )
    monkeypatch.setattr(
        "deepagents_code.thread_titles.generate_thread_name",
        AsyncMock(return_value="Cache repair"),
    )

    async def rename(*_args: object, **_kwargs: object) -> bool:
        await asyncio.sleep(0)
        thread["thread_name"] = "Cache repair"
        return True

    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        selector = ThreadSelectorScreen(filter_cwd=None)
        naming_app.push_screen(selector)
        await pilot.pause()
        name_cell = "ThreadOption .thread-cell-thread_name"
        assert str(selector.query_one(name_cell, Static).render()) == ""
        await naming_app._generate_thread_name(
            "original", "provider:chat", automatic=True
        )
        await pilot.pause()
        assert str(selector.query_one(name_cell, Static).render()) == "Cache repair"
