"""Thread-name command, background generation, and confirmation behavior."""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from textual.widgets import Input, OptionList, Static

from deepagents_code.app import DeepAgentsApp
from deepagents_code.model_config import ThreadConfig
from deepagents_code.tui.modals.thread_name import ThreadNameScreen

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.sessions import ThreadInfo


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


@pytest.mark.parametrize("busy_flag", ["_agent_running", "_shell_running"])
async def test_rename_submission_runs_while_busy(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, busy_flag: str
) -> None:
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        await pilot.pause()
        with monkeypatch.context() as busy:
            busy.setattr(naming_app, busy_flag, True)
            await naming_app._submit_input("/rename Release audit", "command")
            assert naming_app._thread_name == "Release audit"
            assert not naming_app._pending_messages
            rename.assert_awaited_once_with("original", "Release audit")


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TimeoutError(), "Request timed out. Try /rename again."),
        (ValueError(), "ValueError"),
        (ValueError("Provider unavailable"), "Provider unavailable"),
    ],
)
async def test_manual_naming_failure_has_readable_reason(
    naming_app: DeepAgentsApp,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
) -> None:
    notify = MagicMock()
    monkeypatch.setattr(naming_app, "notify", notify)
    monkeypatch.setattr(
        "deepagents_code.thread_titles.generate_thread_name",
        AsyncMock(side_effect=error),
    )

    await naming_app._generate_thread_name("original", "provider:chat", automatic=False)

    notify.assert_called_once_with(
        f"Could not generate a thread name: {reason}", severity="error", markup=False
    )


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


async def test_generated_name_waits_for_thread_selector_input(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed proposal must not steal keys from the thread-selector filter."""
    from deepagents_code.tui.widgets.thread_selector import ThreadSelectorScreen

    started, release = asyncio.Event(), asyncio.Event()

    async def generate(*_args: object, **_kwargs: object) -> str:
        started.set()
        await release.wait()
        return "Cache repair"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(
        "deepagents_code.sessions.list_threads", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("deepagents_code.sessions.get_cached_threads", lambda **_: [])
    monkeypatch.setattr(
        ThreadSelectorScreen, "_load_available_agent_names", AsyncMock()
    )
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        await pilot.press(*"/rename", "enter")
        await asyncio.wait_for(started.wait(), timeout=5)
        task = naming_app._thread_name_tasks["original"]
        await pilot.press(*"/threads", "enter")
        await pilot.pause()
        selector = naming_app.screen
        assert isinstance(selector, ThreadSelectorScreen)
        field = selector.query_one("#thread-filter", Input)
        await pilot.press("c")
        release.set()
        await task
        await pilot.pause()
        await pilot.press("a", "c", "h", "e")
        assert field.value == "cache"
        assert naming_app.screen is selector
        assert selector.focused is field
        rename.assert_not_awaited()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(naming_app.screen, ThreadNameScreen)
        name_field = naming_app.screen.query_one(Input)
        await pilot.press("shift+tab")
        assert naming_app.screen.focused is name_field
        await pilot.press("end", "!", "enter")
        await pilot.pause()
        rename.assert_awaited_once_with("original", "Cache repair!")
        assert naming_app._thread_name == "Cache repair!"


async def test_generated_name_waits_for_nested_auth_modals(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the auth key field must not offer a name over the auth manager."""
    from deepagents_code.tui.widgets.auth import AuthManagerScreen, AuthPromptScreen

    started, release = asyncio.Event(), asyncio.Event()

    async def generate(*_args: object, **_kwargs: object) -> str:
        started.set()
        await release.wait()
        return "Cache repair"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    async with naming_app.run_test() as pilot:
        await pilot.press(*"/rename", "enter")
        await asyncio.wait_for(started.wait(), timeout=5)
        task = naming_app._thread_name_tasks["original"]
        await pilot.press(*"/auth", "enter")
        manager = naming_app.screen
        assert isinstance(manager, AuthManagerScreen)
        options = manager.query_one(OptionList)
        options.highlighted = options.get_option_index("openai")
        await pilot.press("enter")
        prompt = naming_app.screen
        assert isinstance(prompt, AuthPromptScreen)
        release.set()
        await task
        await pilot.pause()
        await pilot.press(*"draft")
        assert naming_app.screen is prompt
        assert prompt.query_one("#auth-prompt-input", Input).value == "draft"
        await pilot.press("escape")
        await pilot.pause()
        assert naming_app.screen is manager
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(naming_app.screen, ThreadNameScreen)
        assert naming_app.screen.query_one(Input).value == "Cache repair"
        await pilot.press("escape")


@pytest.mark.parametrize("manual_name", [False, True])
async def test_deferred_name_is_discarded_when_stale(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, *, manual_name: bool
) -> None:
    """A queued proposal cannot outlive a thread switch or an explicit rename."""
    from deepagents_code.tui.widgets.auth import AuthConfirmScreen

    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    async with naming_app.run_test() as pilot:
        chat_screen = naming_app.screen
        naming_app.push_screen(AuthConfirmScreen(title="Auth", body="Continue?"))
        await pilot.pause()
        naming_app._offer_thread_name("original", "Stale proposal")
        if manual_name:
            await naming_app._save_thread_name("original", "My choice")
        else:
            naming_app._lc_thread_id = "new"
        await pilot.press("escape")
        await pilot.pause()
        assert naming_app.screen is chat_screen
        if manual_name:
            rename.assert_awaited_once_with("original", "My choice")
        else:
            rename.assert_not_awaited()


@pytest.mark.parametrize("automatic", [False, True])
async def test_workspace_switch_cancels_pending_naming(
    naming_app: DeepAgentsApp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    automatic: bool,
) -> None:
    """A delayed state read must not send the old chat to the new workspace."""
    started, release = asyncio.Event(), asyncio.Event()

    async def read_state(_thread_id: str) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"messages": [HumanMessage("Private chat"), AIMessage("Reply")]}

    monkeypatch.setattr(naming_app, "_get_thread_state_values", read_state)
    generate = AsyncMock(return_value="Private chat")
    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(naming_app, "call_after_refresh", MagicMock())
    monkeypatch.setattr(
        naming_app, "_reload_settings_from_environment", AsyncMock(return_value=[])
    )
    monkeypatch.setattr("deepagents_code.model_config.clear_caches", lambda: None)
    naming_app._start_thread_name_generation(
        "original", "provider:chat", automatic=automatic
    )
    task = naming_app._thread_name_tasks["original"]
    await asyncio.wait_for(started.wait(), timeout=5)
    await naming_app._refresh_project_context_for_cwd_switch(tmp_path)
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    generate.assert_not_awaited()
    assert task.cancelled()
    assert not naming_app._thread_name_tasks


@pytest.mark.parametrize("factory_fails", [False, True])
@pytest.mark.parametrize("already_cancelled", [False, True])
async def test_workspace_reload_waits_for_naming_model_initialization(
    naming_app: DeepAgentsApp,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    factory_fails: bool,
    already_cancelled: bool,
) -> None:
    """An old factory cannot mutate provider settings after a workspace reload."""
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    settings = {"workspace": "original"}
    model = AsyncMock()

    def create_model(*_args: object, **_kwargs: object) -> SimpleNamespace:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        settings["workspace"] = "original"
        if factory_fails:
            msg = "Model initialization failed"
            raise ValueError(msg)
        return SimpleNamespace(model=model)

    def reload_settings(*, start_path: Path) -> list[str]:
        settings["workspace"] = str(start_path)
        return []

    reload = AsyncMock(side_effect=reload_settings)
    monkeypatch.setattr("deepagents_code.config.create_model", create_model)
    monkeypatch.setattr(naming_app, "_reload_settings_from_environment", reload)
    monkeypatch.setattr("deepagents_code.model_config.clear_caches", lambda: None)
    naming_app._start_thread_name_generation(
        "original", "provider:chat", automatic=False
    )
    task = naming_app._thread_name_tasks["original"]
    refresh: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if already_cancelled:
            task.cancel()
            await asyncio.sleep(0)
        refresh = asyncio.create_task(
            naming_app._refresh_project_context_for_cwd_switch(tmp_path)
        )
        done, _ = await asyncio.wait({refresh}, timeout=0.05)
        assert not done
        assert not task.done()
        reload.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(
            task, *([refresh] if refresh is not None else []), return_exceptions=True
        )
    await refresh
    assert task.cancelled()
    assert settings["workspace"] == str(tmp_path)
    assert not naming_app._thread_name_tasks
    model.ainvoke.assert_not_awaited()


async def test_naming_cannot_start_during_workspace_reload(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second request during reload cannot read partially changed settings."""
    generate = AsyncMock(return_value="Private chat")
    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    monkeypatch.setattr(naming_app, "call_after_refresh", MagicMock())
    async with naming_app._environment_mutation_lock:
        naming_app._start_thread_name_generation(
            "original", "provider:chat", automatic=False
        )
        await asyncio.gather(*naming_app._thread_name_tasks.values())
    generate.assert_not_awaited()


async def test_auto_name_is_conditional_and_context_isolated(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = contextvars.ContextVar("stream", default="isolated")
    context.set("chat-stream")
    contexts: list[str] = []
    monkeypatch.setattr(
        naming_app,
        "_get_thread_state_values",
        AsyncMock(
            return_value={
                "messages": [
                    HumanMessage("Fix caching"),
                    HumanMessage(
                        "Goal changed", additional_kwargs={"lc_source": "goal_state"}
                    ),
                    HumanMessage(
                        "Hidden context",
                        additional_kwargs={"lc_source": "local_context"},
                    ),
                    AIMessage("Here is the fix"),
                ]
            }
        ),
    )

    async def generate(*_args: object, **_kwargs: object) -> str:
        await asyncio.sleep(0)
        contexts.append(context.get())
        return "Cache repair"

    monkeypatch.setattr("deepagents_code.thread_titles.generate_thread_name", generate)
    rename = AsyncMock(return_value=False)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    task = naming_app._thread_name_tasks["original"]
    await task
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    assert contexts == ["isolated"]
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


def test_auto_name_can_be_disabled(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deepagents_code.model_config.load_thread_config",
        lambda: ThreadConfig({}, True, "updated_at", "cwd", auto_rename=False),
    )
    naming_app._maybe_auto_name_thread("original", "provider:chat")
    assert not naming_app._thread_name_tasks


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
    rename = AsyncMock(return_value=True)
    monkeypatch.setattr("deepagents_code.sessions.rename_thread", rename)
    naming_app._start_thread_name_generation(
        "original", "provider:chat", automatic=False
    )
    task = naming_app._thread_name_tasks["original"]
    await asyncio.wait_for(started.wait(), timeout=5)
    await naming_app._handle_command("/rename My choice")
    await asyncio.gather(task, return_exceptions=True)
    rename.assert_awaited_once_with("original", "My choice")
    assert task.cancelled()
    assert naming_app._thread_name == "My choice"


@pytest.mark.parametrize("manual_name", [False, True])
async def test_stale_load_cannot_overwrite_current_name(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, *, manual_name: bool
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
    if manual_name:
        await naming_app._handle_command("/rename Current name")
    else:
        naming_app._lc_thread_id = "new"
        naming_app._thread_name = "Current name"
    release.set()
    await task
    assert naming_app._thread_name == "Current name"


@pytest.mark.parametrize("automatic", [False, True])
async def test_rename_refreshes_open_thread_selector(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch, *, automatic: bool
) -> None:
    from deepagents_code.tui.widgets.thread_selector import ThreadSelectorScreen

    thread: ThreadInfo = {
        "thread_id": "original",
        "agent_name": "agent",
        "updated_at": None,
        "latest_checkpoint_id": "cp_1",
        "thread_name": None if automatic else "Old name",
    }
    monkeypatch.setattr(
        "deepagents_code.sessions.list_threads",
        AsyncMock(side_effect=lambda **_: [thread.copy()]),
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
        selector = ThreadSelectorScreen(
            initial_threads=[thread.copy()], filter_cwd=None
        )
        naming_app.push_screen(selector)
        await pilot.pause()
        name_cell = "ThreadOption .thread-cell-thread_name"
        assert str(selector.query_one(name_cell, Static).render()) == (
            thread["thread_name"] or ""
        )
        if automatic:
            await naming_app._generate_thread_name(
                "original", "provider:chat", automatic=True
            )
        else:
            await naming_app._handle_command("/rename Cache repair")
        await pilot.pause()
        assert str(selector.query_one(name_cell, Static).render()) == "Cache repair"


async def test_rename_refreshes_thread_autocomplete(
    naming_app: DeepAgentsApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    thread: ThreadInfo = {
        "thread_id": "original",
        "agent_name": "agent",
        "updated_at": None,
        "thread_name": "Old name",
    }
    monkeypatch.setattr(
        "deepagents_code.sessions.list_threads",
        AsyncMock(side_effect=lambda **_: [thread.copy()]),
    )
    monkeypatch.setattr(
        "deepagents_code.sessions.populate_thread_checkpoint_details", AsyncMock()
    )

    def rename(_thread_id: str, name: str) -> bool:
        thread["thread_name"] = name
        return True

    monkeypatch.setattr(
        "deepagents_code.sessions.rename_thread", AsyncMock(side_effect=rename)
    )
    async with naming_app.run_test() as pilot:
        chat = naming_app._chat_input
        assert chat is not None
        assert chat._text_area is not None
        await naming_app.workers.wait_for_complete()
        chat._text_area.insert("compare @@Old")
        await pilot.pause()
        assert chat._current_suggestions[0][0] == "Old name"

        chat._text_area.load_text("/rename Release audit")
        chat._text_area.move_cursor_to_end()
        await pilot.press("enter")
        await pilot.pause()
        await naming_app.workers.wait_for_complete()
        assert naming_app._thread_name == "Release audit"

        chat._text_area.insert("compare @@Release")
        await pilot.pause()
        assert [label for label, _ in chat._current_suggestions] == ["Release audit"]
        await pilot.press("tab")
        assert chat._text_area.text == "compare @@(thread:original) "

        chat._text_area.load_text("compare @@Old")
        chat._text_area.move_cursor_to_end()
        await pilot.pause()
        assert not chat._current_suggestions
