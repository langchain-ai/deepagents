"""Model catalogs and persistent selections use inference-host metadata."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from textual.app import App
from textual.widgets import Input

from deepagents_code import config_manifest, model_catalog, model_config
from deepagents_code._cli_context import INHERIT_SUMMARIZATION_MODEL
from deepagents_code.app import DeepAgentsApp, DeferredAction
from deepagents_code.model_catalog import CatalogProfile, CatalogProvider, ModelCatalog
from deepagents_code.model_config import ProviderAuthState, ProviderAuthStatus
from deepagents_code.model_metadata import ModelMetadata
from deepagents_code.tui.widgets.model_selector import (
    SUMMARIZATION_DEFAULT_SCOPE,
    DefaultModelScope,
    ModelSelectorScreen,
)


@pytest.fixture
def catalog() -> ModelCatalog:
    return ModelCatalog(
        models=["remote:resolved"],
        profiles={
            "remote:resolved": CatalogProfile(
                profile={"name": "Remote Model", "max_input_tokens": 12345},
                overridden_keys=[],
            )
        },
        providers={
            "remote": CatalogProvider(
                state=ProviderAuthState.NOT_REQUIRED,
                display_name="Remote Provider",
                short_name="Remote",
            )
        },
        current_spec="remote:resolved",
    )


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", path)
    model_config.clear_caches()
    yield path
    model_config.clear_caches()


async def test_remote_picker_uses_catalog_for_highlighting_search_and_selection(
    catalog: ModelCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def client_probe(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Remote picker consulted local provider data")

    monkeypatch.setattr(model_config.ModelConfig, "load", client_probe)
    monkeypatch.setattr(
        "deepagents_code.model_catalog.load_model_catalog", client_probe
    )
    selected = Mock()
    screen = ModelSelectorScreen(
        current_model="alias",
        default_scope=None,
        include_recent_models=False,
        check_provider_requirements=False,
        catalog_loader=AsyncMock(return_value=catalog),
    )
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen, selected)
        await pilot.pause()
        assert screen._filtered_models[screen._selected_index] == (
            "remote:resolved",
            "remote",
        )
        assert (
            screen._profiles["remote:resolved"]["profile"]["max_input_tokens"] == 12345
        )
        screen.query_one("#model-filter", Input).value = "Remote Provider Remote Model"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    selected.assert_called_once_with(("remote:resolved", "remote"))


@pytest.mark.parametrize(
    ("provider", "installed"),
    [("groq", False), ("groq", True), ("ollama", False), ("custom", True)],
)
async def test_custom_provider_without_discovered_models_prompts_for_setup(
    provider: str,
    installed: bool,
    isolated_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.tui.widgets.auth import AuthPromptScreen
    from deepagents_code.tui.widgets.install_confirm import InstallProviderConfirmScreen

    await asyncio.to_thread(
        isolated_config.write_text,
        '[models.providers.custom]\napi_key_env = "CUSTOM_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_catalog, "get_available_models", dict)
    monkeypatch.setattr(model_catalog, "get_model_profiles", lambda **_kwargs: {})
    monkeypatch.setattr(
        config_manifest, "is_provider_package_installed", lambda _provider: installed
    )
    monkeypatch.setattr(
        model_catalog,
        "get_provider_auth_status",
        lambda name: ProviderAuthStatus(
            state=ProviderAuthState.MISSING,
            provider=name,
            env_var=f"{name.upper()}_API_KEY",
        ),
    )
    catalog = await asyncio.to_thread(model_catalog.load_model_catalog)
    assert catalog.models == []

    def client_probe(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Remote picker consulted local provider readiness")

    monkeypatch.setattr(config_manifest, "is_provider_package_installed", client_probe)
    monkeypatch.setattr(model_catalog, "get_provider_auth_status", client_probe)
    selected = Mock()
    screen = ModelSelectorScreen(
        default_scope=None,
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
    )
    spec = f"{provider}:custom-model"
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen, selected)
        await pilot.pause()
        screen.query_one("#model-filter", Input).value = spec
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        prompt_type = AuthPromptScreen if installed else InstallProviderConfirmScreen
        assert isinstance(app.screen, prompt_type)
        selected.assert_not_called()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        assert screen.pending_install_extra is None
        if not installed:
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert screen.pending_install_extra == provider
            selected.assert_called_once_with((spec, provider))


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    ("spec", "allowed_models", "blocked"),
    [
        ("groq:custom-model", ["remote:allowed"], True),
        ("groq:custom-model", [], True),
        ("groq:custom-model", ["groq:custom-model"], False),
        ("custom-model", ["remote:allowed"], False),
    ],
)
async def test_catalog_policy_precedes_custom_provider_setup(
    installed: bool,
    spec: str,
    allowed_models: list[str],
    blocked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.tui.widgets.auth import AuthPromptScreen
    from deepagents_code.tui.widgets.install_confirm import InstallProviderConfirmScreen

    catalog = ModelCatalog(
        models=[],
        profiles={},
        providers={
            "groq": CatalogProvider(
                state=ProviderAuthState.MISSING,
                env_var="GROQ_API_KEY",
                install_extra=None if installed else "groq",
            )
        },
        allowed_models=allowed_models,
    )

    def client_probe(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Catalog policy must not consult local provider data")

    monkeypatch.setattr(model_config.ModelConfig, "canonical_model_spec", client_probe)
    monkeypatch.setattr("deepagents_code.config.detect_provider", client_probe)
    selected = Mock()
    notify = Mock()
    screen = ModelSelectorScreen(
        default_scope=None,
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
    )
    monkeypatch.setattr(screen, "notify", notify)
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen, selected)
        await pilot.pause()
        screen.query_one("#model-filter", Input).value = spec
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        if blocked:
            assert app.screen is screen
            assert screen.pending_install_extra is None
            selected.assert_not_called()
            assert "models.allowed" in notify.call_args.args[0]
        elif ":" in spec:
            prompt_type = (
                AuthPromptScreen if installed else InstallProviderConfirmScreen
            )
            assert isinstance(app.screen, prompt_type)
            selected.assert_not_called()
            notify.assert_not_called()
        else:
            selected.assert_called_once_with((spec, ""))
            notify.assert_not_called()


@pytest.mark.parametrize("resolution_fails", [False, True])
async def test_picker_validates_before_saving_default(
    catalog: ModelCatalog,
    resolution_fails: bool,
) -> None:
    stored: dict[str, str | None] = {"model": None}

    def save(spec: str) -> bool:
        stored["model"] = spec
        return True

    screen = ModelSelectorScreen(
        default_scope=DefaultModelScope(
            "default", "set default", lambda: None, save, lambda: True
        ),
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
        resolve_model=AsyncMock(
            return_value=ModelMetadata("canonical", "remote"),
            side_effect=RuntimeError("Server rejected the model")
            if resolution_fails
            else None,
        ),
    )
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert stored["model"] == (None if resolution_fails else "remote:canonical")
        assert screen._default_spec == stored["model"]


@pytest.mark.parametrize(
    ("dismiss", "resolution_fails"), [(False, False), (True, False), (True, True)]
)
async def test_default_validation_keeps_keyboard_responsive(
    catalog: ModelCatalog,
    dismiss: bool,
    resolution_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog.models.append("remote:other")
    started = asyncio.Event()
    release = asyncio.Event()

    async def resolve(_spec: str) -> ModelMetadata:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Simulate a resolver that still returns a late response after the
            # screen's worker has been cancelled on dismissal.
            await release.wait()
        if resolution_fails:
            msg = "Server rejected the model"
            raise RuntimeError(msg)
        return ModelMetadata("canonical", "remote")

    save = Mock(return_value=True)
    resolver = AsyncMock(side_effect=resolve)
    screen = ModelSelectorScreen(
        default_scope=DefaultModelScope(
            "default", "set default", lambda: None, save, lambda: True
        ),
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
        resolve_model=resolver,
    )
    app = App()
    notify = Mock()
    monkeypatch.setattr(screen, "notify", notify)
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        save_key = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            selected = screen._filtered_models[screen._selected_index]
            await asyncio.wait_for(pilot.press("down"), timeout=2)
            assert screen._filtered_models[screen._selected_index] != selected
            await asyncio.wait_for(pilot.press("ctrl+s"), timeout=2)
            resolver.assert_awaited_once_with("remote:resolved")
            if dismiss:
                await asyncio.wait_for(pilot.press("escape"), timeout=2)
                assert app.screen is not screen
            save.assert_not_called()
        finally:
            release.set()
            await save_key
            await app.workers.wait_for_complete()
        await pilot.pause()
        if dismiss:
            save.assert_not_called()
            notify.assert_not_called()
            assert screen._default_spec is None
        else:
            save.assert_called_once_with("remote:canonical")
            assert screen._default_spec == "remote:canonical"


@pytest.mark.parametrize("clear", [False, True])
async def test_default_persistence_keeps_escape_responsive(
    catalog: ModelCatalog, clear: bool
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def persist(*_args: str) -> bool:
        started.set()
        try:
            return release.wait(timeout=5)
        finally:
            finished.set()

    original = "remote:resolved" if clear else None
    screen = ModelSelectorScreen(
        default_scope=DefaultModelScope(
            "default", "set default", lambda: original, persist, persist
        ),
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
    )
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        save_key = asyncio.create_task(pilot.press("ctrl+s"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            await asyncio.wait_for(pilot.press("escape"), timeout=2)
            assert app.screen is not screen
        finally:
            release.set()
            await save_key
            assert await asyncio.to_thread(finished.wait, 2)
        await pilot.pause()
        assert screen._default_spec == original


@pytest.mark.parametrize("resolution_fails", [False, True])
async def test_default_command_validates_without_switching_active_model(
    resolution_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._model_override = "remote:active"
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(
        app,
        "_resolve_model_metadata",
        AsyncMock(
            return_value=ModelMetadata("canonical", "remote"),
            side_effect=RuntimeError("Server rejected the model")
            if resolution_fails
            else None,
        ),
    )
    save = Mock(return_value=True)
    monkeypatch.setattr(model_config, "save_default_model", save)
    await app._set_default_model("alias")
    assert app._model_override == "remote:active"
    if resolution_fails:
        save.assert_not_called()
    else:
        save.assert_called_once_with("remote:canonical")


async def test_summary_default_scope_writes_only_summary_preference(
    catalog: ModelCatalog,
    isolated_config: Path,
) -> None:
    await asyncio.to_thread(
        isolated_config.write_text,
        '[models]\ndefault = "remote:main"\n',
        encoding="utf-8",
    )
    screen = ModelSelectorScreen(
        default_scope=SUMMARIZATION_DEFAULT_SCOPE,
        include_recent_models=False,
        catalog_loader=AsyncMock(return_value=catalog),
        resolve_model=AsyncMock(return_value=ModelMetadata("resolved", "remote")),
    )
    app = App()
    async with app.run_test() as pilot:
        app.push_screen(screen)
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        config = model_config.ModelConfig.load()
        assert config.default_model == "remote:main"
        assert config.summarization_default_model == "remote:resolved"
        await pilot.press("ctrl+s")
        await pilot.pause()
        config = model_config.ModelConfig.load()
        assert config.summarization_default_model is None
        assert config.default_model == "remote:main"


def test_catalog_rejects_incoherent_provider_readiness(catalog: ModelCatalog) -> None:
    payload = catalog.model_dump(mode="json")
    payload["providers"]["remote"]["state"] = "configured"
    with pytest.raises(ValidationError, match="source"):
        ModelCatalog.model_validate(payload)


@pytest.mark.parametrize("confirmed", [True, False])
async def test_deferred_install_switch_finishes_before_thread_switch(
    confirmed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp(thread_id="original")
    app._context_tokens = 1000
    app._model_switch_warning_threshold = 1
    monkeypatch.setattr(app, "notify", Mock())
    monkeypatch.setattr(app, "_remote_agent", lambda: None)
    monkeypatch.setattr(app, "_prepare_model_provider", AsyncMock(return_value=True))
    confirmation_started = asyncio.Event()
    answer: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    models: dict[str, str] = {}

    async def confirm(_screen: object) -> bool:
        confirmation_started.set()
        return await answer

    def switch_model(spec: str, **_kwargs: object) -> None:
        assert app._lc_thread_id is not None
        models[app._lc_thread_id] = spec

    def switch_thread() -> None:
        app._lc_thread_id = "next"

    monkeypatch.setattr(app, "_push_screen_wait", confirm)
    monkeypatch.setattr(app, "_switch_model", AsyncMock(side_effect=switch_model))
    app._agent_running = True
    await app._install_extra_then_switch("remote", "remote:selected")
    app._defer_action(
        DeferredAction(
            kind="thread_switch", execute=AsyncMock(side_effect=switch_thread)
        )
    )
    app._agent_running = False
    drain = asyncio.create_task(app._drain_deferred_actions())
    try:
        await asyncio.wait_for(confirmation_started.wait(), timeout=2)
        assert app._lc_thread_id == "original"
        assert models == {}
        answer.set_result(confirmed)
        await asyncio.wait_for(drain, timeout=2)
        assert app._lc_thread_id == "next"
        assert models == ({"original": "remote:selected"} if confirmed else {})
    finally:
        if not answer.done():
            answer.set_result(False)
        await drain
        await asyncio.gather(*app._modal_command_tasks.values())


@pytest.mark.parametrize("blocked_stage", ["install", "resolve"])
@pytest.mark.parametrize("clear", [False, True])
async def test_newer_summary_choice_survives_older_selection(
    blocked_stage: str, clear: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def wait_for_release() -> None:
        started.set()
        await release.wait()
        finished.set()

    async def prepare(_extra: str, _spec: str) -> bool:
        await wait_for_release()
        return True

    async def resolve(spec: str) -> ModelMetadata:
        if spec == "remote:older" and blocked_stage == "resolve":
            await wait_for_release()
        return ModelMetadata(spec.removeprefix("remote:"), "remote")

    monkeypatch.setattr(app, "_prepare_model_provider", prepare)
    monkeypatch.setattr(app, "_resolve_auxiliary_model", resolve)
    older = asyncio.create_task(
        app._apply_summarization_model_selection(
            "remote:older", "remote" if blocked_stage == "install" else None
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        if clear:
            await app._handle_summarization_model_command("/offload model clear")
        else:
            await app._apply_summarization_model_selection("remote:newer", None)
        expected = INHERIT_SUMMARIZATION_MODEL if clear else "remote:newer"
        assert app._summarization_model_override == expected
        release.set()
        await asyncio.wait_for(older, timeout=2)
        assert finished.is_set()
        assert app._summarization_model_override == expected
    finally:
        release.set()
        await older
