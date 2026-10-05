"""Model catalogs and persistent selections use inference-host metadata."""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from textual.app import App
from textual.widgets import Input

from deepagents_code import model_config
from deepagents_code.app import DeepAgentsApp
from deepagents_code.model_catalog import CatalogProfile, CatalogProvider, ModelCatalog
from deepagents_code.model_config import ProviderAuthState
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
