"""Integration test verifying `/effort` reaches the provider API.

This test uses a real remote agent and model to verify the configured effort is accepted
by the live provider API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_has_anthropic_credentials = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(
    not _has_anthropic_credentials, reason="ANTHROPIC_API_KEY not configured"
)
@pytest.mark.timeout(120)
async def test_effort_high_reaches_real_anthropic_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting `/effort high` should carry `reasoning_effort` to a real call.

    Regression coverage for `current_effort_from_model_params`/
    `_restore_effort_override`: a bug there could silently drop or duplicate
    the effort override before it ever reaches the model, which mocked unit
    tests can't observe end-to-end.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-reasoning-effort"
    # Must be a model whose profile declares `reasoning_effort_levels` (e.g.
    # Opus 4.5+/Sonnet 5), Haiku models have no reasoning-effort support at
    # all, so `/effort` would correctly no-op rather than reveal a real bug.
    model_spec = "anthropic:claude-opus-4-5-20251101"

    home_dir.mkdir()
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    from deepagents_code import model_config
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.client.launch.server_manager import server_session
    from deepagents_code.config import create_model
    from deepagents_code.sessions import generate_thread_id
    from deepagents_code.tui.widgets.messages import AssistantMessage, ErrorMessage

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model(model_spec).apply_to_settings()
        thread_id = generate_thread_id()

        async with server_session(
            assistant_id=assistant_id,
            model_name=model_spec,
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            app = DeepAgentsApp(
                agent=agent,  # ty: ignore
                assistant_id=assistant_id,
                backend=None,
                cwd=project_dir,
                thread_id=thread_id,
            )

            async with app.run_test() as pilot:
                await app._handle_effort_command("/effort high")
                assert app._model_params_override == {"reasoning_effort": "high"}

                await app._handle_user_message("Say hello in one short sentence.")

                for _ in range(300):
                    await pilot.pause(0.1)
                    if app.query(AssistantMessage) or app.query(ErrorMessage):
                        break

                error_messages = [
                    str(widget._content) for widget in app.query(ErrorMessage)
                ]
                assistant_messages = [
                    str(widget._content) for widget in app.query(AssistantMessage)
                ]

            assert not error_messages
            assert assistant_messages
            assert any(msg.strip() for msg in assistant_messages)
    finally:
        model_config.clear_caches()
