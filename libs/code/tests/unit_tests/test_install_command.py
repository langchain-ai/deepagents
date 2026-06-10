"""Tests for the `/install <extra>` slash command and `--install` flag handler.

The CLI-flag side is covered by `test_main_args.TestInstallExtraSubcommand`;
this module focuses on the in-app slash dispatch in `DeepAgentsApp`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_code.app import DeepAgentsApp
from deepagents_code.widgets.messages import AppMessage, ErrorMessage


async def test_install_slash_usage_when_no_extra() -> None:
    """`/install` with no argument prints a usage hint plus the valid extras."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_install_extra",
            new_callable=AsyncMock,
        ) as perform_mock:
            await app._handle_command("/install")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        usage = next(m for m in app_msgs if "Usage: /install" in str(m._content))
        rendered = str(usage._content)
        # The no-arg path must list valid extras so they're discoverable.
        assert "Available extras:" in rendered
        assert "quickjs" in rendered
        assert "daytona" in rendered
        assert "openai" in rendered


async def test_install_slash_known_extra_runs() -> None:
    """A known extra invokes `perform_install_extra`."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        perform_mock.assert_awaited_once()


async def test_install_slash_provider_extra_recommends_restart_slash() -> None:
    """Provider extras advertise `/restart`, not a full relaunch.

    The langgraph subprocess is what imports model-provider packages, so
    respawning that subprocess via `/restart` picks them up without exiting.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        success = next(
            m for m in app_msgs if "Installed extra 'fireworks'" in str(m._content)
        )
        assert "/restart" in str(success._content)


async def test_install_slash_standalone_extra_recommends_full_relaunch() -> None:
    """Standalone extras must require a full relaunch, not `/restart`.

    `quickjs` and other `STANDALONE_EXTRAS` are wired into the TUI parent
    at startup via `verify_interpreter_deps`, so a subprocess respawn
    won't pick them up — the user has to exit and re-run dcode.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        success = next(
            m for m in app_msgs if "Installed extra 'quickjs'" in str(m._content)
        )
        rendered = str(success._content)
        assert "/restart" not in rendered
        assert "relaunch dcode" in rendered


async def test_install_slash_unknown_extra_requires_force() -> None:
    """Unknown extras without `--force` must not call `perform_install_extra`."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/install not-a-real-extra")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("not a known extra" in str(m._content) for m in app_msgs)


async def test_install_slash_unknown_extra_with_force_runs() -> None:
    """`--force` bypasses the unknown-extra confirmation."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
        ):
            await app._handle_command("/install not-a-real-extra --force")
            await pilot.pause()
        perform_mock.assert_awaited_once()


async def test_install_slash_invalid_extra_refuses_even_with_force() -> None:
    """Malformed extras must not reach command construction."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/install quickjs'];touch --force")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Invalid extra name" in str(m._content) for m in app_msgs)


async def test_install_slash_failure_surfaces_log_path_and_manual_cmd() -> None:
    """A failed install renders as `ErrorMessage` with log path + manual cmd.

    The success-styling regression: a previous version mounted `AppMessage`
    on failure, which made it visually indistinguishable from the
    "Installing extra..." status line. Failures must use `ErrorMessage`.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value="/tmp/deepagents-install.log",
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict"),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        error_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        joined = "\n".join(error_msgs)
        assert "Install failed" in joined
        assert "resolver: conflict" in joined
        assert "/tmp/deepagents-install.log" in joined
        assert "uv tool install -U 'deepagents-code" in joined
        assert "quickjs" in joined


async def test_install_slash_exception_surfaces_log_path_and_manual_cmd() -> None:
    """When `perform_install_extra` raises, surface log path + manual cmd."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value="/tmp/deepagents-install.log",
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                side_effect=OSError("disk full"),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        error_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        joined = "\n".join(error_msgs)
        assert "OSError" in joined
        assert "disk full" in joined
        assert "/tmp/deepagents-install.log" in joined
        assert "uv tool install -U 'deepagents-code" in joined
        assert "quickjs" in joined


async def test_install_slash_editable_install_refuses() -> None:
    """Editable installs must not invoke `perform_install_extra` from the TUI.

    Mirrors the editable-install guard for `/update` — running `uv tool
    install` on a dev checkout would clobber the editable install.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Editable install detected" in str(m._content) for m in app_msgs)


async def test_install_slash_package_confirm_runs() -> None:
    """`--package` without `--force` prompts; confirming runs the install."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value=True)
            ) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package")
            await pilot.pause()
        prompt.assert_awaited_once()
        perform_mock.assert_awaited_once()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any(
            "Installed package 'langchain-custom'" in str(m._content) for m in app_msgs
        )


async def test_install_slash_package_cancel_aborts() -> None:
    """Cancelling the prompt must not call `perform_install_package`."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
            ) as perform_mock,
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value=False)
            ) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package")
            await pilot.pause()
        prompt.assert_awaited_once()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        joined = "\n".join(str(m._content) for m in app_msgs)
        assert "Cancelled install" in joined
        # The raw `uv tool` command is never surfaced to the user.
        assert "uv tool" not in joined


async def test_install_slash_package_prompt_timeout_aborts() -> None:
    """A timed-out prompt aborts the install and reports the timeout."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
            ) as perform_mock,
            patch.object(
                app,
                "_push_screen_wait",
                new=AsyncMock(side_effect=TimeoutError()),
            ) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package")
            await pilot.pause()
        prompt.assert_awaited_once()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        joined = "\n".join(str(m._content) for m in app_msgs)
        assert "timed out" in joined
        # A timeout is not a user cancel and must not be reported as one.
        assert "Cancelled install" not in joined


async def test_install_slash_package_prompt_mount_failure_aborts() -> None:
    """A modal that fails to mount aborts the install and surfaces an error."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
            ) as perform_mock,
            patch.object(
                app,
                "_push_screen_wait",
                new=AsyncMock(side_effect=RuntimeError("no screen stack")),
            ) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package")
            await pilot.pause()
        prompt.assert_awaited_once()
        perform_mock.assert_not_awaited()
        err_msgs = [str(m._content) for m in app.query(ErrorMessage)]
        joined = "\n".join(err_msgs)
        assert "Could not show the install confirmation" in joined


async def test_install_slash_package_force_skips_prompt() -> None:
    """`--package --force` must not open the confirmation prompt."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        prompt.assert_not_awaited()
        perform_mock.assert_awaited_once()


async def test_install_slash_package_yes_alias_skips_prompt() -> None:
    """`--package --yes` is an alias for `--force` and skips the prompt."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install langchain-custom --package --yes")
            await pilot.pause()
        prompt.assert_not_awaited()
        perform_mock.assert_awaited_once()


async def test_install_slash_package_with_force_runs() -> None:
    """`--package --force` invokes `perform_install_package` and recommends restart."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ) as perform_mock,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        perform_mock.assert_awaited_once()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        success = next(
            m
            for m in app_msgs
            if "Installed package 'langchain-custom'" in str(m._content)
        )
        assert "/restart" in str(success._content)


async def test_install_slash_package_failure_renders_log() -> None:
    """A failed package install surfaces the detail + log, but no `uv` command."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict"),
            ) as perform_mock,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        perform_mock.assert_awaited_once()
        err_msgs = list(app.query(ErrorMessage))
        joined = "\n".join(str(m._content) for m in err_msgs)
        assert "Install failed" in joined
        assert "resolver: conflict" in joined
        assert "Log:" in joined
        assert "uv tool" not in joined


async def test_install_slash_package_invalid_refuses_even_with_force() -> None:
    """Malformed package names must not reach command construction."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/install custom;touch --package --force")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Invalid package name" in str(m._content) for m in app_msgs)


async def test_install_slash_package_editable_install_refuses() -> None:
    """Editable installs must not invoke `perform_install_package` from the TUI."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
            ) as perform_mock,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        assert any("Editable install detected" in str(m._content) for m in app_msgs)


async def test_install_restart_capable_extra_offers_restart_when_idle() -> None:
    """A provider extra prompts to restart and runs it on accept when idle."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        calls: list[str] = []

        def _reload() -> list[str]:
            calls.append("reload")
            return []

        def _clear() -> None:
            calls.append("clear")

        async def _restart() -> bool:  # noqa: RUF029  # patched async app hook
            calls.append("restart")
            return True

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value="restart")
            ) as prompt,
            patch("deepagents_code.config.settings.reload_from_environment", _reload),
            patch("deepagents_code.model_config.clear_caches", _clear),
            patch.object(
                app,
                "_restart_server_manual",
                new=AsyncMock(side_effect=_restart),
            ) as restart,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_awaited_once()
        assert calls == ["reload", "clear", "restart"]
        app_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        assert any("Restart complete." in m for m in app_msgs)


async def test_install_restart_capable_extra_defer_skips_restart() -> None:
    """Declining the restart prompt leaves the server untouched."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value="later")
            ) as prompt,
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=True)
            ) as restart,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_not_called()


async def test_install_standalone_extra_does_not_offer_restart() -> None:
    """Standalone extras (e.g. `quickjs`) never prompt to restart."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        prompt.assert_not_called()


async def test_install_restart_prompt_skipped_in_remote_server_mode() -> None:
    """Remote-server mode (no owned subprocess) must not offer a restart."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = None
        app._server_kwargs = None
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_not_called()


async def test_install_restart_prompt_skipped_while_agent_running() -> None:
    """A restart cancels in-flight work, so don't prompt mid-run."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        app._agent_running = True
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_not_called()


async def test_install_package_offers_restart_when_idle() -> None:
    """A `--package` install prompts to restart and runs it on accept."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "custom_provider:fake"}
        calls: list[str] = []

        def _reload() -> list[str]:
            calls.append("reload")
            return []

        def _clear() -> None:
            calls.append("clear")

        async def _restart() -> bool:  # noqa: RUF029  # patched async app hook
            calls.append("restart")
            return True

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value="restart")
            ) as prompt,
            patch("deepagents_code.config.settings.reload_from_environment", _reload),
            patch("deepagents_code.model_config.clear_caches", _clear),
            patch.object(
                app,
                "_restart_server_manual",
                new=AsyncMock(side_effect=_restart),
            ) as restart,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_awaited_once()
        assert calls == ["reload", "clear", "restart"]


async def test_install_package_defer_skips_restart() -> None:
    """Declining the prompt after a `--package` install leaves it untouched."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "custom_provider:fake"}
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_package",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value="later")
            ) as prompt,
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=True)
            ) as restart,
        ):
            await app._handle_command("/install langchain-custom --package --force")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_not_called()


async def test_install_restart_prompt_skipped_while_connecting() -> None:
    """A connecting/restarting server has nothing to respawn into, so skip."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        app._connecting = True
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(app, "_push_screen_wait", new=AsyncMock()) as prompt,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_not_called()


async def test_install_restart_prompt_mount_failure_leaves_manual_hint() -> None:
    """If the modal cannot be mounted, fall back to the manual `/restart` hint."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app,
                "_push_screen_wait",
                new=AsyncMock(side_effect=RuntimeError("modal hijacked")),
            ) as prompt,
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=True)
            ) as restart,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_not_called()
        app_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        # The install message keeps the manual recovery path, and no restart
        # was attempted.
        assert any("/restart" in m for m in app_msgs)
        assert not any("Restarting server..." in m for m in app_msgs)


async def test_install_restart_failure_omits_complete_message() -> None:
    """A failed restart shows the attempt but never claims completion."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            patch.object(
                app, "_push_screen_wait", new=AsyncMock(return_value="restart")
            ) as prompt,
            patch("deepagents_code.config.settings.reload_from_environment", list),
            patch("deepagents_code.model_config.clear_caches", lambda: None),
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=False)
            ) as restart,
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        prompt.assert_awaited_once()
        restart.assert_awaited_once()
        app_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        assert any("Restarting server..." in m for m in app_msgs)
        assert not any("Restart complete." in m for m in app_msgs)
