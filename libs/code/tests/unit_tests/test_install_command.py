"""Tests for the `/install <extra>` slash command and `--install` flag handler.

The CLI-flag side is covered by `test_main_args.TestInstallExtraSubcommand`;
this module focuses on the in-app slash dispatch in `DeepAgentsApp`.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage

MANUAL_EXTRA_COMMAND = (
    "curl -LsSf https://langch.in/dcode | DEEPAGENTS_CODE_EXTRAS=quickjs bash"
)


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
        assert "browser" in rendered
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


async def test_install_slash_browser_success_recommends_browser_mode() -> None:
    """Browser install confirms Chromium and gives the relaunch command."""
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
            await app._handle_command("/install browser")
            await pilot.pause()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        contents = " ".join(str(m._content) for m in app_msgs)
        assert "Installed browser support and Chromium" in contents
        assert "dcode --browser" in contents


async def test_install_slash_browser_runtime_failure_is_phase_specific() -> None:
    """A Chromium failure acknowledges that the browser extra was installed."""
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(
                    False,
                    (
                        "The browser extra was installed, but Chromium provisioning "
                        "failed: deps"
                    ),
                ),
            ),
        ):
            await app._handle_command("/install browser")
            await pilot.pause()
        errors = list(app.query(ErrorMessage))
        contents = " ".join(str(m._content) for m in errors)
        assert "Browser provisioning failed" in contents
        assert "browser extra was installed" in contents
        assert "Install failed" not in contents


async def test_install_slash_provider_extra_no_owned_server_recommends_relaunch() -> (
    None
):
    """With no owned server, `/restart` can't respawn — recommend a relaunch.

    The test harness has no app-owned LangGraph subprocess, so the one-keypress
    restart prompt is skipped and a `/restart` would have nothing to respawn;
    the surfaced guidance is a full relaunch, not `/restart`.
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
        contents = " ".join(str(m._content) for m in app_msgs)
        assert "Installed extra 'fireworks'" in contents
        assert "Relaunch dcode" in contents
        # No owned subprocess to respawn, so `/restart` is not recommended.
        assert "/restart" not in contents


async def test_offer_restart_busy_recommends_restart_not_relaunch() -> None:
    """An owned-but-busy server points at `/restart`, never a relaunch.

    `/restart` respawns the owned subprocess (same effect as a relaunch,
    without exiting), so a "relaunch dcode" hint would be redundant noise.
    """
    app = DeepAgentsApp()
    app._server_proc = MagicMock()
    app._server_kwargs = {"model_name": "fireworks:fake"}
    app._agent_running = True
    app._connecting = False
    app._mount_message = AsyncMock()  # ty: ignore

    await app._offer_restart_after_install("fireworks")

    contents = " ".join(
        str(c.args[0]._content)
        for c in app._mount_message.await_args_list  # ty: ignore
    )
    assert "/restart" in contents
    assert "relaunch" not in contents.lower()


async def test_offer_restart_no_owned_server_recommends_relaunch() -> None:
    """A remote/not-owned server can't be `/restart`ed — recommend relaunch."""
    app = DeepAgentsApp()
    app._server_proc = None
    app._server_kwargs = None
    app._mount_message = AsyncMock()  # ty: ignore

    await app._offer_restart_after_install("fireworks")

    contents = " ".join(
        str(c.args[0]._content)
        for c in app._mount_message.await_args_list  # ty: ignore
    )
    assert "Relaunch dcode" in contents
    assert "/restart" not in contents


async def test_offer_restart_state_flip_surfaces_fallback() -> None:
    """An explicit "restart" that can't run (state flipped) isn't a silent no-op.

    The pre-prompt guards pass (owned + idle), but server state can change
    while the user reads the prompt, so `_restart_after_install` returns False.
    The handler must surface a fallback rather than letting the chosen restart
    silently do nothing.
    """
    app = DeepAgentsApp()
    app._server_proc = MagicMock()
    app._server_kwargs = {"model_name": "fireworks:fake"}
    app._agent_running = False
    app._connecting = False
    app._mount_message = AsyncMock()  # ty: ignore
    app._push_screen_wait = AsyncMock(return_value="restart")  # ty: ignore
    app._restart_after_install = AsyncMock(return_value=False)  # ty: ignore

    await app._offer_restart_after_install("fireworks")

    app._restart_after_install.assert_awaited_once_with("fireworks")  # ty: ignore
    contents = " ".join(
        str(c.args[0]._content)
        for c in app._mount_message.await_args_list  # ty: ignore
    )
    assert "Couldn't restart the server automatically to load" in contents


async def test_install_slash_provider_extra_skips_redundant_hint_when_prompted() -> (
    None
):
    """When the restart prompt is offered, no redundant `/restart` hint appears.

    Popping a "restart now?" button while also printing "Run `/restart`" is
    confusing, so the manual hint is reserved for when the prompt can't show.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Pretend dcode owns an idle server so the one-keypress prompt is offered.
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        app._agent_running = False
        app._connecting = False
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(True, ""),
            ),
            # The user dismisses the prompt without restarting now.
            patch.object(app, "_push_screen_wait", new=AsyncMock(return_value="later")),
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=True)
            ),
        ):
            await app._handle_command("/install fireworks")
            await pilot.pause()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        contents = " ".join(str(m._content) for m in app_msgs)
        assert "Installed extra 'fireworks'" in contents
        # The button is the call to action; no inline "Run /restart" hint.
        assert "/restart" not in contents


async def test_install_slash_standalone_extra_recommends_relaunch() -> None:
    """Compatibility standalone extras still point at a full relaunch."""
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
        assert "relaunch dcode" in rendered.lower()
        assert "--interpreter" not in rendered


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
                "deepagents_code.update_check.install_extra_command",
                return_value=MANUAL_EXTRA_COMMAND,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                return_value=MANUAL_EXTRA_COMMAND,
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
        assert "curl -LsSf https://langch.in/dcode" in joined
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in joined
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
                "deepagents_code.update_check.install_extra_command",
                return_value=MANUAL_EXTRA_COMMAND,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                return_value=MANUAL_EXTRA_COMMAND,
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
        assert "curl -LsSf https://langch.in/dcode" in joined
        assert "DEEPAGENTS_CODE_EXTRAS=quickjs bash" in joined
        assert "quickjs" in joined


async def test_install_slash_failure_renders_recovery_bracket_literally() -> None:
    """A uv recovery command's `[extra]` bracket renders literally in the TUI.

    The TUI mounts recovery commands as Textual `Content`, so — unlike the
    Rich-markup CLI path — the bracket must not be backslash-escaped.
    """
    uv_cmd = "uv tool install -U 'deepagents-code[quickjs]'"
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
                "deepagents_code.update_check.install_extra_command",
                return_value=MANUAL_EXTRA_COMMAND,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                return_value=uv_cmd,
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict"),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        joined = "\n".join(str(m._content) for m in app.query(ErrorMessage))
        assert "deepagents-code[quickjs]" in joined
        assert "deepagents-code\\[quickjs]" not in joined


async def test_install_slash_failure_recovery_error_keeps_prior_command() -> None:
    """A recovery-command error on a failed install keeps the prior command.

    The TUI shows the command resolved before the failure rather than crashing
    or showing nothing.
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
                "deepagents_code.update_check.install_extra_command",
                return_value=MANUAL_EXTRA_COMMAND,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                side_effect=ValueError("bad receipt"),
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=(False, "resolver: conflict"),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        joined = "\n".join(str(m._content) for m in app.query(ErrorMessage))
        assert "Install failed" in joined
        assert MANUAL_EXTRA_COMMAND in joined


async def test_install_slash_exception_recovery_error_keeps_prior_command() -> None:
    """A raised install plus a failed recovery command keeps the prior command.

    When `perform_install_extra` raises and the recovery command also fails, the
    TUI still surfaces the command resolved before the failure.
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
                "deepagents_code.update_check.install_extra_command",
                return_value=MANUAL_EXTRA_COMMAND,
            ),
            patch(
                "deepagents_code.update_check.install_extra_recovery_command",
                side_effect=ValueError("bad receipt"),
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                side_effect=OSError("disk full"),
            ),
        ):
            await app._handle_command("/install quickjs")
            await pilot.pause()
        joined = "\n".join(str(m._content) for m in app.query(ErrorMessage))
        assert "OSError" in joined
        assert MANUAL_EXTRA_COMMAND in joined


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
            patch.object(app, "_ensure_restart_prompt_loaded") as preload,
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
        # The modal is preloaded before the upgrade can rewrite our own tree.
        preload.assert_called_once()
        prompt.assert_awaited_once()
        restart.assert_awaited_once()
        assert calls == ["reload", "clear", "restart"]
        app_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        assert any("Restart complete." in m for m in app_msgs)
        # The transient progress status is cleared once the restart succeeds.
        assert not any("Restarting server..." in m for m in app_msgs)


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
            patch.object(app, "_ensure_restart_prompt_loaded") as preload,
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
        # The modal is preloaded before the upgrade can rewrite our own tree.
        preload.assert_called_once()
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
    """A failed restart removes the attempt and never claims completion."""
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
        assert not any("Restarting server..." in m for m in app_msgs)
        assert not any("Restart complete." in m for m in app_msgs)


async def test_install_restart_raising_removes_transient_and_propagates() -> None:
    """A raising restart clears the transient before the exception propagates.

    The transient "Restarting server..." status mounts before
    `_restart_server_manual()` is awaited, so the `try/finally` in
    `_restart_after_install` exists solely to remove it when the restart raises
    (not merely returns `False`). On a raise the transient must be gone, the
    completion banner must never mount, and the exception must propagate.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}

        with (
            patch("deepagents_code.config.settings.reload_from_environment", list),
            patch("deepagents_code.model_config.clear_caches", lambda: None),
            patch.object(
                app,
                "_restart_server_manual",
                new=AsyncMock(side_effect=RuntimeError("respawn exploded")),
            ) as restart,
            pytest.raises(RuntimeError, match="respawn exploded"),
        ):
            await app._restart_after_install("fireworks")

        await pilot.pause()
        restart.assert_awaited_once()
        app_msgs = [
            str(m._content) for m in app.query(AppMessage) if not m._is_markdown
        ]
        assert not any("Restarting server..." in m for m in app_msgs)
        assert not any("Restart complete." in m for m in app_msgs)


async def test_offer_restart_survives_missing_restart_prompt_module() -> None:
    """A missing `restart_prompt` module must degrade, not crash the TUI.

    `/install` runs `uv tool install -U 'deepagents-code[...]'`, which rewrites
    deepagents-code's own on-disk tree mid-session. A first import of the
    restart modal on the post-install path then reads the half-replaced tree
    and raises `ModuleNotFoundError`. The handler must degrade to the manual
    `/restart` hint instead of letting the import crash the app.
    """
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._server_proc = MagicMock()
        app._server_kwargs = {"model_name": "fireworks:fake"}
        app._agent_running = False
        app._connecting = False
        push = AsyncMock(return_value="restart")
        with (
            # `None` in sys.modules makes the `from`-import raise
            # `ModuleNotFoundError` — a deterministic stand-in for the import
            # failure a half-replaced on-disk tree causes after a self-upgrade.
            patch.dict(
                sys.modules,
                {"deepagents_code.tui.widgets.restart_prompt": None},
            ),
            patch.object(app, "_push_screen_wait", new=push),
            patch.object(
                app, "_restart_server_manual", new=AsyncMock(return_value=True)
            ) as restart,
        ):
            # Must not raise despite the unimportable modal.
            await app._offer_restart_after_install("fireworks")
            await pilot.pause()
        # The modal was never mounted and no restart was attempted.
        push.assert_not_awaited()
        restart.assert_not_awaited()


def test_ensure_restart_prompt_loaded_caches_module() -> None:
    """Preloading leaves the restart modal resident in `sys.modules`.

    This is the actual fix: importing the modal before the self-upgrade
    rewrites the on-disk tree means the post-install import in
    `_offer_restart_after_install` resolves from `sys.modules` rather than the
    mutated tree. Dropping the resident copy first forces a real import.
    """
    with patch.dict(sys.modules):
        sys.modules.pop("deepagents_code.tui.widgets.restart_prompt", None)
        DeepAgentsApp._ensure_restart_prompt_loaded()
        assert "deepagents_code.tui.widgets.restart_prompt" in sys.modules


def test_ensure_restart_prompt_loaded_swallows_missing_module() -> None:
    """A failed preload is best-effort and must not raise.

    `None` in `sys.modules` makes `import deepagents_code.tui.widgets.restart_prompt`
    raise `ModuleNotFoundError`, standing in for the half-replaced tree left by
    a self-upgrade. The preload swallows it so the install continues; the
    post-install import then falls back to its own guard.
    """
    with patch.dict(
        sys.modules,
        {"deepagents_code.tui.widgets.restart_prompt": None},
    ):
        # Must not raise despite the unimportable module.
        DeepAgentsApp._ensure_restart_prompt_loaded()
