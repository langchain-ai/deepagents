"""Tests for the `/install <extra>` slash command and `--install` flag handler.

The CLI-flag side is covered by `test_main_args.TestInstallExtraSubcommand`;
this module focuses on the in-app slash dispatch in `DeepAgentsApp`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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


async def test_install_slash_package_requires_force() -> None:
    """`--package` without `--force` must not call `perform_install_package`."""
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
            await app._handle_command("/install langchain-custom --package")
            await pilot.pause()
        perform_mock.assert_not_awaited()
        app_msgs = [m for m in app.query(AppMessage) if not m._is_markdown]
        joined = "\n".join(str(m._content) for m in app_msgs)
        assert "--force" in joined
        assert "third-party code" in joined
        # The raw `uv tool` command is never surfaced to the user.
        assert "uv tool" not in joined


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
