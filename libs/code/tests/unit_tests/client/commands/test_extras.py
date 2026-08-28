"""Tests for the `dcode install` command and shared install helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.client.commands.extras import (
    run_install_command,
    run_install_request,
)
from deepagents_code.update_check import (
    UPDATE_LOCK_CONTENDED_MESSAGE,
    ExtraInstallOutcome,
)


class TestInstallCommandDispatch:
    """Tests for `dcode install` dispatch and shared install helpers."""

    def test_contended_extra_install_omits_manual_recovery(self) -> None:
        """A manual reinstall must not bypass another process's lock."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/deepagents-install.log"),
            ),
            patch(
                "deepagents_code.update_check.install_extra_command",
                return_value="unsafe manual command",
            ),
            patch(
                "deepagents_code.update_check.perform_install_extra",
                new_callable=AsyncMock,
                return_value=ExtraInstallOutcome(
                    False,
                    UPDATE_LOCK_CONTENDED_MESSAGE,
                    manual_recovery_safe=False,
                ),
            ) as perform,
        ):
            code = run_install_request(name="quickjs", package=False, yes=True)

        assert code == 1
        perform.assert_awaited_once()
        printed = " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )
        assert "already running" in printed
        assert "Run manually" not in printed


class TestInstallCliParsing:
    """Parse-level coverage for the install command."""


class TestInstallCliMain:
    """End-to-end control flow through `cli_main` for the install command."""
