"""Tests for optional-extra removal from the CLI and TUI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import contextmanager, nullcontext
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest
from rich.console import Console

from deepagents_code._version import __version__
from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.commands.extras import (
    run_uninstall_command,
    run_uninstall_request,
)
from deepagents_code.tui.widgets.messages import AppMessage
from deepagents_code.update_check import (
    UPDATE_LOCK_CONTENDED_MESSAGE,
    CompositeExtraConflictError,
    ExtraNotInstalledError,
    ExtraRemovalOutcome,
    ProtectedExtraError,
    ToolRequirementIntrospectionError,
    _install_extra_uv_tool_command,
    perform_uninstall_extra,
    removable_extras,
    uninstall_extra_command,
    upgrade_install_command,
)


def _write_receipt(
    root: Path,
    extras: tuple[str, ...],
    *,
    python: str | None = None,
    with_packages: tuple[str, ...] = (),
) -> None:
    extra_values = ", ".join(f'"{extra}"' for extra in extras)
    requirement = (
        f'{{ name = "deepagents-code", extras = [{extra_values}], '
        f'specifier = "=={__version__}" }}'
    )
    requirements = [requirement]
    requirements.extend(f'{{ name = "{package}" }}' for package in with_packages)
    python_line = f'python = "{python}"\n' if python else ""
    root.joinpath("uv-receipt.toml").write_text(
        f"[tool]\n{python_line}requirements = [{', '.join(requirements)}]\n",
        encoding="utf-8",
    )


class TestUninstallExtraCommand:
    """Receipt-aware command generation removes only selected extras."""

    def test_removes_target_and_preserves_tool_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(
            tmp_path,
            ("ollama", "nvidia"),
            python="/opt/Python 3.13/bin/python",
            with_packages=("langchain-custom",),
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        command = uninstall_extra_command("ollama")

        assert command == (
            "uv tool install --reinstall -U --python '/opt/Python 3.13/bin/python' "
            f"'deepagents-code[nvidia]=={__version__}' --with langchain-custom"
        )

    def test_last_extra_reinstalls_plain_pinned_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        assert uninstall_extra_command("ollama") == (
            f"uv tool install --reinstall -U deepagents-code=={__version__}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            'path = "/tmp/deepagents-code.whl"',
            'url = "https://example.com/deepagents-code.whl"',
            'git = "https://example.com/deepagents-code.git"',
        ],
    )
    def test_non_registry_tool_source_refuses_rebuild(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source: str,
    ) -> None:
        """Removing an extra must not replace a custom build with PyPI's."""
        tmp_path.joinpath("uv-receipt.toml").write_text(
            "[tool]\nrequirements = ["
            f'{{ name = "deepagents-code", extras = ["ollama"], {source} }}'
            "]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(
            ToolRequirementIntrospectionError,
            match="source fields that cannot be preserved automatically",
        ):
            uninstall_extra_command("ollama")

    def test_prerelease_install_keeps_prerelease_resolution_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        assert uninstall_extra_command("ollama", version="1.2.0rc1") == (
            "uv tool install --reinstall -U "
            "deepagents-code==1.2.0rc1 --prerelease allow"
        )

    def test_absent_extra_raises_without_building_a_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("nvidia",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ExtraNotInstalledError, match="not installed"):
            uninstall_extra_command("ollama")

    def test_extra_names_use_pep_508_normalization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("ollama", "my-extra"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        command = uninstall_extra_command("MY_EXTRA")

        assert f"deepagents-code[ollama]=={__version__}" in command
        assert "my-extra" not in command

    @pytest.mark.parametrize(
        "extra", ["openai", "anthropic", "google-genai", "quickjs"]
    )
    def test_base_dependency_is_not_removable_even_when_receipt_lists_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        extra: str,
    ) -> None:
        _write_receipt(tmp_path, (extra, "nvidia"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ProtectedExtraError, match="base dependency"):
            uninstall_extra_command(extra)

    def test_removal_survives_a_later_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A removed extra stays removed when the env is next rebuilt.

        Regression guard. The rebuild used to derive its extra set from
        `installed_extra_names`, which reports an extra as installed when *any*
        one of its packages is present. `media` is satisfied by `pillow` and
        `nvidia` by `aiohttp`, both of which arrive as base or transitive
        dependencies, so those extras read as installed on every env — a removal
        could not delete them and the next upgrade silently put them back.
        """
        _write_receipt(tmp_path, ("media", "ollama"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        removal = uninstall_extra_command("media", version=__version__)
        assert "media" not in removal
        assert "ollama" in removal

        # Simulate the receipt uv writes after running `removal`, then confirm
        # the next upgrade and extra install both leave `media` deselected.
        _write_receipt(tmp_path, ("ollama",))
        assert "media" not in upgrade_install_command()
        assert "media" not in _install_extra_uv_tool_command("daytona")

    def test_composite_selection_names_the_providing_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A member of a selected composite is not reported as absent.

        `all-providers` installs `langchain-ollama`, so "not installed" would be
        the opposite of the truth. The error names the composite instead.
        """
        _write_receipt(tmp_path, ("all-providers",))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(
            CompositeExtraConflictError, match="provided by all-providers"
        ):
            uninstall_extra_command("ollama", version=__version__)

    def test_direct_selection_cannot_be_removed_while_composite_provides_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A direct selector cannot remove packages its composite still supplies."""
        _write_receipt(tmp_path, ("all-providers", "ollama"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(
            CompositeExtraConflictError,
            match=r"also provided by all-providers.*cannot be removed independently",
        ):
            uninstall_extra_command("ollama", version=__version__)

    def test_removable_extras_excludes_composite_supplied_direct_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_receipt(tmp_path, ("all-providers", "ollama", "media"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        assert removable_extras() == ["all-providers", "media"]

    def test_absent_extra_lists_what_is_selected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely absent extra reports the selected set for context."""
        _write_receipt(tmp_path, ("nvidia", "ollama"))
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(
            ExtraNotInstalledError,
            match=r"not installed\. Selected extras: nvidia, ollama",
        ):
            uninstall_extra_command("daytona", version=__version__)

    @pytest.mark.parametrize(
        ("receipt", "match"),
        [
            ("[tool]\n", "missing `\\[tool\\].requirements`"),
            ('[tool]\nrequirements = ["bare-string"]\n', "non-table requirement"),
            (
                '[tool]\nrequirements = [{ version = "1.0" }]\n',
                "without a package name",
            ),
            (
                '[tool]\nrequirements = [{ name = "langchain-custom" }]\n',
                "does not contain a 'deepagents-code' requirement",
            ),
        ],
    )
    def test_unusable_receipt_shapes_fail_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        receipt: str,
        match: str,
    ) -> None:
        """Every receipt shape that cannot describe the selection fails closed.

        Guessing an empty selection here would rebuild the tool as a plain
        `deepagents-code`, deselecting everything the user installed.
        """
        tmp_path.joinpath("uv-receipt.toml").write_text(receipt, encoding="utf-8")
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ToolRequirementIntrospectionError, match=match):
            uninstall_extra_command("ollama", version=__version__)

    def test_invalid_name_is_rejected_before_receipt_read(self) -> None:
        with pytest.raises(ValueError, match="Invalid extra name"):
            uninstall_extra_command("ollama']; touch /tmp/pwned; '")

    @pytest.mark.parametrize(
        "main_requirement",
        [
            '{ name = "deepagents-code", extras = "ollama" }',
            '{ name = "deepagents-code", extras = ["ollama;rm -rf /"] }',
            '{ name = "deepagents-code", extras = [1] }',
            # Duplicate canonical spellings: silently collapsing them would make
            # the rebuilt command disagree with the receipt it came from.
            '{ name = "deepagents-code", extras = ["my-extra", "my_extra"] }',
        ],
    )
    def test_malformed_receipt_extras_fail_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        main_requirement: str,
    ) -> None:
        tmp_path.joinpath("uv-receipt.toml").write_text(
            f"[tool]\nrequirements = [{main_requirement}]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

        with pytest.raises(ToolRequirementIntrospectionError):
            uninstall_extra_command("ollama")


class TestPerformUninstallExtra:
    """The performer refuses unsafe/no-op paths before spawning uv."""

    async def test_invalid_name_never_spawns(self) -> None:
        with patch(
            "deepagents_code.update_check._run_install_subprocess",
            new_callable=AsyncMock,
        ) as run:
            outcome = await perform_uninstall_extra("bad;name")
        assert outcome.success is False
        assert "Invalid extra name" in outcome.output
        run.assert_not_awaited()

    @pytest.mark.parametrize(
        ("method", "message"),
        [
            ("unknown", "Editable install detected"),
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    async def test_non_uv_install_is_refused_without_spawning(
        self, method: str, message: str
    ) -> None:
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value=method,
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run,
        ):
            outcome = await perform_uninstall_extra("ollama")
        assert outcome.success is False
        assert message in outcome.output
        run.assert_not_awaited()

    async def test_absent_extra_is_noop_without_uv(self) -> None:
        """An already-completed removal does not require an executable."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch("deepagents_code.update_check.shutil.which", return_value=None),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=ExtraNotInstalledError("Extra 'ollama' is not installed."),
            ),
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run,
        ):
            outcome = await perform_uninstall_extra("ollama")
        assert outcome.success is False
        assert "not installed" in outcome.output
        assert outcome.extra_was_absent is True
        run.assert_not_awaited()

    async def test_selected_extra_requires_uv_before_spawning(self) -> None:
        """A removal that needs a rebuild still fails cleanly without uv."""
        run = AsyncMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch("deepagents_code.update_check.shutil.which", return_value=None),
            patch(
                "deepagents_code.update_check.read_installed_distribution_version",
                return_value="9.8.7",
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install safe-command",
            ),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            outcome = await perform_uninstall_extra("ollama")

        assert outcome.success is False
        assert "`uv` not found" in outcome.output
        assert outcome.extra_was_absent is False
        run.assert_not_awaited()

    async def test_lock_wraps_command_generation_and_subprocess(self) -> None:
        events: list[str] = []

        def run_subprocess(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            events.append("run")
            return False, "resolver failed"

        run = AsyncMock(side_effect=run_subprocess)

        @contextmanager
        def lock() -> Iterator[bool]:
            events.append("acquire")
            try:
                yield True
            finally:
                events.append("release")

        def read_version() -> str:
            events.append("version")
            return "9.8.7"

        def command(_extra: str, *, version: str) -> str:
            events.append("command")
            assert version == "9.8.7"
            return "uv tool install safe-command"

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch("deepagents_code.update_check.update_install_lock", lock),
            patch(
                "deepagents_code.update_check.read_installed_distribution_version",
                side_effect=read_version,
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                side_effect=command,
            ),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            result = await perform_uninstall_extra("ollama", log_path=Path("/tmp/log"))

        assert result.success is False
        assert result.output == "resolver failed"
        assert result.manual_recovery_safe is True
        assert result.manual_recovery_command == "uv tool install safe-command"
        assert events == ["acquire", "version", "command", "run", "release"]
        run.assert_awaited_once_with(
            "uv tool install safe-command", progress=None, log_path=Path("/tmp/log")
        )

    async def test_cancellation_returns_the_locked_recovery_command(self) -> None:
        run = AsyncMock(side_effect=asyncio.CancelledError)
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch(
                "deepagents_code.update_check.read_installed_distribution_version",
                return_value="9.8.7",
            ),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                return_value="uv tool install deepagents-code==9.8.7",
            ),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            outcome = await perform_uninstall_extra("ollama")

        assert outcome.interrupted is True
        assert (
            outcome.manual_recovery_command == "uv tool install deepagents-code==9.8.7"
        )

    async def test_unknown_installed_version_refuses_rebuild(self) -> None:
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch(
                "deepagents_code.update_check.read_installed_distribution_version",
                return_value=None,
            ),
            patch("deepagents_code.update_check.uninstall_extra_command") as command,
            patch(
                "deepagents_code.update_check._run_install_subprocess",
                new_callable=AsyncMock,
            ) as run,
        ):
            outcome = await perform_uninstall_extra("ollama")

        assert outcome.success is False
        assert "Could not determine the installed" in outcome.output
        command.assert_not_called()
        run.assert_not_awaited()

    async def test_contended_lock_skips_receipt_read_and_subprocess(self) -> None:
        command = MagicMock()
        run = AsyncMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.shutil.which", return_value="/usr/bin/uv"
            ),
            patch(
                "deepagents_code.update_check.update_install_lock",
                return_value=nullcontext(False),
            ),
            patch("deepagents_code.update_check.uninstall_extra_command", command),
            patch("deepagents_code.update_check._run_install_subprocess", run),
        ):
            outcome = await perform_uninstall_extra("ollama")

        assert outcome.success is False
        assert "already running" in outcome.output
        # Contention must not advertise the manual `--reinstall` command: running
        # it by hand while the holder is mid-install is what corrupts the env.
        assert outcome.manual_recovery_safe is False
        command.assert_not_called()
        run.assert_not_awaited()


class TestUninstallCli:
    """CLI command and compatibility flag share the same request handler."""

    @pytest.fixture(autouse=True)
    def _uv_install_method(self) -> Iterator[None]:
        with patch(
            "deepagents_code.update_check.detect_install_method", return_value="uv"
        ):
            yield

    def test_subcommand_dispatches_name(self) -> None:
        args = argparse.Namespace(uninstall_target="ollama")
        with patch(
            "deepagents_code.client.commands.extras.run_uninstall_request",
            return_value=0,
        ) as request:
            assert run_uninstall_command(args) == 0
        request.assert_called_once_with(name="ollama")

    def test_help_renders_for_the_subcommand(self, capsys) -> None:
        """`dcode uninstall --help` runs the real help function.

        The subparser wires help through `_lazy_help("show_uninstall_help")`, a
        string the type checker cannot verify — so exercise it rather than
        patching it out.
        """
        from deepagents_code.main import parse_args

        with (
            patch.object(sys, "argv", ["dcode", "uninstall", "--help"]),
            pytest.raises(SystemExit) as exit_info,
        ):
            parse_args()

        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "dcode uninstall NAME" in out
        assert "base dependencies" in out
        assert "quickjs" in out

    def test_missing_subcommand_name_shows_help(self) -> None:
        args = argparse.Namespace(uninstall_target=None)
        with patch("deepagents_code.ui.show_uninstall_help") as show_help:
            assert run_uninstall_command(args) == 2
        show_help.assert_called_once()

    def test_absent_extra_is_successful_noop(self, tmp_path: Path) -> None:
        console = MagicMock()
        log_path = tmp_path / "uninstall.log"
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    "Extra 'ollama' is not installed.",
                    extra_was_absent=True,
                ),
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 0
        perform.assert_awaited_once()
        assert log_path.is_file()
        assert "not installed" in " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )

    def test_composite_conflict_is_a_refusal(self) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False, "Extra 'ollama' is also provided by all-providers."
                ),
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 1
        perform.assert_awaited_once()
        assert "all-providers" in self._console_text(console)

    @pytest.mark.parametrize(
        "extra", ["openai", "anthropic", "google-genai", "quickjs"]
    )
    def test_protected_extra_is_failure(self, extra: str) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    f"Extra {extra!r} is a base dependency and cannot be removed.",
                ),
            ) as perform,
        ):
            assert run_uninstall_request(name=extra) == 1
        perform.assert_awaited_once()
        text = " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )
        assert "cannot be removed" in text

    @pytest.mark.parametrize(
        ("method", "message"),
        [
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    def test_non_uv_install_is_refused_before_receipt_read(
        self, method: str, message: str
    ) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value=method,
            ),
            patch("deepagents_code.update_check.uninstall_extra_command") as command,
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 1
        command.assert_not_called()
        perform.assert_not_awaited()
        text = " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )
        assert message in text

    def test_success_reports_restart_guidance(self) -> None:
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/uninstall.log"),
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(True, ""),
            ) as perform,
            patch(
                "deepagents_code._invocation.invoked_name",
                return_value="deepagents-code",
            ),
        ):
            assert run_uninstall_request(name="ollama") == 0
        perform.assert_awaited_once()
        text = " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )
        assert "Uninstalled extra 'ollama'" in text
        assert "Relaunch deepagents-code" in text

    def _console_text(self, console: MagicMock) -> str:
        return " ".join(
            str(arg) for call in console.print.call_args_list for arg in call.args
        )

    def test_editable_install_refuses_with_nonzero_exit(self) -> None:
        """An editable install is refused before any environment mutation."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            # A uv install so only the editable check can refuse: an editable
            # checkout under a uv tool prefix is detected as "uv".
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value="uv",
            ),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 1
        perform.assert_not_awaited()
        text = self._console_text(console)
        assert "Editable install detected" in text
        assert "uv tool inst" in text

    def test_protected_extra_exits_nonzero(self) -> None:
        """A refused base-provider removal must not report shell success.

        `dcode uninstall openai && echo removed` printing "removed" would tell a
        script the extra is gone when the request was declined outright.
        """
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False, "Extra 'openai' is a base dependency and cannot be removed."
                ),
            ) as perform,
        ):
            assert run_uninstall_request(name="openai") == 1
        perform.assert_awaited_once()
        assert "base dependency" in self._console_text(console)

    @pytest.mark.parametrize("name", ["ollama; rm -rf /", ""])
    def test_invalid_name_exits_two(self, name: str) -> None:
        """A malformed extra name is a usage error, not a failed removal."""
        console = MagicMock()
        with (
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            assert run_uninstall_request(name=name) == 2
        perform.assert_not_awaited()
        assert "Invalid extra name" in self._console_text(console)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("brew", "Homebrew install detected"),
            ("other", "Unsupported install method detected"),
        ],
    )
    def test_unsupported_install_method_skips_receipt_read(
        self, method: str, expected: str
    ) -> None:
        """Method-specific guidance wins over a raw receipt introspection error."""
        console = MagicMock()
        command = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.detect_install_method",
                return_value=method,
            ),
            patch("deepagents_code.update_check.uninstall_extra_command", command),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 1
        command.assert_not_called()
        perform.assert_not_awaited()
        assert expected in self._console_text(console)

    def test_receipt_failure_exits_one(self) -> None:
        """An unreadable receipt is reported with its type, not swallowed."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    "ToolRequirementIntrospectionError: receipt not found",
                ),
            ) as perform,
        ):
            assert run_uninstall_request(name="ollama") == 1
        perform.assert_awaited_once()
        text = self._console_text(console)
        assert "ToolRequirementIntrospectionError" in text
        assert "receipt not found" in text

    def test_failure_reports_log_and_recovery_command(self) -> None:
        """A failed rebuild surfaces the tail of the output, log, and repair cmd."""
        console = MagicMock()
        stale_command = MagicMock(return_value="uv tool install deepagents-code==1.0")
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.uninstall_extra_command",
                stale_command,
            ),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=Path("/tmp/uninstall.log"),
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    "x" * 300 + "resolver boom",
                    manual_recovery_command=("uv tool install deepagents-code==9.8.7"),
                ),
            ),
        ):
            assert run_uninstall_request(name="ollama") == 1
        stale_command.assert_not_called()
        text = self._console_text(console)
        assert "resolver boom" in text
        assert "/tmp/uninstall.log" in text
        assert "uv tool install deepagents-code==9.8.7" in text
        assert "deepagents-code==1.0" not in text
        # Only the last 200 characters of output are echoed.
        assert "x" * 250 not in text

    def test_contention_withholds_manual_command(self) -> None:
        """Contention must not advise hand-running the rebuild."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    UPDATE_LOCK_CONTENDED_MESSAGE,
                    manual_recovery_safe=False,
                ),
            ),
        ):
            assert run_uninstall_request(name="ollama") == 1
        text = self._console_text(console)
        assert "already running" in text
        assert "safe-command" not in text
        assert "Run manually" not in text

    def test_log_creation_failure_omits_log_hint(self) -> None:
        """A failed log create never advertises a path that does not exist."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.create_update_log_file",
                return_value=None,
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(False, "receipt not found"),
            ),
        ):
            assert run_uninstall_request(name="ollama") == 1

        text = self._console_text(console)
        assert "receipt not found" in text
        assert "Uninstall log:" not in text
        assert "Log:" not in text

    def test_interruption_exits_130_with_locked_repair_hint(self) -> None:
        """`Ctrl+C` mid-rebuild reports the partial-rebuild risk."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    "Uninstall interrupted.",
                    manual_recovery_command=("uv tool install deepagents-code==9.8.7"),
                    interrupted=True,
                ),
            ),
        ):
            assert run_uninstall_request(name="ollama") == 130
        text = self._console_text(console)
        assert "Aborted" in text
        assert "partially rebuilt" in text
        assert "uv tool install deepagents-code==9.8.7" in text

    def test_os_error_exits_one_without_an_unlocked_repair_hint(self) -> None:
        """An OSError before a locked outcome never uses a guessed command."""
        console = MagicMock()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                side_effect=OSError("no space left on device"),
            ),
        ):
            assert run_uninstall_request(name="ollama") == 1
        text = self._console_text(console)
        assert "no space left on device" in text
        assert "Run manually" not in text

    @pytest.mark.parametrize(
        ("outcome", "expected_code"),
        [
            (ExtraRemovalOutcome(False, "resolver failed"), 1),
            (
                ExtraRemovalOutcome(
                    False,
                    "Uninstall interrupted.",
                    manual_recovery_command="uv tool install safe-command",
                    interrupted=True,
                ),
                130,
            ),
        ],
    )
    def test_markup_like_log_path_is_safe_in_error_output(
        self, outcome: ExtraRemovalOutcome, expected_code: int
    ) -> None:
        """Failure and interruption paths render bracketed paths literally."""
        output = StringIO()
        console = Console(file=output, color_system=None)
        log_path = Path("/tmp/[/red]/uninstall.log")
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config.console", console, create=True),
            patch(
                "deepagents_code.update_check.create_update_log_path",
                return_value=log_path,
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=outcome,
            ),
        ):
            assert run_uninstall_request(name="ollama") == expected_code

        assert str(log_path) in output.getvalue()

    @pytest.mark.parametrize(
        ("argv", "command", "target"),
        [
            (["dcode", "uninstall", "ollama"], "uninstall", "ollama"),
            (["dcode", "--uninstall", "ollama"], None, "ollama"),
        ],
    )
    def test_parser_accepts_subcommand_and_compatibility_flag(
        self, argv: list[str], command: str | None, target: str
    ) -> None:
        from deepagents_code.main import parse_args

        with patch.object(sys, "argv", argv):
            args = parse_args()
        assert args.command == command
        parsed_target = args.uninstall_target if command else args.uninstall
        assert parsed_target == target

    def test_install_and_uninstall_flags_are_mutually_exclusive(self) -> None:
        from deepagents_code.main import parse_args

        with (
            patch.object(
                sys,
                "argv",
                ["dcode", "--install", "ollama", "--uninstall", "nvidia"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["dcode", "--package", "--uninstall", "ollama"],
            ["dcode", "--uninstall", "ollama", "--package"],
        ],
    )
    def test_package_modifier_is_rejected_for_uninstall(self, argv: list[str]) -> None:
        from deepagents_code.main import parse_args

        with (
            patch.object(sys, "argv", argv),
            pytest.raises(SystemExit) as exc_info,
        ):
            parse_args()
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["dcode", "uninstall", "ollama"],
            ["dcode", "--uninstall", "ollama"],
        ],
    )
    def test_cli_main_dispatches_both_forms(self, argv: list[str]) -> None:
        from deepagents_code.main import cli_main

        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = ""
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "stdin", stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch(
                "deepagents_code.client.commands.extras.run_uninstall_request",
                return_value=7,
            ) as request,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 7
        request.assert_called_once_with(name="ollama")

    def test_cli_main_dispatches_empty_compatibility_value(self) -> None:
        """An explicit empty alias value reaches invalid-name validation."""
        from deepagents_code.main import cli_main

        stdin = MagicMock()
        stdin.isatty.return_value = False
        stdin.read.return_value = ""
        with (
            patch.object(sys, "argv", ["dcode", "--uninstall", ""]),
            patch.object(sys, "stdin", stdin),
            patch("deepagents_code.main.check_cli_dependencies"),
            patch(
                "deepagents_code.client.commands.extras.run_uninstall_request",
                return_value=2,
            ) as request,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli_main()
        assert exc_info.value.code == 2
        request.assert_called_once_with(name="")


async def test_uninstall_slash_usage_does_not_invoke_performer() -> None:
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform:
            await app._handle_command("/uninstall")
            await pilot.pause()
        perform.assert_not_awaited()
        messages = [
            message for message in app.query(AppMessage) if not message._is_markdown
        ]
        assert any("Usage: /uninstall" in str(message._content) for message in messages)


async def test_uninstall_slash_absent_extra_is_noop() -> None:
    app = DeepAgentsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.detect_install_method", return_value="uv"
            ),
            patch(
                "deepagents_code.update_check.perform_uninstall_extra",
                new_callable=AsyncMock,
                return_value=ExtraRemovalOutcome(
                    False,
                    "Extra 'ollama' is not installed.",
                    extra_was_absent=True,
                ),
            ) as perform,
        ):
            await app._handle_command("/uninstall ollama")
            await pilot.pause()
        perform.assert_awaited_once()
        messages = [
            message for message in app.query(AppMessage) if not message._is_markdown
        ]
        assert any("not installed" in str(message._content) for message in messages)


async def test_uninstall_slash_non_uv_install_skips_receipt_read() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch(
            "deepagents_code.update_check.detect_install_method", return_value="brew"
        ),
        patch("deepagents_code.update_check.uninstall_extra_command") as command,
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform,
    ):
        await app._uninstall_extra_unlocked("ollama")

    command.assert_not_called()
    perform.assert_not_awaited()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "Homebrew install detected" in text


async def test_uninstall_slash_editable_install_refuses() -> None:
    """An editable install is refused before any environment mutation.

    This guard is what stops a rebuild from replacing a developer's editable
    checkout, so it must be exercised rather than only ever patched to `False`.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    command = MagicMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=True),
        # Report a uv install so only the editable check can refuse: an editable
        # checkout under a uv tool prefix is detected as "uv", and the method
        # alone would let the removal through.
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch("deepagents_code.update_check.uninstall_extra_command", command),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform,
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    command.assert_not_called()
    perform.assert_not_awaited()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "Editable install detected — cannot remove extras automatically." in text
    assert "uv tool install --editable" in text


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("brew", "Homebrew install detected"),
        ("other", "Unsupported install method detected"),
    ],
)
async def test_uninstall_slash_refuses_unsupported_install_methods(
    method: str, expected: str
) -> None:
    """Brew and unknown installs get method-specific guidance, not a receipt error.

    The method gate must run before `uninstall_extra_command` reads the uv
    receipt; those installs have none, so reading first would surface a raw
    `ToolRequirementIntrospectionError` instead of the tailored message.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    command = MagicMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch(
            "deepagents_code.update_check.detect_install_method", return_value=method
        ),
        patch("deepagents_code.update_check.uninstall_extra_command", command),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
        ) as perform,
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    command.assert_not_called()
    perform.assert_not_awaited()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert expected in text


async def test_uninstall_slash_protected_extra_is_refused(tmp_path: Path) -> None:
    """A base-provider extra is refused, not reported as an ordinary no-op."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    log_path = tmp_path / "uninstall.log"
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.create_update_log_path",
            return_value=log_path,
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(
                False, "Extra 'openai' is a base dependency and cannot be removed."
            ),
        ) as perform,
    ):
        await app._handle_uninstall_command("/uninstall openai")

    perform.assert_awaited_once()
    assert log_path.is_file()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "base dependency" in text


async def test_uninstall_slash_reports_failure_with_recovery_command() -> None:
    """A failed removal surfaces the log path and the repair command."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    stale_command = MagicMock(return_value="uv tool install deepagents-code==1.0")
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.uninstall_extra_command",
            stale_command,
        ),
        patch(
            "deepagents_code.update_check.create_update_log_path",
            return_value=Path("/tmp/uninstall.log"),
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(
                False,
                "resolver exploded",
                manual_recovery_command="uv tool install deepagents-code==9.8.7",
            ),
        ),
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    stale_command.assert_not_called()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "resolver exploded" in text
    assert "/tmp/uninstall.log" in text
    assert "uv tool install deepagents-code==9.8.7" in text
    assert "deepagents-code==1.0" not in text


async def test_uninstall_slash_contention_withholds_manual_command() -> None:
    """Lock contention must not tell the user to hand-run `--reinstall`.

    Running the rebuild by hand while another install holds the lock is how the
    tool environment gets corrupted, so the hint has to be suppressed.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(
                False,
                UPDATE_LOCK_CONTENDED_MESSAGE,
                manual_recovery_safe=False,
            ),
        ),
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "already running" in text
    assert "safe-command" not in text
    assert "Run manually" not in text


async def test_uninstall_slash_cancellation_is_reported_and_reraised() -> None:
    """Cancellation reports the partial-rebuild risk and stays uncaught.

    The rebuild replaces the env before restoring it, so a cancelled removal can
    leave dcode unable to start. Swallowing the cancellation would also break
    task shutdown.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(
                False,
                "Uninstall interrupted.",
                manual_recovery_command="uv tool install deepagents-code==9.8.7",
                interrupted=True,
            ),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "Uninstall interrupted" in text
    assert "partially rebuilt" in text
    assert "uv tool install deepagents-code==9.8.7" in text


async def test_uninstall_slash_os_error_is_reported_without_unlocked_hint() -> None:
    """An OSError before an outcome is surfaced without a guessed command."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            side_effect=OSError("no space left on device"),
        ),
    ):
        await app._handle_uninstall_command("/uninstall ollama")

    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "no space left on device" in text
    assert "Run manually" not in text


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/uninstall a b", "Got: a, b"),
        ("/uninstall --force ollama", "takes no options"),
    ],
)
async def test_uninstall_slash_rejects_bad_arguments(
    command: str, expected: str
) -> None:
    """Extra names and stray flags are reported, never silently dropped."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with patch(
        "deepagents_code.update_check.perform_uninstall_extra",
        new_callable=AsyncMock,
    ) as perform:
        await app._handle_uninstall_command(command)

    perform.assert_not_awaited()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert expected in text


async def test_uninstall_slash_canonicalizes_the_extra_name() -> None:
    """Mixed case and underscores resolve to the canonical extra name."""
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.uninstall_extra_command",
            return_value="uv tool install safe-command",
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(True, ""),
        ) as perform,
    ):
        await app._handle_uninstall_command("/uninstall Google_GenAI")

    assert perform.await_args is not None
    assert perform.await_args.args[0] == "google-genai"


async def test_uninstall_slash_reports_success() -> None:
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.uninstall_extra_command",
            return_value="uv tool install safe-command",
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            new_callable=AsyncMock,
            return_value=ExtraRemovalOutcome(True, ""),
        ) as perform,
        patch("deepagents_code.app.invoked_name", return_value="dcode-worktree"),
    ):
        await app._handle_uninstall_command("/uninstall ollama")
    perform.assert_awaited_once()
    text = " ".join(
        str(call.args[0]._content) for call in app._mount_message.await_args_list
    )
    assert "Uninstalled extra 'ollama'" in text
    assert "already gone" in text
    assert "relaunch dcode-worktree" in text


async def test_uninstall_slash_serializes_environment_mutation() -> None:
    """Two concurrent `/uninstall` runs never overlap inside the performer.

    Asserting only `perform.assert_awaited_once()` would pass with the
    `_environment_mutation_lock` deleted outright, so this drives two handlers at
    once and records enter/exit ordering. A rebuild running against a receipt
    another rebuild is midway through replacing is the race the lock exists for.
    """
    app = DeepAgentsApp()
    app._mount_message = AsyncMock()
    events: list[str] = []
    release = asyncio.Event()
    entered = asyncio.Event()

    async def performer(extra: str, **_kwargs: object) -> ExtraRemovalOutcome:
        events.append(f"enter:{extra}")
        entered.set()
        await release.wait()
        events.append(f"exit:{extra}")
        return ExtraRemovalOutcome(True, "")

    with (
        patch("deepagents_code.config._is_editable_install", return_value=False),
        patch("deepagents_code.update_check.detect_install_method", return_value="uv"),
        patch(
            "deepagents_code.update_check.uninstall_extra_command",
            return_value="uv tool install safe-command",
        ),
        patch(
            "deepagents_code.update_check.perform_uninstall_extra",
            side_effect=performer,
        ),
        patch("deepagents_code.app.invoked_name", return_value="dcode"),
    ):
        first = asyncio.create_task(app._handle_uninstall_command("/uninstall ollama"))
        second = asyncio.create_task(app._handle_uninstall_command("/uninstall nvidia"))
        # Both handlers await `asyncio.to_thread` before the performer, so wait
        # for the first to arrive rather than guessing a number of event-loop
        # turns.
        async with asyncio.timeout(5):
            await entered.wait()
        # Give the second task ample opportunity to slip through; the lock is the
        # only thing that can still be holding it back.
        for _ in range(50):
            await asyncio.sleep(0)
        assert events == ["enter:ollama"]

        release.set()
        await asyncio.gather(first, second)

    # Strict alternation: the second removal starts only after the first returns.
    assert events == [
        "enter:ollama",
        "exit:ollama",
        "enter:nvidia",
        "exit:nvidia",
    ]
