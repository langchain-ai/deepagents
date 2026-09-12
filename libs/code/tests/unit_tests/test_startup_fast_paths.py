"""Tests for lightweight CLI help-only paths.

Each test runs `cli_main` in a subprocess so `sys.modules` reflects only
what that invocation loaded, guarding the startup-perf contract documented
in `CLAUDE.md`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code.main import _show_bare_command_group_help, parse_args

if TYPE_CHECKING:
    import argparse

# Module *prefixes* that must not appear in `sys.modules` after a help-only
# invocation. Using prefixes (rather than an explicit allowlist) catches
# regressions from any new top-level import in `main.py` or `ui.py` that
# pulls in a heavy framework.
_HEAVY_MODULE_PREFIXES = (
    "deepagents.",
    "deepagents_code.agent",
    "deepagents_code.sessions",
    "deepagents_code.model_config",
    "deepagents_code.project_utils",
    "langchain",
    "langgraph",
    "textual",
    "httpx",
)


def _run_cli_main(argv: list[str]) -> subprocess.CompletedProcess[str]:
    # `check_cli_dependencies` is patched purely for environment portability —
    # it only calls `importlib.util.find_spec` (no real imports), so patching
    # it does not hide any heavy module load.
    code = """
        import json
        import sys
        from contextlib import nullcontext
        from unittest.mock import patch

        from deepagents_code.main import cli_main

        argv = ["deepagents", *json.loads(sys.argv[1])]
        # An intentional monorepo SDK pin skew can make `doctor` exit unhealthy;
        # omit that environment-specific check while testing startup bootstrap.
        # Editable installs resolve the pin through `_sdk_requirement_for_cli`.
        requirement_patch = (
            patch(
                "deepagents_code.extras_info._sdk_requirement_for_cli",
                return_value=None,
            )
            if argv[1:] == ["doctor", "--json"]
            else nullcontext()
        )
        try:
            with (
                patch.object(sys, "argv", argv),
                patch("deepagents_code.main.check_cli_dependencies"),
                requirement_patch,
            ):
                cli_main()
        finally:
            prefixes = tuple(json.loads(sys.argv[2]))
            loaded = sorted(
                name for name in sys.modules if name.startswith(prefixes)
            )
            config_module = sys.modules.get("deepagents_code.config")
            bootstrap_state = (
                getattr(config_module, "_bootstrap_state", None)
                if config_module is not None
                else None
            )
            bootstrap_done = (
                bootstrap_state.done if bootstrap_state is not None else None
            )
            print("LOADED_MODULES=" + json.dumps(loaded), file=sys.stderr)
            print("BOOTSTRAP_DONE=" + json.dumps(bootstrap_done), file=sys.stderr)
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code),
            json.dumps(argv),
            json.dumps(_HEAVY_MODULE_PREFIXES),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _read_marker(stderr: str, prefix: str) -> object:
    for line in reversed(stderr.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    msg = f"marker {prefix!r} not found in stderr"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["help"], "Start interactive thread"),
        (["agents"], "dcode agents <command>"),
        (["skills"], "dcode skills <command>"),
        (["threads"], "dcode threads <command>"),
        (["mcp"], "dcode mcp <command>"),
        (["config", "-h"], "dcode config [options]"),
        (["auth"], "dcode auth <command>"),
        (["tools"], "dcode tools <command>"),
    ],
)
def test_help_only_commands_skip_runtime_imports(
    argv: list[str], expected: str
) -> None:
    """Help-only commands must not import heavy runtime modules."""
    result = _run_cli_main(argv)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout

    loaded = _read_marker(result.stderr, "LOADED_MODULES=")
    assert loaded == [], f"unexpected heavy modules loaded: {loaded}"

    bootstrap_done = _read_marker(result.stderr, "BOOTSTRAP_DONE=")
    # Either `deepagents_code.config` was never imported (None) or it was
    # imported transitively but `_ensure_bootstrap()` never ran (False).
    # In neither case may the heavy settings/dotenv path have executed.
    assert bootstrap_done in (None, False), (
        f"settings bootstrap must not run on the fast path; got {bootstrap_done!r}"
    )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["auth", "path"], "/auth.json"),
        (["config", "path", "--json"], '"command": "config path"'),
        (["doctor", "--json"], '"command": "doctor"'),
    ],
)
def test_lightweight_commands_skip_settings_bootstrap(
    argv: list[str], expected: str
) -> None:
    """Lightweight diagnostics commands should avoid settings bootstrap."""
    result = _run_cli_main(argv)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout

    bootstrap_done = _read_marker(result.stderr, "BOOTSTRAP_DONE=")
    assert bootstrap_done in (None, False), (
        f"settings bootstrap must not run for {argv}; got {bootstrap_done!r}"
    )


@pytest.mark.parametrize(
    ("args", "runner"),
    [
        (
            SimpleNamespace(command="config"),
            "deepagents_code.client.commands.config.run_config_command",
        ),
        (
            SimpleNamespace(command="auth", auth_command="path"),
            "deepagents_code.client.commands.auth.run_auth_command",
        ),
        (
            SimpleNamespace(command="doctor"),
            "deepagents_code.doctor.run_doctor_command",
        ),
        (
            SimpleNamespace(command="tools", tools_command="list"),
            "deepagents_code.client.commands.tools.run_tools_command",
        ),
        (
            SimpleNamespace(command="install"),
            "deepagents_code.client.commands.extras.run_install_command",
        ),
    ],
)
def test_lightweight_commands_check_dependency_floors_before_dispatch(
    args: SimpleNamespace, runner: str
) -> None:
    """Every action subcommand warns before an early fast-path exit."""
    from deepagents_code.main import cli_main

    calls: list[str] = []
    with (
        patch.object(sys, "argv", ["dcode", str(args.command)]),
        patch("deepagents_code.main.check_cli_dependencies"),
        patch("deepagents_code.main._install_termination_signal_handlers"),
        patch("deepagents_code.main.parse_args", return_value=args),
        patch(
            "deepagents_code._dep_floor_check.warn_if_editable_deps_stale",
            side_effect=lambda: calls.append("warning"),
        ),
        patch(runner, side_effect=lambda _args: calls.append("dispatch") or 0),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_main()

    assert exc_info.value.code == 0
    assert calls == ["warning", "dispatch"]


def test_auth_credential_resolution_commands_run_settings_bootstrap() -> None:
    """Auth commands that resolve credentials must see dotenv-loaded values."""
    result = _run_cli_main(["auth", "status", "anthropic"])

    assert result.returncode == 0, result.stderr
    bootstrap_done = _read_marker(result.stderr, "BOOTSTRAP_DONE=")
    assert bootstrap_done is True


@pytest.mark.parametrize(
    "argv",
    [
        ["agents", "list"],
        ["skills", "list"],
        ["threads", "list"],
        ["mcp", "login", "example.com"],
        ["config", "get", "interpreter.memory_limit_mb"],
        ["auth", "list"],
        ["tools", "install"],
        ["tools", "list"],
    ],
)
def test_subcommands_bypass_fast_path(argv: list[str]) -> None:
    """When a subcommand is given, the fast path must not fire.

    A `dest=` rename on a subparser would silently swallow the user's
    subcommand if the fast-path's `getattr(..., None) is not None` check
    fell through. This test locks the contract.
    """
    args = parse_args_from(argv)
    assert _show_bare_command_group_help(args) is False


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    """Run `parse_args()` with a controlled argv."""
    from unittest.mock import patch

    with patch.object(sys, "argv", ["deepagents", *argv]):
        return parse_args()
