"""Tests for lightweight CLI help-only paths.

Each test runs `cli_main` in a subprocess so `sys.modules` reflects only
what that invocation loaded, guarding the startup-perf contract documented
in `CLAUDE.md`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap

import pytest

from deepagents_cli.main import _HELP_SPECS, _show_bare_command_group_help, parse_args

# Module *prefixes* that must not appear in `sys.modules` after a help-only
# invocation. Using prefixes (rather than an explicit allowlist) catches
# regressions from any new top-level import in `main.py` or `ui.py` that
# pulls in a heavy framework.
_HEAVY_MODULE_PREFIXES = (
    "deepagents.",
    "deepagents_cli.agent",
    "deepagents_cli.sessions",
    "deepagents_cli.model_config",
    "deepagents_cli.project_utils",
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
        from unittest.mock import patch

        from deepagents_cli.main import cli_main

        argv = ["deepagents", *json.loads(sys.argv[1])]
        with (
            patch.object(sys, "argv", argv),
            patch("deepagents_cli.main.check_cli_dependencies"),
        ):
            cli_main()

        prefixes = tuple(json.loads(sys.argv[2]))
        loaded = sorted(
            name for name in sys.modules if name.startswith(prefixes)
        )
        config_module = sys.modules.get("deepagents_cli.config")
        bootstrap_done = (
            getattr(config_module, "_bootstrap_done", None)
            if config_module is not None
            else None
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
        (["agents"], "deepagents agents <command>"),
        (["skills"], "deepagents skills <command>"),
        (["threads"], "deepagents threads <command>"),
        (["mcp"], "deepagents mcp <command>"),
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
    # Either `deepagents_cli.config` was never imported (None) or it was
    # imported transitively but `_ensure_bootstrap()` never ran (False).
    # In neither case may the heavy settings/dotenv path have executed.
    assert bootstrap_done in (None, False), (
        f"settings bootstrap must not run on the fast path; got {bootstrap_done!r}"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["agents", "list"],
        ["skills", "list"],
        ["threads", "list"],
        ["mcp", "login", "example.com"],
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


def test_unknown_command_bypasses_fast_path() -> None:
    """`command=None` (no command at all) must not trigger help dispatch."""
    args = parse_args_from([])
    assert _show_bare_command_group_help(args) is False


def test_help_specs_covers_every_subparser_group() -> None:
    """Drift guard: every top-level group with sub-subparsers is in `_HELP_SPECS`.

    If a future PR adds a new command group with `add_subparsers(...)` but
    forgets to register it here, the fast path silently regresses for that
    group. This mirrors `test_args.TestHelpScreenDrift`.
    """
    parser = _build_top_level_parser()
    groups_with_subparsers = _top_level_subparser_groups(parser)
    missing = groups_with_subparsers - set(_HELP_SPECS)
    assert not missing, (
        f"Top-level command groups have sub-subparsers but are missing from "
        f"`_HELP_SPECS` in main.py: {sorted(missing)}.\n"
        f"Add an entry mapping each group to its `<group>_command` dest and "
        f"`show_<group>_help` UI function."
    )


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    """Run `parse_args()` with a controlled argv."""
    from unittest.mock import patch

    with patch.object(sys, "argv", ["deepagents", *argv]):
        return parse_args()


def _build_top_level_parser() -> argparse.ArgumentParser:
    """Capture the top-level parser by hooking `ArgumentParser.parse_args`."""
    from typing import Any
    from unittest.mock import patch

    captured: dict[str, argparse.ArgumentParser] = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def _capture(
        self: argparse.ArgumentParser,
        *a: Any,
        **kw: Any,
    ) -> argparse.Namespace:
        captured.setdefault("parser", self)
        return real_parse_args(self, *a, **kw)

    with (
        patch.object(sys, "argv", ["deepagents", "help"]),
        patch.object(argparse.ArgumentParser, "parse_args", _capture),
    ):
        parse_args()

    return captured["parser"]


def _top_level_subparser_groups(parser: argparse.ArgumentParser) -> set[str]:
    """Return names of top-level subparsers that themselves have subparsers."""
    from typing import cast

    groups: set[str] = set()
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        choices = cast("dict[str, argparse.ArgumentParser]", action.choices)
        for name, sub in choices.items():
            for sub_action in sub._actions:
                if isinstance(sub_action, argparse._SubParsersAction):
                    groups.add(name)
                    break
    return groups
