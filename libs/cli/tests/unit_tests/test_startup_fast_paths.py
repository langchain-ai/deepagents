"""Tests for lightweight CLI help-only paths."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_HEAVY_RUNTIME_MODULES = (
    "deepagents_cli.agent",
    "deepagents_cli.sessions",
    "deepagents_cli.model_config",
    "deepagents_cli.project_utils",
)


def _run_cli_main(argv: list[str]) -> subprocess.CompletedProcess[str]:
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

        loaded = [
            name
            for name in json.loads(sys.argv[2])
            if name in sys.modules
        ]
        print("LOADED_MODULES=" + json.dumps(loaded))
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code),
            json.dumps(argv),
            json.dumps(_HEAVY_RUNTIME_MODULES),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


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
def test_bare_help_commands_skip_runtime_imports(
    argv: list[str], expected: str
) -> None:
    """Bare command groups should print help without runtime imports."""
    result = _run_cli_main(argv)

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout

    marker = result.stdout.rsplit("LOADED_MODULES=", maxsplit=1)[-1].splitlines()[0]
    assert json.loads(marker) == []
