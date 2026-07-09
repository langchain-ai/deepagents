"""Tests for the root GitHub Action wrapper."""

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = ROOT / "action.yml"
MAIN_PATH = ROOT / "libs" / "code" / "deepagents_code" / "main.py"

EXPECTED_DCODE_INPUT_FLAGS = {
    "agent_name": ("--agent",),
    "interpreter": ("--interpreter", "--no-interpreter"),
    "interpreter_tools": ("--interpreter-tools",),
    "json": ("--json",),
    "max_retries": ("--max-retries",),
    "max_turns": ("--max-turns",),
    "mcp_config": ("--mcp-config",),
    "model": ("--model",),
    "model_params": ("--model-params",),
    "no_mcp": ("--no-mcp",),
    "no_stream": ("--no-stream",),
    "profile_override": ("--profile-override",),
    "prompt": ("--non-interactive",),
    "quiet": ("--quiet",),
    "rubric": ("--rubric",),
    "rubric_max_iterations": ("--rubric-max-iterations",),
    "rubric_model": ("--rubric-model",),
    "sandbox": ("--sandbox",),
    "sandbox_id": ("--sandbox-id",),
    "sandbox_setup": ("--sandbox-setup",),
    "sandbox_snapshot_name": ("--sandbox-snapshot-name",),
    "shell_allow_list": ("--shell-allow-list",),
    "skill": ("--skill",),
    "startup_cmd": ("--startup-cmd",),
    "stdin": ("--stdin",),
    "task_timeout": ("--timeout",),
    "trust_project_mcp": ("--trust-project-mcp",),
}


def _load_action() -> dict:
    return yaml.safe_load(ACTION_PATH.read_text())


def _action_input_names() -> set[str]:
    return set(_load_action()["inputs"])


def _run_dcode_body() -> str:
    """Return the shell body of the "Run dcode" step.

    Reads the step's ``run:`` block scalar directly from the parsed YAML (which
    yields it already dedented), so later steps can be reordered or renamed
    without truncating the capture.
    """
    for step in _load_action()["runs"]["steps"]:
        if step.get("name") == "Run dcode":
            return step["run"]
    msg = "could not locate the 'Run dcode' step"
    raise AssertionError(msg)


def _root_parser_flags() -> set[str]:
    tree = ast.parse(MAIN_PATH.read_text())
    parse_args = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    flags: set[str] = set()

    for node in ast.walk(parse_args):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "parser"
        ):
            options = {
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            }
            flags.update(options)
            if any(
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "BooleanOptionalAction"
                for keyword in node.keywords
            ):
                flags.update(
                    f"--no-{option.removeprefix('--')}" for option in options
                )
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "add_json_output_arg"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "parser"
        ):
            flags.add("--json")

    return flags


def test_root_action_dcode_flags_match_parser() -> None:
    action_inputs = _action_input_names()
    run_body = _run_dcode_body()
    run_flags = set(re.findall(r'"(--[a-z0-9-]+)"', run_body))
    parser_flags = _root_parser_flags()

    expected_inputs = set(EXPECTED_DCODE_INPUT_FLAGS)
    assert expected_inputs <= action_inputs

    expected_flags = {
        flag for flags in EXPECTED_DCODE_INPUT_FLAGS.values() for flag in flags
    }
    assert expected_flags <= run_flags
    assert run_flags <= parser_flags


def test_root_action_does_not_forward_interactive_auto_approve() -> None:
    assert "auto_approve" not in _action_input_names()
    assert "--auto-approve" not in _run_dcode_body()


# ---------------------------------------------------------------------------
# Behavioral tests: execute the real shell from action.yml in a bash subprocess
# (never dcode itself), so they cover the actual action rather than a
# reimplementation. Two harnesses:
#
#   _run_setup — runs the "Run dcode" body up to `OUTPUT_FILE=$(mktemp)`: the
#     pure setup that defines helpers, validates inputs, and assembles `CMD`,
#     with no side effects beyond `exit 1` on invalid input. Inspects the exit
#     code and the assembled command.
#   _run_full  — runs the entire body with `uvx`/`timeout` stubbed on PATH, so
#     the execution/exit-code region below that marker runs for real without
#     launching dcode.
#
# Both prefix the script with `set -eo pipefail` to match GitHub's composite
# `bash` shell, and skip (rather than error) when bash is unavailable — the
# pure-Python drift tests above stay active on a bash-less runner.
# ---------------------------------------------------------------------------

_SETUP_MARKER = "OUTPUT_FILE=$(mktemp)"

# Valid baseline: prompt is required and non-empty; everything else is unset
# (empty string), matching how GitHub passes omitted inputs.
_BASE_INPUTS = {
    "INPUT_PROMPT": "do the thing",
    "INPUT_AGENT_NAME": "deepagents",
    "INPUT_SHELL_ALLOW_LIST": "recommended,git,gh",
}


def _action_setup_script() -> str:
    body = _run_dcode_body()
    setup = body.split(_SETUP_MARKER, 1)[0]
    # Fail loudly if the marker moved/renamed: without the split the slice would
    # include the real dcode invocation instead of stopping at CMD assembly.
    assert setup != body, f"setup marker {_SETUP_MARKER!r} not found in run body"
    assert "CMD=(" in setup, "setup slice should include CMD assembly"
    # Print the assembled command so tests can assert flag presence/absence.
    return setup + '\nprintf "%s\\n" "${CMD[@]}"\n'


def _run_setup(**overrides: str) -> tuple[int, list[str]]:
    if shutil.which("bash") is None:
        pytest.skip("bash is required for action.yml tests")
    env = {**os.environ, **_BASE_INPUTS, **overrides}
    proc = subprocess.run(
        ["bash", "-c", "set -eo pipefail\n" + _action_setup_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    cmd = proc.stdout.splitlines() if proc.returncode == 0 else []
    return proc.returncode, cmd


def _run_full(
    tmp_path: Path, *, dcode_body: str | None = None, **overrides: str
) -> tuple[int, str]:
    """Run the whole "Run dcode" body with `uvx`/`timeout` stubbed on PATH.

    Substitutes a fake for the real dcode call so the execution/exit-code region
    below ``OUTPUT_FILE=$(mktemp)`` runs for real — the timeout arithmetic, the
    stdin vs ``--non-interactive`` dispatch, the ``PIPESTATUS`` capture, and the
    ``$GITHUB_OUTPUT`` write — without ever launching dcode. Returns the process
    exit code and the text written to ``$GITHUB_OUTPUT``.
    """
    if shutil.which("bash") is None:
        pytest.skip("bash is required for action.yml tests")
    # Default fake dcode: emit a line, then exit with $FAKE_DCODE_EXIT (0).
    agent = dcode_body or 'printf "AGENT OUTPUT\\n"; exit "${FAKE_DCODE_EXIT:-0}"'
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # `timeout` shim drops the duration and execs the rest (also lets this run on
    # macOS, which ships no GNU `timeout`). `uvx` shim ignores the
    # `--from PKG dcode` prefix and runs the fake agent body.
    (bindir / "timeout").write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    (bindir / "uvx").write_text(f"#!/usr/bin/env bash\n{agent}\n")
    (bindir / "timeout").chmod(0o755)
    (bindir / "uvx").chmod(0o755)

    gh_output = tmp_path / "github_output"
    gh_output.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(gh_output),
        **_BASE_INPUTS,
        **overrides,
    }
    proc = subprocess.run(
        ["bash", "-c", "set -eo pipefail\n" + _run_dcode_body()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, gh_output.read_text()


@pytest.mark.parametrize(
    ("value", "should_pass"),
    [
        ("", True),  # unset → skipped
        ("30", True),
        ("1", True),
        ("08", True),  # leading zero must not trip octal parsing
        ("0", False),  # zero is not positive
        ("-1", False),
        ("1.5", False),
        ("abc", False),
    ],
)
def test_validate_positive_int(value: str, should_pass: bool) -> None:
    rc, _ = _run_setup(INPUT_TIMEOUT=value)
    assert (rc == 0) is should_pass


@pytest.mark.parametrize(
    ("value", "should_pass"),
    [
        ("", True),
        ("0", True),  # non-negative accepts zero (the key distinction)
        ("3", True),
        ("-1", False),
        ("abc", False),
    ],
)
def test_validate_non_negative_int(value: str, should_pass: bool) -> None:
    rc, _ = _run_setup(INPUT_MAX_RETRIES=value)
    assert (rc == 0) is should_pass


def test_positive_and_non_negative_validators_differ_on_zero() -> None:
    # Guards against a copy-paste swap of the two validators: max_turns must
    # reject 0 while max_retries must accept it.
    assert _run_setup(INPUT_MAX_TURNS="0")[0] != 0
    assert _run_setup(INPUT_MAX_RETRIES="0")[0] == 0


@pytest.mark.parametrize(
    ("value", "expect_flag", "should_pass"),
    [
        ("true", True, True),
        ("false", False, True),
        ("", False, True),
        ("maybe", False, False),
        ("1", False, False),
    ],
)
def test_append_bool_flag(value: str, expect_flag: bool, should_pass: bool) -> None:
    rc, cmd = _run_setup(INPUT_QUIET=value)
    assert (rc == 0) is should_pass
    if should_pass:
        assert ("--quiet" in cmd) is expect_flag


def test_append_value_flag_omits_when_empty_and_appends_with_value() -> None:
    rc, cmd = _run_setup(INPUT_MODEL="")
    assert rc == 0
    assert "--model" not in cmd

    rc, cmd = _run_setup(INPUT_MODEL="openai:gpt-5.5")
    assert rc == 0
    assert cmd[cmd.index("--model") + 1] == "openai:gpt-5.5"


def test_value_inputs_map_to_expected_flags() -> None:
    # The input→flag mapping is real, not decorative: setting the input places
    # its flag+value in the assembled command.
    rc, cmd = _run_setup(
        INPUT_MODEL_PARAMS='{"temperature":0}',
        INPUT_MAX_RETRIES="2",
        INPUT_RUBRIC_MODEL="anthropic:claude-sonnet-4-6",
    )
    assert rc == 0
    assert cmd[cmd.index("--model-params") + 1] == '{"temperature":0}'
    assert cmd[cmd.index("--max-retries") + 1] == "2"
    assert cmd[cmd.index("--rubric-model") + 1] == "anthropic:claude-sonnet-4-6"


@pytest.mark.parametrize(
    ("value", "expected", "should_pass"),
    [
        ("true", "--interpreter", True),
        ("false", "--no-interpreter", True),
        ("", None, True),
        ("bogus", None, False),
    ],
)
def test_interpreter_tristate(
    value: str, expected: str | None, should_pass: bool
) -> None:
    rc, cmd = _run_setup(INPUT_INTERPRETER=value)
    assert (rc == 0) is should_pass
    if should_pass:
        assert ("--interpreter" in cmd) is (expected == "--interpreter")
        assert ("--no-interpreter" in cmd) is (expected == "--no-interpreter")


@pytest.mark.parametrize(
    ("value", "should_pass"),
    [("true", True), ("false", True), ("", True), ("yes", False)],
)
def test_stdin_validation(value: str, should_pass: bool) -> None:
    rc, _ = _run_setup(INPUT_STDIN=value)
    assert (rc == 0) is should_pass


@pytest.mark.parametrize(
    ("stdin", "skill", "should_pass"),
    [
        ("true", "", True),
        ("false", "review", True),  # skill is fine on the non-stdin path
        ("", "review", True),
        ("true", "review", False),  # stdin + skill would run interactively
    ],
)
def test_stdin_skill_mutually_exclusive(
    stdin: str, skill: str, should_pass: bool
) -> None:
    rc, _ = _run_setup(INPUT_STDIN=stdin, INPUT_SKILL=skill)
    assert (rc == 0) is should_pass


def test_empty_prompt_is_rejected() -> None:
    assert _run_setup(INPUT_PROMPT="")[0] != 0


def test_version_pinning_in_assembled_command() -> None:
    rc, cmd = _run_setup(INPUT_CLI_VERSION="0.1.36")
    assert rc == 0
    assert "deepagents-code==0.1.36" in cmd
    assert cmd[cmd.index("deepagents-code==0.1.36") + 1] == "dcode"

    rc, cmd = _run_setup(INPUT_CLI_VERSION="")
    assert rc == 0
    assert "deepagents-code" in cmd
    assert "dcode" in cmd


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
@pytest.mark.parametrize(
    ("value", "should_pass"),
    [
        ("", True),
        ('{"temperature":0}', True),
        ("{not json", False),
        ("[1, 2]", False),  # valid JSON but not an object
    ],
)
def test_validate_json_object(value: str, should_pass: bool) -> None:
    rc, _ = _run_setup(INPUT_MODEL_PARAMS=value)
    assert (rc == 0) is should_pass


# ---------------------------------------------------------------------------
# Execution-region tests: run the full body (past `OUTPUT_FILE=$(mktemp)`) with
# uvx/timeout stubbed — covering timeout arithmetic, exit-code capture, and the
# stdin pipeline, which the setup-slice tests above deliberately exclude.
# ---------------------------------------------------------------------------


def test_full_run_leading_zero_timeout_does_not_crash(tmp_path: Path) -> None:
    # validate_positive_int accepts "08"; bare $((08 * 60)) would abort as
    # invalid octal, so the consumer must force base-10 (10#).
    rc, out = _run_full(tmp_path, INPUT_TIMEOUT="08")
    assert rc == 0
    assert "exit_code=0" in out


def test_full_run_propagates_agent_exit_code(tmp_path: Path) -> None:
    # A non-zero agent must fail the step (via the final `exit $EXIT_CODE`) and
    # be recorded in the exit_code output, not masked by tee.
    rc, out = _run_full(tmp_path, FAKE_DCODE_EXIT="3")
    assert rc == 3
    assert "exit_code=3" in out


def test_full_run_stdin_ignores_producer_sigpipe(tmp_path: Path) -> None:
    # Agent reads one byte and exits 0 without draining stdin. A classic
    # `printf | timeout` pipe would surface printf's SIGPIPE (141); process
    # substitution + PIPESTATUS[0] must report the agent's real code instead.
    rc, out = _run_full(
        tmp_path,
        INPUT_STDIN="true",
        INPUT_PROMPT="x" * 200_000,
        dcode_body="head -c 1 >/dev/null; exit 0",
    )
    assert rc == 0
    assert "exit_code=0" in out
