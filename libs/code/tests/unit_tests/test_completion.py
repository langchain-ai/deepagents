"""Tests for `dcode completion` script generation."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code.completion import (
    SHELLS,
    CommandSpec,
    OptionSpec,
    build_spec,
    program_arg,
    render_script,
    run_completion_command,
)
from deepagents_code.main import _show_bare_command_group_help, build_parser, parse_args

if TYPE_CHECKING:
    from pathlib import Path


def _find(root: CommandSpec, path: str) -> CommandSpec | None:
    """Return the command level at *path*, if any."""
    if root.path == path:
        return root
    for child in root.subcommands:
        found = _find(child, path)
        if found is not None:
            return found
    return None


def _flags(level: CommandSpec) -> dict[str, OptionSpec]:
    """Map every flag spelling at *level* to its option spec."""
    return {flag: option for option in level.options for flag in option.flags}


@pytest.fixture(scope="module")
def root() -> CommandSpec:
    """The completion tree for the real CLI parser."""
    return build_spec(build_parser())


class TestBuildSpec:
    def test_top_level_commands(self, root: CommandSpec) -> None:
        names = [child.names[0] for child in root.subcommands]
        for expected in (
            "help",
            "agents",
            "skills",
            "threads",
            "completion",
            "doctor",
            "tools",
        ):
            assert expected in names

    def test_aliases_share_a_level(self, root: CommandSpec) -> None:
        plugin = _find(root, "plugin")
        assert plugin is not None
        assert plugin.names == ("plugin", "plugins")

        auth_remove = _find(root, "auth remove")
        assert auth_remove is not None
        assert auth_remove.names == ("remove", "rm", "delete")

    def test_nested_options_and_choices(self, root: CommandSpec) -> None:
        threads_list = _find(root, "threads list")
        assert threads_list is not None
        sort = _flags(threads_list)["--sort"]
        assert sort.choices == ("created", "updated")
        assert sort.takes_value is True

        threads_delete = _find(root, "threads delete")
        assert threads_delete is not None
        assert [positional.name for positional in threads_delete.positionals] == [
            "thread_id"
        ]

    def test_completion_shell_choices(self, root: CommandSpec) -> None:
        completion = _find(root, "completion")
        assert completion is not None
        shell = next(
            positional
            for positional in completion.positionals
            if positional.name == "shell"
        )
        assert shell.choices == SHELLS
        assert "--prog" in _flags(completion)


class TestRenderScript:
    @pytest.mark.parametrize("shell", SHELLS)
    def test_renders_registered_programs(self, shell: str, root: CommandSpec) -> None:
        script = render_script(shell, root, ("dcode", "deepagents-code"))
        assert "dcode" in script
        assert "deepagents-code" in script
        assert "created updated" in script
        assert "bash zsh fish" in script

    def test_bash_registers_both_scripts(self, root: CommandSpec) -> None:
        script = render_script("bash", root, ("dcode", "deepagents-code"))
        assert "complete -o bashdefault -o default -F _dcode_completion dcode" in script
        assert (
            "complete -o bashdefault -o default -F _dcode_completion deepagents-code"
            in script
        )

    def test_zsh_uses_compdef(self, root: CommandSpec) -> None:
        script = render_script("zsh", root, ("dcode", "deepagents-code"))
        assert script.startswith("#compdef dcode deepagents-code")
        assert "compdef _dcode_completion dcode deepagents-code" in script

    def test_fish_registers_both_scripts(self, root: CommandSpec) -> None:
        script = render_script("fish", root, ("dcode", "deepagents-code"))
        assert "complete -c dcode " in script
        assert "complete -c deepagents-code " in script

    def test_custom_program_replaces_defaults(self, root: CommandSpec) -> None:
        script = render_script("bash", root, ("mydcode",))
        assert "_dcode_completion mydcode" in script
        assert "deepagents-code" not in script

    def test_unknown_shell_rejected(self, root: CommandSpec) -> None:
        with pytest.raises(ValueError, match="unsupported shell"):
            render_script("powershell", root, ("dcode",))

    def test_unsafe_program_rejected(self, root: CommandSpec) -> None:
        with pytest.raises(ValueError, match="invalid program name"):
            render_script("bash", root, ("dcode; rm -rf /",))

    @pytest.mark.parametrize(("shell", "binary"), [("bash", "bash"), ("zsh", "zsh")])
    def test_scripts_parse(
        self,
        shell: str,
        binary: str,
        root: CommandSpec,
        tmp_path,
    ) -> None:
        executable = shutil.which(binary)
        if executable is None:
            pytest.skip(f"{binary} is not installed")
        script_path = tmp_path / f"dcode.{shell}"
        script_path.write_text(render_script(shell, root, ("dcode", "deepagents-code")))

        result = subprocess.run(
            [executable, "-n", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr

    def test_custom_program_name_is_not_split(self, root: CommandSpec) -> None:
        script = render_script("bash", root, ("my-dcode",))
        assert "_dcode_completion my-dcode" in script


@pytest.fixture(scope="module")
def bash_script(tmp_path_factory, root: CommandSpec) -> Path:
    """The generated bash script, written once per module."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")
    path = tmp_path_factory.mktemp("completion") / "dcode.bash"
    path.write_text(render_script("bash", root, ("dcode", "deepagents-code")))
    return path


class TestBashCompletion:
    def _complete(self, script_path: Path, words: list[str], cword: int) -> list[str]:
        array = " ".join(shlex.quote(word) for word in words)
        script = "\n".join(
            [
                f"source {shlex.quote(str(script_path))}",
                f"COMP_WORDS=({array})",
                f"COMP_CWORD={cword}",
                "COMPREPLY=()",
                "_dcode_completion",
                (
                    'for candidate in "${COMPREPLY[@]}"; do'
                    ' printf "%s\\n" "$candidate"; done'
                ),
            ]
        )
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.split()

    def test_root_candidates(self, bash_script) -> None:
        candidates = self._complete(bash_script, ["dcode", ""], 1)
        for expected in ("threads", "completion", "agents", "--model", "-h"):
            assert expected in candidates

    def test_subcommand_candidates(self, bash_script) -> None:
        candidates = self._complete(bash_script, ["dcode", "threads", ""], 2)
        assert set(candidates) >= {"list", "ls", "delete", "--json"}

    def test_fixed_choice_value(self, bash_script) -> None:
        candidates = self._complete(
            bash_script,
            ["dcode", "threads", "list", "--sort", ""],
            4,
        )
        assert candidates == ["created", "updated"]

    def test_required_value_is_not_treated_as_subcommand(self, bash_script) -> None:
        candidates = self._complete(
            bash_script,
            ["dcode", "--model", "some-model", "threads", ""],
            4,
        )
        assert "list" in candidates

    def test_positional_choices(self, bash_script) -> None:
        candidates = self._complete(bash_script, ["dcode", "completion", ""], 2)
        assert set(candidates) >= set(SHELLS)

    def test_free_form_value_falls_back_to_files(self, bash_script) -> None:
        assert (
            self._complete(bash_script, ["dcode", "threads", "list", "--limit", ""], 4)
            == []
        )


@pytest.fixture(scope="module")
def zsh_script(tmp_path_factory, root: CommandSpec) -> Path:
    """The generated zsh script, saved under the name `fpath` autoloads."""
    if shutil.which("zsh") is None:
        pytest.skip("zsh is not installed")
    path = tmp_path_factory.mktemp("zfunc") / "_dcode"
    path.write_text(render_script("zsh", root, ("dcode", "deepagents-code")))
    return path


class TestZshCompletion:
    """Run the script the way `compinit` does, not just the syntax check."""

    def _run(self, script_path: Path, body: list[str], home: Path) -> str:
        lines = [
            f"fpath=({shlex.quote(str(script_path.parent))} $fpath)",
            "autoload -Uz compinit",
            "compinit -u",
            'compadd() { print -r -- "COMPADD:$*" }',
            "words=(dcode thr)",
            "CURRENT=2",
            *body,
        ]
        result = subprocess.run(
            ["zsh", "-f", "-c", "\n".join(lines)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "HOME": str(home), "ZDOTDIR": str(home)},
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_autoload_completes_on_every_tab(self, zsh_script, tmp_path) -> None:
        # The first TAB autoloads `_dcode` from `fpath`; later TABs reach
        # whatever `compinit` maps `dcode` to once the script registers itself.
        second_tab = '"${_comps[dcode]}"'
        output = self._run(zsh_script, ["_dcode", second_tab, second_tab], tmp_path)
        assert output.count("COMPADD:") == 3
        assert "COMPADD:--" in output

    def test_sourcing_registers_without_completing(self, zsh_script, tmp_path) -> None:
        output = self._run(
            zsh_script,
            [
                f"source {shlex.quote(str(zsh_script))}",
                'print -r -- "comps=${_comps[dcode]:-<unset>}"',
            ],
            tmp_path,
        )
        assert "COMPADD:" not in output
        assert "comps=_dcode_completion" in output


def _scoped_parser() -> argparse.ArgumentParser:
    """A parser whose subcommands reuse a flag with different meanings."""
    parser = argparse.ArgumentParser(prog="dcode")
    parser.add_argument("-r", "--resume", nargs="?")
    parser.add_argument("--model")
    commands = parser.add_subparsers(dest="command")
    alpha = commands.add_parser("alpha")
    alpha.add_argument("--mode", choices=["fast", "slow"])
    alpha.add_argument("--name")
    beta = commands.add_parser("beta")
    beta.add_argument("--mode", choices=["red", "blue"])
    beta.add_argument("--name", action="store_true")
    beta.add_subparsers(dest="beta_command").add_parser("go")
    return parser


def _complete_bash(script: Path, words: list[str], cwd: Path) -> list[str]:
    """Return bash candidates for *words*, the last being the cursor word."""
    body = "\n".join(
        [
            f"source {shlex.quote(str(script))}",
            f"COMP_WORDS=({' '.join(shlex.quote(word) for word in words)})",
            f"COMP_CWORD={len(words) - 1}",
            "_dcode_completion",
            'printf "%s\\n" "${COMPREPLY[@]}"',
        ]
    )
    return _shell_lines(["bash", "--noprofile", "--norc", "-c", body], cwd)


def _complete_zsh(script: Path, words: list[str], cwd: Path) -> list[str]:
    """Return zsh candidates for *words*, the last being the cursor word."""
    body = "\n".join(
        [
            'compadd() { [[ "$1" == -- ]] && shift; print -rl -- "$@" }',
            "_files() { : }",
            f"source {shlex.quote(str(script))}",
            f"words=({' '.join(shlex.quote(word) for word in words)})",
            f"CURRENT={len(words)}",
            "_dcode_completion",
        ]
    )
    return _shell_lines(["zsh", "-f", "-c", body], cwd)


def _complete_fish(script: Path, words: list[str], cwd: Path) -> list[str]:
    """Return fish candidates for *words*, the last being the cursor word."""
    body = "source $argv[1]; complete -C $argv[2]"
    command = ["fish", "--no-config", "-c", body, str(script), " ".join(words)]
    return [line.split("\t")[0] for line in _shell_lines(command, cwd)]


def _shell_lines(command: list[str], cwd: Path) -> list[str]:
    """Run a shell command and return its non-empty output lines."""
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=cwd
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line]


_COMPLETERS = {"bash": _complete_bash, "zsh": _complete_zsh, "fish": _complete_fish}


class TestArgparseSemantics:
    """Each shell resolves the command level the way argparse parses it."""

    @pytest.fixture(params=SHELLS)
    def complete(self, request, tmp_path):
        shell = request.param
        if shutil.which(shell) is None:
            pytest.skip(f"{shell} is not installed")
        script = tmp_path / f"dcode.{shell}"
        root = build_spec(_scoped_parser(), ("dcode",))
        script.write_text(render_script(shell, root, ("dcode",)))
        cwd = tmp_path / "empty"
        cwd.mkdir()
        return lambda words: set(_COMPLETERS[shell](script, ["dcode", *words], cwd))

    @pytest.mark.parametrize(
        ("words", "present", "absent"),
        [
            # `--resume` consumes `alpha` as its optional value.
            (["--resume", "alpha", ""], {"alpha", "beta"}, {"--mode"}),
            # An option-like word is not an optional value.
            (["--resume", "--model", "m", "alpha", "--m"], {"--mode"}, {"--model"}),
            (["alpha", "--mode", ""], {"fast", "slow"}, {"red", "blue"}),
            (["beta", "--mode", ""], {"red", "blue"}, {"fast", "slow"}),
            # `beta --name` is a switch, so `go` is a subcommand, not its value.
            (["beta", "--name", "go", "-"], {"--help"}, {"--mode", "--name"}),
        ],
    )
    def test_command_level(self, complete, words, present, absent) -> None:
        candidates = complete(words)
        assert present <= candidates
        assert not absent & candidates


class TestProgramArg:
    def test_accepts_command_names(self) -> None:
        assert program_arg("dcode") == "dcode"
        assert program_arg("my.dcode+dev") == "my.dcode+dev"

    @pytest.mark.parametrize("value", ["", "-dcode", "dcode shell", "dcode;ls"])
    def test_rejects_unsafe_names(self, value: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="invalid command name"):
            program_arg(value)


class TestRunCompletionCommand:
    def test_writes_script_to_stdout(self, capsys) -> None:
        assert run_completion_command(argparse.Namespace(shell="bash")) == 0
        assert capsys.readouterr().out.startswith("# bash completion for dcode")

    def test_custom_program(self, capsys) -> None:
        args = argparse.Namespace(shell="bash", prog=["mydcode"])
        assert run_completion_command(args) == 0
        output = capsys.readouterr().out
        assert "mydcode" in output
        assert "deepagents-code" not in output

    def test_missing_shell_shows_help(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "deepagents_code.ui.show_completion_help",
            lambda: calls.append("help"),
        )

        assert run_completion_command(argparse.Namespace()) == 0
        assert calls == ["help"]


class TestParserIntegration:
    def test_shell_argument_skips_bare_help_fast_path(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["deepagents", "completion", "bash"])

        args = parse_args()

        assert args.shell == "bash"
        assert _show_bare_command_group_help(args) is False

    def test_bare_completion_shows_help(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["deepagents", "completion"])

        args = parse_args()

        assert args.shell is None
        with patch("deepagents_code.ui.show_completion_help") as help_fn:
            assert _show_bare_command_group_help(args) is True
        help_fn.assert_called_once_with()

    def test_invalid_shell_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "argv", ["deepagents", "completion", "powershell"])

        with pytest.raises(SystemExit):
            parse_args()
