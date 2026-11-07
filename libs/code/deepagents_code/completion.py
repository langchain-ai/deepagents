"""Shell completion scripts for the `dcode` command line.

`dcode completion <shell>` prints a completion script for bash, zsh, or fish.
The candidates are read from the same argparse parser that handles real
invocations, so a new flag or subcommand appears after regenerating the script
instead of needing a second list kept in sync by hand.

Generated scripts are static: write one to a file and source it from the
shell's startup file. `dcode completion -h` prints the per-shell setup.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

SHELLS: tuple[str, ...] = ("bash", "zsh", "fish")
"""Shell dialects accepted by `dcode completion`."""

DEFAULT_PROGRAMS: tuple[str, ...] = ("dcode", "deepagents-code")
"""Console scripts registered in `pyproject.toml` (`[project.scripts]`)."""

_PROGRAM_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_WORD_RE = re.compile(r"\A[^\s'\"\\`$]+\Z")
"""Characters that can appear unquoted in a generated candidate list."""

_INDENT = "    "


@dataclass(frozen=True)
class OptionSpec:
    """One optional argument exposed to shell completion."""

    flags: tuple[str, ...]
    help: str
    takes_value: bool
    optional_value: bool
    choices: tuple[str, ...]


@dataclass(frozen=True)
class PositionalSpec:
    """One positional argument exposed to shell completion."""

    name: str
    help: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    """One command level: the root, a command group, or a leaf command."""

    names: tuple[str, ...]
    """Canonical name followed by any aliases."""

    path: str
    """Space-joined canonical names leading here; empty for the root."""

    help: str = ""
    options: tuple[OptionSpec, ...] = ()
    positionals: tuple[PositionalSpec, ...] = ()
    subcommands: tuple[CommandSpec, ...] = ()


def build_spec(
    parser: argparse.ArgumentParser,
    programs: Sequence[str] = DEFAULT_PROGRAMS,
) -> CommandSpec:
    """Extract the completion tree from an argparse parser.

    Args:
        parser: Fully built CLI parser.
        programs: Command names the generated script registers.

    Returns:
        The root command spec.
    """
    return _command_spec(parser, names=tuple(programs), path="")


def render_script(
    shell: str,
    root: CommandSpec,
    programs: Sequence[str],
) -> str:
    """Render a completion script.

    Args:
        shell: One of `SHELLS`.
        root: Root command spec from `build_spec`.
        programs: Command names to register.

    Returns:
        The script text.

    Raises:
        ValueError: If *shell* is not supported or a program name is unsafe.
    """
    try:
        renderer = _RENDERERS[shell]
    except KeyError:
        msg = f"unsupported shell: {shell!r}"
        raise ValueError(msg) from None
    checked = tuple(_checked_program(name) for name in programs)
    return renderer(root, checked)


def program_arg(value: str) -> str:
    """Validate a `--prog` value for argparse.

    Args:
        value: Raw command name.

    Returns:
        The validated name.

    Raises:
        argparse.ArgumentTypeError: If the name is not a safe command name.
    """
    if not _PROGRAM_RE.match(value):
        msg = f"invalid command name: {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def run_completion_command(args: argparse.Namespace) -> int:
    """Print a completion script for the shell named in *args*.

    Args:
        args: Parsed CLI namespace with `shell` and optional `prog`.

    Returns:
        Process exit code.
    """
    if getattr(args, "shell", None) is None:
        from deepagents_code import ui

        ui.show_completion_help()
        return 0

    from deepagents_code.main import build_parser

    programs = tuple(args.prog) if getattr(args, "prog", None) else DEFAULT_PROGRAMS
    root = build_spec(build_parser(), programs)
    sys.stdout.write(render_script(args.shell, root, programs))
    return 0


def _command_spec(
    parser: argparse.ArgumentParser,
    *,
    names: tuple[str, ...],
    path: str,
    help_text: str = "",
) -> CommandSpec:
    """Convert one argparse parser into a `CommandSpec`.

    Args:
        parser: Parser for this command level.
        names: Canonical name and aliases that reach this level.
        path: Space-joined canonical names leading here.
        help_text: Help text inherited from the parent subparser action.

    Returns:
        The converted command level.
    """
    options: list[OptionSpec] = []
    positionals: list[PositionalSpec] = []
    subparsers: argparse._SubParsersAction | None = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
        elif action.option_strings:
            option = _option_spec(action)
            if option is not None:
                options.append(option)
        else:
            positional = _positional_spec(action)
            if positional is not None:
                positionals.append(positional)
    subcommands = _subcommand_specs(subparsers, path) if subparsers is not None else ()
    return CommandSpec(
        names=names,
        path=path,
        help=help_text,
        options=tuple(options),
        positionals=tuple(positionals),
        subcommands=subcommands,
    )


def _option_spec(action: argparse.Action) -> OptionSpec | None:
    """Convert an optional action.

    Args:
        action: Option action from an argparse parser.

    Returns:
        The option spec, or `None` when the action is hidden.
    """
    if action.help is argparse.SUPPRESS:
        return None
    return OptionSpec(
        flags=tuple(action.option_strings),
        help=_one_line(action.help),
        takes_value=action.nargs != 0,
        optional_value=action.nargs == argparse.OPTIONAL,
        choices=_choices(action),
    )


def _positional_spec(action: argparse.Action) -> PositionalSpec | None:
    """Convert a positional action.

    Args:
        action: Positional action from an argparse parser.

    Returns:
        The positional spec, or `None` when the action is hidden.
    """
    if action.help is argparse.SUPPRESS:
        return None
    return PositionalSpec(
        name=action.dest,
        help=_one_line(action.help),
        choices=_choices(action),
    )


def _subcommand_specs(
    action: argparse._SubParsersAction,
    parent_path: str,
) -> tuple[CommandSpec, ...]:
    """Convert a subparser action into child specs, grouping aliases.

    Args:
        action: Subparsers action holding the child parsers.
        parent_path: Space-joined canonical names leading to this level.

    Returns:
        One spec per distinct child parser, with aliases grouped.
    """
    help_by_name = {
        pseudo.dest: _one_line(pseudo.help)
        for pseudo in getattr(action, "_choices_actions", ())
    }
    grouped: dict[int, list[str]] = {}
    parsers: dict[int, argparse.ArgumentParser] = {}
    for name, parser in action.choices.items():
        key = id(parser)
        grouped.setdefault(key, []).append(name)
        parsers[key] = parser
    specs = []
    for key, names in grouped.items():
        canonical = names[0]
        specs.append(
            _command_spec(
                parsers[key],
                names=tuple(names),
                path=f"{parent_path} {canonical}".strip(),
                help_text=help_by_name.get(canonical, ""),
            )
        )
    return tuple(specs)


def _one_line(text: str | None) -> str:
    """Collapse argparse help text to one line.

    Args:
        text: Raw argparse help text, possibly `None`.

    Returns:
        The help text with whitespace collapsed.
    """
    return "" if text is None else " ".join(text.split())


def _choices(action: argparse.Action) -> tuple[str, ...]:
    """Return an action's fixed choices as strings.

    Args:
        action: Argument action to read.

    Returns:
        The action's choices, or an empty tuple when it has none.
    """
    choices = action.choices
    if not choices:
        return ()
    if isinstance(choices, dict):
        return tuple(str(name) for name in choices)
    return tuple(str(choice) for choice in choices)


def _checked_program(name: str) -> str:
    """Validate a program name used in generated shell code.

    Args:
        name: Command name to validate.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name contains characters unsafe for shell code.
    """
    if not _PROGRAM_RE.match(name):
        msg = f"invalid program name for completion: {name!r}"
        raise ValueError(msg)
    return name


def _pattern(names: Sequence[str]) -> str:
    """Join names into a shell `case` pattern.

    Args:
        names: Case alternatives.

    Returns:
        The alternatives joined with `|`.
    """
    return "|".join(names)


def _walk_levels(
    level: CommandSpec,
    chain: tuple[tuple[str, ...], ...] = (),
) -> Iterator[tuple[CommandSpec, tuple[tuple[str, ...], ...]]]:
    """Yield each level with the name sets of its non-root ancestors.

    Args:
        level: Level to visit.
        chain: Name sets of the ancestors visited so far, excluding the root.

    Yields:
        Each level and the name sets of its non-root ancestors.
    """
    yield level, chain
    next_chain = chain if not level.path else (*chain, level.names)
    for child in level.subcommands:
        yield from _walk_levels(child, next_chain)


def _iter_levels(root: CommandSpec) -> Iterator[CommandSpec]:
    """Yield the root and every nested command level.

    Args:
        root: Root command spec.

    Yields:
        Every level in pre-order.
    """
    for level, _chain in _walk_levels(root):
        yield level


def _iter_options(root: CommandSpec) -> Iterator[OptionSpec]:
    """Yield every option at every level.

    Args:
        root: Root command spec.

    Yields:
        Every option exposed by the tree.
    """
    for level in _iter_levels(root):
        yield from level.options


def _candidates(level: CommandSpec) -> tuple[str, ...]:
    """Return the flag, subcommand, and positional words for one level.

    Args:
        level: Command level to describe.

    Returns:
        Candidate words in declaration order, without duplicates.
    """
    words = [flag for option in level.options for flag in option.flags]
    words.extend(name for child in level.subcommands for name in child.names)
    words.extend(
        choice for positional in level.positionals for choice in positional.choices
    )
    return tuple(dict.fromkeys(word for word in words if _WORD_RE.match(word)))


def _choice_map(root: CommandSpec) -> dict[str, tuple[str, ...]]:
    """Map each flag to the fixed values it accepts.

    Args:
        root: Root command spec.

    Returns:
        A mapping from flag spelling to its accepted values.
    """
    mapping: dict[str, tuple[str, ...]] = {}
    for option in _iter_options(root):
        values = tuple(choice for choice in option.choices if _WORD_RE.match(choice))
        if not values or not option.takes_value:
            continue
        for flag in option.flags:
            mapping.setdefault(flag, values)
    return mapping


def _required_value_flags(root: CommandSpec) -> tuple[str, ...]:
    """Return flags that always consume the following word.

    Args:
        root: Root command spec.

    Returns:
        Flag spellings whose value is required.
    """
    return tuple(
        dict.fromkeys(
            flag
            for option in _iter_options(root)
            if option.takes_value and not option.optional_value
            for flag in option.flags
        )
    )


def _bash_script(root: CommandSpec, programs: Sequence[str]) -> str:
    """Render a bash completion script.

    Args:
        root: Root command spec.
        programs: Command names to register.

    Returns:
        The bash script.
    """
    choices = _choice_map(root)
    lines = [
        f"# bash completion for {'/'.join(programs)}",
        "# Generated by `dcode completion bash`; regenerate after CLI changes.",
        "",
        "_dcode_completion() {",
        "    local cur prev cmd w i",
        "    COMPREPLY=()",
        '    cur="${COMP_WORDS[COMP_CWORD]}"',
        '    prev="${COMP_WORDS[COMP_CWORD-1]}"',
        '    cmd=""',
        "",
    ]
    lines += _bash_command_scan(root)
    lines.append("")
    lines += _bash_value_case(choices, _required_value_flags(root))
    lines += _bash_candidate_case(root)
    lines += ["}", ""]
    lines += [
        f"complete -o bashdefault -o default -F _dcode_completion {program}"
        for program in programs
    ]
    return "\n".join(lines) + "\n"


def _bash_command_scan(root: CommandSpec) -> list[str]:
    """Emit the loop that walks typed words to find the subcommand path.

    Args:
        root: Root command spec.

    Returns:
        Script lines for the word-scanning loop.
    """
    required = _required_value_flags(root)
    lines = [
        "    # Find the deepest subcommand typed before the cursor.",
        "    for ((i = 1; i < COMP_CWORD; i++)); do",
        '        w="${COMP_WORDS[i]}"',
        '        case "$w" in',
        "            --) break ;;",
    ]
    if required:
        lines += [
            f"            {_pattern(required)})",
            "                i=$((i + 1))",
            "                continue",
            "                ;;",
        ]
    lines += ["            -*) continue ;;", "        esac"]
    lines += _transitions(root, _INDENT * 2)
    lines.append("    done")
    return lines


def _transitions(root: CommandSpec, indent: str) -> list[str]:
    """Emit a `case "$cmd"` block that advances on each subcommand name.

    Args:
        root: Root command spec.
        indent: Leading indentation for the emitted block.

    Returns:
        Script lines for the transition block.
    """
    groups = [level for level in _iter_levels(root) if level.subcommands]
    if not groups:
        return []
    lines = [f'{indent}case "$cmd" in']
    for level in groups:
        lines.extend([f'{indent}    "{level.path}")', f'{indent}        case "$w" in'])
        lines.extend(
            f'{indent}            {_pattern(child.names)}) cmd="{child.path}" ;;'
            for child in level.subcommands
        )
        lines.extend([f"{indent}        esac", f"{indent}        ;;"])
    lines.append(f"{indent}esac")
    return lines


def _bash_value_case(
    choices: dict[str, tuple[str, ...]],
    required: tuple[str, ...],
) -> list[str]:
    """Emit completion for an option value, falling back to file names.

    Args:
        choices: Flag-to-values mapping for options with fixed choices.
        required: Flags whose value is required.

    Returns:
        Script lines for the value branch.
    """
    fallback = [flag for flag in required if flag not in choices]
    if not choices and not fallback:
        return []
    lines = ['    if [[ "$cur" != -* ]]; then', '        case "$prev" in']
    for flag, values in choices.items():
        words = " ".join(values)
        lines.append(
            f'            {flag}) COMPREPLY=( $(compgen -W "{words}" -- "$cur") );'
            " return 0 ;;"
        )
    if fallback:
        lines.append(f"            {_pattern(fallback)}) return 0 ;;")
    lines += ["        esac", "    fi"]
    return lines


def _bash_candidate_case(root: CommandSpec) -> list[str]:
    """Emit candidate words for every command level.

    Args:
        root: Root command spec.

    Returns:
        Script lines for the candidate branch.
    """
    lines = ['    case "$cmd" in']
    for level in _iter_levels(root):
        words = " ".join(_candidates(level))
        lines.extend(
            [
                f'        "{level.path}")',
                f'            COMPREPLY=( $(compgen -W "{words}" -- "$cur") )',
                "            ;;",
            ]
        )
    lines.append("    esac")
    return lines


def _zsh_script(root: CommandSpec, programs: Sequence[str]) -> str:
    """Render a zsh completion script.

    Args:
        root: Root command spec.
        programs: Command names to register.

    Returns:
        The zsh script.
    """
    choices = _choice_map(root)
    lines = [
        f"#compdef {' '.join(programs)}",
        "",
        "_dcode_completion() {",
        "    local cur prev cmd w i",
        "    local -a candidates",
        '    cur="${words[CURRENT]}"',
        '    prev="${words[CURRENT-1]}"',
        '    cmd=""',
        "",
    ]
    lines += _zsh_command_scan(root)
    lines.append("")
    lines += _zsh_value_case(choices, _required_value_flags(root))
    lines += _zsh_candidate_case(root)
    lines += [
        "}",
        "",
        "if (( $+functions[compdef] )); then",
        f"    compdef _dcode_completion {' '.join(programs)}",
        "fi",
        "# When autoloaded from `fpath`, the file body runs as the function",
        "# named after the file, so a fixed function name cannot detect it.",
        "# Match the execution context instead: `loadautofunc` on the first",
        "# call, `shfunc` when a loaded copy is invoked directly later.",
        'if [[ "${zsh_eval_context[-1]}" == (loadautofunc|shfunc) ]]; then',
        '    _dcode_completion "$@"',
        "fi",
    ]
    return "\n".join(lines) + "\n"


def _zsh_command_scan(root: CommandSpec) -> list[str]:
    """Emit the zsh word walk mirroring `_bash_command_scan`.

    Args:
        root: Root command spec.

    Returns:
        Script lines for the word-scanning loop.
    """
    required = _required_value_flags(root)
    lines = [
        "    # Find the deepest subcommand typed before the cursor.",
        "    for ((i = 2; i < CURRENT; i++)); do",
        '        w="${words[i]}"',
        '        case "$w" in',
        "            --) break ;;",
    ]
    if required:
        lines += [
            f"            {_pattern(required)})",
            "                (( i += 1 ))",
            "                continue",
            "                ;;",
        ]
    lines += ["            -*) continue ;;", "        esac"]
    lines += _transitions(root, _INDENT * 2)
    lines.append("    done")
    return lines


def _zsh_value_case(
    choices: dict[str, tuple[str, ...]],
    required: tuple[str, ...],
) -> list[str]:
    """Emit zsh completion for an option value.

    Args:
        choices: Flag-to-values mapping for options with fixed choices.
        required: Flags whose value is required.

    Returns:
        Script lines for the value branch.
    """
    fallback = [flag for flag in required if flag not in choices]
    if not choices and not fallback:
        return []
    lines = ['    if [[ "$cur" != -* ]]; then', '        case "$prev" in']
    for flag, values in choices.items():
        lines.append(f"            {flag}) compadd -- {' '.join(values)}; return ;;")
    if fallback:
        lines.append(f"            {_pattern(fallback)}) _files; return ;;")
    lines += ["        esac", "    fi"]
    return lines


def _zsh_candidate_case(root: CommandSpec) -> list[str]:
    """Emit candidate words for every command level.

    Args:
        root: Root command spec.

    Returns:
        Script lines for the candidate branch.
    """
    lines = ['    case "$cmd" in']
    for level in _iter_levels(root):
        words = " ".join(_candidates(level))
        lines.extend(
            [
                f'        "{level.path}")',
                f"            candidates=({words})",
                '            (( ${#candidates[@]} )) && compadd -- "${candidates[@]}"',
                "            ;;",
            ]
        )
    lines.append("    esac")
    return lines


def _fish_script(root: CommandSpec, programs: Sequence[str]) -> str:
    """Render a fish completion script.

    Args:
        root: Root command spec.
        programs: Command names to register.

    Returns:
        The fish script.
    """
    lines = [
        f"# fish completion for {'/'.join(programs)}",
        "# Generated by `dcode completion fish`; regenerate after CLI changes.",
        "",
        "function __dcode_prev_option",
        "    set -l tokens (commandline -opc)",
        "    test (count $tokens) -gt 0; and contains -- $tokens[-1] $argv",
        "end",
        "",
    ]
    for program in programs:
        for level, chain in _walk_levels(root):
            lines += _fish_level_lines(level, chain, program)
    return "\n".join(lines) + "\n"


def _fish_level_lines(
    level: CommandSpec,
    chain: tuple[tuple[str, ...], ...],
    program: str,
) -> list[str]:
    """Emit fish completions for one command level.

    Args:
        level: Command level to describe.
        chain: Name sets of the level's non-root ancestors.
        program: Command name to register completions for.

    Returns:
        Fish completion lines.
    """
    condition = _fish_condition(level, chain)
    lines: list[str] = []
    for option in level.options:
        for flag in option.flags:
            spec = _fish_flag(flag, option.takes_value and not option.optional_value)
            if spec is not None:
                lines.append(
                    f"complete -c {program} -n {_fish_quote(condition)} "
                    f"{spec} -d {_fish_quote(option.help)}"
                )
        if option.takes_value and option.choices:
            values = _fish_quote(" ".join(option.choices))
            for flag in option.flags:
                prev = _fish_quote(f"__dcode_prev_option {flag}")
                lines.append(f"complete -c {program} -f -n {prev} -a {values}")
    lines.extend(
        f"complete -c {program} -f -n {_fish_quote(condition)} "
        f"-a {_fish_quote(name)} -d {_fish_quote(child.help)}"
        for child in level.subcommands
        for name in child.names
    )
    for positional in level.positionals:
        if positional.choices:
            values = _fish_quote(" ".join(positional.choices))
            lines.append(
                f"complete -c {program} -f -n {_fish_quote(condition)} -a {values}"
            )
    return lines


def _fish_condition(
    level: CommandSpec,
    chain: tuple[tuple[str, ...], ...],
) -> str:
    """Build the fish condition selecting one command level.

    Args:
        level: Command level to describe.
        chain: Name sets of the level's non-root ancestors.

    Returns:
        A fish condition expression.
    """
    if not level.path:
        names = _child_names(level)
        if not names:
            return "true"
        return f"not __fish_seen_subcommand_from {names}"
    parts = [f"__fish_seen_subcommand_from {' '.join(names)}" for names in chain]
    parts.append(f"__fish_seen_subcommand_from {' '.join(level.names)}")
    child_names = _child_names(level)
    if child_names:
        parts.append(f"not __fish_seen_subcommand_from {child_names}")
    return "; and ".join(parts)


def _child_names(level: CommandSpec) -> str:
    """Return every child name at a level, aliases included.

    Args:
        level: Command level to describe.

    Returns:
        Space-separated child names, without duplicates.
    """
    return " ".join(
        dict.fromkeys(name for child in level.subcommands for name in child.names)
    )


def _fish_flag(flag: str, takes_value: bool) -> str | None:
    """Return the fish `-s`/`-l` spec for a flag.

    Args:
        flag: Flag spelling, such as `-M` or `--model`.
        takes_value: Whether the flag always consumes a value.

    Returns:
        The fish flag spec, or `None` for flags it cannot express.
    """
    if flag.startswith("--"):
        spec = f"-l {_fish_quote(flag[2:])}"
    elif flag.startswith("-") and len(flag) == 2:  # noqa: PLR2004  # short option
        spec = f"-s {_fish_quote(flag[1:])}"
    else:
        return None
    if takes_value:
        spec += " -r"
    return spec


def _fish_quote(value: str) -> str:
    """Quote one fish word or description.

    Args:
        value: Text to quote.

    Returns:
        The text wrapped in fish single quotes.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


_RENDERERS: dict[str, Callable[[CommandSpec, Sequence[str]], str]] = {
    "bash": _bash_script,
    "zsh": _zsh_script,
    "fish": _fish_script,
}
