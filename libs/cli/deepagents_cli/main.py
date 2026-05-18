"""Entry point for the `deepagents` CLI.

As of `deepagents-cli==0.1.0` the interactive Textual REPL has moved to the
[`deepagents-code`](https://pypi.org/project/deepagents-code/) package. This
CLI now exposes only the deployment-oriented commands: `init`, `dev`, and
`deploy`. Bare invocations print a deprecation notice and exit non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import TYPE_CHECKING, Any

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_REPL_REDIRECT_MESSAGE = (
    "The interactive `deepagents` REPL has moved to the `deepagents-code` "
    "package.\n"
    "Install it with:\n\n"
    "  pip install deepagents-code\n\n"
    "Then run `deepagents-code` to start an interactive session.\n\n"
    "The `deepagents` CLI now only provides the `init`, `dev`, and `deploy` "
    "subcommands."
)


def _make_help_action(
    help_fn: Callable[[], None],
) -> type[argparse.Action]:
    """Create an argparse Action that calls `help_fn` and exits.

    argparse requires a *class* (not a callable) for custom actions; this
    factory uses a closure so each subcommand can wire `-h` to its own
    `print_help()`.

    Args:
        help_fn: Callable that prints help text to stdout.

    Returns:
        An argparse Action class wired to the given help function.
    """

    class _ShowHelp(argparse.Action):
        def __init__(
            self,
            option_strings: list[str],
            dest: str = argparse.SUPPRESS,
            default: str = argparse.SUPPRESS,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                **kwargs,
            )

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,  # noqa: ARG002
            values: str | Sequence[Any] | None,  # noqa: ARG002
            option_string: str | None = None,  # noqa: ARG002
        ) -> None:
            with contextlib.suppress(BrokenPipeError):
                help_fn()
            parser.exit()

    return _ShowHelp


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser for the deploy/dev/init CLI.

    Returns:
        Configured `ArgumentParser` with `init`, `dev`, and `deploy`
        subparsers registered.
    """
    from deepagents_cli.deploy import setup_deploy_parsers

    parser = argparse.ArgumentParser(
        prog="deepagents",
        description=(
            "Deep Agents - deployment tooling.\n\n"
            "For interactive chat, install `deepagents-code` instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action=_make_help_action(parser.print_help),
        help="show this help message and exit",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"deepagents-cli {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    setup_deploy_parsers(subparsers, make_help_action=_make_help_action)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the resulting namespace.

    Args:
        argv: Optional argument list (defaults to `sys.argv[1:]`).

    Returns:
        Parsed argparse `Namespace`.
    """
    parser = _build_parser()
    return parser.parse_args(argv)


def cli_main() -> None:
    """Entry point for the `deepagents` and `deepagents-cli` console scripts.

    Raises:
        SystemExit: On `--help`/`--version`, after a subcommand finishes,
            on `KeyboardInterrupt`, when invoked without a subcommand (the
            user is redirected to `deepagents-code`), or when a subcommand
            handler raises `SystemExit` directly (e.g. config validation
            failures in `execute_deploy_command`).
    """
    if sys.platform == "darwin":
        # gRPC (pulled in transitively by LangSmith deps) crashes on macOS
        # when the process forks after gRPC has been initialized. Disable
        # fork support to avoid the abort; the current deploy/dev paths
        # don't fork, so disabling fork support is a safe default —
        # reconsider this env var if a future subcommand spawns workers.
        os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

    args = parse_args()

    try:
        if args.command == "init":
            from deepagents_cli.deploy import execute_init_command

            execute_init_command(args)
        elif args.command == "dev":
            from deepagents_cli.deploy import execute_dev_command

            execute_dev_command(args)
        elif args.command == "deploy":
            from deepagents_cli.deploy import execute_deploy_command

            execute_deploy_command(args)
        else:
            sys.stderr.write(_REPL_REDIRECT_MESSAGE + "\n")
            raise SystemExit(1)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        raise SystemExit(130) from None
