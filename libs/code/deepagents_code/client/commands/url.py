"""The `dcode url` command group: own the `dcode://` scheme, and handle a link.

`dcode url install` registers dcode as the operating system's handler for
`dcode://` links, so a page can offer to open a project or resume a thread in
dcode. `uninstall` gives the scheme back, and `status` reports what the desktop
currently does with it.

`dcode url open <link>` is the verb a registration points at. It is what the
browser ends up running, so it takes the link and nothing else: no approval,
model, or sandbox flags exist on this parser to be smuggled in through a crafted
link, and the confirmation in `url_scheme.handler` cannot be turned off.

Help rendering for `dcode url -h` and each subcommand is served by the
`ui.show_url_*_help` screens, which do not import this module, so the help path
stays light.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents_code.output import write_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from deepagents_code.output import OutputFormat
    from deepagents_code.url_scheme import HandlerStatus

logger = logging.getLogger(__name__)


def _lazy_ui_help(fn_name: str) -> Callable[[], None]:
    """Return a callable that lazily imports and invokes a `ui` help function.

    Args:
        fn_name: Name of the `show_*_help` function to invoke.

    Returns:
        The deferred callable.
    """

    def _show() -> None:
        from deepagents_code import ui

        getattr(ui, fn_name)()

    return _show


def setup_url_parser(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the `dcode url` command group.

    Args:
        subparsers: The `argparse` subparsers object from the top-level CLI
            parser, onto which the `url` command group is attached.
        make_help_action: Factory that wraps a `show_*` callable into an
            `argparse.Action` so `-h/--help` renders the hand-maintained help
            screens from `deepagents_code.ui`.
    """
    from deepagents_code.output import add_json_output_arg

    url_parser = subparsers.add_parser(
        "url",
        help="Manage the dcode:// URL scheme",
        add_help=False,
    )
    url_parser.add_argument(
        "-h", "--help", action=make_help_action(_lazy_ui_help("show_url_help"))
    )
    add_json_output_arg(url_parser)
    url_sub = url_parser.add_subparsers(dest="url_command")

    install_parser = url_sub.add_parser(
        "install",
        help="Register dcode as the handler for dcode:// links",
        add_help=False,
    )
    install_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_url_install_help")),
    )
    install_parser.add_argument(
        "--terminal",
        choices=["auto", "terminal", "iterm"],
        default="auto",
        help=(
            "macOS only: terminal a link opens the session in. "
            "'auto' matches the terminal you are running this from."
        ),
    )
    add_json_output_arg(install_parser)

    uninstall_parser = url_sub.add_parser(
        "uninstall",
        help="Remove dcode's dcode:// handler",
        add_help=False,
    )
    uninstall_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_url_uninstall_help")),
    )
    add_json_output_arg(uninstall_parser)

    status_parser = url_sub.add_parser(
        "status",
        help="Show what the system does with dcode:// links",
        add_help=False,
    )
    status_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_url_status_help")),
    )
    add_json_output_arg(status_parser)

    # The verb a registered handler runs. Deliberately the smallest parser in
    # the CLI: one positional and a help flag. A browser-supplied link that
    # smuggles an extra token onto the command line has nothing to reach.
    open_parser = url_sub.add_parser(
        "open",
        help="Open a dcode:// link (used by the system handler)",
        add_help=False,
    )
    open_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_lazy_ui_help("show_url_open_help")),
    )
    open_parser.add_argument("url", help="The dcode:// link to open")


def run_url_command(args: argparse.Namespace) -> int:
    """Dispatch a `dcode url` subcommand.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Process exit code.
    """
    subcommand = getattr(args, "url_command", None)
    if subcommand == "install":
        return _run_install(args)
    if subcommand == "uninstall":
        return _run_uninstall(args)
    if subcommand == "status":
        return _run_status(args)
    if subcommand == "open":
        return _run_open(args)

    # `cli_main`'s bare-group help fast path handles `dcode url` with no
    # subcommand, so this is only reached for an unexpected value.
    from deepagents_code import ui

    ui.show_url_help()
    return 0


def _run_open(args: argparse.Namespace) -> int:
    """Handle one `dcode://` link.

    Args:
        args: Parsed CLI namespace. Only `url` is read.

    Returns:
        Process exit code. On POSIX an approved link replaces this process with
            the session, so this does not return in that case.
    """
    from deepagents_code.url_scheme import open_from_url

    return open_from_url(args.url)


def _run_install(args: argparse.Namespace) -> int:
    """Register dcode as the system's `dcode://` handler.

    Args:
        args: Parsed CLI namespace. Reads `output_format` and `terminal`.

    Returns:
        `0` when the handler is registered, `1` when registration failed.
    """
    from deepagents_code.url_scheme import (
        RegistrationError,
        TerminalChoice,
        install_handler,
    )

    output_format: OutputFormat = getattr(args, "output_format", "text")
    terminal = TerminalChoice(getattr(args, "terminal", "auto"))
    try:
        status = install_handler(terminal=terminal)
    except RegistrationError as exc:
        return _emit_error(output_format, command="url install", message=str(exc))

    return _emit_status(
        output_format,
        command="url install",
        status=status,
        message=_install_message(status),
    )


def _install_message(status: HandlerStatus) -> str:
    """Build the success message for `dcode url install`.

    Args:
        status: Status collected after registering.

    Returns:
        The message, including a link the user can try.
    """
    from deepagents_code.url_scheme import URL_SCHEME, build_open_url

    example = build_open_url("~/your/project")
    return (
        f"Registered dcode as the handler for {URL_SCHEME}:// links.\n"
        f"{status.detail}\n"
        f"Try one: {example}\n"
        "Your browser will ask before handing a link to dcode, and dcode asks "
        "again before opening it."
    )


def _run_uninstall(args: argparse.Namespace) -> int:
    """Remove dcode's `dcode://` handler.

    Args:
        args: Parsed CLI namespace. Only `output_format` is read.

    Returns:
        `0` when nothing is registered any more (including when nothing was),
            `1` when removal failed.
    """
    from deepagents_code.url_scheme import RegistrationError, uninstall_handler

    output_format: OutputFormat = getattr(args, "output_format", "text")
    try:
        status, removed = uninstall_handler()
    except RegistrationError as exc:
        return _emit_error(output_format, command="url uninstall", message=str(exc))

    return _emit_status(
        output_format,
        command="url uninstall",
        status=status,
        message=_uninstall_message(removed),
        removed=removed,
    )


def _uninstall_message(removed: Sequence[str]) -> str:
    """Build the message for `dcode url uninstall`.

    Args:
        removed: Artifacts that were removed.

    Returns:
        The message.
    """
    from deepagents_code.url_scheme import URL_SCHEME

    if not removed:
        return f"No dcode {URL_SCHEME}:// handler was installed; nothing to remove."
    listed = "\n".join(f"  {item}" for item in removed)
    return f"Removed dcode's {URL_SCHEME}:// handler:\n{listed}"


def _run_status(args: argparse.Namespace) -> int:
    """Report what the system does with `dcode://` links.

    Args:
        args: Parsed CLI namespace. Only `output_format` is read.

    Returns:
        `0` always. Status is a diagnostic: "not installed" is an answer, not a
            failure, so scripts read `installed` from `--json` rather than the
            exit code.
    """
    from deepagents_code.url_scheme import handler_status

    output_format: OutputFormat = getattr(args, "output_format", "text")
    status = handler_status()
    return _emit_status(
        output_format, command="url status", status=status, message=status.detail
    )


def _emit_status(
    output_format: OutputFormat,
    *,
    command: str,
    status: HandlerStatus,
    message: str,
    removed: Sequence[str] | None = None,
) -> int:
    """Print a handler status as text or JSON.

    Args:
        output_format: `"json"` for machine-readable output, else text.
        command: Command label for the JSON envelope.
        status: Status to report.
        message: Human-readable summary for text output.
        removed: Artifacts removed, for `uninstall`.

    Returns:
        `0`.
    """
    if output_format == "json":
        payload: dict[str, object] = {
            "ok": True,
            "scheme": status.scheme,
            "platform": status.platform,
            "supported": status.supported,
            "installed": status.installed,
            "handler_path": status.handler_path,
            "launcher": None if status.launcher is None else str(status.launcher),
            "default_handler": status.default_handler,
            "detail": status.detail,
        }
        if removed is not None:
            payload["removed"] = list(removed)
        write_json(command, payload)
        return 0

    from deepagents_code.config import console

    console.print()
    console.print(message, markup=False, highlight=False)
    if status.installed and status.handler_path:
        console.print(f"Handler: {status.handler_path}", style="dim", markup=False)
    console.print()
    return 0


def _emit_error(output_format: OutputFormat, *, command: str, message: str) -> int:
    """Print a failure as text or JSON.

    Args:
        output_format: `"json"` for machine-readable output, else text.
        command: Command label for the JSON envelope.
        message: Explanation of what could not be done.

    Returns:
        `1`.
    """
    if output_format == "json":
        write_json(command, {"ok": False, "error": message})
        return 1

    from deepagents_code.config import console

    console.print()
    console.print(message, style="bold red", markup=False, highlight=False)
    console.print()
    return 1
