"""Handling one `dcode://` link: confirm with the user, then launch.

This runs in the terminal the desktop opened, with the link the browser passed.
Its whole job is the gate between the two.

Why the gate is here and not in the browser: a browser's "open this in dcode?"
prompt names an application, not a request. It does not show which directory
would be opened or what would be typed into the session, and browsers offer to
remember the answer, after which the prompt stops appearing at all. So the
browser's question is "do you trust dcode with links", and this one is "do you
want *this*" — the directory, the agent, the thread, and the prompt text in full,
every time, with the safe answer preselected.

The gate is unconditional. There is no flag, environment variable, or
configuration key that skips it, and no approval-mode parameter a link can
carry: `request` refuses the whole link if it names one. A link therefore cannot
do anything the user could not have done by typing the equivalent command.

Everything a link can ask for is something the session then gates on its own
terms. Opening an unfamiliar directory does not trust it: project hooks, MCP
servers, and extensions still go through their own trust prompts on launch, and
a submitted prompt is a message to the agent, whose tool calls still need
approval.
"""

from __future__ import annotations

import logging
import os
import subprocess  # noqa: S404  # fixed-argv session launch on Windows
import sys
from typing import TYPE_CHECKING

from deepagents_code.url_scheme.registration import RegistrationError, resolve_launcher
from deepagents_code.url_scheme.request import UrlRequestError, parse_open_url

if TYPE_CHECKING:
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent

    from deepagents_code.url_scheme.request import OpenRequest

logger = logging.getLogger(__name__)

EXIT_DECLINED = 1
"""The user declined the request, or there was no way to ask them."""

EXIT_REFUSED = 2
"""The link was malformed or asked for something a link may not ask for."""

_PROMPT_PREVIEW_LINES = 12
"""Prompt lines shown before the rest is summarized as a count."""


def open_from_url(raw: str) -> int:
    """Confirm a `dcode://` link with the user and launch the session it names.

    On POSIX this replaces the process with the session on approval, so a
    successful call does not return.

    Args:
        raw: The link as the operating system delivered it.

    Returns:
        A process exit code: `EXIT_REFUSED` for a link that was not accepted,
            `EXIT_DECLINED` when the user said no or could not be asked, and the
            session's own exit code on Windows.
    """
    try:
        request = parse_open_url(raw)
    except UrlRequestError as exc:
        return _refuse(str(exc))

    if not _confirm(request):
        return _decline()

    try:
        return _launch(request)
    except RegistrationError as exc:
        return _refuse(str(exc))


def _launch(request: OpenRequest) -> int:
    """Start the session the request names.

    The session is launched by the same absolute dcode path a registration
    would record, with only the fields `request` produced. Nothing about the
    approval posture, sandbox, model, or tool set is passed, so the session runs
    with exactly the configuration it would have had if the user had run dcode
    in that directory themselves.

    Propagates `OSError` from `os.execv` when the process cannot be replaced;
    the caller reports it the same way it reports a refused link.

    Args:
        request: The approved request.

    Returns:
        The session's exit code on Windows. On POSIX the process is replaced and
            this does not return.

    Raises:
        RegistrationError: The dcode command could not be located, the directory
            could not be entered, or process replacement failed or returned.
    """
    launcher = resolve_launcher()
    argv = [str(launcher)]
    if request.agent:
        argv += ["-a", request.agent]
    if request.thread:
        argv += ["-r", request.thread]
    if request.prompt:
        argv += ["-m", request.prompt]

    try:
        os.chdir(request.directory)
    except OSError as exc:
        msg = f"Could not enter {request.directory}: {exc}"
        raise RegistrationError(msg) from exc

    if sys.platform == "win32":
        # No `execv` worth having on Windows: it would orphan the console the
        # shell just created for this process.
        completed = subprocess.run(argv, check=False)  # noqa: S603  # resolved dcode path, no shell
        return completed.returncode

    # The argv is dcode's own resolved path plus fields `request` validated, so
    # S606's concern (untrusted arguments to a spawned executable) is covered by
    # the closed parameter set and the approval the user just gave.
    try:
        os.execv(argv[0], argv)  # noqa: S606
    except OSError as exc:
        msg = f"Could not start {launcher}: {exc}"
        raise RegistrationError(msg) from exc
    msg = "os.execv returned unexpectedly"
    raise RegistrationError(msg)


def _confirm(request: OpenRequest) -> bool:
    """Show the request in full and ask whether to open it.

    Args:
        request: The validated request.

    Returns:
        Whether the user explicitly approved. Fail-closed: a terminal that
            cannot be asked, an interrupt, or a picker that will not run all
            return `False`.
    """
    _print_request(request)
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        from deepagents_code.config import console

        console.print(
            "Not opening: a link has to be confirmed in an interactive "
            "terminal, and this one is not interactive.",
            style="bold red",
        )
        return False

    label = (
        "Open session and send this prompt" if request.prompt else "Open this session"
    )
    choice = _pick(label)
    if choice is None:
        return _confirm_by_typing(label)
    return choice


def _print_request(request: OpenRequest) -> None:
    """Print every part of the request, plus where it came from.

    Values are printed with Rich markup disabled: `request` has already refused
    control characters and deceptive Unicode, and this keeps square brackets in
    a path or prompt from being read as styling.

    Args:
        request: The validated request.
    """
    from deepagents_code.config import console

    console.print()
    console.print("A dcode:// link is asking to open a session.", style="bold")
    console.print()
    rows = [("Directory", str(request.directory))]
    if request.agent:
        rows.append(("Agent", request.agent))
    if request.thread:
        rows.append(("Thread", f"resume {request.thread}"))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        console.print(f"  {label.ljust(width)}  {value}", markup=False, highlight=False)

    if request.prompt:
        console.print()
        console.print("  First message, sent as soon as the session starts:")
        for line in _prompt_preview(request.prompt):
            console.print(f"    {line}", markup=False, highlight=False, style="cyan")

    console.print()
    console.print(
        "Any web page can open a link like this. Open it only if you recognize "
        "the directory above.",
        style="yellow",
    )
    console.print()


def _prompt_preview(prompt: str) -> list[str]:
    """Split a prompt into display lines, summarizing an overlong tail.

    Args:
        prompt: The prompt text.

    Returns:
        Lines to print.
    """
    lines = prompt.splitlines() or [prompt]
    if len(lines) <= _PROMPT_PREVIEW_LINES:
        return lines
    hidden = len(lines) - _PROMPT_PREVIEW_LINES
    return [
        *lines[:_PROMPT_PREVIEW_LINES],
        f"... {hidden} more line{'s' if hidden != 1 else ''} not shown",
    ]


def _pick(approve_label: str) -> bool | None:
    """Run the inline two-choice picker, cancel first.

    Args:
        approve_label: Label for the option that opens the session.

    Returns:
        The decision, or `None` when the picker could not run and the caller
            should fall back to typed confirmation.
    """
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.output.defaults import create_output
        from prompt_toolkit.styles import Style

        from deepagents_code.config import get_glyphs

        glyphs = get_glyphs()
        # Cancel leads, so a bare Enter or an Esc declines.
        choices: list[tuple[bool, str]] = [(False, "Cancel"), (True, approve_label)]
        selected = 0

        def rows() -> FormattedText:
            fragments: list[tuple[str, str]] = [
                (
                    "class:prompt.help",
                    (
                        f"{glyphs.arrow_up}/{glyphs.arrow_down}/Tab move "
                        f"{glyphs.separator} Enter select {glyphs.separator} "
                        f"Esc cancel\n"
                    ),
                )
            ]
            for index, (_value, label) in enumerate(choices):
                active = index == selected
                cursor = glyphs.cursor if active else " "
                suffix = "\n" if index < len(choices) - 1 else ""
                fragments.append(
                    (
                        "class:item.current" if active else "class:item",
                        f"{cursor} {label}{suffix}",
                    )
                )
            return FormattedText(fragments)

        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("s-tab")
        def move_up(_event: KeyPressEvent) -> None:
            nonlocal selected
            selected = (selected - 1) % len(choices)

        @bindings.add("down")
        @bindings.add("tab")
        def move_down(_event: KeyPressEvent) -> None:
            nonlocal selected
            selected = (selected + 1) % len(choices)

        @bindings.add("enter")
        def choose(event: KeyPressEvent) -> None:
            event.app.exit(result=choices[selected][0])

        @bindings.add("escape")
        @bindings.add("c-c")
        @bindings.add("c-d")
        def cancel(event: KeyPressEvent) -> None:
            event.app.exit(result=False)

        app: Application[bool] = Application(
            layout=Layout(
                Window(
                    FormattedTextControl(rows, show_cursor=False),
                    height=len(choices) + 1,
                    dont_extend_height=True,
                )
            ),
            key_bindings=bindings,
            style=Style.from_dict(
                {"prompt.help": "ansibrightblack", "item.current": "reverse"}
            ),
            full_screen=False,
            erase_when_done=True,
            output=create_output(stdout=sys.stderr),
        )
    except ImportError:
        logger.debug("Link confirmation picker unavailable", exc_info=True)
        return None

    try:
        return bool(app.run())
    except (EOFError, OSError, RuntimeError):
        logger.debug("Link confirmation picker could not run", exc_info=True)
        return None
    except KeyboardInterrupt:
        return False


def _confirm_by_typing(approve_label: str) -> bool:
    """Ask for a typed confirmation when the picker is unavailable.

    Args:
        approve_label: Label describing what approval does.

    Returns:
        Whether the user typed an explicit yes.
    """
    from deepagents_code.config import console

    console.print(f"{approve_label}? Type 'yes' to continue, anything else to cancel.")
    try:
        answer = input("> ")
    except (EOFError, KeyboardInterrupt, OSError):
        return False
    return answer.strip().lower() == "yes"


def _decline() -> int:
    """Report that nothing was opened.

    Returns:
        `EXIT_DECLINED`.
    """
    from deepagents_code.config import console

    console.print("No session opened.", style="dim")
    return EXIT_DECLINED


def _refuse(message: str) -> int:
    """Report a link that was not accepted, and keep the window readable.

    A link opens a fresh terminal, so a process that exits immediately takes the
    explanation with it. When the terminal is interactive, this waits for a
    keypress so the user gets to read why nothing happened.

    Args:
        message: Explanation to show.

    Returns:
        `EXIT_REFUSED`.
    """
    from deepagents_code.config import console

    console.print()
    console.print(f"Not opening this link: {message}", style="bold red", markup=False)
    console.print()
    if sys.stdin.isatty():
        console.print("Press Enter to close.", style="dim")
        try:
            input()
        except (EOFError, KeyboardInterrupt, OSError):
            logger.debug("Could not wait for acknowledgement", exc_info=True)
    return EXIT_REFUSED
