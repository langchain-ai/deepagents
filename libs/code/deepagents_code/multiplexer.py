"""Terminal multiplexer detection and option probing.

A tmux pane is not the terminal the user is looking at. tmux owns the pty,
rewrites `TERM`, and decides on its own which escape sequences reach the outer
terminal — so several `deepagents-code` features depend on tmux options the
user has to opt into. This module reads those options so `dcode doctor` can
report them.

Probing shells out to `tmux`, so it is deliberately kept out of the startup
path: nothing here runs unless the user invokes `dcode doctor`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

TMUX_QUERY_TIMEOUT_SECONDS = 2.0
"""Ceiling on each `tmux` invocation so a wedged server cannot hang `doctor`."""

FOCUS_EVENTS = "focus-events"
ALLOW_PASSTHROUGH = "allow-passthrough"
SET_CLIPBOARD = "set-clipboard"

REPORTED_OPTIONS: tuple[str, ...] = (FOCUS_EVENTS, ALLOW_PASSTHROUGH, SET_CLIPBOARD)
"""tmux options that change how `deepagents-code` behaves.

- `focus-events` gates `FocusIn`/`FocusOut` delivery, without which an
  unfocused pane keeps drawing a blinking cursor.
- `allow-passthrough` gates the `DCS tmux;` wrapper, without which terminal
  progress and OSC 52 clipboard writes are dropped.
- `set-clipboard` must be `on` for an application's OSC 52 to reach the
  outer terminal's clipboard.
"""

_SHOW_OPTIONS_ARGS: tuple[str, ...] = ("show-options", "-s", ";", "show-options", "-gw")
"""Read both scopes in one invocation.

`focus-events` and `set-clipboard` are server options while
`allow-passthrough` is a window option, so a single `show-options -g` (session
scope) returns none of them. `;` is passed as its own argument, which tmux
reads as a command separator.
"""


@dataclass(frozen=True)
class TmuxStatus:
    """Facts about the tmux server hosting the current pane."""

    version: str | None
    """Reported `tmux -V` string, or `None` when the query failed."""

    options: Mapping[str, str] | None
    """Values of `REPORTED_OPTIONS`, or `None` when the query failed.

    A present mapping missing a key means tmux answered but does not know that
    option — an older server without `allow-passthrough`, say. That is a
    different fact from a failed probe, so the two must not collapse.
    """


def inside_tmux(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the process is running inside a tmux pane.

    Args:
        env: Environment mapping to read. Defaults to the process environment.

    Returns:
        `True` when tmux exported its socket path into this process.
    """
    return bool((env if env is not None else os.environ).get("TMUX"))


def _run_tmux(args: Sequence[str]) -> str | None:
    """Run `tmux` with a fixed argument list and return its stdout.

    Args:
        args: Arguments appended after the `tmux` executable.

    Returns:
        Captured stdout, or `None` when tmux is missing, times out, fails, or
        cannot be executed. Callers treat `None` as "unknown", never as an
        error worth surfacing.
    """
    import subprocess  # noqa: S404  # fixed argv, no shell

    try:
        result = subprocess.run(  # noqa: S603  # fixed argv, no shell, no user input
            ["tmux", *args],  # noqa: S607  # resolved from PATH like `git`
            capture_output=True,
            text=True,
            timeout=TMUX_QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        logger.debug("tmux not on PATH; skipping multiplexer probe")
        return None
    except subprocess.TimeoutExpired:
        logger.debug("tmux %s timed out", " ".join(args))
        return None
    except OSError:
        logger.debug("tmux %s failed", " ".join(args), exc_info=True)
        return None

    if result.returncode != 0:
        logger.debug("tmux %s exited %d", " ".join(args), result.returncode)
        return None
    return result.stdout


def parse_tmux_options(output: str) -> dict[str, str]:
    """Extract the reported options from `tmux show-options` output.

    Args:
        output: Raw stdout, one `<name> <value>` pair per line.

    Returns:
        Values for the `REPORTED_OPTIONS` present in `output`. Options tmux
        does not recognize are simply absent, which is how an older tmux
        without `allow-passthrough` is distinguished from one that has it off.
    """
    options: dict[str, str] = {}
    for line in output.splitlines():
        name, separator, value = line.partition(" ")
        if separator and name in REPORTED_OPTIONS:
            options[name] = value.strip()
    return options


def query_tmux_status() -> TmuxStatus | None:
    """Probe the tmux server hosting this pane.

    Returns:
        The server's version and relevant options, or `None` when not running
        inside tmux. Individual facts degrade to `None` rather than failing the
        whole probe, so a partially answering server still reports what it can.
    """
    if not inside_tmux():
        return None

    version_output = _run_tmux(["-V"])
    options_output = _run_tmux(_SHOW_OPTIONS_ARGS)
    return TmuxStatus(
        version=version_output.strip() if version_output else None,
        options=parse_tmux_options(options_output)
        if options_output is not None
        else None,
    )
