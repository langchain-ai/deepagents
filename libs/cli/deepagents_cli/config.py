"""Project and global `.env` loading for the deploy CLI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _stderr(msg: str) -> None:
    """Print a diagnostic to stderr without a trailing newline corruption.

    The CLI does not configure logging handlers by default, so `logger.warning`
    output is invisible to end users. Diagnostics that the user authored — like
    a malformed `.env` they meant to load — should also surface on stderr.
    """
    sys.stderr.write(f"warning: {msg}\n")


def _find_dotenv_from_start_path(start_path: Path) -> Path | None:
    """Find the nearest `.env` file from an explicit start path upward.

    Args:
        start_path: Directory to start searching from.

    Returns:
        Path to the nearest `.env` file, or `None` if not found.
    """
    current = start_path.expanduser().resolve()
    for parent in [current, *list(current.parents)]:
        candidate = parent / ".env"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            logger.warning("Could not inspect .env candidate %s", candidate)
            continue
    return None


try:
    _GLOBAL_DOTENV_PATH = Path.home() / ".deepagents" / ".env"
except RuntimeError:
    # `Path.home()` raises when `HOME` is unset/misconfigured (e.g. on some
    # CI runners). Fall back to a path that won't exist so `is_file()`
    # short-circuits cleanly, and surface the suppression on stderr so the
    # user knows global defaults will not apply.
    _GLOBAL_DOTENV_PATH = Path("/nonexistent/.deepagents/.env")
    _stderr(
        "could not resolve home directory; "
        "global `~/.deepagents/.env` will not be loaded",
    )
    logger.warning(
        "Could not resolve home directory; global .env will not be loaded",
        exc_info=True,
    )


def _load_dotenv(*, start_path: Path) -> bool:
    """Load environment variables from project and global `.env` files.

    Loads in order (first write wins, `override=False`):

    1. Project `.env` — discovered upward from `start_path`.
    2. `~/.deepagents/.env` — global user defaults.

    Both layers use `override=False` so that shell-exported variables always
    take precedence over dotenv files. Because the project file loads first,
    the effective precedence is:

    ```text
    shell env > project `.env` > global `.env`
    ```

    When a *resolved* dotenv path fails to parse the user is notified on
    stderr in addition to the structured `logger.warning` (the CLI does not
    install logging handlers by default).

    Args:
        start_path: Directory to use for project `.env` discovery. Searched
            upward until either a `.env` is found or the filesystem root is
            reached.

    Returns:
        `True` when at least one dotenv file was loaded, `False` otherwise.
    """
    import dotenv

    loaded = False

    dotenv_path = _find_dotenv_from_start_path(start_path)
    if dotenv_path is not None:
        try:
            loaded = (
                dotenv.load_dotenv(dotenv_path=dotenv_path, override=False) or loaded
            )
        except (OSError, ValueError) as exc:
            _stderr(
                f"could not read project .env at {dotenv_path}: {exc}. "
                "Project env vars will not be loaded.",
            )
            logger.warning(
                "Could not read project dotenv at %s; "
                "project env vars will not be loaded",
                dotenv_path,
                exc_info=True,
            )

    try:
        if _GLOBAL_DOTENV_PATH.is_file() and dotenv.load_dotenv(
            dotenv_path=_GLOBAL_DOTENV_PATH, override=False
        ):
            loaded = True
            logger.debug("Loaded global dotenv: %s", _GLOBAL_DOTENV_PATH)
    except (OSError, ValueError) as exc:
        _stderr(
            f"could not read global .env at {_GLOBAL_DOTENV_PATH}: {exc}. "
            "Global defaults will not be applied.",
        )
        logger.warning(
            "Could not read global dotenv at %s; global defaults will not be applied",
            _GLOBAL_DOTENV_PATH,
            exc_info=True,
        )

    return loaded
