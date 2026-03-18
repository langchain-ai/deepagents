"""Configuration, constants, and model creation for the CLI."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shlex
import sys
import threading
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import dotenv
from rich.console import Console

from deepagents_cli._version import __version__
from deepagents_cli.project_utils import (
    get_server_project_context as _get_server_project_context,
)

logger = logging.getLogger(__name__)


def _find_dotenv_files_from_start_path(start_path: Path) -> list[Path]:
    """Find the nearest `.env` / `.env.local` files from a start path upward.

    Args:
        start_path: Directory to start searching from.

    Returns:
        Ordered dotenv files from the nearest matching directory.
    """
    current = start_path.expanduser().resolve()
    for parent in [current, *list(current.parents)]:
        candidates = [parent / ".env", parent / ".env.local"]
        existing: list[Path] = []
        for candidate in candidates:
            try:
                if candidate.is_file():
                    existing.append(candidate)
            except OSError:
                logger.warning("Could not inspect .env candidate %s", candidate)
                return []
        if existing:
            return existing
    return []


def _load_dotenv(*, start_path: Path | None = None, override: bool = False) -> bool:
    """Load environment variables, optionally anchored to an explicit path.

    Args:
        start_path: Directory to use for `.env` discovery.
        override: Whether loaded values should override existing env vars.

    Returns:
        `True` when a dotenv file was loaded, `False` otherwise.
    """
    if start_path is None:
        loaded = False
        for candidate in (Path(".env"), Path(".env.local")):
            try:
                if candidate.is_file():
                    loaded = (
                        dotenv.load_dotenv(dotenv_path=candidate, override=override)
                        or loaded
                    )
            except OSError:
                logger.warning("Could not inspect .env candidate %s", candidate)
                return loaded
        return loaded or dotenv.load_dotenv(override=override)

    dotenv_paths = _find_dotenv_files_from_start_path(start_path)
    if not dotenv_paths:
        return False

    loaded = False
    for dotenv_path in dotenv_paths:
        loaded = dotenv.load_dotenv(dotenv_path=dotenv_path, override=override) or loaded
    return loaded


_bootstrap_project_context = _get_server_project_context()
_bootstrap_start_path = (
    _bootstrap_project_context.user_cwd if _bootstrap_project_context else None
)

_load_dotenv(start_path=_bootstrap_start_path)

# CRITICAL: Override LANGSMITH_PROJECT to route agent traces to separate project
# LangSmith reads LANGSMITH_PROJECT at invocation time, so we override it here
# and preserve the user's original value for shell commands
_deepagents_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
_original_langsmith_project = os.environ.get("LANGSMITH_PROJECT")
if _deepagents_project:
    # Override LANGSMITH_PROJECT for agent traces
    os.environ["LANGSMITH_PROJECT"] = _deepagents_project

from deepagents_cli.model_config import (  # noqa: E402  # Import after os.environ setup above
    ModelConfig,
    ModelConfigError,
    ModelSpec,
)
from deepagents_cli.project_utils import (  # noqa: E402
    find_project_agent_md as _find_project_agent_md,
    find_project_root as _find_project_root,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig

DOCS_URL = "https://docs.langchain.com/oss/python/deepagents/cli"
"""URL for deepagents-cli documentation."""

COLORS = {
    "primary": "#10b981",
    "primary_dev": "#f97316",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
    "mode_shell": "#ff1493",
    "mode_command": "#8b5cf6",
}
"""App color scheme."""

MODE_PREFIXES: dict[str, str] = {
    "shell": "!",
    "command": "/",
}
"""Maps each non-normal mode to its trigger character."""

MODE_DISPLAY_GLYPHS: dict[str, str] = {
    "shell": "$",
    "command": "/",
}
"""Maps each non-normal mode to its display glyph shown in the prompt/UI."""

if MODE_PREFIXES.keys() != MODE_DISPLAY_GLYPHS.keys():
    _only_prefixes = MODE_PREFIXES.keys() - MODE_DISPLAY_GLYPHS.keys()
    _only_glyphs = MODE_DISPLAY_GLYPHS.keys() - MODE_PREFIXES.keys()
    msg = (
        "MODE_PREFIXES and MODE_DISPLAY_GLYPHS have mismatched keys: "
        f"only in PREFIXES={_only_prefixes}, only in GLYPHS={_only_glyphs}"
    )
    raise ValueError(msg)

PREFIX_TO_MODE: dict[str, str] = {v: k for k, v in MODE_PREFIXES.items()}
"""Reverse lookup: trigger character -> mode name."""


class CharsetMode(StrEnum):
    """Character set mode for TUI display."""

    UNICODE = "unicode"
    ASCII = "ascii"
    AUTO = "auto"


@dataclass(frozen=True)
class Glyphs:
    """Character glyphs for TUI display."""

    tool_prefix: str  # ⏺ vs (*)
    ellipsis: str  # … vs ...
    checkmark: str  # ✓ vs [OK]
    error: str  # ✗ vs [X]
    circle_empty: str  # ○ vs [ ]
    circle_filled: str  # ● vs [*]
    output_prefix: str  # ⎿ vs L
    spinner_frames: tuple[str, ...]  # Braille vs ASCII spinner
    pause: str  # ⏸ vs ||
    newline: str  # ⏎ vs \\n
    warning: str  # ⚠ vs [!]
    question: str  # ? vs [?]
    arrow_up: str  # up arrow vs ^
    arrow_down: str  # down arrow vs v
    bullet: str  # bullet vs -
    cursor: str  # cursor vs >

    # Box-drawing characters
    box_vertical: str  # │ vs |
    box_horizontal: str  # ─ vs -
    box_double_horizontal: str  # ═ vs =

    # Diff-specific
    gutter_bar: str  # ▌ vs |

    # Status bar
    git_branch: str  # "↗" vs "git:"


UNICODE_GLYPHS = Glyphs(
    tool_prefix="⏺",
    ellipsis="…",
    checkmark="✓",
    error="✗",
    circle_empty="○",
    circle_filled="●",
    output_prefix="⎿",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    pause="⏸",
    newline="⏎",
    warning="⚠",
    question="?",
    arrow_up="↑",
    arrow_down="↓",
    bullet="•",
    cursor="›",  # noqa: RUF001  # Intentional Unicode glyph
    # Box-drawing characters
    box_vertical="│",
    box_horizontal="─",
    box_double_horizontal="═",
    gutter_bar="▌",
    git_branch="↗",
)

ASCII_GLYPHS = Glyphs(
    tool_prefix="(*)",
    ellipsis="...",
    checkmark="[OK]",
    error="[X]",
    circle_empty="[ ]",
    circle_filled="[*]",
    output_prefix="L",
    spinner_frames=("(-)", "(\\)", "(|)", "(/)"),
    pause="||",
    newline="\\n",
    warning="[!]",
    question="[?]",
    arrow_up="^",
    arrow_down="v",
    bullet="-",
    cursor=">",
    # Box-drawing characters
    box_vertical="|",
    box_horizontal="-",
    box_double_horizontal="=",
    gutter_bar="|",
    git_branch="git:",
)

_glyphs_cache: Glyphs | None = None
"""Module-level cache for detected glyphs."""

_editable_cache: tuple[bool, str | None] | None = None
"""Module-level cache for editable install info: (is_editable, source_path)."""

_langsmith_url_cache: tuple[str, str] | None = None
"""Module-level cache for successful LangSmith project URL lookups."""

_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS = 2.0
"""Max seconds to wait for LangSmith project URL lookup.

Kept short so tracing metadata can never stall CLI flows.
"""


def _resolve_editable_info() -> tuple[bool, str | None]:
    """Parse PEP 610 `direct_url.json` once and cache both results.

    Returns:
        Tuple of (is_editable, contracted_source_path). The path is
        `~`-contracted when it falls under the user's home directory, or
        `None` when the install is non-editable or the path is unavailable.
    """
    global _editable_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _editable_cache is not None:
        return _editable_cache

    editable = False
    path: str | None = None

    try:
        dist = distribution("deepagents-cli")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            editable = data.get("dir_info", {}).get("editable", False)
            if editable:
                url = data.get("url", "")
                if url.startswith("file://"):
                    path = unquote(urlparse(url).path)
                    home = str(Path.home())
                    if path.startswith(home):
                        path = "~" + path[len(home) :]
    except (PackageNotFoundError, FileNotFoundError, json.JSONDecodeError, TypeError):
        logger.debug(
            "Failed to read editable install info from PEP 610 metadata",
            exc_info=True,
        )

    _editable_cache = (editable, path)
    return _editable_cache


def _is_editable_install() -> bool:
    """Check if deepagents-cli is installed in editable mode.

    Uses PEP 610 `direct_url.json` metadata to detect editable installs.

    Returns:
        `True` if installed in editable mode, `False` otherwise.
    """
    return _resolve_editable_info()[0]


def _get_editable_install_path() -> str | None:
    """Return the `~`-contracted source directory for an editable install.

    Returns `None` for non-editable installs or when the path cannot be
    determined.
    """
    return _resolve_editable_info()[1]


def _detect_charset_mode() -> CharsetMode:
    """Auto-detect terminal charset capabilities.

    Returns:
        The detected CharsetMode based on environment and terminal encoding.
    """
    env_mode = os.environ.get("UI_CHARSET_MODE", "auto").lower()
    if env_mode == "unicode":
        return CharsetMode.UNICODE
    if env_mode == "ascii":
        return CharsetMode.ASCII

    # Auto: check stdout encoding and LANG
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return CharsetMode.UNICODE
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    if "utf" in lang.lower():
        return CharsetMode.UNICODE
    return CharsetMode.ASCII


def get_glyphs() -> Glyphs:
    """Get the glyph set for the current charset mode.

    Returns:
        The appropriate Glyphs instance based on charset mode detection.
    """
    global _glyphs_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _glyphs_cache is not None:
        return _glyphs_cache

    mode = _detect_charset_mode()
    _glyphs_cache = ASCII_GLYPHS if mode == CharsetMode.ASCII else UNICODE_GLYPHS
    return _glyphs_cache


def reset_glyphs_cache() -> None:
    """Reset the glyphs cache (for testing)."""
    global _glyphs_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _glyphs_cache = None


def is_ascii_mode() -> bool:
    """Check whether the terminal is in ASCII charset mode.

    Convenience wrapper so widgets can branch on charset without importing
    both `_detect_charset_mode` and `CharsetMode`.

    Returns:
        `True` when the detected charset mode is ASCII.
    """
    return _detect_charset_mode() == CharsetMode.ASCII


def newline_shortcut() -> str:
    """Return the platform-native label for the newline keyboard shortcut.

    macOS labels the modifier "Option" while other platforms use Ctrl+J
    as the most reliable cross-terminal shortcut.

    Returns:
        A human-readable shortcut string, e.g. `'Option+Enter'` or `'Ctrl+J'`.
    """
    return "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"


# Text art banners (Unicode and ASCII variants)

_UNICODE_BANNER = f"""
██████╗  ███████╗ ███████╗ ██████╗    ▄▓▓▄
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗  ▓•███▙
██║  ██║ █████╗   █████╗   ██████╔╝  ░▀▀████▙▖
██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝      █▓████▙▖
██████╔╝ ███████╗ ███████╗ ██║          ▝█▓█████▙
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝           ░▜█▓████▙
                                          ░█▀█▛▀▀▜▙▄
                                        ░▀░▀▒▛░░  ▝▀▘

 █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                                  v{__version__}
"""
_ASCII_BANNER = f"""
 ____  ____  ____  ____
|  _ \\| ___|| ___||  _ \\
| | | | |_  | |_  | |_) |
| |_| |  _| |  _| |  __/
|____/|____||____||_|

    _    ____  ____  _   _  _____  ____
   / \\  / ___|| ___|| \\ | ||_   _|/ ___|
  / _ \\| |  _ | |_  |  \\| |  | |  \\___ \\
 / ___ \\ |_| ||  _| | |\\  |  | |   ___) |
/_/   \\_\\____||____||_| \\_|  |_|  |____/
                                  v{__version__}
"""


def get_banner() -> str:
    """Get the appropriate banner for the current charset mode.

    Returns:
        The text art banner string (Unicode or ASCII based on charset mode).
        Includes "(local)" suffix when installed in editable mode.
    """
    if _detect_charset_mode() == CharsetMode.ASCII:
        banner = _ASCII_BANNER
    else:
        banner = _UNICODE_BANNER

    if _is_editable_install():
        banner = banner.replace(f"v{__version__}", f"v{__version__} (local)")

    return banner


# Interactive commands
COMMANDS = {
    "clear": "Clear screen and reset conversation",
    "help": "Show help information",
    "remember": "Review conversation and update memory/skills",
    "tokens": "Show token usage for current thread",
    "quit": "Exit the CLI",
    "exit": "Exit the CLI",
}


# Maximum argument length for display
MAX_ARG_LENGTH = 150

# Agent configuration
config: RunnableConfig = {"recursion_limit": 1000}

# Rich console instance
console = Console(highlight=False)


class _ShellAllowAll(list):  # noqa: FURB189  # sentinel type, not a general-purpose list subclass
    """Sentinel subclass for unrestricted shell access.

    Using a dedicated type instead of a plain list lets consumers use
    `isinstance` checks, which survive serialization/copy unlike identity
    checks (`is`).
    """


SHELL_ALLOW_ALL: list[str] = _ShellAllowAll(["__ALL__"])
"""Sentinel value returned by `parse_shell_allow_list` for `--shell-allow-list=all`."""


def parse_shell_allow_list(allow_list_str: str | None) -> list[str] | None:
    """Parse shell allow-list from string.

    Args:
        allow_list_str: Comma-separated list of commands, `'recommended'` for
            safe defaults, or `'all'` to allow any command.

            `'all'` must be the sole value — it is not recognized inside a
            comma-separated list (unlike `'recommended'`).

            Can also include `'recommended'` in the list to merge with custom
            commands.

    Returns:
        List of allowed commands, `SHELL_ALLOW_ALL` if `'all'` was specified,
            or `None` if no allow-list configured.

    Raises:
        ValueError: If `'all'` is combined with other commands.
    """
    if not allow_list_str:
        return None

    # Special value 'all' allows any shell command
    if allow_list_str.strip().lower() == "all":
        return SHELL_ALLOW_ALL

    # Special value 'recommended' uses our curated safe list
    if allow_list_str.strip().lower() == "recommended":
        return list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    # Split by comma and strip whitespace
    commands = [cmd.strip() for cmd in allow_list_str.split(",") if cmd.strip()]

    # Reject ambiguous input: 'all' mixed with other commands
    if any(cmd.lower() == "all" for cmd in commands):
        msg = (
            "Cannot combine 'all' with other commands in --shell-allow-list. "
            "Use '--shell-allow-list all' alone to allow any command."
        )
        raise ValueError(msg)

    # If "recommended" is in the list, merge with recommended commands
    result = []
    for cmd in commands:
        if cmd.lower() == "recommended":
            result.extend(RECOMMENDED_SAFE_SHELL_COMMANDS)
        else:
            result.append(cmd)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in result:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique


@dataclass
class Settings:
    """Global settings and environment detection for deepagents-cli.

    This class is initialized once at startup and provides access to:
    - Available models and API keys
    - Current project information
    - Tool availability (e.g., Tavily)
    - File system paths

    Attributes:
        openai_api_key: OpenAI API key if available.
        anthropic_api_key: Anthropic API key if available.
        google_api_key: Google API key if available.
        nvidia_api_key: NVIDIA API key if available.
        tavily_api_key: Tavily API key if available.
        google_cloud_project: Google Cloud project ID for VertexAI
            authentication.
        deepagents_langchain_project: LangSmith project name for deepagents
            agent tracing.
        user_langchain_project: Original LANGSMITH_PROJECT from environment
            (for user code).
        model_name: Currently active model name (set after model creation).
        model_provider: Provider identifier (e.g., openai, anthropic, google_genai).
        model_context_limit: Maximum input token count from the model profile.
        project_root: Current project root directory (if in a git project).
        shell_allow_list: List of shell commands that don't require approval.
    """

    # API keys
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    nvidia_api_key: str | None
    tavily_api_key: str | None

    # Google Cloud configuration (for VertexAI)
    google_cloud_project: str | None

    # LangSmith configuration
    deepagents_langchain_project: str | None  # For deepagents agent tracing
    user_langchain_project: str | None  # Original LANGSMITH_PROJECT for user code

    # Model configuration
    model_name: str | None = None  # Currently active model name
    model_provider: str | None = None  # Provider name (see PROVIDER_API_KEY_ENV)
    model_context_limit: int | None = None  # Max input tokens from model profile

    # Project information
    project_root: Path | None = None

    # Shell command allow-list for auto-approval
    shell_allow_list: list[str] | None = None

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> Settings:
        """Create settings by detecting the current environment.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            Settings instance with detected configuration
        """
        # Detect API keys (normalize empty strings to None)
        openai_key = os.environ.get("OPENAI_API_KEY") or None
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or None
        google_key = os.environ.get("GOOGLE_API_KEY") or None
        nvidia_key = os.environ.get("NVIDIA_API_KEY") or None
        tavily_key = os.environ.get("TAVILY_API_KEY") or None
        google_cloud_project = os.environ.get("GOOGLE_CLOUD_PROJECT")

        # Detect LangSmith configuration
        # DEEPAGENTS_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # Note: LANGSMITH_PROJECT was already overridden at module import time (above)
        # so we use the saved original value, not the current os.environ value
        deepagents_langchain_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
        user_langchain_project = _original_langsmith_project  # Use saved original!

        # Detect project
        project_root = _find_project_root(start_path)

        # Parse shell command allow-list from environment
        # Format: comma-separated list of commands (e.g., "ls,cat,grep,pwd")
        # Special value "recommended" uses RECOMMENDED_SAFE_SHELL_COMMANDS
        shell_allow_list_str = os.environ.get("DEEPAGENTS_SHELL_ALLOW_LIST")
        shell_allow_list = parse_shell_allow_list(shell_allow_list_str)

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            nvidia_api_key=nvidia_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
            shell_allow_list=shell_allow_list,
        )

    def reload_from_environment(self, *, start_path: Path | None = None) -> list[str]:
        """Reload selected settings from environment variables and project files.

        This refreshes only fields that are expected to change at runtime
        (API keys, Google Cloud project, project root, shell allow-list, and
        LangSmith tracing project).

        Runtime model state (`model_name`, `model_provider`,
        `model_context_limit`) and the original user LangSmith project
        (`user_langchain_project`) are intentionally preserved -- they are
        not in `reloadable_fields` and are never touched by this method.

        Args:
            start_path: Directory to start project detection from (defaults to cwd).

        Returns:
            A list of human-readable change descriptions.
        """
        _load_dotenv(start_path=start_path, override=True)

        api_key_fields = {
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "nvidia_api_key",
            "tavily_api_key",
        }
        reloadable_fields = (
            "openai_api_key",
            "anthropic_api_key",
            "google_api_key",
            "nvidia_api_key",
            "tavily_api_key",
            "google_cloud_project",
            "deepagents_langchain_project",
            "project_root",
            "shell_allow_list",
        )

        previous = {field: getattr(self, field) for field in reloadable_fields}

        try:
            shell_allow_list = parse_shell_allow_list(
                os.environ.get("DEEPAGENTS_SHELL_ALLOW_LIST")
            )
        except ValueError:
            logger.warning(
                "Invalid DEEPAGENTS_SHELL_ALLOW_LIST during reload; "
                "keeping previous value"
            )
            shell_allow_list = previous["shell_allow_list"]

        try:
            project_root = _find_project_root(start_path)
        except OSError:
            logger.warning(
                "Could not detect project root during reload; keeping previous value"
            )
            project_root = previous["project_root"]

        refreshed = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY") or None,
            "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY") or None,
            "google_api_key": os.environ.get("GOOGLE_API_KEY") or None,
            "nvidia_api_key": os.environ.get("NVIDIA_API_KEY") or None,
            "tavily_api_key": os.environ.get("TAVILY_API_KEY") or None,
            "google_cloud_project": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "deepagents_langchain_project": os.environ.get(
                "DEEPAGENTS_LANGSMITH_PROJECT"
            ),
            "project_root": project_root,
            "shell_allow_list": shell_allow_list,
        }

        for field, value in refreshed.items():
            setattr(self, field, value)

        # Sync the LANGSMITH_PROJECT env var so LangSmith tracing picks up
        # the change
        new_project = refreshed["deepagents_langchain_project"]
        if new_project:
            os.environ["LANGSMITH_PROJECT"] = new_project
        elif previous["deepagents_langchain_project"]:
            # Override was previously active but new value is unset; restore.
            if _original_langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = _original_langsmith_project
            else:
                os.environ.pop("LANGSMITH_PROJECT", None)

        def _display(field: str, value: object) -> str:
            if field in api_key_fields:
                return "set" if value else "unset"
            return str(value)

        changes: list[str] = []
        for field in reloadable_fields:
            old_value = previous[field]
            new_value = refreshed[field]
            if old_value != new_value:
                changes.append(
                    f"{field}: {_display(field, old_value)} -> "
                    f"{_display(field, new_value)}"
                )
        return changes

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None

    @property
    def has_nvidia(self) -> bool:
        """Check if NVIDIA API key is configured."""
        return self.nvidia_api_key is not None

    @property
    def has_vertex_ai(self) -> bool:
        """Check if VertexAI is available (Google Cloud project set, no API key).

        VertexAI uses Application Default Credentials (ADC) for authentication,
        so if GOOGLE_CLOUD_PROJECT is set and GOOGLE_API_KEY is not, we assume
        VertexAI.
        """
        return self.google_cloud_project is not None and self.google_api_key is None

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """Get the base user-level .deepagents directory.

        Returns:
            Path to ~/.deepagents
        """
        return Path.home() / ".deepagents"

    @staticmethod
    def get_user_agent_md_path(agent_name: str) -> Path:
        """Get user-level AGENTS.md path for a specific agent.

        Returns path regardless of whether the file exists.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/AGENTS.md
        """
        return Path.home() / ".deepagents" / agent_name / "AGENTS.md"

    def get_project_agent_md_path(self) -> list[Path]:
        """Get project-level AGENTS.md paths.

        Checks both `{project_root}/.deepagents/AGENTS.md` and
        `{project_root}/AGENTS.md`, returning all that exist. If both are
        present, both are loaded and their instructions are combined, with
        `.deepagents/AGENTS.md` first.

        Returns:
            Existing AGENTS.md paths.

                Empty if neither file exists or not in a project, one entry if
                only one is present, or two entries if both locations have the
                file.
        """
        if not self.project_root:
            return []
        return _find_project_agent_md(self.project_root)

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """Validate to prevent invalid filesystem paths and security issues.

        Returns:
            True if the agent name is valid, False otherwise.
        """
        if not agent_name or not agent_name.strip():
            return False
        # Allow only alphanumeric, hyphens, underscores, and whitespace
        return bool(re.match(r"^[a-zA-Z0-9_\-\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the global agent directory path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        return Path.home() / ".deepagents" / agent_name

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """Ensure the global agent directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """Get user-level skills directory path for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """Ensure user-level skills directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        skills_dir = self.get_user_skills_dir(agent_name)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_project_skills_dir(self) -> Path | None:
        """Get project-level skills directory path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """Ensure project-level skills directory exists and return its path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        if skills_dir is None:
            return None
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_user_agents_dir(self, agent_name: str) -> Path:
        """Get user-level agents directory path for custom subagent definitions.

        Args:
            agent_name: Name of the CLI agent (e.g., "deepagents")

        Returns:
            Path to ~/.deepagents/{agent_name}/agents/
        """
        return self.get_agent_dir(agent_name) / "agents"

    def get_project_agents_dir(self) -> Path | None:
        """Get project-level agents directory path for custom subagent definitions.

        Returns:
            Path to {project_root}/.deepagents/agents/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agents"

    @property
    def user_agents_dir(self) -> Path:
        """Get the base user-level `.agents` directory (`~/.agents`).

        Returns:
            Path to `~/.agents`
        """
        return Path.home() / ".agents"

    def get_user_agent_skills_dir(self) -> Path:
        """Get user-level `~/.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `~/.agents/skills/`
        """
        return self.user_agents_dir / "skills"

    def get_project_agent_skills_dir(self) -> Path | None:
        """Get project-level `.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `{project_root}/.agents/skills/`, or `None` if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".agents" / "skills"

    @staticmethod
    def get_built_in_skills_dir() -> Path:
        """Get the directory containing built-in skills that ship with the CLI.

        Returns:
            Path to the `built_in_skills/` directory within the package.
        """
        return Path(__file__).parent / "built_in_skills"


# Global settings instance (initialized once)
settings = Settings.from_environment(start_path=_bootstrap_start_path)


class SessionState:
    """Mutable session state shared across the app, adapter, and agent.

    Tracks runtime flags like auto-approve that can be toggled during a
    session via keybindings or the HITL approval menu's "Auto-approve all"
    option.

    The `auto_approve` flag controls whether tool calls (shell execution, file
    writes/edits, web search, URL fetch) require user confirmation before running.
    """

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        """Initialize session state with optional flags.

        Args:
            auto_approve: Whether to auto-approve tool calls without
                prompting.

                Can be toggled at runtime via Shift+Tab or the HITL
                approval menu.
            no_splash: Whether to skip displaying the splash screen on startup.
        """
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        from deepagents_cli.sessions import generate_thread_id

        self.thread_id = generate_thread_id()

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return the new state.

        Called by the Shift+Tab keybinding in the Textual app.

        When auto-approve is on, all tool calls execute without prompting.

        Returns:
            The new `auto_approve` state after toggling.
        """
        self.auto_approve = not self.auto_approve
        return self.auto_approve


SHELL_TOOL_NAMES: frozenset[str] = frozenset({"bash", "shell", "execute"})
"""Tool names recognized as shell/command-execution tools.

Only `'execute'` is registered by the SDK and CLI backends in practice.
`'bash'` and `'shell'` are legacy names carried over and kept as
backwards-compatible aliases.
"""

DANGEROUS_SHELL_PATTERNS = (
    "$(",  # Command substitution
    "`",  # Backtick command substitution
    "$'",  # ANSI-C quoting (can encode dangerous chars via escape sequences)
    "\n",  # Newline (command injection)
    "\r",  # Carriage return (command injection)
    "\t",  # Tab (can be used for injection in some shells)
    "<(",  # Process substitution (input)
    ">(",  # Process substitution (output)
    "<<<",  # Here-string
    "<<",  # Here-doc (can embed commands)
    ">>",  # Append redirect
    ">",  # Output redirect
    "<",  # Input redirect
    "${",  # Variable expansion with braces (can run commands via ${var:-$(cmd)})
)

# Recommended safe shell commands for non-interactive mode.
# These commands are primarily read-only and do not modify the filesystem
# when used without shell redirection operators (which the dangerous-patterns
# check blocks).
#
# EXCLUDED (dangerous - listed on GTFOBins/LOOBins or can modify system):
# - All shells: bash, sh, zsh, fish, dash, ksh, csh, tcsh, etc.
# - Editors: vim, vi, nano, emacs, ed, etc. (can spawn shells)
# - Interpreters: python, perl, ruby, node, php, lua, awk, gawk, etc.
# - Package managers: pip, npm, gem, apt, yum, brew, etc.
# - Compilers: gcc, cc, make, cmake, etc.
# - Network tools: curl, wget, nc, ssh, scp, ftp, telnet, etc.
# - Archivers with shell escape: tar, zip, 7z, etc.
# - System modifiers: chmod, chown, chattr, mv, rm, cp, dd, etc.
# - Privilege tools: sudo, su, doas, pkexec, etc.
# - Process tools: env, xargs, find (with -exec), etc.
# - Git (can run hooks), docker, kubectl, etc.
#
# SAFE commands included below are primarily readers/formatters. File write and
# injection are prevented by the dangerous-patterns check that blocks redirects,
# command substitution, and other shell metacharacters.
RECOMMENDED_SAFE_SHELL_COMMANDS = (
    # Directory listing
    "ls",
    "dir",
    # File content viewing (read-only)
    "cat",
    "head",
    "tail",
    # Text searching (read-only)
    "grep",
    "wc",
    "strings",
    # Text processing (read-only, no shell execution)
    "cut",
    "tr",
    "diff",
    "md5sum",
    "sha256sum",
    # Path utilities
    "pwd",
    "which",
    # System info (read-only)
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "uptime",
    "nproc",
    "lscpu",
    "lsmem",
    # Process viewing (read-only)
    "ps",
)


def contains_dangerous_patterns(command: str) -> bool:
    """Check if a command contains dangerous shell patterns.

    These patterns can be used to bypass allow-list validation by embedding
    arbitrary commands within seemingly safe commands. The check includes
    both literal substring patterns (redirects, substitution operators, etc.)
    and regex patterns for bare variable expansion (`$VAR`) and the background
    operator (`&`).

    Args:
        command: The shell command to check.

    Returns:
        True if dangerous patterns are found, False otherwise.
    """
    if any(pattern in command for pattern in DANGEROUS_SHELL_PATTERNS):
        return True

    # Bare variable expansion ($VAR without braces) can leak sensitive paths.
    # We already block ${ and $( above; this catches plain $HOME, $IFS, etc.
    if re.search(r"\$[A-Za-z_]", command):
        return True

    # Standalone & (background execution) changes the execution model and
    # should not be allowed.  We check for & that is NOT part of &&.
    return bool(re.search(r"(?<![&])&(?![&])", command))


def is_shell_command_allowed(command: str, allow_list: list[str] | None) -> bool:
    """Check if a shell command is in the allow-list.

    The allow-list matches against the first token of the command (the executable
    name). This allows read-only commands like ls, cat, grep, etc. to be
    auto-approved.

    When `allow_list` is the `SHELL_ALLOW_ALL` sentinel, all non-empty commands
    are approved unconditionally — dangerous pattern checks are skipped.

    SECURITY: For regular allow-lists, this function rejects commands containing
    dangerous shell patterns (command substitution, redirects, process
    substitution, etc.) BEFORE parsing, to prevent injection attacks that could
    bypass the allow-list.

    Args:
        command: The full shell command to check.
        allow_list: List of allowed command names (e.g., `["ls", "cat", "grep"]`),
            the `SHELL_ALLOW_ALL` sentinel to allow any command, or `None`.

    Returns:
        `True` if the command is allowed, `False` otherwise.
    """
    if not allow_list or not command or not command.strip():
        return False

    # SHELL_ALLOW_ALL sentinel — skip pattern and token checks
    if isinstance(allow_list, _ShellAllowAll):
        return True

    # SECURITY: Check for dangerous patterns BEFORE any parsing
    # This prevents injection attacks like: ls "$(rm -rf /)"
    if contains_dangerous_patterns(command):
        return False

    allow_set = set(allow_list)

    # Extract the first command token
    # Handle pipes and other shell operators by checking each command in the pipeline
    # Split by compound operators first (&&, ||), then single-char operators (|, ;).
    # Note: standalone & (background) is blocked by contains_dangerous_patterns above.
    segments = re.split(r"&&|\|\||[|;]", command)

    # Track if we found at least one valid command
    found_command = False

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        try:
            # Try to parse as shell command to extract the executable name
            tokens = shlex.split(segment)
            if tokens:
                found_command = True
                cmd_name = tokens[0]
                # Check if this command is in the allow set
                if cmd_name not in allow_set:
                    return False
        except ValueError:
            # If we can't parse it, be conservative and require approval
            return False

    # All segments are allowed (and we found at least one command)
    return found_command


def get_langsmith_project_name() -> str | None:
    """Resolve the LangSmith project name if tracing is configured.

    Checks for the required API key and tracing environment variables.
    When both are present, resolves the project name with priority:
    `settings.deepagents_langchain_project` (from
    `DEEPAGENTS_LANGSMITH_PROJECT`), then `LANGSMITH_PROJECT` from the
    environment (note: this may already have been overridden at import
    time to match `DEEPAGENTS_LANGSMITH_PROJECT`), then `'default'`.

    Returns:
        Project name string when LangSmith tracing is active, None otherwise.
    """
    langsmith_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get(
        "LANGCHAIN_API_KEY"
    )
    langsmith_tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get(
        "LANGCHAIN_TRACING_V2"
    )
    if not (langsmith_key and langsmith_tracing):
        return None

    return (
        settings.deepagents_langchain_project
        or os.environ.get("LANGSMITH_PROJECT")
        or "default"
    )


def fetch_langsmith_project_url(project_name: str) -> str | None:
    """Fetch the LangSmith project URL via the LangSmith client.

    Successful results are cached at module level so repeated calls do not
    make additional network requests.

    The network call runs in a daemon thread with a hard timeout of
    `_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS`, so this function blocks the
    calling thread for at most that duration even if LangSmith is unreachable.

    Returns None (with a debug log) on any failure: missing `langsmith` package,
    network errors, invalid project names, client initialization issues,
    or timeouts.

    Args:
        project_name: LangSmith project name to look up.

    Returns:
        Project URL string if found, None otherwise.
    """
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement

    if _langsmith_url_cache is not None:
        cached_name, cached_url = _langsmith_url_cache
        if cached_name == project_name:
            return cached_url
        # Different project name — fall through to fetch.

    try:
        from langsmith import Client
    except ImportError:
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=True,
        )
        return None

    result: str | None = None
    lookup_error: Exception | None = None
    done = threading.Event()

    def _lookup_url() -> None:
        nonlocal result, lookup_error
        try:
            project = Client().read_project(project_name=project_name)
            result = project.url or None
        except Exception as exc:  # noqa: BLE001  # LangSmith SDK error types are not stable
            lookup_error = exc
        finally:
            done.set()

    thread = threading.Thread(target=_lookup_url, daemon=True)
    thread.start()

    if not done.wait(_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS):
        logger.debug(
            "Timed out fetching LangSmith project URL for '%s' after %.1fs",
            project_name,
            _LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS,
        )
        return None

    if lookup_error is not None:
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=(
                type(lookup_error),
                lookup_error,
                lookup_error.__traceback__,
            ),
        )
        return None

    if result is not None:
        _langsmith_url_cache = (project_name, result)
    return result


def build_langsmith_thread_url(thread_id: str) -> str | None:
    """Build a full LangSmith thread URL if tracing is configured.

    Combines `get_langsmith_project_name` and `fetch_langsmith_project_url`
    into a single convenience helper.

    Args:
        thread_id: Thread identifier to build the URL for.

    Returns:
        Full thread URL string, or `None` if unavailable (LangSmith is not
            configured or the project URL cannot be resolved.)
    """
    project_name = get_langsmith_project_name()
    if not project_name:
        return None

    project_url = fetch_langsmith_project_url(project_name)
    if not project_url:
        return None

    return f"{project_url.rstrip('/')}/t/{thread_id}?utm_source=deepagents-cli"


def reset_langsmith_url_cache() -> None:
    """Reset the LangSmith URL cache (for testing)."""
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _langsmith_url_cache = None


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (AGENTS.md) is handled separately by the middleware.

    Returns:
        The default agent instructions as a string.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def detect_provider(model_name: str) -> str | None:
    """Auto-detect provider from model name.

    Intentionally duplicates a subset of LangChain's
    `_attempt_infer_model_provider` because we need to resolve the provider
    **before** calling `init_chat_model` in order to:

    1. Build provider-specific kwargs (API base URLs, headers, etc.) that are
       passed *into* `init_chat_model`.
    2. Validate credentials early to surface user-friendly errors.

    Args:
        model_name: Model name to detect provider from.

    Returns:
        Provider name (openai, anthropic, google_genai, google_vertexai,
            nvidia) or `None` if the provider cannot be determined from the
            name alone.
    """
    model_lower = model_name.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"

    if model_lower.startswith("claude"):
        if not settings.has_anthropic and settings.has_vertex_ai:
            return "google_vertexai"
        return "anthropic"

    if model_lower.startswith("gemini"):
        if settings.has_vertex_ai and not settings.has_google:
            return "google_vertexai"
        return "google_genai"

    if model_lower.startswith(("nemotron", "nvidia/")):
        return "nvidia"

    return None


def _get_default_model_spec() -> str:
    """Get default model specification based on available credentials.

    Checks in order:

    1. `[models].default` in config file (user's intentional preference).
    2. `[models].recent` in config file (last `/model` switch).
    3. Auto-detection based on available API credentials.

    Returns:
        Model specification in provider:model format.

    Raises:
        ModelConfigError: If no credentials are configured.
    """
    config = ModelConfig.load()
    if config.default_model:
        return config.default_model

    if config.recent_model:
        return config.recent_model

    if settings.has_openai:
        return "openai:gpt-5.2"
    if settings.has_anthropic:
        return "anthropic:claude-sonnet-4-6"
    if settings.has_google:
        return "google_genai:gemini-3.1-pro-preview"
    if settings.has_vertex_ai:
        return "google_vertexai:gemini-3.1-pro-preview"
    if settings.has_nvidia:
        return "nvidia:nvidia/nemotron-3-super-120b-a12b"

    msg = (
        "No credentials configured. Please set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, "
        "GOOGLE_CLOUD_PROJECT, or NVIDIA_API_KEY"
    )
    raise ModelConfigError(msg)


_OPENROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents CLI"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""


def _apply_openrouter_defaults(kwargs: dict[str, Any]) -> None:
    """Inject default OpenRouter attribution kwargs.

    Sets `app_url` and `app_title` via `setdefault` so that user-supplied
    values in config take precedence. These map to the `HTTP-Referer` and
    `X-Title` headers that `ChatOpenRouter` sends for app attribution
    (see https://openrouter.ai/docs/app-attribution).

    Users can override either value provider-wide or per-model in
    `~/.deepagents/config.toml`:

    ```toml
    # Provider-wide
    [models.providers.openrouter.params]
    app_url = "https://myapp.com"
    app_title = "My App"

    # Per-model (shallow-merges on top of provider-wide)
    [models.providers.openrouter.params."openai/gpt-oss-120b"]
    app_title = "My App (GPT)"
    ```

    Args:
        kwargs: Mutable kwargs dict to update in place.
    """
    kwargs.setdefault("app_url", _OPENROUTER_APP_URL)
    kwargs.setdefault("app_title", _OPENROUTER_APP_TITLE)


def _get_provider_kwargs(
    provider: str, *, model_name: str | None = None
) -> dict[str, Any]:
    """Get provider-specific kwargs from the config file.

    Reads `base_url`, `api_key_env`, and the `params` table from the user's
    `config.toml` for the given provider.

    When `model_name` is provided, per-model overrides from the `params`
    sub-table are shallow-merged on top.

    Args:
        provider: Provider name (e.g., openai, anthropic, fireworks, ollama).
        model_name: Optional model name for per-model overrides.

    Returns:
        Dictionary of provider-specific kwargs.
    """
    config = ModelConfig.load()
    result: dict[str, Any] = config.get_kwargs(provider, model_name=model_name)
    base_url = config.get_base_url(provider)
    if base_url:
        result["base_url"] = base_url
    api_key_env = config.get_api_key_env(provider)
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            result["api_key"] = api_key

    if provider == "openrouter":
        _apply_openrouter_defaults(result)

    return result


def _create_model_from_class(
    class_path: str,
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """Import and instantiate a custom `BaseChatModel` class.

    Args:
        class_path: Fully-qualified class in `module.path:ClassName` format.
        model_name: Model identifier to pass as `model` kwarg.
        provider: Provider name (for error messages).
        kwargs: Additional keyword arguments for the constructor.

    Returns:
        Instantiated `BaseChatModel`.

    Raises:
        ModelConfigError: If the class cannot be imported, is not a
            `BaseChatModel` subclass, or fails to instantiate.
    """
    from langchain_core.language_models import (
        BaseChatModel as _BaseChatModel,  # Runtime import; module level is typing only
    )

    if ":" not in class_path:
        msg = (
            f"Invalid class_path '{class_path}' for provider '{provider}': "
            "must be in module.path:ClassName format"
        )
        raise ModelConfigError(msg)

    module_path, class_name = class_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Could not import module '{module_path}' for provider '{provider}': {e}"
        raise ModelConfigError(msg) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        msg = (
            f"Class '{class_name}' not found in module '{module_path}' "
            f"for provider '{provider}'"
        )
        raise ModelConfigError(msg)

    if not (isinstance(cls, type) and issubclass(cls, _BaseChatModel)):
        msg = (
            f"'{class_path}' is not a BaseChatModel subclass (got {type(cls).__name__})"
        )
        raise ModelConfigError(msg)

    try:
        return cls(model=model_name, **kwargs)
    except Exception as e:
        msg = f"Failed to instantiate '{class_path}' for '{provider}:{model_name}': {e}"
        raise ModelConfigError(msg) from e


def _create_model_via_init(
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """Create a model using langchain's `init_chat_model`.

    Args:
        model_name: Model identifier.
        provider: Provider name (may be empty for auto-detection).
        kwargs: Additional keyword arguments.

    Returns:
        Instantiated `BaseChatModel`.

    Raises:
        ModelConfigError: On import, value, or runtime errors.
    """
    from langchain.chat_models import init_chat_model

    try:
        if provider:
            return init_chat_model(model_name, model_provider=provider, **kwargs)
        return init_chat_model(model_name, **kwargs)
    except ImportError as e:
        import importlib.util

        package_map = {
            "anthropic": "langchain-anthropic",
            "openai": "langchain-openai",
            "google_genai": "langchain-google-genai",
            "google_vertexai": "langchain-google-vertexai",
            "nvidia": "langchain-nvidia-ai-endpoints",
        }
        package = package_map.get(provider, f"langchain-{provider}")
        # Convert pip package name to Python module name for import check.
        module_name = package.replace("-", "_")
        try:
            spec_found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            spec_found = False
        if spec_found:
            # Package is installed but an internal import failed — surface
            # the real error instead of the misleading "missing package" hint.
            msg = (
                f"Provider package '{package}' is installed but failed to "
                f"import for provider '{provider}': {e}"
            )
        else:
            msg = (
                f"Missing package for provider '{provider}'. "
                f"Install: pip install {package}"
            )
        raise ModelConfigError(msg) from e
    except (ValueError, TypeError) as e:
        spec = f"{provider}:{model_name}" if provider else model_name
        msg = f"Invalid model configuration for '{spec}': {e}"
        raise ModelConfigError(msg) from e
    except Exception as e:  # provider SDK auth/network errors
        spec = f"{provider}:{model_name}" if provider else model_name
        msg = f"Failed to initialize model '{spec}': {e}"
        raise ModelConfigError(msg) from e


@dataclass(frozen=True)
class ModelResult:
    """Result of creating a chat model, bundling the model with its metadata.

    This separates model creation from settings mutation so callers can decide
    when to commit the metadata to global settings.

    Attributes:
        model: The instantiated chat model.
        model_name: Resolved model name.
        provider: Resolved provider name.
        context_limit: Max input tokens from the model profile, or `None`.
    """

    model: BaseChatModel
    model_name: str
    provider: str
    context_limit: int | None = None

    def apply_to_settings(self) -> None:
        """Commit this result's metadata to global `settings`."""
        settings.model_name = self.model_name
        settings.model_provider = self.provider
        settings.model_context_limit = self.context_limit


def _apply_profile_overrides(
    model: BaseChatModel,
    overrides: dict[str, Any],
    model_name: str,
    *,
    label: str,
    raise_on_failure: bool = False,
) -> None:
    """Merge `overrides` into `model.profile`.

    If the model already has a dict profile, overrides are layered on top
    so existing keys (e.g., `tool_calling`) are preserved unchanged.

    Args:
        model: The chat model whose profile will be updated.
        overrides: Key/value pairs to merge into the profile.
        model_name: Model name used in log/error messages.
        label: Human-readable source label for messages
            (e.g., `"config.toml"`, `"CLI --profile-override"`).
        raise_on_failure: When `True`, raise `ModelConfigError` instead
            of logging a warning if assignment fails.

    Raises:
        ModelConfigError: If `raise_on_failure` is `True` and the model
            rejects profile assignment.
    """
    logger.debug("Applying %s profile overrides: %s", label, overrides)
    profile = getattr(model, "profile", None)
    merged = {**profile, **overrides} if isinstance(profile, dict) else overrides
    try:
        model.profile = merged  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        if raise_on_failure:
            msg = (
                f"Could not apply {label} to model '{model_name}': {exc}. "
                f"The model may not support profile assignment."
            )
            raise ModelConfigError(msg) from exc
        logger.warning(
            "Could not apply %s profile overrides to model '%s': %s. "
            "Overrides will be ignored.",
            label,
            model_name,
            exc,
        )


def create_model(
    model_spec: str | None = None,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    profile_overrides: dict[str, Any] | None = None,
) -> ModelResult:
    """Create a chat model.

    Uses `init_chat_model` for standard providers, or imports a custom
    `BaseChatModel` subclass when the provider has a `class_path` in config.

    Supports `provider:model` format (e.g., `'anthropic:claude-sonnet-4-5'`)
    for explicit provider selection, or bare model names for auto-detection.

    Args:
        model_spec: Model specification in `provider:model` format (e.g.,
            `'anthropic:claude-sonnet-4-5'`, `'openai:gpt-4o'`) or just the model
            name for auto-detection (e.g., `'claude-sonnet-4-5'`).

                If not provided, uses environment-based defaults.
        extra_kwargs: Additional kwargs to pass to the model constructor.

            These take highest priority, overriding values from the config file.
        profile_overrides: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides (CLI wins).

    Returns:
        A `ModelResult` containing the model and its metadata.

    Raises:
        ModelConfigError: If provider cannot be determined from the model name,
            required provider package is not installed, or no credentials are
            configured.

    Examples:
        >>> model = create_model("anthropic:claude-sonnet-4-5")
        >>> model = create_model("openai:gpt-4o")
        >>> model = create_model("gpt-4o")  # Auto-detects openai
        >>> model = create_model()  # Uses environment defaults
    """
    if not model_spec:
        model_spec = _get_default_model_spec()

    # Parse provider:model syntax
    provider: str
    model_name: str
    parsed = ModelSpec.try_parse(model_spec)
    if parsed:
        # Explicit provider:model (e.g., "anthropic:claude-sonnet-4-5")
        provider, model_name = parsed.provider, parsed.model
    elif ":" in model_spec:
        # Contains colon but ModelSpec rejected it (empty provider or model)
        _, _, after = model_spec.partition(":")
        if after:
            # Leading colon (e.g., ":claude-opus-4-6") — treat as bare model name
            model_name = after
            provider = detect_provider(model_name) or ""
        else:
            msg = (
                f"Invalid model spec '{model_spec}': model name is required "
                "(e.g., 'anthropic:claude-sonnet-4-5' or 'claude-sonnet-4-5')"
            )
            raise ModelConfigError(msg)
    else:
        # Bare model name — auto-detect provider or let init_chat_model infer
        model_name = model_spec
        provider = detect_provider(model_spec) or ""

    # Provider-specific kwargs (with per-model overrides)
    kwargs = _get_provider_kwargs(provider, model_name=model_name)

    # CLI --model-params take highest priority
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Check if this provider uses a custom BaseChatModel class
    config = ModelConfig.load()
    class_path = config.get_class_path(provider) if provider else None

    if class_path:
        model = _create_model_from_class(class_path, model_name, provider, kwargs)
    else:
        model = _create_model_via_init(model_name, provider, kwargs)

    resolved_provider = provider or getattr(model, "_model_provider", provider)

    # Apply profile overrides from config.toml (e.g., max_input_tokens)
    if provider:
        config_profile_overrides = config.get_profile_overrides(
            provider, model_name=model_name
        )
        if config_profile_overrides:
            _apply_profile_overrides(
                model,
                config_profile_overrides,
                model_name,
                label=f"config.toml (provider '{provider}')",
            )

    # CLI --profile-override takes highest priority (on top of config.toml)
    if profile_overrides:
        _apply_profile_overrides(
            model,
            profile_overrides,
            model_name,
            label="CLI --profile-override",
            raise_on_failure=True,
        )

    # Extract context limit from model profile (if available)
    context_limit: int | None = None
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int):
        context_limit = profile["max_input_tokens"]

    return ModelResult(
        model=model,
        model_name=model_name,
        provider=resolved_provider,
        context_limit=context_limit,
    )


def validate_model_capabilities(model: BaseChatModel, model_name: str) -> None:
    """Validate that the model has required capabilities for `deepagents`.

    Checks the model's profile (if available) to ensure it supports tool calling, which
    is required for agent functionality. Issues warnings for models without profiles or
    with limited context windows.

    Args:
        model: The instantiated model to validate.
        model_name: Model name for error/warning messages.

    Note:
        This validation is best-effort. Models without profiles will pass with
        a warning. Exits via sys.exit(1) if model profile explicitly indicates
        tool_calling=False.
    """
    profile = getattr(model, "profile", None)

    if profile is None:
        # Model doesn't have profile data - warn but allow
        console.print(
            f"[dim][yellow]Note:[/yellow] No capability profile for "
            f"'{model_name}'. Cannot verify tool calling support.[/dim]"
        )
        return

    if not isinstance(profile, dict):
        return

    # Check required capability: tool_calling
    tool_calling = profile.get("tool_calling")
    if tool_calling is False:
        console.print(
            f"[bold red]Error:[/bold red] Model '{model_name}' "
            "does not support tool calling."
        )
        console.print(
            "\nDeep Agents requires tool calling for agent functionality. "
            "Please choose a model that supports tool calling."
        )
        console.print("\nSee MODELS.md for supported models.")
        sys.exit(1)

    # Warn about potentially limited context (< 8k tokens)
    max_input_tokens = profile.get("max_input_tokens")
    if max_input_tokens and max_input_tokens < 8000:  # noqa: PLR2004  # Model context window default
        console.print(
            f"[dim][yellow]Warning:[/yellow] Model '{model_name}' has limited context "
            f"({max_input_tokens:,} tokens). Agent performance may be affected.[/dim]"
        )
