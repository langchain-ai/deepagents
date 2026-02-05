"""Configuration, constants, and model creation for the CLI."""

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import dotenv
from rich.console import Console

from deepagents_cli._version import __version__

dotenv.load_dotenv()

# CRITICAL: Override LANGSMITH_PROJECT to route agent traces to separate project
# LangSmith reads LANGSMITH_PROJECT at invocation time, so we override it here
# and preserve the user's original value for shell commands
_deepagents_project = os.environ.get("DEEPAGENTS_LANGSMITH_PROJECT")
_original_langsmith_project = os.environ.get("LANGSMITH_PROJECT")
if _deepagents_project:
    # Override LANGSMITH_PROJECT for agent traces
    os.environ["LANGSMITH_PROJECT"] = _deepagents_project

# E402: Now safe to import LangChain modules
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

# Color scheme
COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}


# Charset mode configuration
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

    # Tree connectors (full prefixes for tree display)
    tree_branch: str  # "├── " vs "+-- "
    tree_last: str  # "└── " vs "`-- "
    tree_vertical: str  # "│   " vs "|   "


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
    arrow_up="↑",
    arrow_down="↓",
    bullet="•",
    cursor="›",  # noqa: RUF001
    # Box-drawing characters
    box_vertical="│",
    box_horizontal="─",
    box_double_horizontal="═",
    gutter_bar="▌",
    tree_branch="├── ",
    tree_last="└── ",
    tree_vertical="│   ",
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
    arrow_up="^",
    arrow_down="v",
    bullet="-",
    cursor=">",
    # Box-drawing characters
    box_vertical="|",
    box_horizontal="-",
    box_double_horizontal="=",
    gutter_bar="|",
    tree_branch="+-- ",
    tree_last="`-- ",
    tree_vertical="|   ",
)

# Module-level cache for detected glyphs
_glyphs_cache: Glyphs | None = None

# Module-level cache for editable install detection
_editable_cache: bool | None = None


def _is_editable_install() -> bool:
    """Check if deepagents-cli is installed in editable mode.

    Uses PEP 610 direct_url.json metadata to detect editable installs.

    Returns:
        True if installed in editable mode, False otherwise.
    """
    global _editable_cache  # noqa: PLW0603
    if _editable_cache is not None:
        return _editable_cache

    try:
        dist = distribution("deepagents-cli")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            data = json.loads(direct_url)
            _editable_cache = data.get("dir_info", {}).get("editable", False)
        else:
            _editable_cache = False
    except (PackageNotFoundError, FileNotFoundError, json.JSONDecodeError, TypeError):
        _editable_cache = False

    return _editable_cache


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
    global _glyphs_cache  # noqa: PLW0603
    if _glyphs_cache is not None:
        return _glyphs_cache

    mode = _detect_charset_mode()
    _glyphs_cache = ASCII_GLYPHS if mode == CharsetMode.ASCII else UNICODE_GLYPHS
    return _glyphs_cache


def reset_glyphs_cache() -> None:
    """Reset the glyphs cache (for testing)."""
    global _glyphs_cache  # noqa: PLW0603
    _glyphs_cache = None


# Text art banners (Unicode and ASCII variants)

_UNICODE_BANNER = f"""
██████╗  ███████╗ ███████╗ ██████╗
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
██║  ██║ █████╗   █████╗   ██████╔╝
██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
██████╔╝ ███████╗ ███████╗ ██║
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

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
        Includes "(local install)" suffix when installed in editable mode.
    """
    if _detect_charset_mode() == CharsetMode.ASCII:
        banner = _ASCII_BANNER
    else:
        banner = _UNICODE_BANNER

    if _is_editable_install():
        banner = banner.replace(f"v{__version__}", f"v{__version__} (local install)")

    return banner


# Legacy alias for backwards compatibility
DEEP_AGENTS_ASCII = _UNICODE_BANNER

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


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from.
            Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def _find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific AGENTS.md file(s).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/AGENTS.md
    2. project_root/AGENTS.md

    Both files will be loaded and combined if both exist.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of paths to project AGENTS.md files (may contain 0, 1, or 2 paths).
    """
    paths = []

    # Check .deepagents/AGENTS.md (preferred)
    deepagents_md = project_root / ".deepagents" / "AGENTS.md"
    if deepagents_md.exists():
        paths.append(deepagents_md)

    # Check root AGENTS.md (fallback, but also include if both exist)
    root_md = project_root / "AGENTS.md"
    if root_md.exists():
        paths.append(root_md)

    return paths


@dataclass
class Settings:
    """Global settings and environment detection for deepagents-cli.

    This class is initialized once at startup and provides access to:
    - Available models and API keys
    - Current project information
    - Tool availability (e.g., Tavily)
    - File system paths

    Attributes:
        project_root: Current project root directory (if in a git project)

        openai_api_key: OpenAI API key if available
        anthropic_api_key: Anthropic API key if available
        deepseek_api_key: DeepSeek API key if available
        dashscope_api_key: Dashscope API key if available
        tavily_api_key: Tavily API key if available
        deepagents_langchain_project: LangSmith project name for deepagents
            agent tracing
        user_langchain_project: Original LANGSMITH_PROJECT from environment
            (for user code)
    """

    # API keys
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    deepseek_api_key: str | None
    dashscope_api_key: str | None
    tavily_api_key: str | None

    # Google Cloud configuration (for VertexAI)
    google_cloud_project: str | None

    # LangSmith configuration
    deepagents_langchain_project: str | None  # For deepagents agent tracing
    user_langchain_project: str | None  # Original LANGSMITH_PROJECT for user code

    # Model configuration
    model_name: str | None = None  # Currently active model name
    model_provider: str | None = None  # Provider (openai, anthropic, google)
    model_context_limit: int | None = None  # Max input tokens from model profile

    # Project information
    project_root: Path | None = None

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> "Settings":
        """Create settings by detecting the current environment.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            Settings instance with detected configuration
        """
        # Detect API keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        qwen_key = os.environ.get("DASHSCOPE_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")
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

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            deepseek_api_key=deepseek_key,
            dashscope_api_key=qwen_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
        )

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
    def has_deepseek(self) -> bool:
        """Check if DeepSeek API key is configured."""
        return self.deepseek_api_key is not None

    @property
    def has_qwen(self) -> bool:
        """Check if Dashscope API key is configured."""
        return self.dashscope_api_key is not None

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
    def has_deepagents_langchain_project(self) -> bool:
        """Check if deepagents LangChain project name is configured."""
        return self.deepagents_langchain_project is not None

    @property
    def has_project(self) -> bool:
        """Check if currently in a git project."""
        return self.project_root is not None

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

    def get_project_agent_md_path(self) -> Path | None:
        """Get project-level AGENTS.md path.

        Returns path regardless of whether the file exists.

        Returns:
            Path to {project_root}/.deepagents/AGENTS.md, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "AGENTS.md"

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

    def ensure_project_deepagents_dir(self) -> Path | None:
        """Ensure the project .deepagents directory exists and return its path.

        Returns:
            Path to project .deepagents directory, or None if not in a project
        """
        if not self.project_root:
            return None

        project_deepagents_dir = self.project_root / ".deepagents"
        project_deepagents_dir.mkdir(parents=True, exist_ok=True)
        return project_deepagents_dir

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


# Global settings instance (initialized once)
settings = Settings.from_environment()


class SessionState:
    """Holds mutable session state (auto-approve mode, etc)."""

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        """Initialize session state with optional flags.

        Args:
            auto_approve: Whether to auto-approve tool calls without prompting.
            no_splash: Whether to skip displaying the splash screen on startup.
        """
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        self.thread_id = str(uuid.uuid4())

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return new state.

        Returns:
            The new auto_approve state.
        """
        self.auto_approve = not self.auto_approve
        return self.auto_approve


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (AGENTS.md) is handled separately by the middleware.

    Returns:
        The default agent instructions as a string.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def _detect_provider(model_name: str) -> str | None:
    """Auto-detect provider from model name.

    Args:
        model_name: Model name to detect provider from

    Returns:
        Provider name (openai, anthropic, google, vertexai, deepseek, qwen) or None if can't detect
    """
    model_lower = model_name.lower()

    # Check for model name patterns
    if any(x in model_lower for x in ["gpt", "o1", "o3"]):
        return "openai"
    if "claude" in model_lower:
        if not settings.has_anthropic and settings.has_vertex_ai:
            return "vertexai"
        return "anthropic"
    if "gemini" in model_lower:
        if settings.has_vertex_ai:
            return "vertexai"
        return "google"
    if any(x in model_lower for x in ["deepseek", "deepseek-chat", "deepseek-coder"]):
        return "deepseek"
    if any(x in model_lower for x in ["qwen", "qwen-turbo", "qwen-plus", "qwen-max", "qwen3-max"]):
        return "qwen"

    return None


def create_model(model_name_override: str | None = None) -> BaseChatModel:
    """Create the appropriate model based on available API keys.

    Uses the global settings instance to determine which model to create.

    Args:
        model_name_override: Optional model name to use instead of environment variable

    Returns:
        ChatModel instance (OpenAI, Anthropic, Google, DeepSeek or Qwen)

    Raises:
        SystemExit if no API key is configured or model provider can't be determined
    """
    # Determine provider and model
    if model_name_override:
        # Use provided model, auto-detect provider
        provider = _detect_provider(model_name_override)
        if not provider:
            console.print(
                "[bold red]Error:[/bold red] Could not detect provider "
                f"from model name: {model_name_override}"
            )
            console.print("\nSupported model name patterns:")
            console.print("  - OpenAI: gpt-*, o1-*, o3-*")
            console.print("  - Anthropic: claude-*")
            console.print("  - Google: gemini-* (requires GOOGLE_API_KEY)")
            console.print("  - DeepSeek: deepseek-*")
            console.print("  - Qwen: qwen-*, qwen3-*")
            console.print(
                "  - VertexAI: claude-*/gemini-* (requires GOOGLE_CLOUD_PROJECT, "
                "uses Application Default Credentials)"
            )
            sys.exit(1)

        # Check if credentials for detected provider are available
        if provider == "openai" and not settings.has_openai:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' "
                "requires OPENAI_API_KEY"
            )
            sys.exit(1)
        elif provider == "anthropic" and not settings.has_anthropic:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' "
                "requires ANTHROPIC_API_KEY"
            )
            sys.exit(1)
        elif provider == "google" and not settings.has_google:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' "
                "requires GOOGLE_API_KEY"
            )
            sys.exit(1)
        elif provider == "vertexai" and not settings.has_vertex_ai:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' requires "
                "GOOGLE_CLOUD_PROJECT to be set"
            )
            console.print("\nPlease set GOOGLE_CLOUD_PROJECT environment variable.")
            console.print("Also ensure you have authenticated with:")
            console.print("  gcloud auth application-default login")
            sys.exit(1)
        elif provider == "deepseek" and not settings.has_deepseek:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' "
                "requires DEEPSEEK_API_KEY"
            )
            sys.exit(1)
        elif provider == "qwen" and not settings.has_qwen:
            console.print(
                f"[bold red]Error:[/bold red] Model '{model_name_override}' "
                "requires DASHSCOPE_API_KEY"
            )
            sys.exit(1)

        model_name = model_name_override
    # Use environment variable defaults, detect provider by API key priority
    elif settings.has_openai:
        provider = "openai"
        model_name = os.environ.get("OPENAI_MODEL", "gpt-5.2")
    elif settings.has_anthropic:
        provider = "anthropic"
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    elif settings.has_google:
        provider = "google"
        model_name = os.environ.get("GOOGLE_MODEL", "gemini-3-pro-preview")
    elif settings.has_vertex_ai:
        provider = "vertexai"
        model_name = os.environ.get("VERTEX_AI_MODEL", "gemini-3-pro-preview")
    elif settings.has_deepseek:
        provider = "deepseek"
        model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    elif settings.has_qwen:
        provider = "qwen"
        model_name = os.environ.get("QWEN_MODEL", "qwen-plus")
    else:
        console.print("[bold red]Error:[/bold red] No credentials configured.")
        console.print("\nPlease set one of the following environment variables:")
        console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5.2)")
        console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
        console.print("  - GOOGLE_API_KEY     (for Google Gemini models)")
        console.print("  - DEEPSEEK_API_KEY   (for DeepSeek models)")
        console.print("  - DASHSCOPE_API_KEY  (for Qwen models)")
        console.print(
            "  - GOOGLE_CLOUD_PROJECT (for VertexAI models, "
            "with Application Default Credentials)"
        )
        console.print("\nExample:")
        console.print("  export OPENAI_API_KEY=your_api_key_here")
        console.print("\nOr add it to your .env file.")
        sys.exit(1)

    # Store model info in settings for display
    settings.model_name = model_name
    settings.model_provider = provider

    # Create the model
    model: BaseChatModel
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name)  # type: ignore[call-arg]
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name=model_name,
            max_tokens=20_000,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
        )
    elif provider == "vertexai":
        model_lower = model_name.lower()

        if "claude" in model_lower:
            try:
                from langchain_google_vertexai.model_garden import (  # type: ignore[unresolved-import]
                    ChatAnthropicVertex,
                )
            except ImportError:
                console.print(
                    "[bold red]Error:[/bold red] langchain-google-vertexai "
                    "package is required for this model"
                )
                console.print("\nInstall it with:")
                console.print("  pip install deepagents-cli[vertexai]", markup=False)
                sys.exit(1)

            model = ChatAnthropicVertex(
                # Remove version tag (e.g., "claude-haiku-4-5@20251015" ->
                # "claude-haiku-4-5"). ChatAnthropicVertex expects just the base
                # model name without the @version suffix.
                model_name=model_name,
                project=settings.google_cloud_project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
                max_tokens=20_000,
            )
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=model_name,
                project=settings.google_cloud_project,
                vertexai=True,
                temperature=0,
                max_tokens=None,
            )
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model=model_name,
            temperature=0,
        )
    elif provider == "qwen":
        from langchain_qwq import ChatQwen
        model = ChatQwen(
            model=model_name,
            temperature=0,
            max_tokens=20_000,
        )
    else:
        # Should not reach here due to earlier validation
        console.print(f"[bold red]Error:[/bold red] Unknown provider: {provider}")
        sys.exit(1)

    # Extract context limit from model profile (if available)
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and isinstance(profile.get("max_input_tokens"), int):
        settings.model_context_limit = profile["max_input_tokens"]

    return model


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
    if max_input_tokens and max_input_tokens < 8000:
        console.print(
            f"[dim][yellow]Warning:[/yellow] Model '{model_name}' has limited context "
            f"({max_input_tokens:,} tokens). Agent performance may be affected.[/dim]"
        )
