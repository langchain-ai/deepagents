"""Deploy configuration model for deepagents.

Reads and validates a ``deepagents.json`` deployment manifest. The config
is a high-level description of an agent deployment that gets compiled down
to a ``langgraph.json`` plus supporting files at deploy time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default scope for namespace resolution
DEFAULT_NAMESPACE_SCOPE = "assistant"
DEFAULT_NAMESPACE_PREFIX = "filesystem"
DEFAULT_SANDBOX_PROVIDER = "langsmith"
DEFAULT_SANDBOX_SCOPE = "thread"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_BACKEND_TYPE = "store"


@dataclass(frozen=True)
class NamespaceConfig:
    """Namespace scoping configuration for the Store backend.

    The scope determines how namespaces are resolved at runtime:
    - ``"assistant"``: ``(assistant_id, prefix)`` — shared across all users/threads
    - ``"user"``: ``(user_id, prefix)`` — per-user, persists across threads
    - ``"thread"``: ``(thread_id, prefix)`` — per-conversation, isolated
    - ``"user+thread"``: ``(user_id, thread_id, prefix)`` — per-user per-conversation
    """

    scope: str = DEFAULT_NAMESPACE_SCOPE
    prefix: str = DEFAULT_NAMESPACE_PREFIX

    def __post_init__(self) -> None:
        valid_scopes = {"assistant", "user", "thread", "user+thread"}
        if self.scope not in valid_scopes:
            msg = f"Invalid namespace scope {self.scope!r}. Must be one of: {', '.join(sorted(valid_scopes))}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NamespaceConfig:
        if data is None:
            return cls()
        return cls(
            scope=data.get("scope", DEFAULT_NAMESPACE_SCOPE),
            prefix=data.get("prefix", DEFAULT_NAMESPACE_PREFIX),
        )


@dataclass(frozen=True)
class BackendConfig:
    """Backend configuration for file storage.

    The backend type determines how files are stored:
    - ``"store"``: LangGraph Store (persistent cross-thread, default)
    - ``"sandbox"``: Files live inside the sandbox filesystem
    - ``"custom"``: Custom backend via ``module:variable`` import path
    """

    type: str = DEFAULT_BACKEND_TYPE
    namespace: NamespaceConfig = field(default_factory=NamespaceConfig)
    path: str | None = None  # For custom backends: "module.py:factory"

    def __post_init__(self) -> None:
        valid_types = {"store", "sandbox", "custom"}
        if self.type not in valid_types:
            msg = f"Invalid backend type {self.type!r}. Must be one of: {', '.join(sorted(valid_types))}"
            raise ValueError(msg)
        if self.type == "custom" and not self.path:
            msg = "Custom backend requires a 'path' field (e.g., './my_backend.py:create_backend')"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BackendConfig:
        if data is None:
            return cls()
        return cls(
            type=data.get("type", DEFAULT_BACKEND_TYPE),
            namespace=NamespaceConfig.from_dict(data.get("namespace")),
            path=data.get("path"),
        )


@dataclass(frozen=True)
class MemoryConfig:
    """Memory (AGENTS.md) configuration."""

    scope: str = DEFAULT_NAMESPACE_SCOPE
    sources: list[str] = field(default_factory=lambda: [".deepagents/AGENTS.md"])

    def __post_init__(self) -> None:
        valid_scopes = {"assistant", "user", "thread", "user+thread"}
        if self.scope not in valid_scopes:
            msg = f"Invalid memory scope {self.scope!r}. Must be one of: {', '.join(sorted(valid_scopes))}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MemoryConfig:
        if data is None:
            return cls()
        return cls(
            scope=data.get("scope", DEFAULT_NAMESPACE_SCOPE),
            sources=data.get("sources", [".deepagents/AGENTS.md"]),
        )


@dataclass(frozen=True)
class SkillsConfig:
    """Skills configuration."""

    sources: list[str] = field(default_factory=lambda: [".deepagents/skills"])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SkillsConfig:
        if data is None:
            return cls()
        return cls(
            sources=data.get("sources", [".deepagents/skills"]),
        )


@dataclass(frozen=True)
class ToolsConfig:
    """Tools configuration."""

    shell: bool = True
    shell_allow_list: list[str] | None = None
    web_search: bool = True
    fetch_url: bool = True
    http_request: bool = True
    mcp: str | bool = False  # Path to .mcp.json or False to disable
    custom: str | None = None  # "module.py:variable" for custom tools

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ToolsConfig:
        if data is None:
            return cls()

        shell_allow_list = data.get("shell_allow_list")
        if isinstance(shell_allow_list, str):
            shell_allow_list = [s.strip() for s in shell_allow_list.split(",")]

        return cls(
            shell=data.get("shell", True),
            shell_allow_list=shell_allow_list,
            web_search=data.get("web_search", True),
            fetch_url=data.get("fetch_url", True),
            http_request=data.get("http_request", True),
            mcp=data.get("mcp", False),
            custom=data.get("custom"),
        )


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox configuration for code execution.

    The scope determines sandbox lifecycle:
    - ``"assistant"``: One sandbox shared across all threads
    - ``"user"``: Per-user sandbox, persists across threads
    - ``"thread"``: Fresh sandbox per conversation (default)
    - ``"user+thread"``: Per-user per-thread sandbox
    """

    provider: str = DEFAULT_SANDBOX_PROVIDER
    scope: str = DEFAULT_SANDBOX_SCOPE
    template: str | None = None
    image: str | None = None
    setup_script: str | None = None

    def __post_init__(self) -> None:
        valid_providers = {"langsmith", "modal", "daytona", "runloop"}
        if self.provider not in valid_providers:
            msg = f"Invalid sandbox provider {self.provider!r}. Must be one of: {', '.join(sorted(valid_providers))}"
            raise ValueError(msg)
        valid_scopes = {"assistant", "user", "thread", "user+thread"}
        if self.scope not in valid_scopes:
            msg = f"Invalid sandbox scope {self.scope!r}. Must be one of: {', '.join(sorted(valid_scopes))}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SandboxConfig | None:
        """Parse sandbox config. Returns None if sandbox is explicitly disabled."""
        if data is None:
            # Default: langsmith sandbox with thread scope
            return cls()
        if data is False:
            return None
        return cls(
            provider=data.get("provider", DEFAULT_SANDBOX_PROVIDER),
            scope=data.get("scope", DEFAULT_SANDBOX_SCOPE),
            template=data.get("template"),
            image=data.get("image"),
            setup_script=data.get("setup_script"),
        )


@dataclass(frozen=True)
class DeployConfig:
    """Full deployment configuration.

    Read from ``deepagents.json`` at the project root.
    """

    # Identity
    agent: str = "agent"
    description: str = ""

    # Model
    model: str = "anthropic:claude-sonnet-4-6"
    model_params: dict[str, Any] = field(default_factory=dict)

    # Prompt
    prompt: str | None = None  # Override BASE_AGENT_PROMPT

    # Components
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    sandbox: SandboxConfig | None = field(default_factory=SandboxConfig)

    # Environment
    env: str = ".env"
    python_version: str = DEFAULT_PYTHON_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeployConfig:
        """Parse a deploy config from a dict (typically loaded from JSON)."""
        return cls(
            agent=data.get("agent", "agent"),
            description=data.get("description", ""),
            model=data.get("model", "anthropic:claude-sonnet-4-6"),
            model_params=data.get("model_params", {}),
            prompt=data.get("prompt"),
            memory=MemoryConfig.from_dict(data.get("memory")),
            skills=SkillsConfig.from_dict(data.get("skills")),
            tools=ToolsConfig.from_dict(data.get("tools")),
            backend=BackendConfig.from_dict(data.get("backend")),
            sandbox=SandboxConfig.from_dict(data.get("sandbox")),
            env=data.get("env", ".env"),
            python_version=data.get("python_version", DEFAULT_PYTHON_VERSION),
        )

    @classmethod
    def load(cls, config_path: Path | None = None) -> DeployConfig:
        """Load deploy config from a file.

        Args:
            config_path: Path to ``deepagents.json``. If None, searches for
                ``deepagents.json`` in the current directory.

        Returns:
            Parsed deploy configuration.
        """
        if config_path is None:
            config_path = Path.cwd() / "deepagents.json"

        if not config_path.exists():
            logger.info(
                "No %s found, using defaults",
                config_path.name,
            )
            return cls()

        try:
            data = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {config_path}: {exc}"
            raise ValueError(msg) from exc

        logger.info("Loaded deploy config from %s", config_path)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dict for JSON output."""
        result: dict[str, Any] = {
            "agent": self.agent,
            "model": self.model,
        }
        if self.description:
            result["description"] = self.description
        if self.model_params:
            result["model_params"] = self.model_params
        if self.prompt:
            result["prompt"] = self.prompt
        return result
