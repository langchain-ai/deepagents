"""Deploy configuration parsing and validation.

Reads ``deepagents.toml`` and produces a validated :class:`DeployConfig`
dataclass used by the bundler and deploy commands.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Valid sandbox providers (mirrors sandbox_factory._PROVIDER_TO_WORKING_DIR)
VALID_SANDBOX_PROVIDERS = frozenset(
    {"none", "agentcore", "daytona", "langsmith", "modal", "runloop"}
)

DEFAULT_CONFIG_FILENAME = "deepagents.toml"


@dataclass(frozen=True)
class AgentConfig:
    """``[agent]`` section — core agent identity."""

    name: str
    model: str = "anthropic:claude-sonnet-4-6"
    system_prompt: str = ""


VALID_MEMORY_SCOPES = frozenset({"agent", "user"})


@dataclass(frozen=True)
class MemoryConfig:
    """``[memory]`` section — AGENTS.md file paths."""

    sources: list[str] = field(default_factory=list)
    scope: str = "agent"
    """Memory scope: ``"agent"`` shares memory across all users,
    ``"user"`` isolates memory per user via ``ctx.runtime.context.user_id``."""


@dataclass(frozen=True)
class SkillsConfig:
    """``[skills]`` section — skill directory paths."""

    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolsConfig:
    """``[tools]`` section — Python file with @tool functions."""

    python_file: str | None = None
    functions: list[str] | None = None


@dataclass(frozen=True)
class McpConfig:
    """``[mcp]`` section — MCP server configuration."""

    config: str | None = None


@dataclass(frozen=True)
class SandboxConfig:
    """``[sandbox]`` section — sandbox provider settings."""

    provider: str = "langsmith"
    template: str = "deepagents-deploy"
    image: str = "python:3"


@dataclass(frozen=True)
class DeploySettingsConfig:
    """``[deploy]`` section — deployment build settings."""

    python_version: str = "3.12"
    dependencies: list[str] = field(default_factory=list)
    env_file: str | None = None


@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from ``deepagents.toml``."""

    agent: AgentConfig
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    deploy: DeploySettingsConfig = field(default_factory=DeploySettingsConfig)

    def validate(self, project_root: Path) -> list[str]:
        """Validate config against the filesystem.

        Args:
            project_root: Directory containing ``deepagents.toml``.

        Returns:
            List of validation error strings. Empty if valid.
        """
        errors: list[str] = []

        # Memory sources must exist
        for src in self.memory.sources:
            if not (project_root / src).exists():
                errors.append(f"Memory source not found: {src}")

        # Skills sources must exist
        for src in self.skills.sources:
            p = project_root / src
            if not p.exists():
                errors.append(f"Skills source not found: {src}")
            elif p.is_file():
                errors.append(f"Skills source must be a directory: {src}")

        # Tools python_file must exist
        if self.tools.python_file:
            if not (project_root / self.tools.python_file).exists():
                errors.append(f"Tools python_file not found: {self.tools.python_file}")

        # MCP config must exist and contain only http/sse servers
        if self.mcp.config:
            mcp_path = project_root / self.mcp.config
            if not mcp_path.exists():
                errors.append(f"MCP config not found: {self.mcp.config}")
            else:
                mcp_errors = _validate_mcp_for_deploy(mcp_path)
                errors.extend(mcp_errors)

        # Sandbox provider must be valid
        if self.sandbox.provider not in VALID_SANDBOX_PROVIDERS:
            errors.append(
                f"Unknown sandbox provider: {self.sandbox.provider}. "
                f"Valid: {', '.join(sorted(VALID_SANDBOX_PROVIDERS))}"
            )

        # Env file must exist if specified
        if self.deploy.env_file:
            if not (project_root / self.deploy.env_file).exists():
                errors.append(f"Env file not found: {self.deploy.env_file}")

        return errors


def _validate_mcp_for_deploy(mcp_path: Path) -> list[str]:
    """Validate that MCP config only uses http/sse transports (no stdio).

    Args:
        mcp_path: Path to ``.mcp.json`` file.

    Returns:
        List of error strings.
    """
    import json

    errors: list[str] = []
    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [f"Could not read MCP config: {e}"]

    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        return ["MCP config 'mcpServers' must be a dictionary"]

    for name, server_config in servers.items():
        transport = server_config.get("type", server_config.get("transport", "stdio"))
        if transport == "stdio":
            errors.append(
                f"MCP server '{name}' uses stdio transport, which is not "
                "supported in deployed context. Use http or sse instead."
            )

    return errors


def load_config(config_path: Path) -> DeployConfig:
    """Load and parse a ``deepagents.toml`` file.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        Parsed :class:`DeployConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is missing required fields or has invalid values.
    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> DeployConfig:
    """Parse raw TOML dict into a DeployConfig.

    Args:
        data: Parsed TOML data.

    Returns:
        Validated :class:`DeployConfig`.

    Raises:
        ValueError: If required fields are missing.
    """
    # [agent] section — required
    agent_data = data.get("agent", {})
    if "name" not in agent_data:
        msg = "[agent].name is required in deepagents.toml"
        raise ValueError(msg)

    agent = AgentConfig(
        name=agent_data["name"],
        model=agent_data.get("model", "anthropic:claude-sonnet-4-6"),
        system_prompt=agent_data.get("system_prompt", ""),
    )

    # [memory] section — optional
    memory_data = data.get("memory", {})
    memory_scope = memory_data.get("scope", "agent")
    if memory_scope not in VALID_MEMORY_SCOPES:
        msg = (
            f"[memory].scope must be one of {sorted(VALID_MEMORY_SCOPES)}, "
            f"got {memory_scope!r}"
        )
        raise ValueError(msg)
    memory = MemoryConfig(
        sources=memory_data.get("sources", []),
        scope=memory_scope,
    )

    # [skills] section — optional
    skills_data = data.get("skills", {})
    skills = SkillsConfig(sources=skills_data.get("sources", []))

    # [tools] section — optional
    tools_data = data.get("tools", {})
    tools = ToolsConfig(
        python_file=tools_data.get("python_file"),
        functions=tools_data.get("functions"),
    )

    # [mcp] section — optional
    mcp_data = data.get("mcp", {})
    mcp = McpConfig(config=mcp_data.get("config"))

    # [sandbox] section — optional
    sandbox_data = data.get("sandbox", {})
    sandbox = SandboxConfig(
        provider=sandbox_data.get("provider", "langsmith"),
        template=sandbox_data.get("template", "deepagents-deploy"),
        image=sandbox_data.get("image", "python:3"),
    )

    # [deploy] section — optional
    deploy_data = data.get("deploy", {})
    deploy_settings = DeploySettingsConfig(
        python_version=deploy_data.get("python_version", "3.12"),
        dependencies=deploy_data.get("dependencies", []),
        env_file=deploy_data.get("env_file"),
    )

    return DeployConfig(
        agent=agent,
        memory=memory,
        skills=skills,
        tools=tools,
        mcp=mcp,
        sandbox=sandbox,
        deploy=deploy_settings,
    )


def generate_starter_config() -> str:
    """Generate a starter ``deepagents.toml`` template.

    Returns:
        TOML string with commented example configuration.
    """
    return '''\
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"
# system_prompt = "You are a helpful assistant"

[memory]
sources = ["./AGENTS.md"]
scope = "agent"  # "agent" = shared across all users, "user" = per-user isolation

[skills]
sources = ["./skills/"]

# [tools]
# python_file = "./tools.py"
# functions = ["my_tool"]  # Optional: explicit list (else auto-discover @tool functions)

# [mcp]
# config = "./.mcp.json"  # Only http/sse servers supported in deployed context

[sandbox]
provider = "langsmith"
# template = "deepagents-deploy"
# image = "python:3"

[deploy]
python_version = "3.12"
# dependencies = []
# env_file = ".env"
'''
