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


VALID_AGENT_MEMORIES_BACKENDS = frozenset({"hub", "store"})


@dataclass(frozen=True)
class AgentMemoriesConfig:
    """``[agent_memories]`` section — agent-scoped memory files.

    Mounted by the runtime composite at ``/agent_memories/``. The
    ``backend`` field selects between ``"hub"`` (a LangSmith Prompt Hub
    repo, versioned and visible in the UI) and ``"store"`` (the LangGraph
    persistent store, namespaced by ``(agent_name, "agent_memories")``).
    Both options keep the data agent-scoped — every user of the agent
    sees the same files.

    ``sources`` are local paths the bundler ships into the chosen
    backend at deploy time.
    """

    backend: str = "hub"
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UserMemoriesConfig:
    """``[user_memories]`` section — user-scoped memory files.

    Always backed by the LangGraph store with a namespace of
    ``(agent_name, user_id, "user_memories")``. Mounted by the runtime
    composite at ``/user_memories/``. ``sources`` are seed file paths
    the bundler walks; the runtime middleware reads/writes the same
    paths under the user-scoped namespace at request time.
    """

    sources: list[str] = field(default_factory=list)


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
    agent_memories: AgentMemoriesConfig = field(default_factory=AgentMemoriesConfig)
    user_memories: UserMemoriesConfig = field(default_factory=UserMemoriesConfig)
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

        if self.agent_memories.backend not in VALID_AGENT_MEMORIES_BACKENDS:
            errors.append(
                f"[agent_memories].backend must be one of "
                f"{sorted(VALID_AGENT_MEMORIES_BACKENDS)}, "
                f"got {self.agent_memories.backend!r}"
            )

        for src in self.agent_memories.sources:
            if not (project_root / src).exists():
                errors.append(f"Agent memory source not found: {src}")

        for src in self.user_memories.sources:
            if not (project_root / src).exists():
                errors.append(f"User memory source not found: {src}")

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

    # [agent_memories] section — optional. Replaces legacy [memory].
    agent_memories_data = data.get("agent_memories", {})
    agent_memories = AgentMemoriesConfig(
        backend=agent_memories_data.get("backend", "hub"),
        sources=agent_memories_data.get("sources", []),
    )

    # [user_memories] section — optional. Always store-backed with the
    # user identity in the namespace.
    user_memories_data = data.get("user_memories", {})
    user_memories = UserMemoriesConfig(
        sources=user_memories_data.get("sources", []),
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
        agent_memories=agent_memories,
        user_memories=user_memories,
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

[agent_memories]
backend = "hub"  # or "store" — chooses the storage for /agent_memories/
sources = ["./AGENTS.md"]

# [user_memories]
# sources = ["./preferences.md"]  # always store-backed, per-user namespace

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
