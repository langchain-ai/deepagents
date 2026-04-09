"""Deploy configuration parsing and validation.

Reads `deepagents.toml` and produces a validated `DeployConfig`.

The new minimal surface has exactly two sections:

- `[agent]`: name + model
- `[sandbox]`: sandbox provider settings

`AGENTS.md` is always seeded into a shared memory namespace so the agent can
read it at runtime, but writes/edits to that path are blocked by a read-only
middleware in the generated graph.

Skills (`src/skills/`) and MCP servers (`src/mcp.json`) are auto-detected
from the project layout. The agent's system prompt is read from
`src/AGENTS.md` at bundle time — there is no `system_prompt` key.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_SANDBOX_PROVIDERS = frozenset(
    {"none", "daytona", "langsmith", "modal", "runloop"}
)
"""Valid sandbox providers (mirrors sandbox_factory._PROVIDER_TO_WORKING_DIR)"""

DEFAULT_CONFIG_FILENAME = "deepagents.toml"

# Canonical filenames inside the project root.
AGENTS_MD_FILENAME = "AGENTS.md"
SKILLS_DIRNAME = "skills"
MCP_FILENAME = "mcp.json"
AGENTS_DIRNAME = "agents"


@dataclass(frozen=True)
class AgentConfig:
    """``[agent]`` section — core agent identity."""

    name: str
    model: str = "anthropic:claude-sonnet-4-6"


VALID_SANDBOX_SCOPES = frozenset({"thread", "assistant"})


@dataclass(frozen=True)
class SandboxConfig:
    """`[sandbox]` section — sandbox provider settings.

    The whole section is optional. When omitted (or `provider = "none"`)
    the runtime falls back to an in-process `StateBackend` and tools
    like `execute` become no-ops.

    `scope` controls how the sandbox cache keys are built:

    - `"thread"` (default): one sandbox per thread. Different threads
        get different sandboxes, same thread reuses across turns.
    - `"assistant"`: one sandbox per assistant. All threads of the
        same assistant share a single sandbox and its filesystem.
    """

    provider: str = "none"
    template: str = "deepagents-deploy"
    image: str = "python:3"
    scope: str = "thread"


@dataclass(frozen=True)
class SubagentConfig:
    """A single subagent parsed from agents/{name}/."""

    agent: AgentConfig
    sandbox: SandboxConfig | None  # None means inherit from parent
    system_prompt: str
    description: str
    skills_dir: Path | None  # absolute path to subagent's skills/ if present
    mcp_path: Path | None  # absolute path to subagent's mcp.json if present


@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from `deepagents.toml`."""

    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    subagents: list[SubagentConfig] = field(default_factory=list)

    def validate(self, project_root: Path) -> list[str]:
        """Validate config against the filesystem.

        Args:
            project_root: Directory containing `deepagents.toml` (i.e.
                the `src/` dir in the canonical layout).

        Returns:
            List of validation error strings. Empty if valid.
        """
        errors: list[str] = []

        # AGENTS.md is required — it's the system prompt.
        agents_md = project_root / AGENTS_MD_FILENAME
        if not agents_md.is_file():
            errors.append(
                f"{AGENTS_MD_FILENAME} not found in {project_root}. "
                f"This file is required — it provides the agent's system prompt."
            )

        # skills/ is optional; if present it must be a directory.
        skills_dir = project_root / SKILLS_DIRNAME
        if skills_dir.exists() and not skills_dir.is_dir():
            errors.append(f"{SKILLS_DIRNAME} must be a directory if present")

        # mcp.json is optional; if present it must be a file with only
        # http/sse transports (stdio is unsupported in deployed contexts).
        mcp_path = project_root / MCP_FILENAME
        if mcp_path.exists():
            if not mcp_path.is_file():
                errors.append(f"{MCP_FILENAME} must be a file if present")
            else:
                errors.extend(_validate_mcp_for_deploy(mcp_path))

        if self.sandbox.provider not in VALID_SANDBOX_PROVIDERS:
            errors.append(
                f"Unknown sandbox provider: {self.sandbox.provider}. "
                f"Valid: {', '.join(sorted(VALID_SANDBOX_PROVIDERS))}"
            )

        if self.sandbox.scope not in VALID_SANDBOX_SCOPES:
            errors.append(
                f"Unknown sandbox scope: {self.sandbox.scope}. "
                f"Valid: {', '.join(sorted(VALID_SANDBOX_SCOPES))}"
            )

        # Validate credentials for model provider.
        errors.extend(_validate_model_credentials(self.agent.model))

        # Validate credentials for sandbox provider.
        errors.extend(_validate_sandbox_credentials(self.sandbox.provider))

        # Validate subagents.
        seen_names: set[str] = set()
        for sa in self.subagents:
            if sa.agent.name in seen_names:
                errors.append(f"Duplicate subagent name: '{sa.agent.name}'")
            seen_names.add(sa.agent.name)

            if sa.agent.name == "general-purpose":
                errors.append("Subagent name 'general-purpose' is reserved")

            if sa.mcp_path and sa.mcp_path.is_file():
                errors.extend(_validate_mcp_for_deploy(sa.mcp_path))

            errors.extend(_validate_model_credentials(sa.agent.model))

            if sa.sandbox is not None:
                if sa.sandbox.provider not in VALID_SANDBOX_PROVIDERS:
                    errors.append(
                        f"Subagent '{sa.agent.name}': unknown sandbox provider "
                        f"{sa.sandbox.provider}. Valid: {', '.join(sorted(VALID_SANDBOX_PROVIDERS))}"
                    )

        return errors


def _validate_mcp_for_deploy(mcp_path: Path) -> list[str]:
    """Validate that MCP config only uses http/sse transports (no stdio)."""
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
    """Load and parse a `deepagents.toml` file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is missing required fields or has an
            unknown top-level section.
    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    return _parse_config(data)


_ALLOWED_SECTIONS = frozenset({"agent", "sandbox"})


def _parse_config(data: dict[str, Any]) -> DeployConfig:
    """Parse raw TOML dict into a `DeployConfig`."""
    # Reject unknown top-level sections up front — the old surface had
    # many more, and silently ignoring them would hide migration bugs.
    unknown = set(data.keys()) - _ALLOWED_SECTIONS
    if unknown:
        msg = (
            f"Unknown section(s) in deepagents.toml: {sorted(unknown)}. "
            f"The new surface only accepts: {sorted(_ALLOWED_SECTIONS)}. "
            f"Skills, MCP, and tools are auto-detected from the project layout."
        )
        raise ValueError(msg)

    agent_data = data.get("agent", {})
    if "name" not in agent_data:
        msg = "[agent].name is required in deepagents.toml"
        raise ValueError(msg)

    agent = AgentConfig(
        name=agent_data["name"],
        model=agent_data.get("model", "anthropic:claude-sonnet-4-6"),
    )

    sandbox_data = data.get("sandbox", {})
    sandbox = SandboxConfig(
        provider=sandbox_data.get("provider", "none"),
        template=sandbox_data.get("template", "deepagents-deploy"),
        image=sandbox_data.get("image", "python:3"),
        scope=sandbox_data.get("scope", "thread"),
    )

    return DeployConfig(agent=agent, sandbox=sandbox)


_MODEL_PROVIDER_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_CLOUD_PROJECT",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "perplexity": "PPLX_API_KEY",
}

_SANDBOX_PROVIDER_ENV: dict[str, list[str]] = {
    "langsmith": [
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_SANDBOX_API_KEY",
    ],
    "daytona": ["DAYTONA_API_KEY"],
    "runloop": ["RUNLOOP_API_KEY"],
    # Modal falls back to default auth if env vars are not set.
}


def _validate_model_credentials(model: str) -> list[str]:
    """Check that the API key env var is set for the model provider."""
    if ":" not in model:
        return []
    provider = model.split(":", 1)[0]
    env_var = _MODEL_PROVIDER_ENV.get(provider)
    if env_var is None:
        return []
    if os.environ.get(env_var):
        return []
    return [
        (
            f"Missing API key for model provider '{provider}': "
            f"set {env_var} in your .env file or environment."
        ),
    ]


def _validate_sandbox_credentials(provider: str) -> list[str]:
    """Check that the API key env var is set for the sandbox provider."""
    required_vars = _SANDBOX_PROVIDER_ENV.get(provider)
    if required_vars is None:
        return []
    if any(os.environ.get(v) for v in required_vars):
        return []
    return [
        (
            f"Missing API key for sandbox provider '{provider}': "
            f"set one of {', '.join(required_vars)} in your .env file or environment."
        ),
    ]


def find_config(start_path: Path | None = None) -> Path | None:
    """Find `deepagents.toml` in the current directory.

    Returns the path if found, or ``None`` otherwise.
    """
    current = (start_path or Path.cwd()).resolve()
    candidate = current / DEFAULT_CONFIG_FILENAME
    if candidate.is_file():
        return candidate
    return None


def _parse_subagent_frontmatter(agents_md_path: Path) -> tuple[str, str, str]:
    """Parse AGENTS.md frontmatter for name, description, and system_prompt.

    Returns:
        Tuple of (name, description, system_prompt).

    Raises:
        ValueError: If frontmatter is missing or invalid.
    """
    content = agents_md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if not match:
        msg = f"{agents_md_path}: missing YAML frontmatter (--- delimiters)"
        raise ValueError(msg)

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        msg = f"{agents_md_path}: invalid YAML frontmatter: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(frontmatter, dict):
        msg = f"{agents_md_path}: frontmatter must be a YAML mapping"
        raise ValueError(msg)

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name:
        msg = f"{agents_md_path}: missing required frontmatter field 'name'"
        raise ValueError(msg)

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description:
        msg = f"{agents_md_path}: missing required frontmatter field 'description'"
        raise ValueError(msg)

    system_prompt = match.group(2).strip()
    return name, description, system_prompt


def load_subagents(project_root: Path) -> list[SubagentConfig]:
    """Discover and load subagent configs from agents/ directory.

    Args:
        project_root: Directory containing the main deepagents.toml.

    Returns:
        List of SubagentConfig, sorted by name.

    Raises:
        ValueError: If a subagent has invalid config.
    """
    agents_dir = project_root / AGENTS_DIRNAME
    if not agents_dir.is_dir():
        return []

    subagents: list[SubagentConfig] = []
    seen_names: set[str] = set()

    for entry in sorted(agents_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Require AGENTS.md
        agents_md_path = entry / AGENTS_MD_FILENAME
        if not agents_md_path.is_file():
            msg = f"agents/{entry.name}/AGENTS.md not found — every subagent directory must contain an AGENTS.md"
            raise ValueError(msg)

        # Parse frontmatter
        fm_name, description, system_prompt = _parse_subagent_frontmatter(agents_md_path)

        # Load optional toml
        toml_path = entry / DEFAULT_CONFIG_FILENAME
        toml_data: dict[str, Any] = {}
        if toml_path.exists():
            with toml_path.open("rb") as f:
                toml_data = tomllib.load(f)

        # Build AgentConfig from toml (with frontmatter name as fallback)
        agent_data = toml_data.get("agent", {})
        toml_name = agent_data.get("name", fm_name)

        # Validate name consistency
        if toml_data and "agent" in toml_data and toml_name != fm_name:
            msg = (
                f"Subagent '{fm_name}' in agents/{entry.name}/ has mismatched "
                f"name in deepagents.toml: '{toml_name}'"
            )
            raise ValueError(msg)

        # Check reserved name
        if fm_name == "general-purpose":
            msg = "Subagent name 'general-purpose' is reserved"
            raise ValueError(msg)

        # Check for duplicates
        if fm_name in seen_names:
            msg = f"Duplicate subagent name: '{fm_name}'"
            raise ValueError(msg)
        seen_names.add(fm_name)

        agent_config = AgentConfig(
            name=fm_name,
            model=agent_data.get("model", "anthropic:claude-sonnet-4-6"),
        )

        # Sandbox: None means inherit from parent
        sandbox_data = toml_data.get("sandbox")
        sandbox_config: SandboxConfig | None = None
        if sandbox_data is not None:
            sandbox_config = SandboxConfig(
                provider=sandbox_data.get("provider", "none"),
                template=sandbox_data.get("template", "deepagents-deploy"),
                image=sandbox_data.get("image", "python:3"),
                scope=sandbox_data.get("scope", "thread"),
            )

        # Detect optional dirs/files
        skills_dir = entry / SKILLS_DIRNAME
        mcp_path = entry / MCP_FILENAME

        subagents.append(
            SubagentConfig(
                agent=agent_config,
                sandbox=sandbox_config,
                system_prompt=system_prompt,
                description=description,
                skills_dir=skills_dir if skills_dir.is_dir() else None,
                mcp_path=mcp_path if mcp_path.is_file() else None,
            )
        )

    return subagents


def generate_starter_config() -> str:
    """Generate a starter `deepagents.toml` template."""
    return """\
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

# [sandbox] is optional. Omit if not needed for skills or code execution.
# [sandbox]
# provider = "langsmith"   # langsmith | daytona | modal | runloop
# scope = "thread"         # thread | assistant
"""


def generate_starter_agents_md() -> str:
    """Generate a starter `AGENTS.md` template."""
    return """\
# Agent Instructions

You are a helpful AI agent.

## Guidelines

- Follow the user's instructions carefully.
- Ask for clarification when the request is ambiguous.
"""


def generate_starter_env() -> str:
    """Generate a starter `.env` template."""
    return """\
# Model provider API key (required)
ANTHROPIC_API_KEY=

# LangSmith API key (required for deploy and sandbox)
LANGSMITH_API_KEY=
"""


def generate_starter_mcp_json() -> str:
    """Generate a starter `mcp.json` template."""
    return """\
{
  "mcpServers": {}
}
"""


# Starter skill name and content.
STARTER_SKILL_NAME = "review"

# Starter subagent name and content.
STARTER_SUBAGENT_NAME = "researcher"


def generate_starter_skill_md() -> str:
    """Generate a starter `skills/review/SKILL.md` for code review."""
    return """\
---
name: review
description: >-
  Review code for bugs, security issues, and improvements.
  Use when the user asks to: (1) review code or a diff,
  (2) check code quality, (3) find bugs or issues,
  (4) audit for security problems.
  Trigger on phrases like 'review this', 'check my code',
  'any issues with this', 'code review'.
---

# Code Review

Review the provided code or diff with focus on:

1. **Correctness** — Logic errors, off-by-one bugs, unhandled edge cases
2. **Security** — Injection, auth issues, secrets in code, unsafe deserialization
3. **Performance** — Unnecessary allocations, N+1 queries, missing indexes
4. **Readability** — Unclear naming, overly complex logic, missing context

## Process

1. Read the code or diff carefully
2. Identify concrete issues (not style nitpicks)
3. For each issue: state what's wrong, why it matters, and suggest a fix
4. If the code looks good, say so — don't invent problems

## Output format

For each issue found:

- **File:line** — Brief description of the problem
  - Why it matters
  - Suggested fix

Keep feedback actionable. Skip praise for things that are simply correct.
"""


def generate_starter_subagent_agents_md() -> str:
    """Generate a starter subagent `AGENTS.md`."""
    return """\
---
name: researcher
description: Research topics on the web before writing content
---

You are a research assistant. Search for relevant information
and summarize your findings clearly and concisely.
"""


def generate_starter_subagent_config() -> str:
    """Generate a starter subagent `deepagents.toml`."""
    return """\
[agent]
name = "researcher"
model = "anthropic:claude-haiku-4-5-20251001"
"""
