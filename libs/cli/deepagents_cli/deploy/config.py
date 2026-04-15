"""Deploy configuration parsing and validation.

Reads `deepagents.toml` and produces a validated `DeployConfig`.

The new minimal surface has exactly two sections:

- `[agent]`: name + model
- `[sandbox]`: sandbox provider settings

`AGENTS.md` is always seeded into a shared memory namespace so the agent can
read it at runtime, but writes/edits to that path are blocked by a read-only
middleware in the generated graph.

Skills (`skills/`) and MCP servers (`mcp.json`) are auto-detected from the
project layout. The agent's system prompt is read from `AGENTS.md` at bundle
time — there is no `system_prompt` key.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

SandboxProvider = Literal["none", "daytona", "langsmith", "modal", "runloop"]
"""Valid sandbox provider identifiers."""

SandboxScope = Literal["thread", "assistant"]
"""Valid sandbox scope values."""

VALID_SANDBOX_PROVIDERS: frozenset[str] = frozenset(get_args(SandboxProvider))
"""Valid sandbox providers for deploy (subset of sandbox_factory, plus `"none"`)."""

VALID_SANDBOX_SCOPES: frozenset[str] = frozenset(get_args(SandboxScope))

DEFAULT_CONFIG_FILENAME = "deepagents.toml"

# Canonical filenames inside the project root.
AGENTS_MD_FILENAME = "AGENTS.md"
SKILLS_DIRNAME = "skills"
USER_DIRNAME = "user"
MCP_FILENAME = "mcp.json"


@dataclass(frozen=True)
class AgentConfig:
    """`[agent]` section — core agent identity."""

    name: str
    model: str = "anthropic:claude-sonnet-4-6"

    def __post_init__(self) -> None:  # noqa: D105 — simple guard, not a public API
        if not self.name.strip():
            msg = "AgentConfig.name must be non-empty"
            raise ValueError(msg)


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

    provider: SandboxProvider = "none"
    template: str = "deepagents-deploy"
    image: str = "python:3"
    scope: SandboxScope = "thread"


@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from `deepagents.toml`."""

    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    def validate(self, project_root: Path) -> list[str]:
        """Validate config against the filesystem.

        Args:
            project_root: Directory containing `deepagents.toml`.

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

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Syntax error in {config_path}: {exc}"
        raise ValueError(msg) from exc

    return _parse_config(data)


_ALLOWED_SECTIONS = frozenset({"agent", "sandbox"})
_ALLOWED_AGENT_KEYS = frozenset({"name", "model"})
_ALLOWED_SANDBOX_KEYS = frozenset({"provider", "template", "image", "scope"})


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

    unknown_agent = set(agent_data.keys()) - _ALLOWED_AGENT_KEYS
    if unknown_agent:
        msg = (
            f"Unknown key(s) in [agent]: {sorted(unknown_agent)}. "
            f"Allowed: {sorted(_ALLOWED_AGENT_KEYS)}"
        )
        raise ValueError(msg)

    # Only pass keys present in TOML; dataclass defaults handle the rest.
    agent_kwargs: dict[str, Any] = {"name": agent_data["name"]}
    if "model" in agent_data:
        agent_kwargs["model"] = agent_data["model"]
    agent = AgentConfig(**agent_kwargs)

    sandbox_data = data.get("sandbox", {})
    unknown_sandbox = set(sandbox_data.keys()) - _ALLOWED_SANDBOX_KEYS
    if unknown_sandbox:
        msg = (
            f"Unknown key(s) in [sandbox]: {sorted(unknown_sandbox)}. "
            f"Allowed: {sorted(_ALLOWED_SANDBOX_KEYS)}"
        )
        raise ValueError(msg)

    sandbox_kwargs: dict[str, Any] = {
        k: sandbox_data[k] for k in _ALLOWED_SANDBOX_KEYS if k in sandbox_data
    }
    sandbox = SandboxConfig(**sandbox_kwargs)

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
    """Check that at least one required API key env var is set for the provider."""
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
    """Find `deepagents.toml` in *start_path* (or cwd if not given).

    Only checks the single directory — does not walk parent directories.

    Returns the path if found, or `None` otherwise.
    """
    current = (start_path or Path.cwd()).resolve()
    candidate = current / DEFAULT_CONFIG_FILENAME
    if candidate.is_file():
        return candidate
    return None


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
