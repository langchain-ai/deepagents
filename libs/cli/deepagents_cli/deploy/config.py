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

AuthProvider = Literal["supabase", "clerk", "anonymous"]
"""Valid auth provider identifiers.

`"anonymous"` ships a permissive auth handler that overrides LangSmith
Cloud's default `x-api-key` requirement so the bundled frontend can
reach `/threads`. The API is open to anyone with the deploy URL —
per-browser thread scoping is enforced client-side via a UUID cookie.
"""

VALID_AUTH_PROVIDERS: frozenset[str] = frozenset(get_args(AuthProvider))
"""Valid auth providers for deploy."""

DEFAULT_CONFIG_FILENAME = "deepagents.toml"

# Canonical filenames inside the project root.
AGENTS_MD_FILENAME = "AGENTS.md"
SKILLS_DIRNAME = "skills"
USER_DIRNAME = "user"
MCP_FILENAME = "mcp.json"
SUBAGENTS_DIRNAME = "subagents"


@dataclass(frozen=True)
class AgentConfig:
    """`[agent]` section — core agent identity."""

    name: str
    description: str = ""
    model: str = "anthropic:claude-sonnet-4-6"

    def __post_init__(self) -> None:  # noqa: D105 — simple guard, not a public API
        if not self.name.strip():
            msg = "AgentConfig.name must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class SubAgentConfig:
    """Parsed from a subagent's deepagents.toml."""

    agent: AgentConfig


@dataclass(frozen=True)
class SubAgentProject:
    """A discovered subagent directory with its parsed config."""

    config: SubAgentConfig
    root: Path


@dataclass(frozen=True)
class SandboxConfig:
    """`[sandbox]` section — sandbox provider settings.

    The whole section is optional. When omitted (or `provider = "none"`)
    the runtime falls back to an in-process `StateBackend` and tools
    like `execute` become no-ops.
    """

    provider: SandboxProvider = "none"
    """Sandbox backend identifier (`"none"` disables the sandbox)."""

    template: str = "deepagents-deploy"
    """LangSmith snapshot name the deployed graph boots from.

    The TOML key is kept as `template` for backward compatibility with
    existing `deepagents.toml` files — LangSmith's API renamed "template"
    to "snapshot" in 0.7.32, but this field name did not. The default
    `"deepagents-deploy"` is distinct from the interactive CLI default
    (`deepagents-cli`) so production deployments can be rebuilt
    independently of local-CLI snapshots.
    """

    image: str = "python:3"
    """Docker image used to build the snapshot when it does not yet exist."""

    scope: SandboxScope = "thread"
    """How sandbox cache keys are built.

    - `"thread"` (default): one sandbox per thread. Different threads get
        different sandboxes; the same thread reuses across turns.
    - `"assistant"`: one sandbox per assistant. All threads of the same
        assistant share a single sandbox and its filesystem.
    """


@dataclass(frozen=True)
class AuthConfig:
    """`[auth]` section — authentication provider settings.

    The whole section is optional. When omitted, the deployed agent has
    no authentication and all requests are anonymous.
    """

    provider: AuthProvider


@dataclass(frozen=True)
class FrontendConfig:
    """`[frontend]` section — bundled default frontend settings.

    When `enabled = True`, `deepagent deploy` copies a pre-built React
    chat UI into the deployment alongside the agent. `[auth]` is
    optional; omitting it ships in anonymous mode.
    """

    enabled: bool = False
    app_name: str | None = None
    subtitle: str | None = None
    """Subtitle shown under the app name in the header and on the
    empty-state hero. Falls back to a generic default when unset."""
    prompts: tuple[str, ...] | None = None
    """Suggestion chips shown on the empty-state. Falls back to the
    bundled defaults when unset."""


@dataclass(frozen=True)
class DeployConfig:
    """Top-level deploy configuration parsed from `deepagents.toml`."""

    agent: AgentConfig
    """Parsed `[agent]` section — core agent identity (name + model)."""

    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    """Parsed `[sandbox]` section — provider, snapshot name, image, scope."""
    auth: AuthConfig | None = None
    frontend: FrontendConfig | None = None

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

        # Validate credentials for auth provider.
        if self.auth is not None:
            errors.extend(_validate_auth_credentials(self.auth.provider))

        if self.frontend is not None and self.frontend.enabled:
            if self.auth is None:
                errors.append(
                    "[frontend].enabled requires [auth] to be configured. "
                    'Add an [auth] section with provider = "supabase", '
                    '"clerk", or "anonymous".'
                )
            else:
                errors.extend(_validate_frontend_credentials(self.auth.provider))

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


_ALLOWED_SUBAGENT_SECTIONS = frozenset({"agent"})


def _parse_subagent_config(data: dict[str, Any], subagent_dir: Path) -> SubAgentConfig:
    """Parse a subagent's deepagents.toml into a `SubAgentConfig`.

    Raises:
        ValueError: If the config has disallowed sections, missing required
            fields, or unknown keys.
    """
    # Reject disallowed sections.
    unknown = set(data.keys()) - _ALLOWED_SUBAGENT_SECTIONS
    if unknown:
        disallowed = sorted(unknown)
        msg = (
            f"Section(s) {disallowed} not allowed in subagent config "
            f"({subagent_dir}). Only {sorted(_ALLOWED_SUBAGENT_SECTIONS)} "
            f"is permitted."
        )
        raise ValueError(msg)

    agent_data = data.get("agent", {})

    # Reject unknown agent keys.
    unknown_agent = set(agent_data.keys()) - _ALLOWED_AGENT_KEYS
    if unknown_agent:
        msg = (
            f"Unknown key(s) in [agent]: {sorted(unknown_agent)}. "
            f"Allowed: {sorted(_ALLOWED_AGENT_KEYS)}"
        )
        raise ValueError(msg)

    # Require name.
    if "name" not in agent_data:
        msg = f"[agent].name is required in subagent deepagents.toml ({subagent_dir})"
        raise ValueError(msg)

    # Require description (non-empty).
    desc = agent_data.get("description", "")
    if not isinstance(desc, str) or not desc.strip():
        msg = (
            f"[agent].description is required (non-empty) in subagent "
            f"deepagents.toml ({subagent_dir})"
        )
        raise ValueError(msg)

    agent_kwargs: dict[str, Any] = {
        "name": agent_data["name"],
        "description": desc,
    }
    if "model" in agent_data:
        agent_kwargs["model"] = agent_data["model"]

    return SubAgentConfig(agent=AgentConfig(**agent_kwargs))


def load_subagents(project_root: Path) -> dict[str, SubAgentProject]:
    """Discover and load subagent projects from `subagents/`.

    Returns a dict keyed by subagent name. If the `subagents/` directory
    does not exist or is empty, returns an empty dict.

    Raises:
        ValueError: On any structural or config validation error.
    """
    subagents_dir = project_root / SUBAGENTS_DIRNAME
    if not subagents_dir.is_dir():
        return {}

    result: dict[str, SubAgentProject] = {}

    for entry in sorted(subagents_dir.iterdir()):
        # Skip dotfiles and non-directories.
        if entry.name.startswith(".") or not entry.is_dir():
            continue

        # Reject nested subagents/.
        if (entry / SUBAGENTS_DIRNAME).exists():
            msg = (
                f"Nested subagents/ not allowed inside subagent "
                f"directory '{entry.name}'"
            )
            raise ValueError(msg)

        # Require deepagents.toml.
        toml_path = entry / DEFAULT_CONFIG_FILENAME
        if not toml_path.is_file():
            msg = f"deepagents.toml is required in subagent directory '{entry.name}'"
            raise ValueError(msg)

        # Require AGENTS.md.
        agents_md = entry / AGENTS_MD_FILENAME
        if not agents_md.is_file():
            msg = f"AGENTS.md is required in subagent directory '{entry.name}'"
            raise ValueError(msg)

        # Parse the subagent config.
        try:
            with toml_path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Syntax error in {toml_path}: {exc}"
            raise ValueError(msg) from exc

        config = _parse_subagent_config(data, entry)

        # Validate MCP if present.
        mcp_path = entry / MCP_FILENAME
        if mcp_path.is_file():
            errors = _validate_mcp_for_deploy(mcp_path)
            if errors:
                msg = f"MCP validation errors in subagent '{entry.name}': " + "; ".join(
                    errors
                )
                raise ValueError(msg)

        result[config.agent.name] = SubAgentProject(config=config, root=entry)

    return result


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


_ALLOWED_SECTIONS = frozenset({"agent", "sandbox", "auth", "frontend"})
_ALLOWED_AGENT_KEYS = frozenset({"name", "description", "model"})
_ALLOWED_SANDBOX_KEYS = frozenset({"provider", "template", "image", "scope"})
_ALLOWED_AUTH_KEYS = frozenset({"provider"})
_ALLOWED_FRONTEND_KEYS = frozenset({"enabled", "app_name", "subtitle", "prompts"})


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
    if "description" in agent_data:
        agent_kwargs["description"] = agent_data["description"]
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

    auth: AuthConfig | None = None
    auth_data = data.get("auth")
    if auth_data is not None:
        unknown_auth = set(auth_data.keys()) - _ALLOWED_AUTH_KEYS
        if unknown_auth:
            msg = (
                f"Unknown key(s) in [auth]: {sorted(unknown_auth)}. "
                f"Allowed: {sorted(_ALLOWED_AUTH_KEYS)}"
            )
            raise ValueError(msg)

        if "provider" not in auth_data:
            msg = "[auth].provider is required in deepagents.toml"
            raise ValueError(msg)

        auth_provider = auth_data["provider"]
        if auth_provider not in VALID_AUTH_PROVIDERS:
            msg = (
                f"Unknown auth provider: {auth_provider}. "
                f"Valid: {', '.join(sorted(VALID_AUTH_PROVIDERS))}"
            )
            raise ValueError(msg)

        auth = AuthConfig(provider=auth_provider)

    frontend: FrontendConfig | None = None
    frontend_data = data.get("frontend")
    if frontend_data is not None:
        unknown_frontend = set(frontend_data.keys()) - _ALLOWED_FRONTEND_KEYS
        if unknown_frontend:
            msg = (
                f"Unknown key(s) in [frontend]: {sorted(unknown_frontend)}. "
                f"Allowed: {sorted(_ALLOWED_FRONTEND_KEYS)}"
            )
            raise ValueError(msg)

        frontend_kwargs: dict[str, Any] = {
            k: frontend_data[k] for k in _ALLOWED_FRONTEND_KEYS if k in frontend_data
        }
        # FrontendConfig is frozen=True; coerce list -> tuple so the
        # dataclass stays hashable.
        if "prompts" in frontend_kwargs:
            prompts_raw = frontend_kwargs["prompts"]
            if not isinstance(prompts_raw, list) or not all(
                isinstance(p, str) for p in prompts_raw
            ):
                msg = "[frontend].prompts must be a list of strings"
                raise ValueError(msg)
            frontend_kwargs["prompts"] = tuple(prompts_raw)
        frontend = FrontendConfig(**frontend_kwargs)

    return DeployConfig(agent=agent, sandbox=sandbox, auth=auth, frontend=frontend)


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

_AUTH_PROVIDER_ENV: dict[str, list[str]] = {
    "supabase": ["SUPABASE_URL", "SUPABASE_PUBLISHABLE_DEFAULT_KEY"],
    "clerk": ["CLERK_SECRET_KEY"],
}

_FRONTEND_EXTRA_ENV: dict[str, list[str]] = {
    # Supabase reuses `SUPABASE_URL` + `SUPABASE_PUBLISHABLE_DEFAULT_KEY`
    # from [auth] — no extra browser-facing env vars needed.
    "supabase": [],
    # Clerk's browser-facing publishable key is distinct from
    # `CLERK_SECRET_KEY` (which [auth] uses for JWKS validation).
    "clerk": ["CLERK_PUBLISHABLE_KEY"],
}
"""Additional env vars the frontend bundle needs beyond what `[auth]` requires."""


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


def _validate_auth_credentials(provider: str) -> list[str]:
    """Check that all required env vars are set for the auth provider."""
    required_vars = _AUTH_PROVIDER_ENV.get(provider)
    if required_vars is None:
        return []
    missing = [v for v in required_vars if not os.environ.get(v)]
    if not missing:
        return []
    return [
        (
            f"Auth provider '{provider}' requires {' and '.join(missing)}. "
            f"Add them to your .env file or environment."
        ),
    ]


def _validate_frontend_credentials(provider: str) -> list[str]:
    """Check that all extra env vars are set for the frontend bundle."""
    required = _FRONTEND_EXTRA_ENV.get(provider)
    if required is None:
        return []
    missing = [v for v in required if not os.environ.get(v)]
    if not missing:
        return []
    return [
        (
            f"Frontend for '{provider}' requires {' and '.join(missing)}. "
            f"Add it to your .env file so the bundler can write it "
            f"into index.html at deploy time."
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

# [auth] is optional. Add to enable user authentication.
# [auth]
# provider = "supabase"   # supabase | clerk

# [frontend] is optional. Add to ship a bundled chat UI on the same
# deployment as the agent. Pair with [auth] for authenticated deploys; see below.
# [frontend]
# enabled = true
# app_name = "My Agent"

# [frontend] without [auth] runs the UI in anonymous mode — anyone
# with the deploy URL can use the agent. Use only for private/dev
# deploys; add [auth] for any user-facing deployment.
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

# Auth provider (optional, uncomment for [auth])
# SUPABASE_URL=
# SUPABASE_PUBLISHABLE_DEFAULT_KEY=
# CLERK_SECRET_KEY=

# Frontend (optional, uncomment for [frontend] + matching [auth])
# Clerk only — browser-facing publishable key. Supabase reuses the keys above.
# CLERK_PUBLISHABLE_KEY=
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
