"""Typed configuration for the CLI-to-server subprocess communication channel.

The CLI spawns a `langgraph dev` subprocess and passes configuration via
environment variables prefixed with `DEEPAGENTS_CLI_SERVER_`. This module
provides a single
`ServerConfig` dataclass that both sides share so that the set of variables,
their serialization format, and their default values are defined in one place.
The CLI writes config with `to_env()` and the server graph reads it back
with `from_env()`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_cli._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from deepagents_cli.project_utils import ProjectContext

logger = logging.getLogger(__name__)

_DEFAULT_ASSISTANT_ID = "agent"


def _read_env_bool(suffix: str, *, default: bool = False) -> bool:
    """Read a `DEEPAGENTS_CLI_SERVER_*` boolean from the environment.

    Boolean env vars use the `'true'` / `'false'` convention (case insensitive).
    Missing variables fall back to *default*.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CLI_SERVER_` prefix.
        default: Value when the variable is absent.

    Returns:
        Parsed boolean.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    return raw.lower() == "true"


def _read_env_json(suffix: str) -> Any:  # noqa: ANN401
    """Read a JSON-encoded `DEEPAGENTS_CLI_SERVER_*` variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CLI_SERVER_` prefix.

    Returns:
        Parsed JSON value, or `None` if the variable is absent.

    Raises:
        ValueError: If the variable is present but not valid JSON.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = (
            f"Failed to parse {SERVER_ENV_PREFIX}{suffix} as JSON: {exc}. "
            f"Value was: {raw[:200]!r}"
        )
        raise ValueError(msg) from exc


def _read_env_str(suffix: str) -> str | None:
    """Read an optional `DEEPAGENTS_CLI_SERVER_*` string variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CLI_SERVER_` prefix.

    Returns:
        The string value, or `None` if absent.
    """
    return os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")


def _read_env_optional_bool(suffix: str) -> bool | None:
    """Read a tri-state `DEEPAGENTS_CLI_SERVER_*` boolean (`True` / `False` / `None`).

    Used for settings where `None` carries a distinct meaning (e.g. "not
    specified, use default logic").

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CLI_SERVER_` prefix.

    Returns:
        `True`, `False`, or `None` when the variable is absent.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    return raw.lower() == "true"


@dataclass(frozen=True)
class ServerConfig:
    """Full configuration payload passed from the CLI to the server subprocess.

    Serialized to/from `DEEPAGENTS_CLI_SERVER_*` environment variables so
    that the server
    graph (which runs in a separate Python interpreter) can reconstruct the
    CLI's intent without sharing memory.
    """

    model: str | None = None
    model_params: dict[str, Any] | None = None
    assistant_id: str = _DEFAULT_ASSISTANT_ID
    system_prompt: str | None = None
    auto_approve: bool = False
    interrupt_shell_only: bool = False
    shell_allow_list: list[str] | None = None
    interactive: bool = True
    enable_shell: bool = True
    enable_ask_user: bool = False
    enable_memory: bool = True
    enable_skills: bool = True
    sandbox_type: str | None = None
    sandbox_id: str | None = None
    sandbox_setup: str | None = None
    cwd: str | None = None
    project_root: str | None = None
    mcp_config_path: str | None = None
    no_mcp: bool = False
    trust_project_mcp: bool | None = None

    def __post_init__(self) -> None:
        """Normalize fields that have canonical representations."""
        if self.sandbox_type == "none":
            object.__setattr__(self, "sandbox_type", None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_env(self) -> dict[str, str | None]:
        """Serialize this config to a `DEEPAGENTS_CLI_SERVER_*` env-var mapping.

        `None` values signal that the variable should be *cleared* from the
        environment (rather than set to an empty string), so callers can
        iterate and set or clear each variable in `os.environ`.

        Returns:
            Dict mapping env-var suffixes (without the prefix) to their
                string values or `None`.
        """
        return {
            "MODEL": self.model,
            "MODEL_PARAMS": (
                json.dumps(self.model_params) if self.model_params is not None else None
            ),
            "ASSISTANT_ID": self.assistant_id,
            "SYSTEM_PROMPT": self.system_prompt,
            "AUTO_APPROVE": str(self.auto_approve).lower(),
            "INTERRUPT_SHELL_ONLY": str(self.interrupt_shell_only).lower(),
            "SHELL_ALLOW_LIST": (
                ",".join(self.shell_allow_list)
                if self.shell_allow_list is not None
                else None
            ),
            "INTERACTIVE": str(self.interactive).lower(),
            "ENABLE_SHELL": str(self.enable_shell).lower(),
            "ENABLE_ASK_USER": str(self.enable_ask_user).lower(),
            "ENABLE_MEMORY": str(self.enable_memory).lower(),
            "ENABLE_SKILLS": str(self.enable_skills).lower(),
            "SANDBOX_TYPE": self.sandbox_type,
            "SANDBOX_ID": self.sandbox_id,
            "SANDBOX_SETUP": self.sandbox_setup,
            "CWD": self.cwd,
            "PROJECT_ROOT": self.project_root,
            "MCP_CONFIG_PATH": self.mcp_config_path,
            "NO_MCP": str(self.no_mcp).lower(),
            "TRUST_PROJECT_MCP": (
                str(self.trust_project_mcp).lower()
                if self.trust_project_mcp is not None
                else None
            ),
        }

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Reconstruct a `ServerConfig` from `DEEPAGENTS_CLI_SERVER_*` env vars.

        This is the inverse of `to_env()` and is called inside the server
        subprocess to recover the CLI's configuration.

        Returns:
            A `ServerConfig` populated from the environment.
        """
        return cls(
            model=_read_env_str("MODEL"),
            model_params=_read_env_json("MODEL_PARAMS"),
            assistant_id=_read_env_str("ASSISTANT_ID") or _DEFAULT_ASSISTANT_ID,
            system_prompt=_read_env_str("SYSTEM_PROMPT"),
            auto_approve=_read_env_bool("AUTO_APPROVE"),
            interrupt_shell_only=_read_env_bool("INTERRUPT_SHELL_ONLY"),
            shell_allow_list=(
                raw.split(",") if (raw := _read_env_str("SHELL_ALLOW_LIST")) else None
            ),
            interactive=_read_env_bool("INTERACTIVE", default=True),
            enable_shell=_read_env_bool("ENABLE_SHELL", default=True),
            enable_ask_user=_read_env_bool("ENABLE_ASK_USER"),
            enable_memory=_read_env_bool("ENABLE_MEMORY", default=True),
            enable_skills=_read_env_bool("ENABLE_SKILLS", default=True),
            sandbox_type=_read_env_str("SANDBOX_TYPE"),
            sandbox_id=_read_env_str("SANDBOX_ID"),
            sandbox_setup=_read_env_str("SANDBOX_SETUP"),
            cwd=_read_env_str("CWD"),
            project_root=_read_env_str("PROJECT_ROOT"),
            mcp_config_path=_read_env_str("MCP_CONFIG_PATH"),
            no_mcp=_read_env_bool("NO_MCP"),
            trust_project_mcp=_read_env_optional_bool("TRUST_PROJECT_MCP"),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_cli_args(
        cls,
        *,
        project_context: ProjectContext | None,
        model_name: str | None,
        model_params: dict[str, Any] | None,
        assistant_id: str,
        auto_approve: bool,
        interrupt_shell_only: bool = False,
        shell_allow_list: list[str] | None = None,
        sandbox_type: str = "none",
        sandbox_id: str | None,
        sandbox_setup: str | None,
        enable_shell: bool,
        enable_ask_user: bool,
        mcp_config_path: str | None,
        no_mcp: bool,
        trust_project_mcp: bool | None,
        interactive: bool,
    ) -> ServerConfig:
        """Build a `ServerConfig` from parsed CLI arguments.

        Handles path normalization (e.g. resolving relative MCP config paths
        against the user's working directory) so that the raw serialized values
        are always absolute and unambiguous.

        Args:
            project_context: Explicit user/project path context.
            model_name: Model spec string.
            model_params: Extra model kwargs.
            assistant_id: Agent identifier.
            auto_approve: Auto-approve all tools.
            interrupt_shell_only: Validate shell commands via middleware instead
                of HITL.
            shell_allow_list: Restrictive shell allow-list to forward to the
                server subprocess for `ShellAllowListMiddleware`.
            sandbox_type: Sandbox type.
            sandbox_id: Existing sandbox ID to reuse.
            sandbox_setup: Path to setup script for the sandbox.
            enable_shell: Enable shell execution tools.
            enable_ask_user: Enable ask_user tool.
            mcp_config_path: Path to MCP config.
            no_mcp: Disable MCP.
            trust_project_mcp: Trust project MCP servers.
            interactive: Whether the agent is interactive.

        Returns:
            A fully resolved `ServerConfig`.
        """
        normalized_mcp = _normalize_path(mcp_config_path, project_context, "MCP config")

        return cls(
            model=model_name,
            model_params=model_params,
            assistant_id=assistant_id,
            auto_approve=auto_approve,
            interrupt_shell_only=interrupt_shell_only,
            shell_allow_list=shell_allow_list,
            interactive=interactive,
            enable_shell=enable_shell,
            enable_ask_user=enable_ask_user,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_setup=_normalize_path(
                sandbox_setup, project_context, "sandbox setup"
            ),
            cwd=(
                str(project_context.user_cwd) if project_context is not None else None
            ),
            project_root=(
                str(project_context.project_root)
                if project_context is not None
                and project_context.project_root is not None
                else None
            ),
            mcp_config_path=normalized_mcp,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
        )


def _normalize_path(
    raw_path: str | None,
    project_context: ProjectContext | None,
    label: str,
) -> str | None:
    """Resolve a possibly-relative path to absolute.

    The server subprocess runs in a different working directory, so relative
    paths must be resolved against the user's original cwd before serialization.

    Args:
        raw_path: Path from CLI arguments (may be relative).
        project_context: User/project context for path resolution.
        label: Human-readable label for error messages (e.g. "MCP config").

    Returns:
        Absolute path string, or `None` when *raw_path* is `None` or empty.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    if not raw_path:
        return None
    try:
        if project_context is not None:
            return str(project_context.resolve_user_path(raw_path))
        return str(Path(raw_path).expanduser().resolve())
    except OSError as exc:
        msg = (
            f"Could not resolve {label} path {raw_path!r}: {exc}. "
            "Ensure the path exists and is accessible."
        )
        raise ValueError(msg) from exc
