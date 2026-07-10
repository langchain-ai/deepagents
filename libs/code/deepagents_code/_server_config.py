"""Typed configuration for the app-to-server subprocess communication channel.

The app spawns a `langgraph dev` subprocess and passes configuration via
environment variables prefixed with `DEEPAGENTS_CODE_SERVER_`. This module
provides a single
`ServerConfig` dataclass that both sides share so that the set of variables,
their serialization format, and their default values are defined in one place.
The app writes config with `to_env()` and the server graph reads it back
with `from_env()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_code._constants import DEFAULT_AGENT_NAME as DEFAULT_ASSISTANT_ID
from deepagents_code._env_vars import SERVER_ENV_PREFIX

if TYPE_CHECKING:
    from deepagents_code.project_utils import ProjectContext


def _read_env_bool(suffix: str, *, default: bool = False) -> bool:
    """Read a `DEEPAGENTS_CODE_SERVER_*` boolean from the environment.

    Boolean env vars use the `'true'` / `'false'` convention (case insensitive).
    Missing variables fall back to *default*.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.
        default: Value when the variable is absent.

    Returns:
        Parsed boolean.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    return raw.lower() == "true"


def _read_env_json(suffix: str) -> Any:  # noqa: ANN401
    """Read a JSON-encoded `DEEPAGENTS_CODE_SERVER_*` variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

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
    """Read an optional `DEEPAGENTS_CODE_SERVER_*` string variable.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

    Returns:
        The string value, or `None` if absent.
    """
    return os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")


def _read_env_int(suffix: str, *, default: int | None) -> int | None:
    """Read a `DEEPAGENTS_CODE_SERVER_*` integer from the environment.

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.
        default: Value when the variable is absent or malformed.

    Returns:
        Parsed integer, or the default when absent or parsing fails.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_env_optional_bool(suffix: str) -> bool | None:
    """Read a tri-state `DEEPAGENTS_CODE_SERVER_*` boolean (`True` / `False` / `None`).

    Used for settings where `None` carries a distinct meaning (e.g. "not
    specified, use default logic").

    Args:
        suffix: Variable name suffix after the `DEEPAGENTS_CODE_SERVER_` prefix.

    Returns:
        `True`, `False`, or `None` when the variable is absent.
    """
    raw = os.environ.get(f"{SERVER_ENV_PREFIX}{suffix}")
    if raw is None:
        return None
    return raw.lower() == "true"


def _resolve_enable_interpreter(
    enable_interpreter: bool | None, sandbox_type: str | None
) -> bool:
    """Resolve the interpreter's tri-state caller option to a concrete boolean.

    Args:
        enable_interpreter: Explicit caller preference, or `None` to use the
            sandbox-aware default.
        sandbox_type: Sandbox backend identifier. Any falsy value (`None`, `""`)
            or `"none"` is treated as local execution.

    Returns:
        The explicit `enable_interpreter` value when not `None`; `False` for
            remote-sandbox defaults; otherwise the configured local default
            (`settings.enable_interpreter`).
    """
    if enable_interpreter is not None:
        return enable_interpreter
    if sandbox_type and sandbox_type != "none":
        return False

    from deepagents_code.config import settings

    return settings.enable_interpreter


def _interpreter_suppressed_by_sandbox(
    *, enable_interpreter: bool | None, sandbox_type: str | None, local_default: bool
) -> bool:
    """Whether a remote sandbox suppressed the otherwise-default interpreter.

    Used to decide whether to surface an advisory: returns `True` only when the
    user made no explicit choice, a remote sandbox is active, and the local
    default would have enabled it — i.e. the sandbox (not an explicit
    `--no-interpreter` opt-out, nor a disabled `[interpreter]` config) is why
    `js_eval` is unavailable.

    Takes the *raw* tri-state caller intent rather than the resolved boolean: a
    sandbox-suppressed default and an explicit `--no-interpreter` both resolve to
    `False`, so the resolved value cannot distinguish them. Any explicit choice
    (`not None`) is the user's own decision and is left unannounced.

    Args:
        enable_interpreter: The raw tri-state caller intent (`--interpreter` →
            `True`, `--no-interpreter` → `False`, unset → `None`).
        sandbox_type: Sandbox backend identifier. Any falsy value (`None`, `""`)
            or `"none"` is treated as local execution.
        local_default: The local-mode default (`settings.enable_interpreter`);
            gating on it keeps the advisory quiet for users who disabled the
            interpreter in config.

    Returns:
        `True` when the advisory should be shown, otherwise `False`.
    """
    if enable_interpreter is not None:
        return False
    if not (sandbox_type and sandbox_type != "none"):
        return False
    return local_default


@dataclass(frozen=True)
class ServerConfig:
    """Full configuration payload passed from the app to the server subprocess.

    Serialized to/from `DEEPAGENTS_CODE_SERVER_*` environment variables so
    that the server graph (which runs in a separate Python interpreter)
    can reconstruct the app's intent without sharing memory.
    """

    model: str | None = None
    """Model spec string (e.g. `'anthropic:claude-opus-4-7'`); `None` lets the
    server pick its default."""

    model_params: dict[str, Any] | None = None
    """Extra kwargs forwarded to the chat model constructor (temperature,
    max_tokens, etc.)."""

    assistant_id: str = DEFAULT_ASSISTANT_ID
    """Identifier of the agent graph to invoke on the server."""

    system_prompt: str | None = None
    """Override for the agent's system prompt; `None` uses the agent's default."""

    auto_approve: bool = False
    """Auto-approve every tool call without human-in-the-loop interrupts."""

    interrupt_shell_only: bool = False
    """Route only shell tool calls through HITL; validate others via middleware."""

    shell_allow_list: list[str] | None = None
    """Restrictive allow-list of shell commands; `None` disables the allow-list.

    Must be non-empty when set.
    """

    interactive: bool = True
    """Whether the agent runs in an interactive session (vs.
    one-shot/non-interactive)."""

    enable_shell: bool = True
    """Enable the shell execution tool on the server."""

    enable_ask_user: bool = False
    """Enable the `ask_user` tool that lets the agent prompt the user mid-run."""

    enable_memory: bool = True
    """Enable the long-term memory subsystem."""

    enable_skills: bool = True
    """Enable the skills subsystem (SKILL.md loading and skill tools)."""

    enable_interpreter: bool = False
    """Enable `CodeInterpreterMiddleware` (`js_eval` REPL) on the main agent.

    Always the resolved concrete value: `from_cli_args` collapses the tri-state
    caller option via `_resolve_enable_interpreter` before constructing the
    config, so the `bool | None` "defer to default" sentinel never reaches this
    field. The `False` default here is only the bare-constructor/`from_env`
    fallback; the user-facing default (on in local mode) lives in
    `settings.enable_interpreter`.

    Local-mode only; the server graph raises if a sandbox is configured and
    this flag is `True`.
    """

    interpreter_ptc: str | list[str] | None = None
    """Override for `settings.interpreter_ptc`.

    `None` means "fall through to whatever `settings.interpreter_ptc` resolves
    to from `~/.deepagents/config.toml`". A string is one of `"safe"`/`"all"`;
    a list is an explicit allowlist of tool names that may also include the
    `"safe"` preset (expanded at agent-build time); `"all"` is rejected inside
    a list.
    """

    interpreter_ptc_acknowledge_unsafe: bool = False
    """Mirror of `settings.interpreter_ptc_acknowledge_unsafe` — required when
    `interpreter_ptc="all"` is paired with non-`auto_approve` mode.
    """

    allow_fs_tools: str | list[str] | None = None
    """Allowlist for `FilesystemMiddleware`'s `tools` param, from
    `--allow-fs-tools`.

    `None` leaves the SDK default (all filesystem tools). A string is
    `"all"`; a list is an explicit allowlist of filesystem tool names and
    must include `"read_file"`.
    """

    rubric_model: str | None = None
    """Grader model spec for `RubricMiddleware` (e.g. `'anthropic:...'`).

    `None` reuses the main agent model.
    """

    rubric_max_iterations: int | None = None
    """Explicit grader iterations per rubric attempt; `None` uses the SDK default."""

    sandbox_type: str | None = None
    """Sandbox backend identifier (e.g. `'daytona'`); `None` runs tools on the
    host. `'none'` is normalized to `None` in `__post_init__`."""

    sandbox_id: str | None = None
    """Existing sandbox ID to attach to; `None` creates a fresh sandbox."""

    sandbox_snapshot_name: str | None = None
    """Sandbox snapshot (langsmith) or blueprint (runloop) name; must be `None`
    when `sandbox_id` is set."""

    sandbox_setup: str | None = None
    """Absolute path to a setup script executed inside the sandbox on first attach."""

    cwd: str | None = None
    """User's original working directory, serialized as an absolute path."""

    project_root: str | None = None
    """Detected project root (e.g. nearest git/uv/npm boundary), or `None` when
    outside a project."""

    mcp_config_path: str | None = None
    """Absolute path to the MCP server config file; `None` disables
    MCP-from-config."""

    no_mcp: bool = False
    """Disable all MCP server connections regardless of other config."""

    trust_project_mcp: bool | None = None
    """Tri-state trust flag for project-scoped MCP servers: `True`/`False`/`None`
    (prompt user)."""

    def __post_init__(self) -> None:
        """Normalize fields and validate invariants.

        Raises:
            TypeError: If `rubric_max_iterations` is a boolean.
            ValueError: If `shell_allow_list` is an empty list or
                `rubric_max_iterations` is non-positive.
        """
        if self.sandbox_type == "none":
            object.__setattr__(self, "sandbox_type", None)
        if self.shell_allow_list is not None and len(self.shell_allow_list) == 0:
            msg = "shell_allow_list must be None or non-empty"
            raise ValueError(msg)
        if isinstance(self.rubric_max_iterations, bool):
            msg = "rubric_max_iterations must be None or a positive integer"
            raise TypeError(msg)
        if self.rubric_max_iterations is not None and self.rubric_max_iterations <= 0:
            msg = "rubric_max_iterations must be None or a positive integer"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_env(self) -> dict[str, str | None]:
        """Serialize this config to a `DEEPAGENTS_CODE_SERVER_*` env-var mapping.

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
            "ENABLE_INTERPRETER": str(self.enable_interpreter).lower(),
            "INTERPRETER_PTC": (
                json.dumps(self.interpreter_ptc)
                if self.interpreter_ptc is not None
                else None
            ),
            "INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE": str(
                self.interpreter_ptc_acknowledge_unsafe
            ).lower(),
            "ALLOW_FS_TOOLS": (
                json.dumps(self.allow_fs_tools)
                if self.allow_fs_tools is not None
                else None
            ),
            "RUBRIC_MODEL": self.rubric_model,
            "RUBRIC_MAX_ITERATIONS": (
                str(self.rubric_max_iterations)
                if self.rubric_max_iterations is not None
                else None
            ),
            "SANDBOX_TYPE": self.sandbox_type,
            "SANDBOX_ID": self.sandbox_id,
            "SANDBOX_SNAPSHOT_NAME": self.sandbox_snapshot_name,
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
        """Reconstruct a `ServerConfig` from `DEEPAGENTS_CODE_SERVER_*` env vars.

        This is the inverse of `to_env()` and is called inside the server
        subprocess to recover the app's configuration.

        Returns:
            A `ServerConfig` populated from the environment.
        """
        return cls(
            model=_read_env_str("MODEL"),
            model_params=_read_env_json("MODEL_PARAMS"),
            assistant_id=_read_env_str("ASSISTANT_ID") or DEFAULT_ASSISTANT_ID,
            system_prompt=_read_env_str("SYSTEM_PROMPT"),
            auto_approve=_read_env_bool("AUTO_APPROVE"),
            interrupt_shell_only=_read_env_bool("INTERRUPT_SHELL_ONLY"),
            shell_allow_list=(
                [cmd.strip() for cmd in raw.split(",") if cmd.strip()]
                if (raw := _read_env_str("SHELL_ALLOW_LIST"))
                else None
            )
            or None,
            interactive=_read_env_bool("INTERACTIVE", default=True),
            enable_shell=_read_env_bool("ENABLE_SHELL", default=True),
            enable_ask_user=_read_env_bool("ENABLE_ASK_USER"),
            enable_memory=_read_env_bool("ENABLE_MEMORY", default=True),
            enable_skills=_read_env_bool("ENABLE_SKILLS", default=True),
            enable_interpreter=_read_env_bool("ENABLE_INTERPRETER"),
            interpreter_ptc=_read_env_json("INTERPRETER_PTC"),
            interpreter_ptc_acknowledge_unsafe=_read_env_bool(
                "INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE"
            ),
            allow_fs_tools=_read_env_json("ALLOW_FS_TOOLS"),
            rubric_model=_read_env_str("RUBRIC_MODEL") or None,
            rubric_max_iterations=_read_env_int("RUBRIC_MAX_ITERATIONS", default=None),
            sandbox_type=_read_env_str("SANDBOX_TYPE"),
            sandbox_id=_read_env_str("SANDBOX_ID"),
            sandbox_snapshot_name=_read_env_str("SANDBOX_SNAPSHOT_NAME") or None,
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
        sandbox_snapshot_name: str | None,
        sandbox_setup: str | None,
        enable_shell: bool,
        enable_ask_user: bool,
        enable_interpreter: bool | None = None,
        interpreter_ptc: str | list[str] | None = None,
        interpreter_ptc_acknowledge_unsafe: bool = False,
        allow_fs_tools: str | list[str] | None = None,
        rubric_model: str | None = None,
        rubric_max_iterations: int | None = None,
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
            sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop)
                name to use or create.
            sandbox_setup: Path to setup script for the sandbox.
            enable_shell: Enable shell execution tools.
            enable_ask_user: Enable ask_user tool.
            enable_interpreter: Enable `CodeInterpreterMiddleware` on the main
                agent. `None` uses the sandbox-aware default.
            interpreter_ptc: Override for `settings.interpreter_ptc`.
            interpreter_ptc_acknowledge_unsafe: Mirror of
                `settings.interpreter_ptc_acknowledge_unsafe`.
            allow_fs_tools: Allowlist for `FilesystemMiddleware`'s `tools`
                param to forward to the server subprocess. `None` leaves the
                SDK default (all tools).
            rubric_model: Grader model spec; `None` reuses the main model.
            rubric_max_iterations: Explicit grader iterations per rubric attempt;
                `None` uses the SDK default.
            mcp_config_path: Path to MCP config.
            no_mcp: Disable MCP.
            trust_project_mcp: Trust project MCP servers.
            interactive: Whether the agent is interactive.

        Returns:
            A fully resolved `ServerConfig`.
        """
        normalized_mcp = _normalize_path(mcp_config_path, project_context, "MCP config")

        resolved_enable_interpreter = _resolve_enable_interpreter(
            enable_interpreter, sandbox_type
        )

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
            enable_interpreter=resolved_enable_interpreter,
            interpreter_ptc=interpreter_ptc,
            interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
            allow_fs_tools=allow_fs_tools,
            rubric_model=rubric_model,
            rubric_max_iterations=rubric_max_iterations,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_snapshot_name=sandbox_snapshot_name,
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
