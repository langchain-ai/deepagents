"""`RestrictedShellBackend`: Filesystem backend with restricted local shell execution."""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class RestrictedShellBackend(LocalShellBackend):
    """Filesystem backend with restricted local shell command execution.

    This backend extends `LocalShellBackend` to add safety guardrails:
    - Whitelist of allowed base commands.
    - Blocking of shell metacharacters (;, |, &, etc.) to prevent command injection.

    !!! warning "Security"

        While more secure than `LocalShellBackend`, this still runs on the host system.
        For untrusted workloads, use a containerized or sandboxed backend.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        allowed_commands: list[str] | None = None,
        allow_metacharacters: bool = False,
        virtual_mode: bool | None = None,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize restricted shell backend.

        Args:
            root_dir: Working directory.
            allowed_commands: List of allowed base commands (e.g., ["ls", "git"]).
            allow_metacharacters: If True, allows shell characters like |, ;, &.
            virtual_mode: Enable virtual path mode.
            timeout: Command timeout.
            max_output_bytes: Output limit.
            env: Environment variables.
            inherit_env: Inherit parent environment.
        """
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=env,
            inherit_env=inherit_env,
        )
        self._allowed_commands = set(allowed_commands) if allowed_commands is not None else set()
        self._allow_metacharacters = allow_metacharacters

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a restricted shell command.

        Validates the command against the whitelist and checks for metacharacters.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        # 1. Check for shell metacharacters if disallowed
        if not self._allow_metacharacters:
            meta = {";", "|", "&", ">", "<", "`", "$", "(", ")"}
            if any(c in command for c in meta):
                return ExecuteResponse(
                    output=f"Security Error: Shell metacharacters are not allowed in command: {command}",
                    exit_code=1,
                    truncated=False,
                )

        # 2. Extract base command and check whitelist
        try:
            parts = shlex.split(command)
            if not parts:
                return ExecuteResponse(
                    output="Error: Empty command after parsing.",
                    exit_code=1,
                    truncated=False,
                )

            base_cmd = parts[0]
            if base_cmd not in self._allowed_commands:
                return ExecuteResponse(
                    output=f"Security Error: Command '{base_cmd}' is not in the allowed whitelist.",
                    exit_code=1,
                    truncated=False,
                )
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error parsing command: {e}",
                exit_code=1,
                truncated=False,
            )

        return super().execute(command, timeout=timeout)
