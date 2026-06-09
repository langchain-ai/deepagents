"""`SandboxedShellBackend`: a Landlock-confined `LocalShellBackend`.

`LocalShellBackend` runs ``execute()`` as a raw host subprocess with no
filesystem confinement -- ``virtual_mode`` only scopes the *file* tools, not the
shell (see that backend's own security note). This subclass keeps every
file-tool behavior identical and overrides only ``execute()`` so each shell
command runs under a Linux Landlock exec wrapper (``_landlock_exec.py``),
confined to the backend's ``root_dir`` plus a read-only system allowlist.

The wrapper is a separate process that restricts *itself* and then ``execve``s
the command, so the backend process is never restricted and we avoid the
fork-safety hazards of ``preexec_fn`` in a threaded async server.

Filesystem boundary only -- network is not restricted (the Landlock network ABI
is not present on most deployed kernels, and many CLIs need outbound access) and
this is Linux-only (Landlock requires kernel >= 5.13). On kernels without
Landlock the wrapper fails closed: the command is reported as failed rather than
run unconfined.
"""

from __future__ import annotations

import base64
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

if TYPE_CHECKING:
    from collections.abc import Sequence

_HELPER_PATH = str(Path(__file__).with_name("_landlock_exec.py"))

_EXIT_SANDBOX_SETUP_FAILED = 127


def _python_prefix_roots() -> tuple[str, ...]:
    """Return the active interpreter's install prefixes, de-duplicated.

    Granting read+execute on these keeps ``python3``, the standard library, and
    site-packages (including the current virtualenv) reachable inside the
    sandbox without hard-coding a venv path. Venvs add a prefix outside the
    system roots; system installs resolve under ``/usr`` and are covered anyway.
    """
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    return tuple(dict.fromkeys(p for p in prefixes if p))


# Read-only roots the confined shell may read/execute from. Generous on purpose:
# interpreters, system tools, CA certs, and the active Python prefixes. Missing
# entries are skipped by the helper, so this is safe across images. Override via
# the ``readonly_roots`` constructor argument when you need to tighten or extend.
DEFAULT_READONLY_ROOTS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc",
    "/opt",
    *_python_prefix_roots(),
)

# Extra writable roots beyond ``root_dir`` (which is always writable). ``/tmp``
# is included for tool compatibility (many tools write scratch files there), but
# it is shared across processes; pass ``extra_writable_roots=[]`` for stricter
# isolation.
DEFAULT_EXTRA_WRITABLE_ROOTS: tuple[str, ...] = ()  # noqa: S108  # opt-in scratch root, documented and overridable

# Device roots granted read+write to existing nodes (no exec/create). Required
# for a usable shell: ``>/dev/null``, ``/dev/urandom``, ttys, ``/dev/std*``.
# Not user-configurable -- these are a floor for a working shell.
DEFAULT_DEVICE_ROOTS: tuple[str, ...] = ("/dev",)


class SandboxedShellBackend(LocalShellBackend):
    """`LocalShellBackend` whose ``execute()`` is confined with Linux Landlock.

    File-tool behavior (``read``/``write``/``ls``/``grep``/``glob``/``edit``) is
    inherited unchanged from `LocalShellBackend`. Only ``execute()`` is
    overridden: each shell command (and every process it spawns) is confined to
    ``root_dir`` plus ``extra_writable_roots`` (writable), ``readonly_roots``
    (read + execute), and ``/dev`` (read + write existing nodes).

    Requires Linux with Landlock (kernel >= 5.13). On any other platform or an
    older kernel the sandbox setup fails and ``execute()`` returns a non-zero
    `ExecuteResponse` *without running the command* -- it never silently falls
    back to an unconfined shell.

    !!! warning

        This is a **filesystem** boundary only. It does **not** restrict
        network access, CPU/memory, or in-place truncation of already-readable
        files on Landlock ABI < 3. For untrusted code, combine it with network
        controls and Human-in-the-Loop review, or use a container/VM sandbox.

    Examples:
        ```python
        from deepagents import create_deep_agent
        from deepagents.backends import SandboxedShellBackend

        backend = SandboxedShellBackend(root_dir="/workspace/session-123", virtual_mode=True)
        agent = create_deep_agent(model=model, tools=tools, backend=backend)

        # Reads inside root_dir succeed; reads outside are denied by the kernel.
        backend.execute("cat notes.txt")  # ok
        backend.execute("cat /etc/shadow")  # permission denied

        # Tighten isolation by dropping the shared /tmp grant, or extend the
        # read-only allowlist for tools that live elsewhere.
        backend = SandboxedShellBackend(
            root_dir="/workspace/session-123",
            virtual_mode=True,
            extra_writable_roots=[],
            readonly_roots=[*SandboxedShellBackend.DEFAULT_READONLY_ROOTS, "/srv/models"],
        )
        ```
    """

    DEFAULT_READONLY_ROOTS = DEFAULT_READONLY_ROOTS
    DEFAULT_EXTRA_WRITABLE_ROOTS = DEFAULT_EXTRA_WRITABLE_ROOTS

    def __init__(
        self,
        *args: object,
        readonly_roots: Sequence[str] | None = None,
        extra_writable_roots: Sequence[str] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the backend, forwarding all other arguments to the parent.

        Args:
            *args: Positional arguments forwarded to `LocalShellBackend`
                (e.g. ``root_dir``).
            readonly_roots: Directories the shell may read and execute from. If
                ``None``, uses `DEFAULT_READONLY_ROOTS`. Absent paths are skipped.
            extra_writable_roots: Writable directories *in addition to*
                ``root_dir``. If ``None``, uses `DEFAULT_EXTRA_WRITABLE_ROOTS`
                (``/tmp``). Pass ``[]`` to make ``root_dir`` the only writable
                location.
            **kwargs: Keyword arguments forwarded to `LocalShellBackend`
                (e.g. ``virtual_mode``, ``timeout``, ``env``, ``inherit_env``).
        """
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._readonly_roots = list(DEFAULT_READONLY_ROOTS if readonly_roots is None else readonly_roots)
        self._extra_writable_roots = list(DEFAULT_EXTRA_WRITABLE_ROOTS if extra_writable_roots is None else extra_writable_roots)

    def _writable_roots(self) -> list[str]:
        # ``self.cwd`` is the resolved root_dir (set by FilesystemBackend and
        # used as the subprocess cwd by LocalShellBackend.execute).
        return [str(self.cwd), *self._extra_writable_roots]

    def _wrap_command(self, command: str) -> str:
        """Build the ``python3 _landlock_exec.py ... --cmd-b64 <cmd>`` wrapper.

        The original command is base64-encoded so arbitrary shell content
        (quotes, pipes, heredocs) survives untouched; the helper decodes it and
        re-runs it under ``/bin/sh -c``.
        """
        cmd_b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
        parts = [shlex.quote(sys.executable), shlex.quote(_HELPER_PATH)]
        for root in self._writable_roots():
            parts += ["--rw", shlex.quote(root)]
        for root in self._readonly_roots:
            parts += ["--ro", shlex.quote(root)]
        for root in DEFAULT_DEVICE_ROOTS:
            parts += ["--dev", shlex.quote(root)]
        parts += ["--cmd-b64", cmd_b64]
        return " ".join(parts)

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Run ``command`` under the Landlock wrapper; see `LocalShellBackend.execute`."""
        # Defer non-string / empty validation to the parent's error contract.
        if not command or not isinstance(command, str):
            return super().execute(command, timeout=timeout)
        if not Path(_HELPER_PATH).exists():  # pragma: no cover - packaging guard
            return ExecuteResponse(
                output=f"Error: Landlock helper missing at {_HELPER_PATH}; refusing to run unconfined.",
                exit_code=_EXIT_SANDBOX_SETUP_FAILED,
                truncated=False,
            )
        return super().execute(self._wrap_command(command), timeout=timeout)


__all__ = ["SandboxedShellBackend"]
