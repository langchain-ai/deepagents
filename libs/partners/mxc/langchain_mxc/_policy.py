"""Build MXC request documents (contract version 1.0.0) for `MXCSandbox`.

The policy is computed once per backend instance and turned into a request
per command. All host-dependent inputs (platform, environment, `which`,
`exists`, `realpath`) are injectable so the policy for every operating system
can be unit-tested on any machine.

Policy summary (see the README for the table):

- read-write: `root_dir` and a per-instance scratch directory, plus
  user-supplied `read_write_paths`;
- read-only: the OS toolchain (system directories plus the install prefixes
  of `python`, `python3`, `git` and `node` found on `PATH`), plus
  user-supplied `read_only_paths`;
- denied: well-known credential folders under the home directory, unless the
  user granted one explicitly;
- network: egress and ingress denied unless `allow_network=True`;
- UI: MXC's default-deny values (no GUI, clipboard `none`, no input
  injection) unless `allow_ui`, `clipboard` or `allow_input_injection` opt in;
- environment: a minimal one built from scratch, with user `env` layered on top.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = "1.0.0"
CONTAINMENT = "process"

CLIPBOARD_LEVELS: tuple[str, ...] = ("none", "read", "write", "all")
"""Values MXC accepts for `ui.clipboard` (contract 1.0.0)."""

TOOLCHAIN_TOOLS: tuple[str, ...] = ("python", "python3", "git", "node")
"""Tools whose install locations are made readable inside the sandbox."""

SYSTEM_READONLY_PATHS: dict[str, tuple[str, ...]] = {
    # `/dev` is deliberately absent: listing it on macOS overrides the
    # baseline read-write rule for /dev/null (observed with mxc-sdk 1.0.1).
    "darwin": (
        "/bin",
        "/sbin",
        "/usr",
        "/System",
        "/Library",
        "/private/etc",
        "/private/var/db/timezone",
        "/private/var/select",
        "/opt/homebrew",
        "/opt/local",
        "/nix/store",
    ),
    "linux": (
        "/bin",
        "/sbin",
        "/usr",
        "/lib",
        "/lib32",
        "/lib64",
        "/libx32",
        "/etc",
        "/opt",
        "/snap",
        "/nix/store",
        "/run/current-system",
    ),
}

PACKAGE_MANAGER_ROOTS: tuple[str, ...] = (
    "/opt/homebrew",
    "/usr/local",
    "/home/linuxbrew/.linuxbrew",
    "/nix/store",
)
"""If a tool resolves under one of these, the whole root is made readable.

Homebrew binaries link against `<root>/opt/<dep>/lib/*.dylib`, so the
tool's own prefix is not enough (verified with node).
"""

POSIX_PATH_DIRS: tuple[str, ...] = ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")
DARWIN_EXTRA_PATH_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/opt/homebrew/sbin")

DEFAULT_DENIED_HOME_ENTRIES: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".azure",
    ".gnupg",
    ".kube",
    ".docker",
    ".config/gcloud",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
)
"""Credential locations denied even when a broader read-only grant covers them."""

MACOS_BASELINE_READABLE: tuple[str, ...] = (
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/usr/libexec",
    "/usr/share",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db/timezone",
    "/private/var/db/dyld",
    "/private/var/select",
    "/dev",
)
"""What the Seatbelt profile allows before any `readonlyPaths` (from `--debug`)."""


@dataclass(frozen=True)
class HostInfo:
    """Everything about the host that influences the policy.

    Tests construct this directly to simulate Windows, macOS or Linux.
    """

    platform: str
    environ: Mapping[str, str]
    home: str
    which: Callable[[str], str | None]
    exists: Callable[[str], bool]
    realpath: Callable[[str], str]
    sys_prefixes: tuple[str, ...] = ()
    """`sys.prefix` and `sys.base_prefix` of the interpreter running the agent."""
    executable: str | None = None
    """`sys.executable`: its directory goes first on the sandbox `PATH`.

    This makes `python` inside the sandbox resolve to the interpreter the
    agent itself runs with (typically a virtual environment), even when that
    environment is not activated in the host's `PATH`.
    """

    @classmethod
    def current(cls) -> HostInfo:
        """Describe the machine this process runs on."""
        return cls(
            platform=sys.platform,
            environ=dict(os.environ),
            home=os.path.expanduser("~"),
            which=shutil.which,
            exists=os.path.exists,
            realpath=os.path.realpath,
            sys_prefixes=(sys.prefix, sys.base_prefix),
            executable=sys.executable or None,
        )

    @property
    def is_windows(self) -> bool:
        """Whether the host is Windows."""
        return self.platform == "win32"

    @property
    def pathmod(self) -> Any:
        """`ntpath` or `posixpath`, matching the host."""
        return ntpath if self.is_windows else posixpath


def _norm(host: HostInfo, path: str) -> str:
    pm = host.pathmod
    return pm.normcase(pm.normpath(path)) if host.is_windows else pm.normpath(path)


def is_under(host: HostInfo, path: str, parent: str) -> bool:
    """Whether `path` equals `parent` or lies beneath it (lexically)."""
    pm = host.pathmod
    p = _norm(host, path)
    q = _norm(host, parent)
    if p == q:
        return True
    if not q.endswith(pm.sep):
        q += pm.sep
    return p.startswith(q)


def _is_root_like(host: HostInfo, path: str) -> bool:
    pm = host.pathmod
    n = pm.normpath(path)
    _drive, tail = pm.splitdrive(n)
    return tail in {pm.sep, ""} or n == pm.sep


def _expand(host: HostInfo, path: str, base: str) -> str:
    pm = host.pathmod
    if path == "~" or path.startswith(("~/", "~\\")):
        path = pm.join(host.home, path[2:]) if len(path) > 1 else host.home
    if not pm.isabs(path):
        path = pm.join(base, path)
    return pm.normpath(path)


def _add(seq: list[str], host: HostInfo, path: str | None) -> None:
    if not path:
        return
    path = host.pathmod.normpath(path)
    if all(_norm(host, path) != _norm(host, existing) for existing in seq):
        seq.append(path)


def _drop_nested(host: HostInfo, paths: Sequence[str]) -> list[str]:
    """Keep only the shallowest entries (a deeper duplicate adds nothing)."""
    kept: list[str] = []
    for path in sorted(paths, key=lambda p: (len(_norm(host, p)), p)):
        if not any(is_under(host, path, parent) for parent in kept):
            kept.append(path)
    return kept


def _install_prefix(host: HostInfo, exe: str) -> str | None:
    """Directory to make readable so `exe` can run, or None if that is unsafe."""
    pm = host.pathmod
    bin_dir = pm.dirname(exe)
    # `bin` (POSIX, venvs), `Scripts` (Windows venvs), `cmd` (Git for Windows).
    prefix = pm.dirname(bin_dir) if pm.basename(bin_dir).lower() in {"bin", "scripts", "cmd"} else bin_dir
    for candidate in (prefix, bin_dir):
        if _is_root_like(host, candidate):
            continue
        if is_under(host, host.home, candidate):  # candidate is the home dir or above it
            continue
        return candidate
    return None


def toolchain_paths(host: HostInfo) -> tuple[list[str], list[str]]:  # noqa: C901 - one branch per OS and per toolchain layout
    """Detect the toolchain.

    Returns:
        `(readonly_paths, path_dirs)`: directories to grant read-only, and
        directories to put on the sandbox `PATH`.
    """
    pm = host.pathmod
    readonly: list[str] = []
    path_dirs: list[str] = []

    executables: list[tuple[str, str | None]] = [(tool, host.which(tool)) for tool in TOOLCHAIN_TOOLS]
    if host.executable:
        executables.insert(0, ("python", host.executable))

    for _tool, located in executables:
        if not located:
            continue
        found = pm.normpath(located)
        _add(path_dirs, host, pm.dirname(found))
        try:
            real = pm.normpath(host.realpath(found))
        except OSError:
            real = found
        for candidate in (found, real):
            _add(readonly, host, _install_prefix(host, candidate))
            if not host.is_windows:
                for root in PACKAGE_MANAGER_ROOTS:
                    if is_under(host, candidate, root):
                        _add(readonly, host, root)

    for prefix in host.sys_prefixes:
        if prefix and not _is_root_like(host, prefix) and not is_under(host, host.home, prefix):
            _add(readonly, host, prefix)

    return readonly, path_dirs


def system_readonly_paths(host: HostInfo) -> list[str]:
    """Standard system locations for the host OS (only those that exist)."""
    if host.is_windows:
        system_root = host.environ.get("SYSTEMROOT") or host.environ.get("SystemRoot") or r"C:\Windows"
        return [p for p in (ntpath.join(system_root, "System32"),) if host.exists(p)]
    return [p for p in SYSTEM_READONLY_PATHS.get(host.platform, SYSTEM_READONLY_PATHS["linux"]) if host.exists(p)]


def wrap_command(command: str, host: HostInfo) -> str:
    """Wrap a shell command string in the OS shell for `process.commandLine`."""
    if host.is_windows:
        system_root = host.environ.get("SYSTEMROOT") or host.environ.get("SystemRoot") or r"C:\Windows"
        cmd_exe = ntpath.join(system_root, "System32", "cmd.exe")
        # `/s /c "<command>"`: cmd strips exactly the outer pair of quotes and
        # runs the rest verbatim, so a command that itself starts with a quote
        # (`"C:\\Program Files\\x.exe" arg`) is not mangled. `subprocess.run(shell=True)`,
        # which `LocalShellBackend` relies on, wraps the command in one pair of quotes the same way.
        return f'"{cmd_exe}" /d /s /c "{command}"'
    return f"/bin/sh -c {shlex.quote(command)}"


def build_env(host: HostInfo, scratch_dir: str, path_dirs: Sequence[str], user_env: Mapping[str, str] | None) -> list[str]:
    """Build the minimal sandbox environment as `KEY=VALUE` strings."""
    pm = host.pathmod
    env: dict[str, str] = {}
    if host.is_windows:
        system_root = host.environ.get("SYSTEMROOT") or host.environ.get("SystemRoot") or r"C:\Windows"
        system32 = ntpath.join(system_root, "System32")
        path_entries: list[str] = []
        for entry in (*path_dirs, system32, system_root, ntpath.join(system32, "Wbem"), ntpath.join(system32, "WindowsPowerShell", "v1.0")):
            _add(path_entries, host, entry)
        env["SYSTEMROOT"] = system_root
        env["SYSTEMDRIVE"] = host.environ.get("SYSTEMDRIVE") or host.environ.get("SystemDrive") or ntpath.splitdrive(system_root)[0] or "C:"
        env["LOCALAPPDATA"] = host.environ.get("LOCALAPPDATA") or ntpath.join(scratch_dir, "AppData", "Local")
        env["TEMP"] = scratch_dir
        env["TMP"] = scratch_dir
        env["USERPROFILE"] = scratch_dir
        env["PATH"] = ";".join(path_entries)
        env["PATHEXT"] = host.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
        env["COMSPEC"] = ntpath.join(system32, "cmd.exe")
    else:
        path_entries = []
        extra = DARWIN_EXTRA_PATH_DIRS if host.platform == "darwin" else ()
        for entry in (*path_dirs, *POSIX_PATH_DIRS, *extra):
            if entry in path_dirs or host.exists(entry):
                _add(path_entries, host, entry)
        env["PATH"] = ":".join(path_entries)
        env["HOME"] = scratch_dir
        env["TMPDIR"] = scratch_dir
        env["TEMP"] = scratch_dir
        env["TMP"] = scratch_dir
        env["LANG"] = host.environ.get("LANG") or "C.UTF-8"

    if user_env:
        for key, value in user_env.items():
            k = key.upper() if host.is_windows else key
            env[k] = str(value)
    del pm
    return [f"{key}={value}" for key, value in env.items()]


@dataclass(frozen=True)
class SandboxPolicy:
    """The resolved policy for one `MXCSandbox` instance."""

    host: HostInfo
    root_dir: str
    scratch_dir: str
    read_write_paths: tuple[str, ...]
    read_only_paths: tuple[str, ...]
    denied_paths: tuple[str, ...]
    allow_network: bool
    env: tuple[str, ...]
    dropped_paths: tuple[str, ...] = field(default=())
    """User-supplied paths that were ignored because they do not exist."""
    allow_ui: bool = False
    """Let commands create windows (`ui.disable = false`)."""
    clipboard: str = "none"
    """Clipboard access for commands: one of `CLIPBOARD_LEVELS`."""
    allow_input_injection: bool = False
    """Let commands send synthetic keyboard and mouse input (`ui.injection`)."""

    @classmethod
    def create(  # noqa: C901 - assembles each policy section in turn
        cls,
        *,
        root_dir: str,
        scratch_dir: str,
        read_only_paths: Iterable[str] = (),
        read_write_paths: Iterable[str] = (),
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
        host: HostInfo | None = None,
        allow_ui: bool = False,
        clipboard: str = "none",
        allow_input_injection: bool = False,
    ) -> SandboxPolicy:
        """Resolve the policy for `root_dir` on `host`.

        Raises:
            ValueError: If `clipboard` is not one of `CLIPBOARD_LEVELS`.
        """
        if clipboard not in CLIPBOARD_LEVELS:
            msg = f"clipboard must be one of {', '.join(CLIPBOARD_LEVELS)}, got {clipboard!r}"
            raise ValueError(msg)
        host = host or HostInfo.current()
        pm = host.pathmod
        root = pm.normpath(root_dir)
        scratch = pm.normpath(scratch_dir)
        dropped: list[str] = []

        def user_paths(raw: Iterable[str]) -> list[str]:
            out: list[str] = []
            for item in raw:
                path = _expand(host, str(item), root)
                if not host.exists(path):
                    dropped.append(str(item))
                    continue
                _add(out, host, path)
                try:
                    real = pm.normpath(host.realpath(path))
                except OSError:
                    real = path
                _add(out, host, real)
            return out

        rw: list[str] = []
        for p in (root, scratch, *user_paths(read_write_paths)):
            _add(rw, host, p)
        user_rw = rw[2:]

        user_ro = user_paths(read_only_paths)
        toolchain_ro, path_dirs = toolchain_paths(host)
        ro: list[str] = []
        for p in (*system_readonly_paths(host), *toolchain_ro, *user_ro):
            if any(is_under(host, p, w) for w in rw):
                continue
            _add(ro, host, p)
        ro = _drop_nested(host, ro)
        rw = _drop_nested(host, rw)

        explicit_grants = [*user_rw, *user_ro]
        denied: list[str] = []
        for entry in DEFAULT_DENIED_HOME_ENTRIES:
            candidate = pm.normpath(pm.join(host.home, *entry.split("/")))
            if not host.exists(candidate):
                continue
            if is_under(host, root, candidate):
                continue
            if any(is_under(host, grant, candidate) for grant in explicit_grants):
                continue
            _add(denied, host, candidate)

        return cls(
            host=host,
            root_dir=root,
            scratch_dir=scratch,
            read_write_paths=tuple(rw),
            read_only_paths=tuple(ro),
            denied_paths=tuple(denied),
            allow_network=bool(allow_network),
            env=tuple(build_env(host, scratch, path_dirs, env)),
            dropped_paths=tuple(dropped),
            allow_ui=bool(allow_ui),
            clipboard=clipboard,
            allow_input_injection=bool(allow_input_injection),
        )

    # -- request -----------------------------------------------------------

    def build_request(self, command: str, timeout: float | None = None) -> dict[str, Any]:
        """Build the MXC 1.0.0 one-shot request for `command`.

        Args:
            command: The shell command string, as the agent wrote it.
            timeout: Seconds; becomes `process.timeout` in milliseconds.
                `None` or `0` means no executor-side timeout.
        """
        process: dict[str, Any] = {
            "commandLine": wrap_command(command, self.host),
            "cwd": self.root_dir,
            "env": list(self.env),
        }
        if timeout:
            process["timeout"] = round(timeout * 1000)

        filesystem: dict[str, Any] = {
            "readwritePaths": list(self.read_write_paths),
            "readonlyPaths": list(self.read_only_paths),
        }
        if self.denied_paths:
            filesystem["deniedPaths"] = list(self.denied_paths)

        # Network is on or off, and "on" means outbound only. Inbound stays
        # denied because that is the only shape every backend accepts
        # (verified): bubblewrap rejects `ingress.default: allow`, and Seatbelt
        # rejects `hostLoopback: allow` while the inbound default is deny.
        network = {
            "egress": {"default": "allow" if self.allow_network else "deny"},
            "ingress": {"default": "deny", "hostLoopback": "deny"},
        }

        return {
            "version": SCHEMA_VERSION,
            "containment": CONTAINMENT,
            "process": process,
            "filesystem": filesystem,
            "network": network,
            # MXC's own defaults are deny for all three; these only loosen them
            # when the user opted in. Enforced on Windows and macOS; bubblewrap
            # ignores the section. On Windows, `disable: true` makes MXC ignore
            # the other two fields.
            "ui": {"disable": not self.allow_ui, "clipboard": self.clipboard, "injection": self.allow_input_injection},
        }

    # -- queries used for the denial hint -----------------------------------

    @property
    def allowed_paths(self) -> tuple[str, ...]:
        """Every path the policy grants (read-write and read-only)."""
        return (*self.read_write_paths, *self.read_only_paths)

    def is_denied(self, path: str) -> bool:
        """Whether `path` is under an explicit deny."""
        return any(is_under(self.host, path, d) for d in self.denied_paths)

    def is_allowed(self, path: str) -> bool:
        """Whether the policy (or the OS baseline) grants access to `path`."""
        if self.is_denied(path):
            return False
        if any(is_under(self.host, path, a) for a in self.allowed_paths):
            return True
        if self.host.platform == "darwin":
            return any(is_under(self.host, path, b) for b in MACOS_BASELINE_READABLE)
        return False


__all__ = [
    "CONTAINMENT",
    "DEFAULT_DENIED_HOME_ENTRIES",
    "PACKAGE_MANAGER_ROOTS",
    "SCHEMA_VERSION",
    "SYSTEM_READONLY_PATHS",
    "TOOLCHAIN_TOOLS",
    "HostInfo",
    "SandboxPolicy",
    "build_env",
    "is_under",
    "system_readonly_paths",
    "toolchain_paths",
    "wrap_command",
]
