"""Policy JSON for Windows, macOS and Linux with the host fully mocked."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from langchain_mxc._policy import (
    DEFAULT_DENIED_HOME_ENTRIES,
    HostInfo,
    SandboxPolicy,
    is_under,
    wrap_command,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def make_host(
    platform: str,
    *,
    home: str,
    which: dict[str, str],
    exists: Iterable[str],
    realpath: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
    sys_prefixes: tuple[str, ...] = (),
    executable: str | None = None,
) -> HostInfo:
    existing = set(exists)
    real = realpath or {}

    def _exists(path: str) -> bool:
        return path in existing

    return HostInfo(
        platform=platform,
        environ=environ or {},
        home=home,
        which=which.get,
        exists=_exists,
        realpath=lambda p: real.get(p, p),
        sys_prefixes=sys_prefixes,
        executable=executable,
    )


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


@pytest.fixture
def macos_host() -> HostInfo:
    home = "/Users/me"
    return make_host(
        "darwin",
        home=home,
        environ={"LANG": "en_US.UTF-8"},
        which={
            "python": "/Users/me/proj/.venv/bin/python",
            "python3": "/opt/homebrew/bin/python3",
            "git": "/usr/bin/git",
            "node": "/opt/homebrew/bin/node",
        },
        realpath={
            "/Users/me/proj/.venv/bin/python": "/Users/me/.local/share/uv/python/cpython-3.12/bin/python3.12",
            "/opt/homebrew/bin/node": "/opt/homebrew/Cellar/node/26.3.0/bin/node",
            "/opt/homebrew/bin/python3": "/opt/homebrew/Cellar/python@3.13/3.13.1/bin/python3.13",
        },
        exists=[
            "/bin",
            "/sbin",
            "/usr",
            "/System",
            "/Library",
            "/private/etc",
            "/opt/homebrew",
            "/opt/homebrew/bin",
            "/dev",
            "/Users/me/.ssh",
            "/Users/me/.aws",
            "/Users/me/data",
            "/Users/me/proj",
            "/Users/me/proj/.venv",
        ],
        sys_prefixes=("/Users/me/proj/.venv", "/Users/me/.local/share/uv/python/cpython-3.12"),
        executable="/Users/me/proj/.venv/bin/python",
    )


def test_macos_request_shape(macos_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/Users/me/proj", scratch_dir="/private/var/folders/x/T/scratch", host=macos_host)
    request = policy.build_request("echo hi", 5)

    assert request["version"] == "1.0.0"
    assert request["containment"] == "process"
    assert request["process"]["commandLine"] == "/bin/sh -c 'echo hi'"
    assert request["process"]["cwd"] == "/Users/me/proj"
    assert request["process"]["timeout"] == 5000
    assert set(request) == {"version", "containment", "process", "filesystem", "network", "ui"}
    assert set(request["process"]) == {"commandLine", "cwd", "env", "timeout"}

    fs = request["filesystem"]
    assert fs["readwritePaths"] == ["/private/var/folders/x/T/scratch", "/Users/me/proj"] or set(fs["readwritePaths"]) == {
        "/Users/me/proj",
        "/private/var/folders/x/T/scratch",
    }
    for expected in (
        "/bin",
        "/sbin",
        "/usr",
        "/System",
        "/Library",
        "/private/etc",
        "/opt/homebrew",
        "/Users/me/.local/share/uv/python/cpython-3.12",
    ):
        assert expected in fs["readonlyPaths"], expected
    # the venv lives inside root_dir, which is already read-write
    assert "/Users/me/proj/.venv" not in fs["readonlyPaths"]
    # /dev must never be listed on macOS (it would break >/dev/null)
    assert "/dev" not in fs["readonlyPaths"]
    assert fs["deniedPaths"] == ["/Users/me/.ssh", "/Users/me/.aws"]

    assert request["network"] == {"egress": {"default": "deny"}, "ingress": {"default": "deny", "hostLoopback": "deny"}}
    assert request["ui"] == {"disable": True, "clipboard": "none", "injection": False}


def test_macos_env_is_minimal_and_prefers_agent_python(macos_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/Users/me/proj", scratch_dir="/scratch", host=macos_host)
    env = dict(entry.split("=", 1) for entry in policy.env)
    assert set(env) == {"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG"}
    assert env["HOME"] == "/scratch"
    assert env["TMPDIR"] == "/scratch"
    assert env["LANG"] == "en_US.UTF-8"
    path = env["PATH"].split(":")
    assert path[0] == "/Users/me/proj/.venv/bin"
    assert "/opt/homebrew/bin" in path
    assert "/usr/bin" in path
    assert "/bin" in path


def test_macos_shell_quoting(macos_host: HostInfo) -> None:
    assert wrap_command("""echo "it's" | grep 's'""", macos_host) == """/bin/sh -c 'echo "it'"'"'s" | grep '"'"'s'"'"''"""


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------


@pytest.fixture
def linux_host() -> HostInfo:
    return make_host(
        "linux",
        home="/home/me",
        environ={},
        which={
            "python3": "/usr/bin/python3",
            "git": "/usr/bin/git",
            "node": "/home/me/.nvm/versions/node/v24.1.0/bin/node",
        },
        exists=["/bin", "/sbin", "/usr", "/lib", "/lib64", "/etc", "/opt", "/home/me/.ssh", "/home/me/.config/gcloud", "/home/me/proj"],
        sys_prefixes=("/usr", "/usr"),
        executable="/usr/bin/python3",
    )


def test_linux_request_shape(linux_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/home/me/proj", scratch_dir="/tmp/scratch", host=linux_host)
    request = policy.build_request("ls -la", None)

    assert request["process"]["commandLine"] == "/bin/sh -c 'ls -la'"
    assert "timeout" not in request["process"]
    fs = request["filesystem"]
    for expected in ("/bin", "/sbin", "/usr", "/lib", "/lib64", "/etc", "/opt", "/home/me/.nvm/versions/node/v24.1.0"):
        assert expected in fs["readonlyPaths"], expected
    # nonexistent standard dirs are not listed (bubblewrap rejects them)
    assert "/snap" not in fs["readonlyPaths"]
    assert "/nix/store" not in fs["readonlyPaths"]
    # the home directory itself is never granted just because a tool lives under it
    assert "/home/me" not in fs["readonlyPaths"]
    assert "/home" not in fs["readonlyPaths"]
    assert fs["deniedPaths"] == ["/home/me/.ssh", "/home/me/.config/gcloud"]
    env = dict(entry.split("=", 1) for entry in request["process"]["env"])
    assert env["PATH"].split(":")[0] == "/usr/bin"
    assert "/home/me/.nvm/versions/node/v24.1.0/bin" in env["PATH"].split(":")
    assert env["HOME"] == "/tmp/scratch"
    assert env["LANG"] == "C.UTF-8"


def test_linux_env_merges_user_overrides(linux_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/home/me/proj", scratch_dir="/tmp/scratch", env={"FOO": "bar", "PATH": "/custom/bin"}, host=linux_host)
    env = dict(entry.split("=", 1) for entry in policy.env)
    assert env["FOO"] == "bar"
    assert env["PATH"] == "/custom/bin"
    assert env["HOME"] == "/tmp/scratch"


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@pytest.fixture
def windows_host() -> HostInfo:
    return make_host(
        "win32",
        home=r"C:\Users\me",
        environ={
            "SYSTEMROOT": r"C:\Windows",
            "SystemDrive": "C:",
            "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD;.PY",
            "Path": r"C:\Windows\System32",
        },
        which={
            "python": r"C:\Users\me\proj\.venv\Scripts\python.exe",
            "git": r"C:\Program Files\Git\cmd\git.exe",
            "node": r"C:\Program Files\nodejs\node.exe",
        },
        exists=[
            r"C:\Windows\System32",
            r"C:\Users\me\.ssh",
            r"C:\Users\me\proj",
            r"C:\Users\me\proj\.venv",
            r"C:\Python312",
            r"C:\Program Files\Git",
            r"C:\Program Files\nodejs",
        ],
        sys_prefixes=(r"C:\Users\me\proj\.venv", r"C:\Python312"),
        executable=r"C:\Users\me\proj\.venv\Scripts\python.exe",
    )


def test_windows_request_shape(windows_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir=r"C:\Users\me\proj", scratch_dir=r"C:\Users\me\AppData\Local\Temp\scratch", host=windows_host)
    request = policy.build_request("python -m pytest", 60)

    assert request["process"]["commandLine"] == r'"C:\Windows\System32\cmd.exe" /d /s /c "python -m pytest"'
    assert request["process"]["cwd"] == r"C:\Users\me\proj"
    assert request["process"]["timeout"] == 60000

    fs = request["filesystem"]
    assert r"C:\Users\me\proj" in fs["readwritePaths"]
    assert r"C:\Users\me\AppData\Local\Temp\scratch" in fs["readwritePaths"]
    assert r"C:\Windows\System32" in fs["readonlyPaths"]
    assert r"C:\Program Files\Git" in fs["readonlyPaths"]  # cmd\git.exe is a launcher for mingw64\bin\git.exe
    assert r"C:\Program Files\nodejs" in fs["readonlyPaths"]
    assert r"C:\Python312" in fs["readonlyPaths"]
    assert r"C:\Users\me\proj\.venv" not in fs["readonlyPaths"]
    assert r"C:\Users\me" not in fs["readonlyPaths"]
    assert fs["deniedPaths"] == [r"C:\Users\me\.ssh"]

    env = dict(entry.split("=", 1) for entry in request["process"]["env"])
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["LOCALAPPDATA"] == r"C:\Users\me\AppData\Local"
    assert env["TEMP"] == r"C:\Users\me\AppData\Local\Temp\scratch"
    assert env["TMP"] == r"C:\Users\me\AppData\Local\Temp\scratch"
    assert env["USERPROFILE"] == r"C:\Users\me\AppData\Local\Temp\scratch"
    assert env["PATHEXT"] == ".COM;.EXE;.BAT;.CMD;.PY"
    assert env["COMSPEC"] == r"C:\Windows\System32\cmd.exe"
    path = env["PATH"].split(";")
    assert path[0] == r"C:\Users\me\proj\.venv\Scripts"
    assert r"C:\Windows\System32" in path
    assert r"C:\Program Files\Git\cmd" in path
    assert "HOME" not in env


def test_windows_env_override_is_case_insensitive(windows_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir=r"C:\Users\me\proj", scratch_dir=r"C:\scratch", env={"path": r"D:\only"}, host=windows_host)
    env = dict(entry.split("=", 1) for entry in policy.env)
    assert env["PATH"] == r"D:\only"
    assert sum(1 for k in env if k.upper() == "PATH") == 1


# ---------------------------------------------------------------------------
# Options shared by all platforms
# ---------------------------------------------------------------------------


def test_allow_network_opens_outbound_only(macos_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/Users/me/proj", scratch_dir="/scratch", allow_network=True, host=macos_host)
    assert policy.build_request("x", 1)["network"] == {"egress": {"default": "allow"}, "ingress": {"default": "deny", "hostLoopback": "deny"}}


def test_user_paths_expand_and_drop_missing(macos_host: HostInfo) -> None:
    policy = SandboxPolicy.create(
        root_dir="/Users/me/proj",
        scratch_dir="/scratch",
        read_only_paths=["~/data", "/nope/missing"],
        read_write_paths=["/Users/me/.ssh"],  # explicit grant wins over the default deny
        host=macos_host,
    )
    assert "/Users/me/data" in policy.read_only_paths
    assert policy.dropped_paths == ("/nope/missing",)
    assert "/Users/me/.ssh" in policy.read_write_paths
    assert "/Users/me/.ssh" not in policy.denied_paths
    assert "/Users/me/.aws" in policy.denied_paths


def test_default_denied_entries_are_only_credential_locations() -> None:
    assert ".ssh" in DEFAULT_DENIED_HOME_ENTRIES
    assert all(entry.startswith(".") for entry in DEFAULT_DENIED_HOME_ENTRIES)


def test_is_allowed_and_is_denied(macos_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir="/Users/me/proj", scratch_dir="/scratch", host=macos_host)
    assert policy.is_allowed("/Users/me/proj/src/x.py")
    assert policy.is_allowed("/scratch/tmpfile")
    assert policy.is_allowed("/usr/bin/git")
    assert policy.is_allowed("/dev/null")  # macOS baseline
    assert not policy.is_allowed("/Users/me/Desktop/secret.txt")
    assert not policy.is_allowed("/Users/me/.ssh/id_ed25519")
    assert policy.is_denied("/Users/me/.ssh/id_ed25519")
    assert not policy.is_denied("/Users/me/proj")


def test_is_under_is_lexical_and_case_insensitive_on_windows(windows_host: HostInfo, linux_host: HostInfo) -> None:
    assert is_under(windows_host, r"c:\users\ME\proj\x", r"C:\Users\me\proj")
    assert not is_under(windows_host, r"C:\Users\me\project", r"C:\Users\me\proj")
    assert is_under(linux_host, "/home/me/proj", "/home/me/proj")
    assert not is_under(linux_host, "/home/me/project", "/home/me/proj")


def test_request_never_contains_enforcement_off_modes(macos_host: HostInfo, linux_host: HostInfo, windows_host: HostInfo) -> None:
    for host, root in ((macos_host, "/Users/me/proj"), (linux_host, "/home/me/proj"), (windows_host, r"C:\Users\me\proj")):
        policy = SandboxPolicy.create(
            root_dir=root, scratch_dir="/scratch" if host.platform != "win32" else r"C:\scratch", allow_network=True, host=host
        )
        text = json.dumps(policy.build_request("echo", 1)).lower()
        for forbidden in ("learningmode", "capturedenials", "audit", "permissive", "profileoverride", "leastprivilege"):
            assert forbidden not in text, forbidden


def test_windows_command_starting_with_a_quote_survives_cmd(windows_host: HostInfo) -> None:
    policy = SandboxPolicy.create(root_dir=r"C:\proj", scratch_dir=r"C:\scratch", host=windows_host)
    command = r'"C:\Program Files\Git\cmd\git.exe" log -1 --format="%H"'
    line = policy.build_request(command, 5)["process"]["commandLine"]
    # cmd /s strips only the outermost pair, leaving the command untouched.
    assert line == rf'"C:\Windows\System32\cmd.exe" /d /s /c "{command}"'
