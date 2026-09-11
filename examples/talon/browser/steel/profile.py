"""Validate and exclusively lease the persistent Chromium profile."""

import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

PROFILE = Path("/var/lib/steel/profile")


def validate(profile: Path) -> None:
    """Reject dirty, linked, or corrupt profiles without repairing them."""
    for name in (".talon-dirty", "SingletonLock", "SingletonCookie", "SingletonSocket"):
        if os.path.lexists(profile / name):
            raise ValueError("profile_dirty_or_locked")
    for root, directories, files in os.walk(profile):
        for name in directories + files:
            if (Path(root) / name).is_symlink():
                raise ValueError("profile_symlink")
    for path in [profile / "Local State", *profile.glob("*/Preferences")]:
        if path.exists():
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("profile_json_invalid")
            state = data.get("profile", {})
            if not isinstance(state, dict):
                raise ValueError("profile_json_invalid")
            if (
                state.get("exit_type", "Normal") != "Normal"
                or state.get("exited_cleanly") is False
            ):
                raise ValueError("profile_unclean_exit")
    for path in profile.rglob("Cookies"):
        with sqlite3.connect(
            f"{path.as_uri()}?mode=ro", uri=True, timeout=0
        ) as database:
            if database.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("profile_sqlite_corrupt")


def prepare(profile: Path) -> int:
    """Acquire the mount lease before any profile writes."""
    if not os.path.ismount(profile) or profile.is_symlink():
        raise ValueError("profile_mount_required")
    metadata = profile.stat()
    privileged = os.getuid() == 0
    if not privileged and (os.getuid() != 1000 or metadata.st_uid != 1000):
        raise ValueError("profile_owner_invalid")
    if privileged and metadata.st_uid == 1000:
        os.setegid(1000)
        os.seteuid(1000)
    descriptor = os.open(profile, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate(profile)
    if metadata.st_uid != 1000 and any(profile.iterdir()):
        raise ValueError("profile_owner_invalid")
    os.fchmod(descriptor, 0o700)
    if privileged:
        os.seteuid(0)
        os.setegid(0)
        os.fchown(descriptor, 1000, 1000)
    return descriptor


def wait_for_egress(stopped: threading.Event) -> None:
    """Gate application launch on the host-installed namespace firewall."""
    if os.environ.get("STEEL_REQUIRE_EGRESS_READY", "false").lower() != "true":
        return
    deadline = time.monotonic() + 60
    while not Path("/run/steel-network/ready").is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("profile_egress_ready_timeout")
        if stopped.wait(min(0.1, remaining)):
            return


def main() -> int:
    """Keep the lease until the unprivileged browser supervisor exits."""
    os.umask(0o077)
    child = None
    descriptor = None
    pending = 0
    stopped = threading.Event()

    def forward(signum: int, frame: object) -> None:
        nonlocal pending
        pending = signum
        stopped.set()
        if child is not None:
            child.send_signal(signum)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, forward)
    try:
        descriptor = prepare(PROFILE)
        wait_for_egress(stopped)
        if pending:
            return 128 + pending
        command = ["setpriv", "--no-new-privs"]
        if os.getuid() == 0:
            command += [
                "--reuid=1000",
                "--regid=1000",
                "--clear-groups",
                "--bounding-set=-all",
                "--inh-caps=-all",
                "--ambient-caps=-all",
            ]
        child = subprocess.Popen([*command, "node", "/app/api/talon/bootstrap.mjs"])
        if os.getuid() == 0:
            os.setgroups([])
            os.setgid(1000)
            os.setuid(1000)
        if pending:
            child.send_signal(pending)
        return child.wait()
    except (OSError, ValueError, sqlite3.Error) as error:
        code = (
            str(error)
            if isinstance(error, ValueError) and str(error).startswith("profile_")
            else type(error).__name__
        )
        print(
            json.dumps(
                {
                    "error": "profile_start_failed",
                    "reason": code,
                    "action": "inspect_profile_offline_do_not_replace",
                }
            ),
            flush=True,
        )
        return 1
    finally:
        if descriptor is not None:
            os.close(descriptor)


if __name__ == "__main__":
    sys.exit(main())
