"""Launch the rootful Linux browser overlay with fail-closed namespace firewalls."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DENIED = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.88.99.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "168.63.129.16/32",
)


def run(command: list[str], env: dict[str, str]) -> str:
    """Keep subprocess output, including Compose interpolation, out of logs."""
    capture = any(item in command for item in ("ps", "inspect", "info"))
    result = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode:
        msg = f"Deployment command failed ({command[0]}, exit {result.returncode})"
        raise RuntimeError(msg)
    return (result.stdout or "").strip()


def preflight(env: dict[str, str]) -> None:
    """Reject unsupported hosts and remote or rootless Docker daemons."""
    if sys.platform != "linux" or os.geteuid() != 0:
        msg = (
            "Linux host root is required; use sudo with explicit --home and --env-file"
        )
        raise RuntimeError(msg)
    for executable in ("docker", "nsenter", "iptables", "ip6tables"):
        if not shutil.which(executable):
            msg = f"Required host executable is missing: {executable}"
            raise RuntimeError(msg)
    context = json.loads(run(["docker", "context", "inspect"], env))[0]
    endpoint = env.get("DOCKER_HOST", context["Endpoints"]["docker"]["Host"])
    if not endpoint.startswith("unix://"):
        msg = "A local Unix-socket Docker daemon is required"
        raise RuntimeError(msg)
    info = json.loads(run(["docker", "info", "--format", "{{json .}}"], env))
    if info.get("OSType") != "linux" or any(
        "rootless" in item or "userns" in item
        for item in info.get("SecurityOptions", [])
    ):
        msg = "Rootful Linux Docker without user namespace remapping is required"
        raise RuntimeError(msg)


def firewall(pid: int, proxy: bool, env: dict[str, str]) -> None:
    """Install deny-first OUTPUT policy using host binaries in only the target netns."""
    prefix = ["nsenter", "--target", str(pid), "--net"]
    for binary in ("ip6tables", "iptables"):
        run([*prefix, binary, "-w", "-P", "OUTPUT", "DROP"], env)
    rule = [*prefix, "iptables", "-w", "-A", "OUTPUT"]
    run(
        [*rule, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        env,
    )
    if not proxy:
        for address, port in (("172.30.14.3", "8080"), ("127.0.0.1", "9222")):
            run(
                [*rule, "-p", "tcp", "-d", address, "--dport", port, "-j", "ACCEPT"],
                env,
            )
        return
    for protocol in ("udp", "tcp"):
        run(
            [
                *rule,
                "-p",
                protocol,
                "-d",
                "127.0.0.11",
                "-m",
                "conntrack",
                "--ctorigdst",
                "127.0.0.11",
                "--ctorigdstport",
                "53",
                "-j",
                "ACCEPT",
            ],
            env,
        )
    for network in DENIED:
        run([*rule, "-d", network, "-j", "DROP"], env)
    run(
        [*rule, "-p", "tcp", "-m", "multiport", "--dports", "80,443", "-j", "ACCEPT"],
        env,
    )


def start_gated(service: str, compose: list[str], env: dict[str, str]) -> int:
    """Start only the waiting supervisor and obtain its host PID."""
    run([*compose, "start", service], env)
    identifier = run([*compose, "ps", "-q", service], env)
    if not identifier or "\n" in identifier:
        msg = f"Expected one running container for {service}"
        raise RuntimeError(msg)
    state = json.loads(
        run(["docker", "inspect", "--format", "{{json .State}}", identifier], env)
    )
    if (
        not state.get("Running")
        or not isinstance(state.get("Pid"), int)
        or state["Pid"] <= 0
    ):
        msg = f"No running namespace for {service}"
        raise RuntimeError(msg)
    return state["Pid"]


def runtime_files(runtime: Path) -> None:
    """Create a fresh credential and separate empty readiness mounts."""
    runtime.chmod(0o700)
    token = runtime / "service-token"
    descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_urlsafe(32))
        os.fchown(stream.fileno(), 1000, 1000)
    for name in ("steel-network", "egress-network"):
        (runtime / name).mkdir(mode=0o755)
        (runtime / name).chmod(0o755)


def deploy(home: Path, env_file: Path) -> None:
    """Own one complete create, gate, run, and teardown lifecycle."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COMPOSE_")
    }
    env["HOME"] = str(home)
    env["TALON_ENV_FILE"] = str(env_file)
    preflight(env)
    with open("/run/talon-browser-deploy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(
            prefix="talon-browser-", dir="/dev/shm"
        ) as directory:
            runtime = Path(directory)
            env["BROWSER_RUNTIME_DIR"] = directory
            compose = [
                "docker",
                "compose",
                "--project-name",
                "talon-browser",
                "--env-file",
                str(env_file),
                "-f",
                str(BASE / "docker-compose.yml"),
                "-f",
                str(BASE / "browser/compose.yml"),
            ]
            try:
                runtime_files(runtime)
                run([*compose, "down", "--timeout", "40"], env)
                run([*compose, "up", "--no-start", "--build", "--force-recreate"], env)
                steel = start_gated("steel", compose, env)
                firewall(steel, False, env)
                proxy = start_gated("browser-egress", compose, env)
                firewall(proxy, True, env)
                for name in ("egress-network", "steel-network"):
                    (runtime / name / "ready").touch(mode=0o444)
                run([*compose, "up", "--no-recreate", "--abort-on-container-exit"], env)
            finally:
                for name in ("steel-network", "egress-network"):
                    (runtime / name / "ready").unlink(missing_ok=True)
                run([*compose, "down", "--timeout", "40"], env)


def main() -> int:
    """Require explicit user paths rather than accidentally mounting root's home."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.home.is_absolute() or not args.home.is_dir():
        parser.error("--home must be an existing absolute directory")
    if not args.env_file.is_absolute() or not args.env_file.is_file():
        parser.error("--env-file must be an existing absolute file")

    def stop(number: int, frame: object) -> None:
        for event in (signal.SIGINT, signal.SIGTERM):
            signal.signal(event, signal.SIG_IGN)
        raise SystemExit(128 + number)

    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, stop)
    try:
        deploy(args.home, args.env_file)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"Browser deployment failed: {type(error).__name__}; check host prerequisites and container state",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
