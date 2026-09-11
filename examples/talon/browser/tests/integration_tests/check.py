"""Run a credential-free, exclusive live Docker deployment; requires host root."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

BROWSER = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("browser_deploy", BROWSER / "deploy.py")
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def run(command: list[str], env: dict[str, str], timeout: int = 180) -> str:
    """Capture output without exposing runtime credentials."""
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"{command[:2]} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def probe(compose: list[str], env: dict[str, str], service: str, source: str) -> None:
    run([*compose, "exec", "-T", service, "python3", "-c", source], env)


def token_files(runtime: Path) -> None:
    deploy.runtime_files(runtime)
    token = runtime / "service-token"
    assert stat.S_IMODE(token.stat().st_mode) == 0o400
    assert token.stat().st_uid == 1000
    with tempfile.TemporaryDirectory(prefix="jkb89-rotation-", dir="/dev/shm") as other:
        deploy.runtime_files(Path(other))
        assert token.read_bytes() != (Path(other) / "service-token").read_bytes()
    print("PASS token mode 0400, owner 1000, fresh deployment rotation", flush=True)


def configuration(compose: list[str], env: dict[str, str]) -> None:
    services = json.loads(run([*compose, "config", "--format", "json"], env))[
        "services"
    ]
    steel = services["steel"]
    assert steel["user"] == "0:0" and steel["read_only"]
    assert steel["cap_drop"] == ["ALL"]
    assert set(steel["cap_add"]) == {"CHOWN", "SETUID", "SETGID", "SETPCAP"}
    talon = services["talon"]
    assert not talon.get("build") and not talon.get("env_file")
    assert not talon.get("volumes") and not talon.get("environment")
    print(
        "PASS merged Compose: Steel root without FOWNER; Talon has no credentials/mounts/build",
        flush=True,
    )


def wait_healthy(compose: list[str], env: dict[str, str]) -> None:
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        identifiers = run([*compose, "ps", "-aq"], env).splitlines()
        states = json.loads(run(["docker", "inspect", *identifiers], env))
        if any(item["State"]["Status"] in {"exited", "dead"} for item in states):
            summary = [
                (item["Name"], item["State"]["Status"], item["State"]["ExitCode"])
                for item in states
            ]
            raise RuntimeError(f"Container exited: {summary}")
        bridge = [
            item
            for item in states
            if item["Config"]["Labels"]["com.docker.compose.service"]
            == "browser-bridge"
        ]
        if bridge and bridge[0]["State"].get("Health", {}).get("Status") == "healthy":
            print("PASS bridge health (including upstream Steel)", flush=True)
            return
        time.sleep(2)
    raise RuntimeError("Bridge did not become healthy within 150 seconds")


def routes(compose: list[str], env: dict[str, str]) -> None:
    probe(
        compose,
        env,
        "browser-bridge",
        """
import http.client
from pathlib import Path

def check(host, port, path, expected, headers=None):
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request("GET", path, headers=headers or {})
    response = connection.getresponse()
    assert response.status == expected, (host, path, response.status)
    response.read()
    connection.close()

for host, port in (("172.30.12.3", 8081), ("172.30.13.3", 8080)):
    check(host, port, "/health", 200)
    check(host, port, "/unknown", 404)
    check(host, port, "/health", 404, {"Connection": "Upgrade", "Upgrade": "websocket"})
check("172.30.13.3", 8080, "/internal/browser/status", 404)
check("172.30.12.3", 8081, "/internal/browser/status", 401)
token = Path("/run/browser/service-token").read_text().strip()
check("172.30.12.3", 8081, "/internal/browser/status", 200, {"Authorization": "Bearer " + token})
""",
    )
    probe(
        compose,
        env,
        "talon",
        """
import http.client
connection = http.client.HTTPConnection("172.30.12.3", 8081, timeout=5)
connection.request("GET", "/internal/browser/status")
assert connection.getresponse().status == 401
connection.close()
""",
    )
    print(
        "PASS viewer route/upgrade denial; control unauthenticated 401/authenticated 200",
        flush=True,
    )


def isolation(compose: list[str], env: dict[str, str]) -> None:
    deny = """
import socket
for host, port in TARGETS:
    try:
        connection = socket.create_connection((host, port), timeout=2)
    except OSError:
        continue
    connection.close()
    raise AssertionError(("Unexpected direct connection", host, port))
"""
    probe(
        compose,
        env,
        "talon",
        deny.replace("TARGETS", repr([("172.30.14.2", 3000), ("172.30.14.2", 9222)])),
    )
    targets = [
        ("172.30.14.4", 8081),
        ("172.30.12.3", 8081),
        ("169.254.169.254", 80),
        ("1.1.1.1", 443),
        ("127.0.0.11", 53),
        ("1.1.1.1", 53),
    ]
    probe(compose, env, "steel", deny.replace("TARGETS", repr(targets)))
    probe(
        compose,
        env,
        "steel",
        """
import socket
query = b"\\x12\\x34\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x07example\\x03com\\x00\\x00\\x01\\x00\\x01"
for host in ("127.0.0.11", "1.1.1.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(2)
        try:
            connection.sendto(query, (host, 53))
            connection.recv(4096)
        except OSError:
            continue
        raise AssertionError("Unexpected DNS response")
with socket.create_connection(("172.30.14.3", 8080), timeout=5):
    pass
""",
    )
    print(
        "PASS Talon cannot reach Steel; Steel direct private/public/bridge/TCP+UDP DNS denied; proxy reachable",
        flush=True,
    )


def public_navigation(compose: list[str], env: dict[str, str]) -> None:
    addresses = run(
        [
            *compose,
            "exec",
            "-T",
            "browser-egress",
            "python3",
            "-c",
            "import socket, json; print(json.dumps(sorted({item[4][0] for item in socket.getaddrinfo('example.com', 443, type=socket.SOCK_STREAM)})))",
        ],
        env,
    )
    print(f"Public probe DNS answers for example.com: {addresses}", flush=True)
    errors = []
    try:
        probe(
            compose,
            env,
            "steel",
            """
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://172.30.14.3:8080"}))
with opener.open("https://example.com", timeout=30) as response:
    assert response.status == 200
    assert b"Example Domain" in response.read(100000)
""",
        )
    except RuntimeError as error:
        errors.append(str(error))
    else:
        print("PASS public HTTPS through egress proxy", flush=True)
    source = """
const puppeteer = (await import('puppeteer-core')).default;
const browser = await puppeteer.connect({browserURL: 'http://127.0.0.1:9222'});
try {
  const page = await browser.newPage();
  try {
    const response = await page.goto('https://example.com', {waitUntil: 'domcontentloaded', timeout: 45000});
    if (response.status() !== 200 || await page.title() !== 'Example Domain') throw new Error('public_navigation_failed');
  } finally { await page.close(); }
} finally { await browser.disconnect(); }
"""
    try:
        run(
            [
                *compose,
                "exec",
                "-T",
                "steel",
                "node",
                "--input-type=module",
                "-e",
                source,
            ],
            env,
        )
    except RuntimeError as error:
        errors.append(str(error))
    else:
        print("PASS Chromium navigated example.com via enforced proxy", flush=True)
    if errors:
        raise RuntimeError("Public connectivity failures:\n" + "\n".join(errors))


def lifecycle(directory: Path, runtime: Path, env: dict[str, str]) -> None:
    project = "jkb89-" + uuid.uuid4().hex[:12]
    env_file = directory / ".env"
    env_file.touch(mode=0o600)
    env.update(
        HOME=str(directory),
        TALON_ENV_FILE=str(env_file),
        BROWSER_RUNTIME_DIR=str(runtime),
        COMPOSE_PARALLEL_LIMIT="1",
    )
    override = directory / "override.yml"
    override.write_text("""services:
  talon:
    image: talon-browser-stdlib:local
    build: !reset null
    entrypoint: [python3]
    command: ["-c", "import time; time.sleep(3600)"]
    working_dir: /tmp
    env_file: !reset []
    environment: !override {}
    volumes: !override []
""")
    compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        str(env_file),
        "-f",
        str(BROWSER.parent / "docker-compose.yml"),
        "-f",
        str(BROWSER / "compose.yml"),
        "-f",
        str(override),
    ]
    token_files(runtime)
    configuration(compose, env)
    try:
        run(
            [*compose, "build", "steel", "browser-egress", "browser-bridge"],
            env,
            timeout=600,
        )
        image = run(
            [
                "docker",
                "image",
                "inspect",
                "talon-browser-stdlib:local",
                "--format",
                "{{.Id}}",
            ],
            env,
        )
        override.write_text(
            override.read_text().replace("talon-browser-stdlib:local", image)
        )
        run([*compose, "create", "--no-build"], env)
        for service, proxy, gate in (
            ("steel", False, "steel-network"),
            ("browser-egress", True, "egress-network"),
        ):
            pid = deploy.start_gated(service, compose, env)
            assert not (runtime / gate / "ready").exists()
            try:
                deploy.firewall(pid, proxy, env)
            except RuntimeError as error:
                diagnostic = subprocess.run(
                    [
                        "nsenter",
                        "--target",
                        str(pid),
                        "--net",
                        "ip6tables",
                        "-w",
                        "-S",
                        "OUTPUT",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=15,
                )
                raise RuntimeError(
                    f"Real namespace firewall failed for {service}: {diagnostic.stderr.strip()}"
                ) from error
        print("PASS real gated namespace firewalls installed", flush=True)
        for name in ("steel-network", "egress-network"):
            (runtime / name / "ready").touch(mode=0o444)
        run([*compose, "start", "browser-bridge", "talon"], env)
        wait_healthy(compose, env)
        routes(compose, env)
        isolation(compose, env)
        public_navigation(compose, env)
    finally:
        for name in ("steel-network", "egress-network"):
            (runtime / name / "ready").unlink(missing_ok=True)
        run([*compose, "down", "--volumes", "--timeout", "40"], env)
        for resource in ("container", "network", "volume"):
            assert not run(
                [
                    "docker",
                    resource,
                    "ls",
                    "-q",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ],
                env,
            )
        print(
            f"PASS cleanup: no containers/networks/volumes remain for {project}",
            flush=True,
        )


def main() -> None:
    """Serialize fixed-subnet deployments and remove only this test's resources."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COMPOSE_")
    }
    deploy.preflight(env)
    with open("/run/talon-browser-deploy.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if run(["docker", "ps", "-aq"], env):
            raise RuntimeError(
                "Refusing live test while existing containers are present"
            )

        def stop(number: int, frame: object) -> None:
            raise SystemExit(128 + number)

        signal.signal(signal.SIGTERM, stop)
        with (
            tempfile.TemporaryDirectory(prefix="jkb89-", dir="/tmp") as directory,
            tempfile.TemporaryDirectory(prefix="jkb89-", dir="/dev/shm") as runtime,
        ):
            lifecycle(Path(directory), Path(runtime), env)
    print(
        "PASS complete live deployment slice (no controlled redirect or cookie test)",
        flush=True,
    )


if __name__ == "__main__":
    main()
