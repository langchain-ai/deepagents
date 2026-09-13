"""Network-free deployment lifecycle and fail-closed policy checks."""

import importlib.util
import io
from contextlib import ExitStack
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "deploy", Path(__file__).parents[2] / "deploy.py"
)
assert SPEC and SPEC.loader
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class DeployTests(unittest.TestCase):
    """Exercise the real lifecycle through a fake subprocess boundary."""

    def lifecycle(
        self, failure: str = "", *, local_viewer: bool = False
    ) -> tuple[list[list[str]], bytes]:
        """Assert marker visibility and credential destruction on each exit path."""
        commands = []
        runtime = None
        credential = b""

        def execute(
            command: list[str], **kwargs: object
        ) -> subprocess.CompletedProcess:
            nonlocal runtime, credential
            commands.append(command)
            env = kwargs["env"]
            self.assertEqual(
                env["TALON_BROWSER_LOCAL_VIEWER"], str(local_viewer).lower()
            )
            self.assertEqual(
                env["TALON_BROWSER_VIEWER_TOKEN_FILE"],
                "/run/browser/viewer-token" if local_viewer else "",
            )
            runtime = Path(env["BROWSER_RUNTIME_DIR"])
            token = runtime / "service-token"
            if token.exists():
                credential = token.read_bytes()
                self.assertEqual(token.stat().st_mode & 0o777, 0o400)
                self.assertEqual(token.stat().st_uid, 1000)
                self.assertEqual(runtime.stat().st_mode & 0o777, 0o700)
                self.assertNotIn(credential.decode(), str(env))
                self.assertNotIn(credential.decode(), str(command))
            markers = [
                runtime / name / "ready" for name in ("steel-network", "egress-network")
            ]
            if "nsenter" in command or "start" in command or "--no-start" in command:
                self.assertTrue(all(not marker.exists() for marker in markers))
            if "--abort-on-container-exit" in command:
                self.assertTrue(all(marker.exists() for marker in markers))
                policies = [
                    item for item in commands if "OUTPUT" in item and "-P" in item
                ]
                self.assertEqual(len(policies), 4)
                if failure == "signal":
                    raise SystemExit(128 + signal.SIGTERM)
            if "down" in command:
                self.assertTrue(all(not marker.exists() for marker in markers))
                self.assertNotIn("-v", command)
            output = ""
            if "ps" in command:
                output = "container-id"
            if "inspect" in command:
                output = json.dumps({"Running": True, "Pid": 4321})
            failed = (failure == "firewall" and "80,443" in command) or (
                failure == "teardown" and "down" in command and len(commands) > 1
            )
            return subprocess.CompletedProcess(
                command, int(failed), output, "secret must not leak"
            )

        with tempfile.TemporaryDirectory() as root:
            real_temporary = tempfile.TemporaryDirectory
            with (
                patch.object(deploy, "preflight"),
                patch.object(deploy, "start_local_viewer") as viewer,
                patch.object(deploy.subprocess, "run", side_effect=execute),
                patch.object(
                    deploy,
                    "open",
                    create=True,
                    side_effect=lambda *args: open(Path(root) / "lock", "a"),
                ),
                patch.object(
                    deploy.tempfile,
                    "TemporaryDirectory",
                    side_effect=lambda **kwargs: real_temporary(dir=root),
                ),
            ):
                if failure:
                    with self.assertRaises(
                        SystemExit if failure == "signal" else RuntimeError
                    ):
                        deploy.deploy(
                            Path(root), Path(root) / ".env", local_viewer=local_viewer
                        )
                else:
                    deploy.deploy(
                        Path(root), Path(root) / ".env", local_viewer=local_viewer
                    )
                self.assertEqual(viewer.call_count, int(local_viewer))
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime.exists())
        self.assertIn("down", commands[-1])
        return commands, credential

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_startup_and_rotation(self) -> None:
        """Applications see readiness only after both policies, and secrets rotate."""
        commands, first = self.lifecycle()
        _, second = self.lifecycle()
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(len(first), 43)
        create = next(
            i for i, command in enumerate(commands) if "--no-start" in command
        )
        start = next(i for i, command in enumerate(commands) if "start" in command)
        self.assertLess(create, start)

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_firewall_failure_never_releases_gate(self) -> None:
        """A failed rule tears everything down without releasing either application."""
        commands, _ = self.lifecycle("firewall")
        self.assertFalse(
            any("--abort-on-container-exit" in command for command in commands)
        )

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_signal_cleans_secrets_and_containers(self) -> None:
        """Interruption after release still removes both markers and the secret."""
        self.lifecycle("signal")

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_teardown_failure_still_removes_secrets(self) -> None:
        """Docker failure cannot preserve host credentials or readiness files."""
        self.lifecycle("teardown")

    def test_firewall_has_no_management_or_ipv6_bypass(self) -> None:
        """Only the proxy and API-to-CDP new TCP flows are allowed for Steel."""
        with patch.object(deploy, "run", return_value="") as runner:
            deploy.firewall(4321, False, {})
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(commands[0][-4:], ["-w", "-P", "OUTPUT", "DROP"])
        accepted = [command for command in commands if "--dport" in command]
        self.assertEqual(
            [command[command.index("--dport") + 1] for command in accepted],
            ["8080", "9222"],
        )
        self.assertFalse(
            any("3000" in command or "8081" in command for command in commands)
        )
        self.assertTrue(
            all(
                command[:5] == ["nsenter", "--target", "4321", "--net", command[4]]
                for command in commands
            )
        )

    def test_proxy_drops_metadata_before_web_allow(self) -> None:
        """Private/reserved destinations cannot reach the general web-port allow."""
        with patch.object(deploy, "run", return_value="") as runner:
            deploy.firewall(4321, True, {})
        commands = [call.args[0] for call in runner.call_args_list]
        allow = next(i for i, command in enumerate(commands) if "80,443" in command)
        for network in (
            "169.254.0.0/16",
            "168.63.129.16/32",
            "100.64.0.0/10",
            "240.0.0.0/4",
        ):
            index = next(i for i, command in enumerate(commands) if network in command)
            self.assertLess(index, allow)
            self.assertEqual(commands[index][-1], "DROP")
        dns = [command for command in commands if "--ctorigdstport" in command]
        self.assertEqual(len(dns), 2)
        self.assertTrue(all("127.0.0.11" in command for command in dns))

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_viewer_credentials_are_separate(self) -> None:
        """Both files satisfy the reader contract without sharing credentials."""
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            deploy.runtime_files(runtime)
            service = (runtime / "service-token").read_text()
            viewer = runtime / "viewer-token"
            self.assertRegex(viewer.read_text(), r"^[A-Za-z0-9_-]{43}$")
            self.assertNotEqual(service, viewer.read_text())
            self.assertEqual(viewer.stat().st_mode & 0o777, 0o400)
            self.assertEqual(viewer.stat().st_uid, 1000)

    @unittest.skipUnless(os.geteuid() == 0, "credential ownership requires root")
    def test_opt_in_lifecycle(self) -> None:
        """Only explicit opt-in activates local viewer deployment settings."""
        self.lifecycle(local_viewer=True)
        self.lifecycle("signal", local_viewer=True)

    def test_password_is_written_only_to_tty(self) -> None:
        """The password is not placed in the URL, stdout, or stderr."""
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            password = "v" * 43
            (runtime / "viewer-token").write_text(password)
            terminal = io.StringIO()
            with (
                ExitStack() as stack,
                patch.object(deploy, "open", create=True, return_value=terminal),
                patch.object(terminal, "isatty", return_value=True),
                patch.object(deploy.importlib.util, "module_from_spec") as module,
                patch.object(deploy.importlib.util, "spec_from_file_location"),
                patch.object(deploy.sys, "stdout", new_callable=io.StringIO) as stdout,
                patch.object(deploy.sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                deploy.start_local_viewer(stack, runtime)
                self.assertEqual(
                    terminal.getvalue(),
                    f"Local browser: http://127.0.0.1:8765\nLaunch password: {password}\n",
                )
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), "")
                module.return_value.LocalRelay.return_value.__enter__.assert_called_once()

    def test_viewer_requires_controlling_tty(self) -> None:
        """Missing or redirected terminals fail before loading the relay."""
        for terminal in (OSError("no tty"), io.StringIO()):
            with (
                ExitStack() as stack,
                patch.object(deploy, "open", create=True) as opener,
                patch.object(
                    deploy.importlib.util, "spec_from_file_location"
                ) as loader,
            ):
                if isinstance(terminal, OSError):
                    opener.side_effect = terminal
                else:
                    opener.return_value = terminal
                with self.assertRaises((OSError, RuntimeError)):
                    deploy.start_local_viewer(stack, Path("/unused"))
                loader.assert_not_called()
                opener.assert_called_once_with("/dev/tty", "w")

    def test_command_errors_do_not_disclose_output(self) -> None:
        """Compose output may contain user credentials and must not reach errors."""
        result = subprocess.CompletedProcess(
            ["docker"], 1, "token-value", "token-value"
        )
        with (
            patch.object(deploy.subprocess, "run", return_value=result),
            self.assertRaisesRegex(RuntimeError, "Deployment command failed") as caught,
        ):
            deploy.run(["docker"], {})
        self.assertNotIn("token-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
