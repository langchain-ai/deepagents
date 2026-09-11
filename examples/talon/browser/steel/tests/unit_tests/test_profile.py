"""Network-free profile rejection tests."""

import importlib.util
import signal
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "steel_profile", Path(__file__).parents[2] / "profile.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileTests(unittest.TestCase):
    def test_clean_and_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            module.validate(profile)
            state = profile / "Local State"
            state.write_text("{}")
            module.validate(profile)
            state.write_text("broken")
            with self.assertRaises(ValueError):
                module.validate(profile)
            self.assertEqual(state.read_text(), "broken")

    def test_invalid_nested_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for value in ("null", "[]", '"bad"', "42"):
                (Path(directory) / "Local State").write_text(
                    '{"profile":' + value + "}"
                )
                with self.assertRaisesRegex(ValueError, "profile_json_invalid"):
                    module.validate(Path(directory))

    def test_readiness(self) -> None:
        with (
            patch.dict(module.os.environ, {"STEEL_REQUIRE_EGRESS_READY": "true"}),
            patch.object(module.Path, "is_file", return_value=False),
            patch.object(module.time, "monotonic", side_effect=[0, 61]),
            self.assertRaisesRegex(ValueError, "profile_egress_ready_timeout"),
        ):
            module.wait_for_egress(threading.Event())
        with (
            patch.dict(module.os.environ, {"STEEL_REQUIRE_EGRESS_READY": "false"}),
            patch.object(module.Path, "is_file", side_effect=AssertionError),
        ):
            module.wait_for_egress(threading.Event())

    def test_signal_during_spawn(self) -> None:
        handlers = {}
        child = Mock()
        child.wait.return_value = 0

        def spawn(command):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return child

        with (
            patch.object(
                module.signal,
                "signal",
                side_effect=lambda number, handler: handlers.update({number: handler}),
            ),
            patch.object(module, "prepare", return_value=7),
            patch.object(module, "wait_for_egress"),
            patch.object(module.subprocess, "Popen", side_effect=spawn),
            patch.object(module.os, "getuid", return_value=1000),
            patch.object(module.os, "close"),
        ):
            self.assertEqual(module.main(), 0)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        child.wait.assert_called_once()

    def test_dirty_and_singleton(self) -> None:
        for name in (".talon-dirty", "SingletonLock"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                profile = Path(directory)
                (profile / name).touch()
                with self.assertRaises(ValueError):
                    module.validate(profile)
                self.assertTrue((profile / name).exists())

    def test_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            cookies = profile / "Cookies"
            with sqlite3.connect(cookies) as database:
                database.execute("CREATE TABLE cookies (value TEXT)")
            module.validate(profile)
            cookies.write_bytes(b"corrupt database")
            with self.assertRaises(sqlite3.DatabaseError):
                module.validate(profile)
            self.assertEqual(cookies.read_bytes(), b"corrupt database")

    def test_unclean_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            (profile / "Default").mkdir()
            (profile / "Default/Preferences").write_text(
                '{"profile":{"exit_type":"Crashed"}}'
            )
            with self.assertRaises(ValueError):
                module.validate(profile)


if __name__ == "__main__":
    unittest.main()
