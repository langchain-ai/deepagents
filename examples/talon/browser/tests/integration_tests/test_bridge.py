"""Exercise fail-closed bridge routes over real loopback HTTP connections."""

import http.client
import importlib.util
import json
import threading
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "browser_bridge", Path(__file__).resolve().parents[2] / "main.py"
)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
TOKEN = "synthetic-test-token"


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = bridge.Bridge(
            ("127.0.0.1", 0), control=True, token=TOKEN, steel="http://127.0.0.1:1"
        )
        self.addCleanup(self.server.server_close)
        worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 5)
        self.addCleanup(self.server.shutdown)

    def check_response(
        self, path: str, headers: list[tuple[str, str]], status: int, error: str
    ) -> None:
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.putrequest("GET", path)
            for name, value in headers:
                connection.putheader(name, value)
            connection.endheaders()
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, status)
            self.assertEqual(json.loads(body), {"error": error})
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(response.getheader("Connection"), "close")
            self.assertNotIn(TOKEN.encode(), body)
        finally:
            connection.close()

    def test_malformed_nonascii_and_duplicate_authorization(self) -> None:
        valid = ("Authorization", "Bearer " + TOKEN)
        cases = [
            [("Authorization", value)]
            for value in ("", "Bearer", "Basic " + TOKEN, "Bearer wrong", "Bearer \xff")
        ]
        cases.extend(
            (
                [valid, valid],
                [valid, ("authorization", "Bearer wrong")],
                [("authorization", "Bearer wrong"), valid],
            )
        )
        for headers in cases:
            with self.subTest(headers=headers):
                self.check_response(
                    "/internal/browser/status", headers, 401, "unauthorized"
                )

    def test_unknown_routes_and_upgrades_do_not_disclose(self) -> None:
        for control in (True, False):
            self.server.control = control
            for path in ("/unknown?token=" + TOKEN, "/internal/browser/status"):
                with self.subTest(control=control, path=path):
                    self.check_response(
                        path,
                        [
                            ("Authorization", "Bearer " + TOKEN),
                            ("Connection", "Upgrade"),
                            ("Upgrade", "unknown-protocol"),
                        ],
                        404,
                        "not_found",
                    )
            self.check_response("/unknown?token=" + TOKEN, [], 404, "not_found")


if __name__ == "__main__":
    unittest.main()
