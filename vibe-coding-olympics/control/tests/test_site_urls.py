import os
import unittest
from unittest.mock import patch

from control_server import site_urls


class TestSiteUrls(unittest.TestCase):
    def test_relay_host_derivation(self) -> None:
        env = {"VIBE_PLAYER_3001_RELAY": "http://192.168.1.21:9771"}
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                site_urls.site_url_for("3001"),
                "http://192.168.1.21:3001",
            )

    def test_explicit_override_wins_over_relay(self) -> None:
        env = {
            "VIBE_PLAYER_3001_RELAY": "http://192.168.1.21:9771",
            "VIBE_PLAYER_3001_SITE_URL": "https://alice.fly.dev",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                site_urls.site_url_for("3001"),
                "https://alice.fly.dev",
            )

    def test_returns_none_without_config(self) -> None:
        env = {
            "VIBE_PLAYER_3001_RELAY": "",
            "VIBE_PLAYER_3001_SITE_URL": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertIsNone(site_urls.site_url_for("3001"))

    def test_override_strips_trailing_slash(self) -> None:
        env = {"VIBE_PLAYER_3001_SITE_URL": "https://alice.fly.dev/"}
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                site_urls.site_url_for("3001"),
                "https://alice.fly.dev",
            )

    def test_site_urls_aggregates_configured_ports(self) -> None:
        env = {
            "VIBE_PLAYER_3001_RELAY": "http://192.168.1.21:9771",
            "VIBE_PLAYER_3002_SITE_URL": "https://bob.example/",
            "VIBE_PLAYER_3003_RELAY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            out = site_urls.site_urls(["3001", "3002", "3003"])
        self.assertEqual(
            out,
            {
                "3001": "http://192.168.1.21:3001",
                "3002": "https://bob.example",
            },
        )


if __name__ == "__main__":
    unittest.main()
