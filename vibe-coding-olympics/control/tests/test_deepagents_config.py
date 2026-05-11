from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from control_server.deepagents_config import clear_recent_model


class TestDeepAgentsConfig(unittest.TestCase):
    def test_clear_recent_model_removes_only_models_recent(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        "[models]",
                        'default = "provider:model-a"',
                        'recent = "provider:model-b"',
                        "",
                        "[models.providers.openai]",
                        'recent = "provider-local-value"',
                        'models = ["gpt-5.2"]',
                        "",
                        "[agents]",
                        'recent = "coder"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            cleared = clear_recent_model(path)

            self.assertTrue(cleared)
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("recent", data["models"])
            self.assertEqual(
                data["models"]["providers"]["openai"]["recent"],
                "provider-local-value",
            )
            self.assertEqual(data["agents"]["recent"], "coder")

    def test_clear_recent_model_is_noop_when_absent(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "config.toml"
            content = '[models]\ndefault = "provider:model-a"\n'
            path.write_text(content, encoding="utf-8")

            cleared = clear_recent_model(path)

            self.assertTrue(cleared)
            self.assertEqual(path.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
