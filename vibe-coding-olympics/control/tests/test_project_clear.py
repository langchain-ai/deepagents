from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from control_server.project_clear import clear_round_project


class TestProjectClear(unittest.TestCase):
    def test_clear_round_project_blanks_vite_app_files(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp)
            src = path / "src"
            src.mkdir()
            (path / "package.json").write_text('{"scripts":{}}\n', encoding="utf-8")
            (path / "index.html").write_text("<h1>old</h1>", encoding="utf-8")
            (src / "main.js").write_text("console.log('old')", encoding="utf-8")
            (src / "style.css").write_text("body{color:red}", encoding="utf-8")
            (src / "counter.js").write_text("old", encoding="utf-8")

            cleared = clear_round_project(path)

            self.assertTrue(cleared)
            index_html = (path / "index.html").read_text(encoding="utf-8")
            self.assertIn("/src/main.js", index_html)
            self.assertIn("Site will be built here...", index_html)
            self.assertIn("LangChain x Interrupt 2026", index_html)
            self.assertIn("ready for build", index_html)
            self.assertNotIn("<h1>old</h1>", index_html)
            self.assertEqual(
                (src / "main.js").read_text(encoding="utf-8"),
                'import "./style.css";\n\nlocalStorage.clear();\nsessionStorage.clear();\n',
            )
            self.assertIn(".terminal", (src / "style.css").read_text(encoding="utf-8"))
            self.assertNotIn(
                "body{color:red}", (src / "style.css").read_text(encoding="utf-8")
            )
            self.assertEqual((src / "counter.js").read_text(encoding="utf-8"), "")

    def test_clear_round_project_rejects_non_vite_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            self.assertFalse(clear_round_project(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
