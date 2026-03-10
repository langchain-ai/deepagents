"""Tests for the JSON output utility."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from deepagents_cli.output import write_json


class TestWriteJson:
    """Tests for write_json envelope format."""

    def test_envelope_structure(self) -> None:
        """Output has version, command, and data keys."""
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_json("list", [])
        result = json.loads(buf.getvalue())
        assert result == {"version": 1, "command": "list", "data": []}

    def test_list_data(self) -> None:
        """Array data is serialized correctly."""
        buf = StringIO()
        items = [{"name": "a"}, {"name": "b"}]
        with patch("sys.stdout", buf):
            write_json("skills list", items)
        result = json.loads(buf.getvalue())
        assert result["command"] == "skills list"
        assert result["data"] == items

    def test_dict_data(self) -> None:
        """Dict data is serialized correctly."""
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_json("reset", {"agent": "coder", "reset_to": "default"})
        result = json.loads(buf.getvalue())
        assert result["data"]["agent"] == "coder"

    def test_path_serialization(self) -> None:
        """Path objects are serialized via default=str."""
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_json("test", {"path": Path("/tmp/foo")})
        result = json.loads(buf.getvalue())
        assert result["data"]["path"] == "/tmp/foo"

    def test_trailing_newline(self) -> None:
        """Output ends with a single newline."""
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_json("test", {})
        assert buf.getvalue().endswith("\n")
        assert not buf.getvalue().endswith("\n\n")

    def test_single_line(self) -> None:
        """Output is a single line (no pretty-printing)."""
        buf = StringIO()
        with patch("sys.stdout", buf):
            write_json("test", {"a": 1, "b": [2, 3]})
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 1
