"""Tests for the `just-bash` sandbox backend."""

from __future__ import annotations

import base64

from langchain_quickjs import JustBashSandbox


class FakeJustBashClient:
    """Small in-memory bridge fake for backend contract tests."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def request(self, payload):
        op = payload["op"]
        if op == "upload":
            for item in payload["files"]:
                self.files[item["path"]] = base64.b64decode(item["content"])
            return {
                "ok": True,
                "responses": [
                    {"path": item["path"], "error": None}
                    for item in payload["files"]
                ],
            }
        if op == "download":
            responses = []
            for path in payload["paths"]:
                content = self.files.get(path)
                responses.append(
                    {
                        "path": path,
                        "content": (
                            base64.b64encode(content).decode("ascii")
                            if content is not None
                            else None
                        ),
                        "error": None if content is not None else "file_not_found",
                    }
                )
            return {"ok": True, "responses": responses}
        if op == "files":
            return {
                "ok": True,
                "files": [
                    {
                        "path": path,
                        "content": base64.b64encode(content).decode("ascii"),
                        "stat": {"mtime": "2026-01-01T00:00:00+00:00"},
                    }
                    for path, content in self.files.items()
                ],
            }
        if op == "execute":
            command = payload["command"]
            if command.startswith("echo ") and " > " in command:
                text, path = command.removeprefix("echo ").split(" > ", 1)
                self.files[path] = text.strip('"').encode("utf-8") + b"\n"
                return {"ok": True, "output": "", "exitCode": 0}
            if command.startswith("cat "):
                path = command.removeprefix("cat ")
                return {
                    "ok": True,
                    "output": self.files.get(path, b"").decode("utf-8"),
                    "exitCode": 0,
                }
            return {"ok": True, "output": "ran", "exitCode": 0}
        msg = f"unknown op: {op}"
        raise AssertionError(msg)


def make_sandbox() -> JustBashSandbox:
    """Create a sandbox with the fake bridge."""
    return JustBashSandbox(_client=FakeJustBashClient())


def test_write_file_is_visible_to_execute() -> None:
    sandbox = make_sandbox()

    assert sandbox.write("/work/hello.txt", "hello").error is None
    result = sandbox.execute("cat /work/hello.txt")

    assert result.exit_code == 0
    assert result.output == "hello"


def test_execute_created_file_is_visible_to_read_file() -> None:
    sandbox = make_sandbox()

    result = sandbox.execute('echo "hello" > /work/out.txt')
    read = sandbox.read("/work/out.txt")

    assert result.exit_code == 0
    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "hello\n"


def test_edit_updates_virtual_filesystem() -> None:
    sandbox = make_sandbox()

    sandbox.write("/src/app.py", "print('old')\n")
    edit = sandbox.edit("/src/app.py", "old", "new")
    read = sandbox.read("/src/app.py")

    assert edit.error is None
    assert edit.occurrences == 1
    assert read.file_data is not None
    assert read.file_data["content"] == "print('new')\n"


def test_glob_grep_and_ls_use_bridge_files() -> None:
    sandbox = make_sandbox()

    sandbox.write("/src/app.py", "needle\n")
    sandbox.write("/src/readme.md", "notes\n")

    ls = sandbox.ls("/src")
    glob = sandbox.glob("*.py", "/src")
    grep = sandbox.grep("needle", "/src")

    assert ls.entries is not None
    assert [entry["path"] for entry in ls.entries] == ["/src/app.py", "/src/readme.md"]
    assert glob.matches is not None
    assert [match["path"] for match in glob.matches] == ["/src/app.py"]
    assert grep.matches == [{"path": "/src/app.py", "line": 1, "text": "needle"}]
