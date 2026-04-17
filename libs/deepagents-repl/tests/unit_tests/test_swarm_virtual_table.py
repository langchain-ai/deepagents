"""Port of ``libs/deepagents/src/swarm/virtual-table.test.ts``."""

from __future__ import annotations

from typing import Any

from deepagents.backends.protocol import BackendProtocol, GlobResult

from deepagents_repl._swarm import VirtualTableInput, resolve_virtual_table_tasks


class _MockBackend(BackendProtocol):
    """Minimal backend: only implements ``aglob``; rest is not exercised."""

    def __init__(self, files: dict[str, list[dict[str, str]]] | None = None) -> None:
        self._files = files or {}
        self._errors: dict[str, str] = {}

    def set_error(self, pattern: str, error: str) -> None:
        self._errors[pattern] = error

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        if pattern in self._errors:
            return GlobResult(error=self._errors[pattern])
        if pattern in self._files:
            return GlobResult(matches=[{"path": f["path"]} for f in self._files[pattern]])
        return GlobResult(matches=[])

    # Unused in tests but required to satisfy the abstract base.
    def ls(self, path: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write(self, file_path: str, content: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def edit(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def grep(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def glob(self, pattern: str, path: str = "/") -> Any:  # pragma: no cover
        raise NotImplementedError

    def upload_files(self, files: list) -> Any:  # pragma: no cover
        raise NotImplementedError

    def download_files(self, paths: list) -> Any:  # pragma: no cover
        raise NotImplementedError


class TestExplicitFilePaths:
    async def test_creates_one_task_per_file(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Classify this file",
                file_paths=["data/a.txt", "data/b.txt"],
            ),
            _MockBackend(),
        )
        assert result.error is None
        assert result.tasks is not None
        assert len(result.tasks) == 2
        assert result.tasks[0].description == "Classify this file\n\nFile: data/a.txt"
        assert result.tasks[1].description == "Classify this file\n\nFile: data/b.txt"

    async def test_uses_basename_as_id(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Summarize", file_paths=["data/report.txt"]),
            _MockBackend(),
        )
        assert result.tasks is not None
        assert result.tasks[0].id == "report.txt"

    async def test_disambiguates_basename_collisions(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Translate",
                file_paths=["en/readme.md", "fr/readme.md"],
            ),
            _MockBackend(),
        )
        assert result.tasks is not None
        ids = {t.id for t in result.tasks}
        assert "en-readme.md" in ids
        assert "fr-readme.md" in ids

    async def test_deduplicates(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Analyze",
                file_paths=["data/a.txt", "data/a.txt", "data/a.txt"],
            ),
            _MockBackend(),
        )
        assert result.tasks is not None
        assert len(result.tasks) == 1

    async def test_empty_file_paths_returns_error(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Analyze", file_paths=[]),
            _MockBackend(),
        )
        assert result.error is not None
        assert "No files matched" in result.error


class TestGlob:
    async def test_single_pattern(self) -> None:
        backend = _MockBackend(
            {"feedback/*.txt": [{"path": "feedback/001.txt"}, {"path": "feedback/002.txt"}]}
        )
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Classify", glob="feedback/*.txt"),
            backend,
        )
        assert result.tasks is not None
        assert len(result.tasks) == 2

    async def test_multiple_patterns(self) -> None:
        backend = _MockBackend(
            {
                "feedback/*.txt": [{"path": "feedback/001.txt"}],
                "reports/*.csv": [{"path": "reports/q1.csv"}],
            }
        )
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Analyze", glob=["feedback/*.txt", "reports/*.csv"]
            ),
            backend,
        )
        assert result.tasks is not None
        assert len(result.tasks) == 2

    async def test_deduplicates_across_globs(self) -> None:
        backend = _MockBackend(
            {
                "data/*.txt": [{"path": "data/a.txt"}, {"path": "data/b.txt"}],
                "data/a.*": [{"path": "data/a.txt"}],
            }
        )
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Read", glob=["data/*.txt", "data/a.*"]),
            backend,
        )
        assert result.tasks is not None
        assert len(result.tasks) == 2

    async def test_no_matches_returns_error(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Read", glob="nothing/*.txt"),
            _MockBackend(),
        )
        assert result.error is not None
        assert "No files matched" in result.error
        assert "nothing/*.txt" in result.error

    async def test_glob_error_propagates(self) -> None:
        backend = _MockBackend()
        backend.set_error("secret/*", "permission denied")
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Read", glob="secret/*"),
            backend,
        )
        assert result.error is not None
        assert "permission denied" in result.error


class TestCombined:
    async def test_merges_paths_and_globs(self) -> None:
        backend = _MockBackend({"feedback/*.txt": [{"path": "feedback/001.txt"}]})
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Process",
                file_paths=["extra/manual.txt"],
                glob="feedback/*.txt",
            ),
            backend,
        )
        assert result.tasks is not None
        assert len(result.tasks) == 2
        paths = [t.description.split("\n\nFile: ")[1] for t in result.tasks]
        assert "extra/manual.txt" in paths
        assert "feedback/001.txt" in paths


class TestSubagentType:
    async def test_included_when_provided(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Analyze",
                file_paths=["a.txt"],
                subagent_type="analyst",
            ),
            _MockBackend(),
        )
        assert result.tasks is not None
        assert result.tasks[0].subagent_type == "analyst"

    async def test_none_when_not_provided(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Analyze", file_paths=["a.txt"]),
            _MockBackend(),
        )
        assert result.tasks is not None
        assert result.tasks[0].subagent_type is None


class TestTasksJsonl:
    async def test_returns_valid_jsonl(self) -> None:
        import json

        result = await resolve_virtual_table_tasks(
            VirtualTableInput(instruction="Process", file_paths=["a.txt", "b.txt"]),
            _MockBackend(),
        )
        assert result.tasks_jsonl is not None
        lines = [ln for ln in result.tasks_jsonl.split("\n") if ln.strip()]
        assert len(lines) == 2
        parsed = [json.loads(ln) for ln in lines]
        assert parsed[0]["id"] == "a.txt"
        assert parsed[1]["id"] == "b.txt"


class TestSorting:
    async def test_returns_sorted(self) -> None:
        result = await resolve_virtual_table_tasks(
            VirtualTableInput(
                instruction="Read", file_paths=["z.txt", "a.txt", "m.txt"]
            ),
            _MockBackend(),
        )
        assert result.tasks is not None
        ids = [t.id for t in result.tasks]
        assert ids == ["a.txt", "m.txt", "z.txt"]
