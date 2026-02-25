"""End-to-end backwards compatibility tests for file format v1 ↔ v2.

Scenarios covered:
1. V1 style writes: write, read, edit, grep, download all work end-to-end
   for both StateBackend and StoreBackend with file_format="v1".
2. V2 mode loading V1 checkpoint data: a backend running in v2 mode can
   seamlessly read/edit/grep/download data that was stored in v1 format
   (e.g. from a restored checkpoint or migrated store).
"""

import warnings

from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import _to_legacy_file_data, create_file_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_runtime():
    return ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t1",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def _make_state_runtime(files=None):
    return ToolRuntime(
        state={"messages": [], "files": files or {}},
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


# ===================================================================
# Scenario 1: V1-style writes still work end-to-end
# ===================================================================


class TestV1StyleWritesStateBackend:
    """Full lifecycle using StateBackend with file_format='v1'."""

    def test_write_read_roundtrip(self):
        """Write a file in v1 mode, then read it back successfully."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        result = be.write("/project/main.py", "import os\nprint('hello')\n")
        assert result.error is None
        assert result.path == "/project/main.py"

        # Verify v1 storage shape: list[str], no encoding key
        fd = result.files_update["/project/main.py"]
        assert isinstance(fd["content"], list)
        assert "encoding" not in fd

        # Simulate state update (as LangGraph would apply it)
        rt2 = _make_state_runtime(files=result.files_update)
        be2 = StateBackend(rt2, file_format="v1")

        read_result = be2.read("/project/main.py")
        assert "import os" in read_result
        assert "print('hello')" in read_result

    def test_write_edit_read_lifecycle(self):
        """Write → edit → read cycle works entirely in v1 mode."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        write_res = be.write("/app.py", "def greet():\n    return 'hi'\n")
        assert write_res.error is None

        # Apply write to state, then edit
        rt2 = _make_state_runtime(files=write_res.files_update)
        be2 = StateBackend(rt2, file_format="v1")

        edit_res = be2.edit("/app.py", "'hi'", "'hello'")
        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify edit result is still v1 format
        fd = edit_res.files_update["/app.py"]
        assert isinstance(fd["content"], list)
        assert "encoding" not in fd

        # Apply edit and read
        rt3 = _make_state_runtime(files=edit_res.files_update)
        be3 = StateBackend(rt3, file_format="v1")

        read_result = be3.read("/app.py")
        assert "'hello'" in read_result
        assert "'hi'" not in read_result

    def test_grep_works_with_v1_data(self):
        """Grep can search through v1-formatted file data."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        write_res = be.write("/src/utils.py", "import sys\ndef helper():\n    pass\nimport os\n")
        assert write_res.error is None

        rt2 = _make_state_runtime(files=write_res.files_update)
        be2 = StateBackend(rt2, file_format="v1")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be2.grep_raw("import", path="/")

        assert isinstance(matches, list)
        assert len(matches) == 2
        paths = {m["text"] for m in matches}
        assert "import sys" in paths
        assert "import os" in paths

    def test_download_works_with_v1_data(self):
        """download_files can retrieve v1-formatted data as bytes."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        write_res = be.write("/data.txt", "line1\nline2\nline3")
        assert write_res.error is None

        rt2 = _make_state_runtime(files=write_res.files_update)
        be2 = StateBackend(rt2, file_format="v1")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be2.download_files(["/data.txt"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"line1\nline2\nline3"

    def test_ls_works_with_v1_data(self):
        """ls_info works correctly with v1-formatted file data."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        write_res = be.write("/dir/file.txt", "content here")
        assert write_res.error is None

        rt2 = _make_state_runtime(files=write_res.files_update)
        be2 = StateBackend(rt2, file_format="v1")

        infos = be2.ls_info("/dir")
        assert len(infos) == 1
        assert infos[0]["path"] == "/dir/file.txt"

    def test_glob_works_with_v1_data(self):
        """glob_info works correctly with v1-formatted file data."""
        rt = _make_state_runtime()
        be = StateBackend(rt, file_format="v1")

        w1 = be.write("/src/a.py", "aaa")
        w2_rt = _make_state_runtime(files=w1.files_update)
        be2 = StateBackend(w2_rt, file_format="v1")
        w2 = be2.write("/src/b.txt", "bbb")

        merged = {**w1.files_update, **w2.files_update}
        rt3 = _make_state_runtime(files=merged)
        be3 = StateBackend(rt3, file_format="v1")

        infos = be3.glob_info("**/*.py", path="/")
        paths = [fi["path"] for fi in infos]
        assert "/src/a.py" in paths
        assert "/src/b.txt" not in paths


class TestV1StyleWritesStoreBackend:
    """Full lifecycle using StoreBackend with file_format='v1'."""

    def test_write_read_roundtrip(self):
        """Write a file in v1 mode, then read it back successfully."""
        rt = _make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("fs",), file_format="v1")

        result = be.write("/project/main.py", "import os\nprint('hello')\n")
        assert result.error is None

        # Verify v1 shape in store
        item = rt.store.get(("fs",), "/project/main.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        # Read back
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            read_result = be.read("/project/main.py")

        assert "import os" in read_result
        assert "print('hello')" in read_result

    def test_write_edit_read_lifecycle(self):
        """Write → edit → read cycle works entirely in v1 mode."""
        rt = _make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("fs",), file_format="v1")

        be.write("/app.py", "def greet():\n    return 'hi'\n")

        edit_res = be.edit("/app.py", "'hi'", "'hello'")
        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify store still has v1 format
        item = rt.store.get(("fs",), "/app.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            read_result = be.read("/app.py")

        assert "'hello'" in read_result
        assert "'hi'" not in read_result

    def test_grep_works_with_v1_data(self):
        """Grep can search through v1-formatted store data."""
        rt = _make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("fs",), file_format="v1")

        be.write("/src/utils.py", "import sys\ndef helper():\n    pass\nimport os\n")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep_raw("import", path="/")

        assert isinstance(matches, list)
        assert len(matches) == 2

    def test_download_works_with_v1_data(self):
        """download_files can retrieve v1-formatted store data as bytes."""
        rt = _make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("fs",), file_format="v1")

        be.write("/data.txt", "line1\nline2\nline3")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.txt"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"line1\nline2\nline3"


# ===================================================================
# Scenario 2: V2 backend loading V1-style checkpoint/store data
# ===================================================================


class TestV2LoadsV1CheckpointStateBackend:
    """V2-mode StateBackend reading data originally stored in v1 format.

    Simulates restoring a checkpoint that was written by a v1-era system.
    """

    def _make_v1_file_data(self, content: str) -> dict:
        """Create a v1-format file data dict (list[str], no encoding)."""
        return _to_legacy_file_data(create_file_data(content))

    def test_read_v1_checkpoint_data(self):
        """V2 backend can read files from a v1-era checkpoint."""
        v1_data = self._make_v1_file_data("hello\nworld")
        rt = _make_state_runtime(files={"/old/file.txt": v1_data})
        be = StateBackend(rt, file_format="v2")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/old/file.txt")
            assert "hello" in result
            assert "world" in result
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_edit_v1_checkpoint_data(self):
        """V2 backend can edit files from a v1-era checkpoint.

        After editing, the result should be in v2 format (str, with encoding).
        """
        v1_data = self._make_v1_file_data("foo\nbar\nbaz")
        rt = _make_state_runtime(files={"/old/code.py": v1_data})
        be = StateBackend(rt, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/old/code.py", "bar", "qux")

        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # The edit result should now be v2 format
        fd = edit_res.files_update["/old/code.py"]
        assert isinstance(fd["content"], str)
        assert fd["encoding"] == "utf-8"
        assert "qux" in fd["content"]
        assert "bar" not in fd["content"]

    def test_grep_v1_checkpoint_data(self):
        """V2 backend can grep through v1-era checkpoint data."""
        v1_data = self._make_v1_file_data("def foo():\n    return 1\ndef bar():\n    return 2")
        rt = _make_state_runtime(files={"/src/funcs.py": v1_data})
        be = StateBackend(rt, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep_raw("def", path="/")

        assert isinstance(matches, list)
        assert len(matches) == 2
        assert matches[0]["text"] == "def foo():"
        assert matches[1]["text"] == "def bar():"

    def test_download_v1_checkpoint_data(self):
        """V2 backend can download v1-era checkpoint data as bytes."""
        v1_data = self._make_v1_file_data("alpha\nbeta\ngamma")
        rt = _make_state_runtime(files={"/data.csv": v1_data})
        be = StateBackend(rt, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.csv"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"alpha\nbeta\ngamma"

    def test_ls_v1_checkpoint_data(self):
        """V2 backend can list v1-era checkpoint data."""
        v1_data = self._make_v1_file_data("some content")
        rt = _make_state_runtime(files={"/dir/file.txt": v1_data})
        be = StateBackend(rt, file_format="v2")

        infos = be.ls_info("/dir")
        assert len(infos) == 1
        assert infos[0]["path"] == "/dir/file.txt"
        # size should still be computed correctly from list content
        assert infos[0]["size"] == len("some content")

    def test_glob_v1_checkpoint_data(self):
        """V2 backend can glob through v1-era checkpoint data."""
        v1_py = self._make_v1_file_data("print('hi')")
        v1_txt = self._make_v1_file_data("notes")
        rt = _make_state_runtime(files={"/src/a.py": v1_py, "/src/b.txt": v1_txt})
        be = StateBackend(rt, file_format="v2")

        infos = be.glob_info("**/*.py", path="/")
        paths = [fi["path"] for fi in infos]
        assert "/src/a.py" in paths
        assert "/src/b.txt" not in paths

    def test_write_new_file_alongside_v1_data(self):
        """V2 backend can write new v2 files alongside v1 checkpoint data."""
        v1_data = self._make_v1_file_data("old content")
        rt = _make_state_runtime(files={"/old/file.txt": v1_data})
        be = StateBackend(rt, file_format="v2")

        result = be.write("/new/file.txt", "new content")
        assert result.error is None

        # New file should be in v2 format
        fd = result.files_update["/new/file.txt"]
        assert isinstance(fd["content"], str)
        assert fd["encoding"] == "utf-8"

    def test_full_lifecycle_v1_checkpoint_to_v2_operations(self):
        """Complete lifecycle: load v1 checkpoint → read → edit → write new → read all.

        This simulates a real upgrade scenario where an agent resumes from
        a checkpoint created by an older v1 system.
        """
        # Step 1: "Restore" v1 checkpoint data
        v1_config = self._make_v1_file_data("DB_HOST=localhost\nDB_PORT=5432")
        v1_code = self._make_v1_file_data("def connect():\n    pass")
        checkpoint_files = {
            "/config.env": v1_config,
            "/src/db.py": v1_code,
        }

        # Step 2: V2 backend reads v1 data
        rt = _make_state_runtime(files=checkpoint_files)
        be = StateBackend(rt)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config_content = be.read("/config.env")
        assert "DB_HOST=localhost" in config_content
        assert "DB_PORT=5432" in config_content

        # Step 3: Edit v1 data (result upgrades to v2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/config.env", "DB_HOST=localhost", "DB_HOST=prod.example.com")
        assert edit_res.error is None

        # Step 4: Write a brand new file in v2
        new_write = be.write("/src/migrations.py", "# migration scripts\n")
        assert new_write.error is None

        # Step 5: Merge everything and verify
        merged_files = {
            **checkpoint_files,
            **edit_res.files_update,
            **new_write.files_update,
        }
        rt2 = _make_state_runtime(files=merged_files)
        be2 = StateBackend(rt2)

        # Edited file is now v2
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edited = be2.read("/config.env")
        assert "prod.example.com" in edited

        # Untouched v1 file still readable
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            code = be2.read("/src/db.py")
        assert "def connect():" in code

        # New v2 file readable
        new_file = be2.read("/src/migrations.py")
        assert "migration scripts" in new_file


class TestV2LoadsV1CheckpointStoreBackend:
    """V2-mode StoreBackend reading data originally stored in v1 format.

    Simulates a store that contains legacy v1 items (e.g. from before
    the format migration).
    """

    def _seed_v1_store_item(self, store, namespace, path, content):
        """Manually put a v1-format item into the store."""
        v1_data = _to_legacy_file_data(create_file_data(content))
        store.put(namespace, path, v1_data)

    def test_read_v1_store_data(self):
        """V2 backend can read v1-format items from the store."""
        rt = _make_store_runtime()
        ns = ("fs",)
        self._seed_v1_store_item(rt.store, ns, "/old/file.txt", "hello\nworld")

        be = StoreBackend(rt, namespace=lambda _ctx: ns, file_format="v2")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/old/file.txt")
            assert "hello" in result
            assert "world" in result
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_edit_v1_store_data(self):
        """V2 backend can edit v1-format items in the store.

        After editing, the stored item should be upgraded to v2 format.
        """
        rt = _make_store_runtime()
        ns = ("fs",)
        self._seed_v1_store_item(rt.store, ns, "/code.py", "foo\nbar\nbaz")

        be = StoreBackend(rt, namespace=lambda _ctx: ns, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/code.py", "bar", "qux")

        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify the store now has v2 format
        item = rt.store.get(ns, "/code.py")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"
        assert "qux" in item.value["content"]

    def test_grep_v1_store_data(self):
        """V2 backend can grep through v1-format store items."""
        rt = _make_store_runtime()
        ns = ("fs",)
        self._seed_v1_store_item(rt.store, ns, "/funcs.py", "def foo():\n    pass\ndef bar():\n    pass")

        be = StoreBackend(rt, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep_raw("def", path="/")

        assert isinstance(matches, list)
        assert len(matches) == 2

    def test_download_v1_store_data(self):
        """V2 backend can download v1-format store data as bytes."""
        rt = _make_store_runtime()
        ns = ("fs",)
        self._seed_v1_store_item(rt.store, ns, "/data.txt", "line1\nline2")

        be = StoreBackend(rt, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.txt"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"line1\nline2"

    def test_full_lifecycle_v1_store_to_v2_operations(self):
        """Complete lifecycle: v1 store data → read → edit → write new → read all."""
        rt = _make_store_runtime()
        ns = ("fs",)

        # Seed v1 data
        self._seed_v1_store_item(rt.store, ns, "/config.env", "DB_HOST=localhost\nDB_PORT=5432")
        self._seed_v1_store_item(rt.store, ns, "/src/db.py", "def connect():\n    pass")

        be = StoreBackend(rt, namespace=lambda _ctx: ns, file_format="v2")

        # Read v1 data
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config = be.read("/config.env")
        assert "DB_HOST=localhost" in config

        # Edit v1 data (upgrades to v2 in store)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/config.env", "DB_HOST=localhost", "DB_HOST=prod.example.com")
        assert edit_res.error is None

        # Write brand new v2 file
        new_write = be.write("/src/migrations.py", "# migration scripts\n")
        assert new_write.error is None

        # Verify edited file is now v2 in store
        item = rt.store.get(ns, "/config.env")
        assert isinstance(item.value["content"], str)
        assert "encoding" in item.value

        # Verify untouched v1 file still readable
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            code = be.read("/src/db.py")
        assert "def connect():" in code

        # Verify new file
        new = be.read("/src/migrations.py")
        assert "migration scripts" in new


# ===================================================================
# Scenario 3: V1 data without encoding field (bare minimum legacy)
# ===================================================================


class TestBareV1DataNoEncodingField:
    """Test with v1 data that has NO encoding field at all.

    This is the most minimal legacy format — just content as list[str]
    plus timestamps. No encoding key present.
    """

    def test_state_backend_reads_bare_v1(self):
        """StateBackend (v2 mode) handles v1 data missing the encoding field."""
        bare_v1 = {
            "content": ["line1", "line2", "line3"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        rt = _make_state_runtime(files={"/legacy.txt": bare_v1})
        be = StateBackend(rt)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/legacy.txt")
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_store_backend_reads_bare_v1(self):
        """StoreBackend (v2 mode) handles v1 data missing the encoding field."""
        rt = _make_store_runtime()
        ns = ("fs",)

        # Manually insert bare v1 data (no encoding key)
        rt.store.put(
            ns,
            "/legacy.txt",
            {
                "content": ["line1", "line2", "line3"],
                "created_at": "2024-06-01T00:00:00+00:00",
                "modified_at": "2024-06-01T00:00:00+00:00",
            },
        )

        be = StoreBackend(rt, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/legacy.txt")
            assert "line1" in result
            assert "line2" in result
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_state_backend_edits_bare_v1(self):
        """Editing bare v1 data upgrades it to v2 with encoding field."""
        bare_v1 = {
            "content": ["old", "content"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        rt = _make_state_runtime(files={"/legacy.txt": bare_v1})
        be = StateBackend(rt, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/legacy.txt", "old", "new")

        assert edit_res.error is None
        fd = edit_res.files_update["/legacy.txt"]
        assert isinstance(fd["content"], str)
        # update_file_data uses .get("encoding", "utf-8") so bare v1 gets default
        assert fd["encoding"] == "utf-8"
        assert "new" in fd["content"]

    def test_state_backend_download_bare_v1(self):
        """Downloading bare v1 data (no encoding key) defaults to utf-8."""
        bare_v1 = {
            "content": ["hello", "world"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        rt = _make_state_runtime(files={"/legacy.txt": bare_v1})
        be = StateBackend(rt)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/legacy.txt"])

        assert responses[0].error is None
        assert responses[0].content == b"hello\nworld"

    def test_store_backend_download_bare_v1(self):
        """Downloading bare v1 store data (no encoding key) defaults to utf-8."""
        rt = _make_store_runtime()
        ns = ("fs",)

        rt.store.put(
            ns,
            "/legacy.txt",
            {
                "content": ["hello", "world"],
                "created_at": "2024-06-01T00:00:00+00:00",
                "modified_at": "2024-06-01T00:00:00+00:00",
            },
        )

        be = StoreBackend(rt, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/legacy.txt"])

        assert responses[0].error is None
        assert responses[0].content == b"hello\nworld"
