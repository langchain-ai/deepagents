"""End-to-end backwards compatibility tests for file format v1 ↔ v2.

Scenarios covered:
1. V1 style writes: write, read, edit, grep, download all work end-to-end
   for StoreBackend with file_format="v1".
2. V2 mode loading V1 checkpoint data: a backend running in v2 mode can
   seamlessly read/edit/grep/download data that was stored in v1 format
   (e.g. from a restored checkpoint or migrated store).
"""

import warnings

from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import ReadResult
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import _to_legacy_file_data, create_file_data

# ===================================================================
# Scenario 1: V1-style writes still work end-to-end
# ===================================================================


class TestV1StyleWritesStoreBackendDirect:
    """Full lifecycle using StoreBackend with file_format='v1' (direct store)."""

    def test_write_read_roundtrip(self):
        """Write a file in v1 mode, then read it back successfully."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        result = be.write("/project/main.py", "import os\nprint('hello')\n")
        assert result.error is None
        assert result.path == "/project/main.py"

        # Verify v1 storage shape: list[str], no encoding key
        item = mem_store.get(("filesystem",), "/project/main.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        # Read back
        read_result = be.read("/project/main.py")
        assert isinstance(read_result, ReadResult)
        assert read_result.file_data is not None
        assert "import os" in read_result.file_data["content"]
        assert "print('hello')" in read_result.file_data["content"]

    def test_write_edit_read_lifecycle(self):
        """Write -> edit -> read cycle works entirely in v1 mode."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        write_res = be.write("/app.py", "def greet():\n    return 'hi'\n")
        assert write_res.error is None

        edit_res = be.edit("/app.py", "'hi'", "'hello'")
        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify stored data is still v1 format
        item = mem_store.get(("filesystem",), "/app.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        # Read back
        read_result = be.read("/app.py")
        assert isinstance(read_result, ReadResult)
        assert read_result.file_data is not None
        assert "'hello'" in read_result.file_data["content"]
        assert "'hi'" not in read_result.file_data["content"]

    def test_grep_works_with_v1_data(self):
        """Grep can search through v1-formatted file data."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        write_res = be.write("/src/utils.py", "import sys\ndef helper():\n    pass\nimport os\n")
        assert write_res.error is None

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep("import", path="/").matches

        assert matches is not None
        assert len(matches) == 2
        paths = {m["text"] for m in matches}
        assert "import sys" in paths
        assert "import os" in paths

    def test_download_works_with_v1_data(self):
        """download_files can retrieve v1-formatted data as bytes."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        write_res = be.write("/data.txt", "line1\nline2\nline3")
        assert write_res.error is None

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.txt"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"line1\nline2\nline3"

    def test_ls_works_with_v1_data(self):
        """ls_info works correctly with v1-formatted file data."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        write_res = be.write("/dir/file.txt", "content here")
        assert write_res.error is None

        infos = be.ls("/dir").entries
        assert infos is not None
        assert len(infos) == 1
        assert infos[0]["path"] == "/dir/file.txt"

    def test_glob_works_with_v1_data(self):
        """Glob works correctly with v1-formatted file data."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v1")

        be.write("/src/a.py", "aaa")
        be.write("/src/b.txt", "bbb")

        infos = be.glob("**/*.py", path="/").matches
        paths = [fi["path"] for fi in infos]
        assert "/src/a.py" in paths
        assert "/src/b.txt" not in paths


class TestV1StyleWritesStoreBackend:
    """Full lifecycle using StoreBackend with file_format='v1'."""

    def test_write_read_roundtrip(self):
        """Write a file in v1 mode, then read it back successfully."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("fs",), file_format="v1")

        result = be.write("/project/main.py", "import os\nprint('hello')\n")
        assert result.error is None

        # Verify v1 shape in store
        item = mem_store.get(("fs",), "/project/main.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        # Read back
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            read_result = be.read("/project/main.py")

        assert isinstance(read_result, ReadResult)
        assert read_result.file_data is not None
        assert "import os" in read_result.file_data["content"]
        assert "print('hello')" in read_result.file_data["content"]

    def test_write_edit_read_lifecycle(self):
        """Write -> edit -> read cycle works entirely in v1 mode."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("fs",), file_format="v1")

        be.write("/app.py", "def greet():\n    return 'hi'\n")

        edit_res = be.edit("/app.py", "'hi'", "'hello'")
        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify store still has v1 format
        item = mem_store.get(("fs",), "/app.py")
        assert isinstance(item.value["content"], list)
        assert "encoding" not in item.value

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            read_result = be.read("/app.py")

        assert isinstance(read_result, ReadResult)
        assert read_result.file_data is not None
        assert "'hello'" in read_result.file_data["content"]
        assert "'hi'" not in read_result.file_data["content"]

    def test_grep_works_with_v1_data(self):
        """Grep can search through v1-formatted store data."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("fs",), file_format="v1")

        be.write("/src/utils.py", "import sys\ndef helper():\n    pass\nimport os\n")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep("import", path="/").matches

        assert matches is not None
        assert len(matches) == 2

    def test_download_works_with_v1_data(self):
        """download_files can retrieve v1-formatted store data as bytes."""
        mem_store = InMemoryStore()
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("fs",), file_format="v1")

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


class TestV2LoadsV1CheckpointStoreBackendDirect:
    """V2-mode StoreBackend reading data originally stored in v1 format.

    Simulates restoring a checkpoint that was written by a v1-era system.
    """

    def _make_v1_file_data(self, content: str) -> dict:
        """Create a v1-format file data dict (list[str], no encoding)."""
        return _to_legacy_file_data(create_file_data(content))

    def test_read_v1_checkpoint_data(self):
        """V2 backend can read files from a v1-era checkpoint."""
        v1_data = self._make_v1_file_data("hello\nworld")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/old/file.txt", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/old/file.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "hello" in result.file_data["content"]
            assert "world" in result.file_data["content"]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_edit_v1_checkpoint_data(self):
        """V2 backend can edit files from a v1-era checkpoint.

        After editing, the result should be in v2 format (str, with encoding).
        """
        v1_data = self._make_v1_file_data("foo\nbar\nbaz")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/old/code.py", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/old/code.py", "bar", "qux")

        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # The stored data should now be v2 format
        item = mem_store.get(("filesystem",), "/old/code.py")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"
        assert "qux" in item.value["content"]
        assert "bar" not in item.value["content"]

    def test_grep_v1_checkpoint_data(self):
        """V2 backend can grep through v1-era checkpoint data."""
        v1_data = self._make_v1_file_data("def foo():\n    return 1\ndef bar():\n    return 2")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/src/funcs.py", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep("def", path="/").matches

        assert matches is not None
        assert len(matches) == 2
        assert matches[0]["text"] == "def foo():"
        assert matches[1]["text"] == "def bar():"

    def test_download_v1_checkpoint_data(self):
        """V2 backend can download v1-era checkpoint data as bytes."""
        v1_data = self._make_v1_file_data("alpha\nbeta\ngamma")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/data.csv", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.csv"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"alpha\nbeta\ngamma"

    def test_ls_v1_checkpoint_data(self):
        """V2 backend can list v1-era checkpoint data."""
        v1_data = self._make_v1_file_data("some content")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/dir/file.txt", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        infos = be.ls("/dir").entries
        assert infos is not None
        assert len(infos) == 1
        assert infos[0]["path"] == "/dir/file.txt"
        # size should still be computed correctly from list content
        assert infos[0]["size"] == len("some content")

    def test_glob_v1_checkpoint_data(self):
        """V2 backend can glob through v1-era checkpoint data."""
        v1_py = self._make_v1_file_data("print('hi')")
        v1_txt = self._make_v1_file_data("notes")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/src/a.py", v1_py)
        mem_store.put(("filesystem",), "/src/b.txt", v1_txt)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        infos = be.glob("**/*.py", path="/").matches
        paths = [fi["path"] for fi in infos]
        assert "/src/a.py" in paths
        assert "/src/b.txt" not in paths

    def test_write_new_file_alongside_v1_data(self):
        """V2 backend can write new v2 files alongside v1 checkpoint data."""
        v1_data = self._make_v1_file_data("old content")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/old/file.txt", v1_data)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        result = be.write("/new/file.txt", "new content")
        assert result.error is None

        # New file should be in v2 format
        item = mem_store.get(("filesystem",), "/new/file.txt")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"

    def test_full_lifecycle_v1_checkpoint_to_v2_operations(self):
        """Complete lifecycle: load v1 checkpoint -> read -> edit -> write new -> read all.

        This simulates a real upgrade scenario where an agent resumes from
        a checkpoint created by an older v1 system.
        """
        # Step 1: "Restore" v1 checkpoint data
        v1_config = self._make_v1_file_data("DB_HOST=localhost\nDB_PORT=5432")
        v1_code = self._make_v1_file_data("def connect():\n    pass")
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/config.env", v1_config)
        mem_store.put(("filesystem",), "/src/db.py", v1_code)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))

        # Step 2: V2 backend reads v1 data
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config_content = be.read("/config.env")
        assert isinstance(config_content, ReadResult)
        assert config_content.file_data is not None
        assert "DB_HOST=localhost" in config_content.file_data["content"]
        assert "DB_PORT=5432" in config_content.file_data["content"]

        # Step 3: Edit v1 data (result upgrades to v2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/config.env", "DB_HOST=localhost", "DB_HOST=prod.example.com")
        assert edit_res.error is None

        # Step 4: Write a brand new file in v2
        new_write = be.write("/src/migrations.py", "# migration scripts\n")
        assert new_write.error is None

        # Step 5: Verify everything via reads

        # Edited file is now v2
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edited = be.read("/config.env")
        assert isinstance(edited, ReadResult)
        assert edited.file_data is not None
        assert "prod.example.com" in edited.file_data["content"]

        # Untouched v1 file still readable
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            code = be.read("/src/db.py")
        assert isinstance(code, ReadResult)
        assert code.file_data is not None
        assert "def connect():" in code.file_data["content"]

        # New v2 file readable
        new_file = be.read("/src/migrations.py")
        assert isinstance(new_file, ReadResult)
        assert new_file.file_data is not None
        assert "migration scripts" in new_file.file_data["content"]


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
        mem_store = InMemoryStore()
        ns = ("fs",)
        self._seed_v1_store_item(mem_store, ns, "/old/file.txt", "hello\nworld")

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns, file_format="v2")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/old/file.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "hello" in result.file_data["content"]
            assert "world" in result.file_data["content"]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_edit_v1_store_data(self):
        """V2 backend can edit v1-format items in the store.

        After editing, the stored item should be upgraded to v2 format.
        """
        mem_store = InMemoryStore()
        ns = ("fs",)
        self._seed_v1_store_item(mem_store, ns, "/code.py", "foo\nbar\nbaz")

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns, file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/code.py", "bar", "qux")

        assert edit_res.error is None
        assert edit_res.occurrences == 1

        # Verify the store now has v2 format
        item = mem_store.get(ns, "/code.py")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"
        assert "qux" in item.value["content"]

    def test_grep_v1_store_data(self):
        """V2 backend can grep through v1-format store items."""
        mem_store = InMemoryStore()
        ns = ("fs",)
        self._seed_v1_store_item(mem_store, ns, "/funcs.py", "def foo():\n    pass\ndef bar():\n    pass")

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            matches = be.grep("def", path="/").matches

        assert matches is not None
        assert len(matches) == 2

    def test_download_v1_store_data(self):
        """V2 backend can download v1-format store data as bytes."""
        mem_store = InMemoryStore()
        ns = ("fs",)
        self._seed_v1_store_item(mem_store, ns, "/data.txt", "line1\nline2")

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/data.txt"])

        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"line1\nline2"

    def test_full_lifecycle_v1_store_to_v2_operations(self):
        """Complete lifecycle: v1 store data -> read -> edit -> write new -> read all."""
        mem_store = InMemoryStore()
        ns = ("fs",)

        # Seed v1 data
        self._seed_v1_store_item(mem_store, ns, "/config.env", "DB_HOST=localhost\nDB_PORT=5432")
        self._seed_v1_store_item(mem_store, ns, "/src/db.py", "def connect():\n    pass")

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns, file_format="v2")

        # Read v1 data
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            config = be.read("/config.env")
        assert isinstance(config, ReadResult)
        assert config.file_data is not None
        assert "DB_HOST=localhost" in config.file_data["content"]

        # Edit v1 data (upgrades to v2 in store)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/config.env", "DB_HOST=localhost", "DB_HOST=prod.example.com")
        assert edit_res.error is None

        # Write brand new v2 file
        new_write = be.write("/src/migrations.py", "# migration scripts\n")
        assert new_write.error is None

        # Verify edited file is now v2 in store
        item = mem_store.get(ns, "/config.env")
        assert isinstance(item.value["content"], str)
        assert "encoding" in item.value

        # Verify untouched v1 file still readable
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            code = be.read("/src/db.py")
        assert isinstance(code, ReadResult)
        assert code.file_data is not None
        assert "def connect():" in code.file_data["content"]

        # Verify new file
        new = be.read("/src/migrations.py")
        assert isinstance(new, ReadResult)
        assert new.file_data is not None
        assert "migration scripts" in new.file_data["content"]


# ===================================================================
# Scenario 3: V1 data without encoding field (bare minimum legacy)
# ===================================================================


class TestBareV1DataNoEncodingField:
    """Test with v1 data that has NO encoding field at all.

    This is the most minimal legacy format -- just content as list[str]
    plus timestamps. No encoding key present.
    """

    def test_store_backend_reads_bare_v1_direct(self):
        """StoreBackend (v2 mode) handles v1 data missing the encoding field (direct store)."""
        bare_v1 = {
            "content": ["line1", "line2", "line3"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/legacy.txt", bare_v1)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/legacy.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "line1" in result.file_data["content"]
            assert "line2" in result.file_data["content"]
            assert "line3" in result.file_data["content"]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_store_backend_reads_bare_v1(self):
        """StoreBackend (v2 mode) handles v1 data missing the encoding field."""
        mem_store = InMemoryStore()
        ns = ("fs",)

        # Manually insert bare v1 data (no encoding key)
        mem_store.put(
            ns,
            "/legacy.txt",
            {
                "content": ["line1", "line2", "line3"],
                "created_at": "2024-06-01T00:00:00+00:00",
                "modified_at": "2024-06-01T00:00:00+00:00",
            },
        )

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = be.read("/legacy.txt")
            assert isinstance(result, ReadResult)
            assert result.file_data is not None
            assert "line1" in result.file_data["content"]
            assert "line2" in result.file_data["content"]
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_store_backend_edits_bare_v1(self):
        """Editing bare v1 data upgrades it to v2 with encoding field."""
        bare_v1 = {
            "content": ["old", "content"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/legacy.txt", bare_v1)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",), file_format="v2")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            edit_res = be.edit("/legacy.txt", "old", "new")

        assert edit_res.error is None
        item = mem_store.get(("filesystem",), "/legacy.txt")
        assert isinstance(item.value["content"], str)
        assert item.value["encoding"] == "utf-8"
        assert "new" in item.value["content"]

    def test_store_backend_download_bare_v1_direct(self):
        """Downloading bare v1 data (no encoding key) defaults to utf-8 (direct store)."""
        bare_v1 = {
            "content": ["hello", "world"],
            "created_at": "2024-06-01T00:00:00+00:00",
            "modified_at": "2024-06-01T00:00:00+00:00",
        }
        mem_store = InMemoryStore()
        mem_store.put(("filesystem",), "/legacy.txt", bare_v1)
        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/legacy.txt"])

        assert responses[0].error is None
        assert responses[0].content == b"hello\nworld"

    def test_store_backend_download_bare_v1(self):
        """Downloading bare v1 store data (no encoding key) defaults to utf-8."""
        mem_store = InMemoryStore()
        ns = ("fs",)

        mem_store.put(
            ns,
            "/legacy.txt",
            {
                "content": ["hello", "world"],
                "created_at": "2024-06-01T00:00:00+00:00",
                "modified_at": "2024-06-01T00:00:00+00:00",
            },
        )

        be = StoreBackend(store=mem_store, namespace=lambda _ctx: ns)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            responses = be.download_files(["/legacy.txt"])

        assert responses[0].error is None
        assert responses[0].content == b"hello\nworld"
