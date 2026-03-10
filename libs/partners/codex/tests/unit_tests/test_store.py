import json
import time

import pytest

from deepagents_codex.store import CodexAuthStore, CodexCredentials


@pytest.fixture
def tmp_store(tmp_path):
    return CodexAuthStore(path=tmp_path / "codex.json")


@pytest.fixture
def sample_creds():
    return CodexCredentials(
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        expires_at=time.time() + 3600,
        user_email="test@example.com",
    )


class TestCodexCredentials:
    def test_is_expired_false(self) -> None:
        creds = CodexCredentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() + 3600,
        )
        assert creds.is_expired is False

    def test_is_expired_true(self) -> None:
        creds = CodexCredentials(
            access_token="tok",
            refresh_token="ref",
            expires_at=time.time() - 100,
        )
        assert creds.is_expired is True


class TestCodexAuthStore:
    def test_load_missing_file(self, tmp_store) -> None:
        assert tmp_store.load() is None

    def test_save_and_load(self, tmp_store, sample_creds) -> None:
        tmp_store.save(sample_creds)
        loaded = tmp_store.load()
        assert loaded is not None
        assert loaded.access_token == sample_creds.access_token
        assert loaded.refresh_token == sample_creds.refresh_token
        assert loaded.user_email == sample_creds.user_email

    def test_file_permissions(self, tmp_store, sample_creds) -> None:
        tmp_store.save(sample_creds)
        stat = tmp_store.path.stat()
        assert stat.st_mode & 0o777 == 0o600

    def test_directory_permissions(self, tmp_store, sample_creds) -> None:
        tmp_store.save(sample_creds)
        stat = tmp_store.path.parent.stat()
        assert stat.st_mode & 0o777 == 0o700

    def test_delete_existing(self, tmp_store, sample_creds) -> None:
        tmp_store.save(sample_creds)
        assert tmp_store.delete() is True
        assert tmp_store.load() is None

    def test_delete_missing(self, tmp_store) -> None:
        assert tmp_store.delete() is False

    def test_load_corrupt_json(self, tmp_store) -> None:
        tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_store.path.write_text("not valid json{{{", encoding="utf-8")
        assert tmp_store.load() is None

    def test_load_missing_fields(self, tmp_store) -> None:
        tmp_store.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_store.path.write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")
        assert tmp_store.load() is None

    def test_save_creates_parent_dirs(self, tmp_path, sample_creds) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "codex.json"
        store = CodexAuthStore(path=deep_path)
        store.save(sample_creds)
        assert store.load() is not None
