"""Tests for live approval-mode store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code.approval_mode import (
    APPROVAL_MODE_NAMESPACE,
    AUTO_NOTICE_VERSION,
    YOLO_ACKNOWLEDGEMENT_POLICY_VERSION,
    ApprovalMode,
    approval_mode_key,
    approval_mode_payload,
    aread_approval_mode_from_store,
    awrite_approval_mode,
    has_auto_mode_notice,
    has_yolo_acknowledgement,
    next_approval_mode,
    read_approval_mode_from_store,
    save_auto_mode_notice,
    save_yolo_acknowledgement,
)


@dataclass
class _StoreItem:
    value: object


class _Store:
    def __init__(self, item: object = None) -> None:
        self.item = item

    def get(self, namespace: tuple[str, ...], key: str) -> object:
        assert namespace == APPROVAL_MODE_NAMESPACE
        assert key
        return self.item


class _FailingStore:
    def get(self, namespace: tuple[str, ...], key: str) -> object:
        _ = (namespace, key)
        msg = "store unavailable"
        raise RuntimeError(msg)


class _AsyncOnlyStore:
    def __init__(self, item: object = None) -> None:
        self.item = item

    async def aget(self, namespace: tuple[str, ...], key: str) -> object:
        assert namespace == APPROVAL_MODE_NAMESPACE
        assert key
        return self.item

    def get(self, namespace: tuple[str, ...], key: str) -> object:
        _ = (namespace, key)
        msg = "synchronous Store access is forbidden on the event loop"
        raise AssertionError(msg)


class _AsyncFailingStore:
    async def aget(self, namespace: tuple[str, ...], key: str) -> object:
        _ = (namespace, key)
        msg = "store unavailable"
        raise RuntimeError(msg)


class _Writer:
    def __init__(self) -> None:
        self.items: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        self.items.append((namespace, key, value))


def test_read_approval_mode_from_store_accepts_mapping_item() -> None:
    key = approval_mode_key("thread-1")
    item = {"value": {"mode": "auto"}}

    assert read_approval_mode_from_store(_Store(item), key) is ApprovalMode.AUTO


def test_read_approval_mode_from_store_accepts_attribute_item() -> None:
    key = approval_mode_key("thread-1")
    item = _StoreItem({"mode": "yolo"})

    assert read_approval_mode_from_store(_Store(item), key) is ApprovalMode.YOLO


@pytest.mark.parametrize(
    ("store", "key"),
    [
        (None, approval_mode_key("thread-1")),
        (object(), approval_mode_key("thread-1")),  # store has no get()
        (_Store(None), approval_mode_key("thread-1")),
        (_Store(_StoreItem(["not", "a", "mapping"])), approval_mode_key("thread-1")),
        (_Store(_StoreItem({"auto_approve": "yes"})), approval_mode_key("thread-1")),
        (_Store(_StoreItem({"auto_approve": 1})), approval_mode_key("thread-1")),
        (_Store(_StoreItem({"auto_approve": True})), ""),
        (_Store(_StoreItem({"auto_approve": True})), None),
    ],
)
def test_read_approval_mode_from_store_fails_closed(
    store: object,
    key: str | None,
) -> None:
    assert read_approval_mode_from_store(store, key) is None


def test_read_approval_mode_from_store_non_string_key_fails_closed() -> None:
    """A non-string key still fails closed via the runtime guard.

    The declared `key` type is `str | None`, but the value crosses the
    JSON/RemoteGraph boundary, so the `isinstance` guard remains as
    defense-in-depth against a malformed payload.
    """
    item = _StoreItem({"auto_approve": True})
    assert read_approval_mode_from_store(_Store(item), cast("str", object())) is None


def test_read_approval_mode_from_store_exception_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="deepagents_code.approval_mode"):
        assert (
            read_approval_mode_from_store(
                _FailingStore(),
                approval_mode_key("thread-1"),
            )
            is None
        )

    assert "Could not read approval-mode store item" in caplog.text


async def test_awrite_approval_mode_returns_none_without_writer() -> None:
    assert (
        await awrite_approval_mode(object(), "thread-1", mode=ApprovalMode.AUTO)
    ) is None


def test_auto_mode_notice_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".state" / "approval.json"

    assert not has_auto_mode_notice(path)
    assert save_auto_mode_notice(path)
    assert has_auto_mode_notice(path)


def test_auto_mode_notice_rejects_stale_version(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    path.write_text(
        '{"version":1,"auto_notice_version":"old","auto_notice_shown":true}\n'
    )

    assert not has_auto_mode_notice(path)


def test_auto_mode_notice_rejects_missing_or_corrupt_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "approval.json"
    corrupt = tmp_path / "approval.json"
    corrupt.write_text("not-json\n", encoding="utf-8")

    assert not has_auto_mode_notice(missing)
    assert not has_yolo_acknowledgement(missing)
    assert not has_auto_mode_notice(corrupt)
    assert not has_yolo_acknowledgement(corrupt)


def test_save_fails_open_on_lock_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock-wait timeout returns False without raising (fail open)."""
    from filelock import Timeout

    from deepagents_code import approval_mode

    path = tmp_path / "approval.json"

    class _TimingOutLock:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> None:
            raise Timeout(str(path))

        def __exit__(self, *_exc: object) -> bool:
            return False

    monkeypatch.setattr(approval_mode, "FileLock", _TimingOutLock)

    assert save_auto_mode_notice(path) is False
    assert save_yolo_acknowledgement(path) is False
    # The timed-out writes never touched disk, so nothing is recorded.
    assert not path.exists()


def test_load_corrupt_state_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt state is treated as empty but logged (not silently swallowed)."""
    import logging

    path = tmp_path / "approval.json"
    path.write_text("not-json\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="deepagents_code.approval_mode"):
        assert not has_auto_mode_notice(path)
        assert not has_yolo_acknowledgement(path)

    assert any("corrupt" in record.getMessage() for record in caplog.records)


def test_load_missing_state_does_not_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing file is the normal first-run case and stays silent."""
    import logging

    path = tmp_path / "missing" / "approval.json"

    with caplog.at_level(logging.WARNING, logger="deepagents_code.approval_mode"):
        assert not has_auto_mode_notice(path)

    assert caplog.records == []
