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


@pytest.mark.parametrize("mode", list(ApprovalMode))
def test_approval_mode_payload_shape(mode: ApprovalMode) -> None:
    payload = approval_mode_payload(mode=mode)

    assert payload == {"mode": mode.value}
    assert "auto_approve" not in payload


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


async def test_aread_approval_mode_prefers_async_store_api() -> None:
    key = approval_mode_key("thread-1")
    item = _StoreItem({"mode": "auto"})

    assert (
        await aread_approval_mode_from_store(_AsyncOnlyStore(item), key)
        is ApprovalMode.AUTO
    )


async def test_aread_approval_mode_falls_back_to_sync_get() -> None:
    """A store exposing only sync `get()` is still read via the fallback branch."""
    key = approval_mode_key("thread-1")
    item = _StoreItem({"mode": "yolo"})

    assert await aread_approval_mode_from_store(_Store(item), key) is ApprovalMode.YOLO


@pytest.mark.parametrize(
    ("store", "key"),
    [
        (None, approval_mode_key("thread-1")),
        (object(), approval_mode_key("thread-1")),  # no get()/aget()
        (_AsyncOnlyStore(None), approval_mode_key("thread-1")),  # missing item
        (
            _AsyncOnlyStore(_StoreItem(["not", "a", "mapping"])),
            approval_mode_key("thread-1"),
        ),
        (
            _AsyncOnlyStore(_StoreItem({"auto_approve": "yes"})),
            approval_mode_key("thread-1"),
        ),
        (_AsyncOnlyStore(_StoreItem({"mode": "not-a-mode"})), approval_mode_key("x")),
        (_AsyncOnlyStore(_StoreItem({"mode": "auto"})), ""),
        (_AsyncOnlyStore(_StoreItem({"mode": "auto"})), None),
    ],
)
async def test_aread_approval_mode_fails_closed(
    store: object,
    key: str | None,
) -> None:
    """The async reader re-implements the sync fail-closed guards; verify each."""
    assert await aread_approval_mode_from_store(store, key) is None


async def test_aread_approval_mode_exception_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="deepagents_code.approval_mode"):
        assert (
            await aread_approval_mode_from_store(
                _AsyncFailingStore(),
                approval_mode_key("thread-1"),
            )
            is None
        )

    assert "Could not read approval-mode store item" in caplog.text


async def test_awrite_approval_mode_writes_payload() -> None:
    writer = _Writer()
    key = await awrite_approval_mode(writer, "thread-1", mode=ApprovalMode.AUTO)

    assert key == approval_mode_key("thread-1")
    assert writer.items == [
        (APPROVAL_MODE_NAMESPACE, approval_mode_key("thread-1"), {"mode": "auto"})
    ]


async def test_awrite_approval_mode_returns_none_without_writer() -> None:
    assert (
        await awrite_approval_mode(object(), "thread-1", mode=ApprovalMode.AUTO)
    ) is None


def test_yolo_acknowledgement_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".state" / "approval.json"

    assert not has_yolo_acknowledgement(path)
    assert save_yolo_acknowledgement(path)
    assert has_yolo_acknowledgement(path)


def test_yolo_acknowledgement_rejects_stale_policy(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    path.write_text('{"version":1,"policy_version":"old","acknowledged":true}\n')

    assert not has_yolo_acknowledgement(path)


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


def test_yolo_and_auto_notice_coexist(tmp_path: Path) -> None:
    import json

    path = tmp_path / "approval.json"

    assert save_auto_mode_notice(path)
    assert save_yolo_acknowledgement(path)
    assert has_auto_mode_notice(path)
    assert has_yolo_acknowledgement(path)

    # Reverse order: YOLO first, then Auto, still keeps both records.
    path.unlink()
    assert save_yolo_acknowledgement(path)
    assert save_auto_mode_notice(path)
    assert has_auto_mode_notice(path)
    assert has_yolo_acknowledgement(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["policy_version"] == YOLO_ACKNOWLEDGEMENT_POLICY_VERSION
    assert data["acknowledged"] is True
    assert data["auto_notice_version"] == AUTO_NOTICE_VERSION
    assert data["auto_notice_shown"] is True
