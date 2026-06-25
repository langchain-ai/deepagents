"""Tests for live approval-mode store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from deepagents_code.approval_mode import (
    APPROVAL_MODE_NAMESPACE,
    approval_mode_key,
    approval_mode_payload,
    awrite_approval_mode,
    read_approval_mode_from_store,
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


def test_approval_mode_payload_shape() -> None:
    assert approval_mode_payload(auto_approve=True) == {"auto_approve": True}


def test_read_approval_mode_from_store_accepts_mapping_item() -> None:
    key = approval_mode_key("thread-1")
    item = {"value": {"auto_approve": True}}

    assert read_approval_mode_from_store(_Store(item), key) is True


def test_read_approval_mode_from_store_accepts_attribute_item() -> None:
    key = approval_mode_key("thread-1")
    item = _StoreItem({"auto_approve": False})

    assert read_approval_mode_from_store(_Store(item), key) is False


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


async def test_awrite_approval_mode_writes_payload() -> None:
    writer = _Writer()
    key = await awrite_approval_mode(writer, "thread-1", auto_approve=True)

    assert key == approval_mode_key("thread-1")
    assert writer.items == [
        (APPROVAL_MODE_NAMESPACE, approval_mode_key("thread-1"), {"auto_approve": True})
    ]


async def test_awrite_approval_mode_returns_none_without_writer() -> None:
    assert (await awrite_approval_mode(object(), "thread-1", auto_approve=True)) is None
