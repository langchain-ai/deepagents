"""Tests for the OOLONG loader's fetch dispatch (datasets-server -> Hub parquet).

Offline: the two underlying fetch paths are monkeypatched, so no network is
touched — only the dispatch/fallback logic is exercised.
"""

from __future__ import annotations

import pytest

from harbor_adapters.oolong import loader

_ROW: dict[str, object] = {"id": "1", "dataset": "trec_coarse", "context_len": 1024}


def test_uses_filter_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _filter(*_a: object, **_k: object) -> list[dict[str, object]]:
        calls.append("filter")
        return [_ROW]

    def _parquet(*_a: object, **_k: object) -> list[dict[str, object]]:
        calls.append("parquet")
        return []

    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_filter", _filter)
    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_parquet", _parquet)

    rows = loader._fetch_oolong_rows("trec_coarse", 1024, "validation")

    assert rows == [_ROW]
    assert calls == ["filter"]  # parquet fallback not touched


def test_falls_back_to_parquet_on_filter_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _filter(*_a: object, **_k: object) -> list[dict[str, object]]:
        calls.append("filter")
        msg = "datasets-server request failed after 4 attempts"
        raise RuntimeError(msg)

    def _parquet(*_a: object, **_k: object) -> list[dict[str, object]]:
        calls.append("parquet")
        return [_ROW]

    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_filter", _filter)
    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_parquet", _parquet)

    rows = loader._fetch_oolong_rows("trec_coarse", 1024, "validation")

    assert rows == [_ROW]
    assert calls == ["filter", "parquet"]


def test_raises_combined_error_when_both_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _filter(*_a: object, **_k: object) -> list[dict[str, object]]:
        msg = "server 503"
        raise RuntimeError(msg)

    def _parquet(*_a: object, **_k: object) -> list[dict[str, object]]:
        msg = "no parquet shards"
        raise RuntimeError(msg)

    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_filter", _filter)
    monkeypatch.setattr(loader, "_fetch_oolong_rows_via_parquet", _parquet)

    with pytest.raises(RuntimeError, match="both paths"):
        loader._fetch_oolong_rows("trec_coarse", 1024, "validation")
