# Copyright (c) 2026 Mainbrella
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest
from mainbrella import Mainbrella

from langchain_mainbrella import MainbrellaProvider

GENERATION = "2026-10-05T12:00:00.000Z"
REPLACEMENT = "2026-10-05T12:30:00.000Z"
IDENTITY = f"small@{GENERATION}"


def test_create_attach_and_cleanup_exact_generation() -> None:
    def transport(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        assert urlsplit(url).path == "/containers"
        if method == "POST":
            assert json.loads(body or b"") == {"catalogId": "python", "size": "lite"}
            assert headers["Idempotency-Key"] == "test-creation"
            return 200, json.dumps(
                {
                    "creation": {
                        "id": "creation",
                        "containerId": "small",
                        "createdAt": GENERATION,
                        "status": "running",
                    },
                    "containers": [
                        {"id": "small", "createdAt": GENERATION, "status": "running"}
                    ],
                }
            ).encode()
        if method == "DELETE":
            assert parse_qs(urlsplit(url).query) == {
                "id": ["small"],
                "createdAt": [GENERATION],
            }
            return 200, b'{"containers":[]}'
        return 200, json.dumps(
            {
                "containers": [
                    {"id": "small", "createdAt": GENERATION, "status": "running"}
                ]
            }
        ).encode()

    provider = MainbrellaProvider(api_key="mb_" + "a" * 64)
    provider._client = Mainbrella("mb_" + "a" * 64, transport=transport)
    sandbox = provider.get_or_create(idempotency_key="test-creation")
    assert sandbox.id == IDENTITY
    assert provider.get_or_create(sandbox_id=sandbox.id).id == IDENTITY
    provider.delete(sandbox_id=sandbox.id)


@pytest.mark.parametrize("operation", ["attach", "delete"])
def test_stale_generation_cannot_affect_replacement(operation: str) -> None:
    def transport(
        _url: str,
        method: str,
        _headers: dict[str, str],
        _body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        assert method == "GET", "A stale generation must never be mutated"
        return 200, json.dumps(
            {
                "containers": [
                    {"id": "small", "createdAt": REPLACEMENT, "status": "running"}
                ]
            }
        ).encode()

    provider = MainbrellaProvider(api_key="mb_" + "a" * 64)
    provider._client = Mainbrella("mb_" + "a" * 64, transport=transport)
    action = provider.get_or_create if operation == "attach" else provider.delete
    with pytest.raises(KeyError):
        action(sandbox_id=IDENTITY)


def test_bare_slot_is_rejected() -> None:
    provider = MainbrellaProvider(api_key="mb_" + "a" * 64)
    with pytest.raises(ValueError, match="slot@createdAt"):
        provider.get_or_create(sandbox_id="small")


def test_custom_image_overrides_python_catalog() -> None:
    def transport(
        _url: str,
        method: str,
        _headers: dict[str, str],
        body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        assert method == "POST"
        assert json.loads(body or b"") == {"imageId": "custom-python", "size": "medium"}
        return 200, json.dumps(
            {
                "creation": {
                    "id": "creation",
                    "containerId": "small",
                    "createdAt": GENERATION,
                    "status": "running",
                },
                "containers": [
                    {"id": "small", "createdAt": GENERATION, "status": "running"}
                ],
            }
        ).encode()

    provider = MainbrellaProvider(api_key="mb_" + "a" * 64)
    provider._client = Mainbrella("mb_" + "a" * 64, transport=transport)
    assert (
        provider.get_or_create(image_id="custom-python", size="medium").id == IDENTITY
    )
