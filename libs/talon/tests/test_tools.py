from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import requests

from deepagents_talon import tools as runtime_tools
from deepagents_talon.tools import http_request

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        url: str,
        headers: dict[str, str] | None = None,
        json_content: object | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers = headers or {}
        self._json_content = json_content
        self.text = text

    def json(self) -> object:
        if self._json_content is None:
            msg = "not json"
            raise ValueError(msg)
        return self._json_content


def test_http_request_returns_response_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request_with_redirects(**kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse(
            status_code=201,
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            json_content={"ok": True},
        )

    monkeypatch.setattr(runtime_tools, "_request_with_redirects", fake_request_with_redirects)

    result = http_request(
        "https://example.com/api",
        method="post",
        headers={"X-Test": "1"},
        data={"value": "ok"},
        params={"page": "1"},
        timeout=5,
    )

    assert captured == {
        "url": "https://example.com/api",
        "method": "POST",
        "headers": {"X-Test": "1"},
        "data": {"value": "ok"},
        "params": {"page": "1"},
        "timeout": 5,
    }
    assert result == {
        "success": True,
        "status_code": 201,
        "headers": {"Content-Type": "application/json"},
        "content": {"ok": True},
        "url": "https://example.com/api",
    }


def test_http_request_blocks_invalid_url_scheme() -> None:
    result = http_request("file:///etc/passwd")

    assert result["success"] is False
    assert result["status_code"] == 0
    assert result["url"] == "file:///etc/passwd"
    assert result["category"] == "validation"
    assert "URL scheme not allowed" in str(result["content"])


def test_request_with_redirects_revalidates_each_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RecordingSession(
        [
            FakeResponse(
                status_code=302,
                url="https://example.com/start",
                headers={"Location": "/next"},
            ),
            FakeResponse(status_code=200, url="https://example.com/next", text="ok"),
        ],
    )
    validated: list[str] = []
    pinned: list[tuple[str, tuple[str, ...]]] = []

    def fake_validate_url(url: str) -> list[str]:
        validated.append(url)
        return ["93.184.216.34"]

    @contextmanager
    def fake_pinned_dns(hostname: str, allowed_ips: list[str]) -> Iterator[None]:
        pinned.append((hostname, tuple(allowed_ips)))
        yield

    monkeypatch.setattr(requests, "Session", lambda: session)
    monkeypatch.setattr(runtime_tools, "_validate_url", fake_validate_url)
    monkeypatch.setattr(runtime_tools, "_pinned_dns", fake_pinned_dns)

    response = runtime_tools._request_with_redirects(
        url="https://example.com/start",
        method="GET",
        headers=None,
        data=None,
        params=None,
        timeout=5,
    )

    assert response.url == "https://example.com/next"
    assert session.trust_env is False
    assert validated == ["https://example.com/start", "https://example.com/next"]
    assert pinned == [
        ("example.com", ("93.184.216.34",)),
        ("example.com", ("93.184.216.34",)),
    ]
    assert [call["url"] for call in session.calls] == [
        "https://example.com/start",
        "https://example.com/next",
    ]


class RecordingSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.trust_env = True

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)
