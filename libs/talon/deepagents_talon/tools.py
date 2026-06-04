"""Optional runtime tools shared by Talon channel agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urljoin, urlparse

from deepagents_code.tools import (
    _MAX_FETCH_REDIRECTS,
    _pinned_dns,
    _UrlValidationError,
    _validate_url,
    fetch_url,
    web_search,
)

MIN_ERROR_STATUS = 400
MIN_REDIRECT_STATUS = 300
MAX_REDIRECT_STATUS = 400

if TYPE_CHECKING:
    from collections.abc import Mapping


class _HttpResponse(Protocol):
    """Small response shape used by the HTTP request helper."""

    status_code: int
    url: str
    text: str

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""

    def json(self) -> object:
        """Parse response content as JSON."""


def build_web_tools() -> list[Any]:
    """Return web tools available to a Talon runtime.

    Returns:
        Tools for URL fetch, web search, and direct HTTP requests.
    """
    return [fetch_url, web_search, http_request]


def http_request(  # noqa: PLR0913  # agent tool exposes common HTTP request fields
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make an HTTP request to an API or web service.

    Args:
        url: Target URL.
        method: HTTP method such as `GET`, `POST`, `PUT`, or `DELETE`.
        headers: Optional request headers.
        data: Optional request body. Dictionaries are sent as JSON.
        params: Optional URL query parameters.
        timeout: Request timeout in seconds.

    Returns:
        Response metadata and content, or an error dictionary.
    """
    try:
        import requests  # noqa: PLC0415  # keep missing dependency as a tool output
    except ImportError as exc:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Required package not installed: {exc.name}.",
            "url": url,
        }

    try:
        response = _request_with_redirects(
            url=url,
            method=method.upper(),
            headers=headers,
            data=data,
            params=params,
            timeout=timeout,
        )
    except _UrlValidationError as exc:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request URL error: {exc}",
            "url": url,
            "category": "validation",
        }
    except requests.exceptions.TooManyRedirects as exc:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request redirect error: {exc}",
            "url": url,
            "category": "redirects",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {exc}",
            "url": url,
        }

    try:
        content: Any = response.json()
    except ValueError:
        content = response.text

    return {
        "success": response.status_code < MIN_ERROR_STATUS,
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "content": content,
        "url": response.url,
    }


def _request_with_redirects(  # noqa: PLR0913  # mirrors `http_request` tool parameters
    *,
    url: str,
    method: str,
    headers: dict[str, str] | None,
    data: str | dict[str, Any] | None,
    params: dict[str, str] | None,
    timeout: int,
) -> _HttpResponse:
    """Run an HTTP request while re-validating each redirect hop."""
    import requests  # noqa: PLC0415  # requests is only needed when the tool runs

    kwargs: dict[str, Any] = {}
    if headers:
        kwargs["headers"] = headers
    if params:
        kwargs["params"] = params
    if data is not None:
        if isinstance(data, dict):
            kwargs["json"] = data
        else:
            kwargs["data"] = data

    current_url = url
    session = requests.Session()
    session.trust_env = False
    for _hop in range(_MAX_FETCH_REDIRECTS + 1):
        validated_ips = _validate_url(current_url)
        hostname = urlparse(current_url).hostname
        assert hostname is not None  # noqa: S101  # invariant from `_validate_url`
        encoded_hostname = hostname.encode("idna").decode("ascii")
        with _pinned_dns(encoded_hostname, validated_ips):
            response = session.request(
                method,
                current_url,
                timeout=timeout,
                allow_redirects=False,
                **kwargs,
            )

        if MIN_REDIRECT_STATUS <= response.status_code < MAX_REDIRECT_STATUS:
            location = response.headers.get("Location")
            if not location:
                msg = (
                    f"Redirect response (status {response.status_code}) at "
                    f"{current_url!r} is missing a Location header"
                )
                raise _UrlValidationError(msg)
            current_url = urljoin(current_url, location)
            continue
        return response

    msg = f"Exceeded {_MAX_FETCH_REDIRECTS} redirects starting from {url!r}"
    raise requests.exceptions.TooManyRedirects(msg)
