"""Remote configuration fetching and provider integration.

Fetches TOML config from HTTPS URLs and presents the result as a ranked
config provider between managed and environment tiers. URLs are stored in
`config.toml` under `[remote]` and fetched at startup and on `/reload`.
"""

from __future__ import annotations

import logging
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue

logger = logging.getLogger(__name__)

_REMOTE_TIMEOUT_SECONDS = 10
_MAX_RESPONSE_BYTES = 1_048_576  # 1 MiB


@dataclass(frozen=True, slots=True)
class RemoteConfigUrl:
    """A single remote config URL with optional authentication."""

    url: str
    """The HTTPS URL to fetch config from."""

    bearer_token: str | None = None
    """Optional Bearer token for the Authorization header."""


@dataclass(frozen=True, slots=True)
class RemoteConfigResult:
    """Result of fetching one remote config URL."""

    url: str
    snapshot: TomlSnapshot


def _redact_url(url: str) -> str:
    """Return a URL safe for logging, with credentials stripped."""
    parsed = urlparse(url)
    if parsed.password or parsed.username:
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc += f":{parsed.port}"
        return parsed._replace(netloc=netloc).geturl()
    return url


def fetch_remote_config(
    url: str,
    *,
    bearer_token: str | None = None,
    timeout: int = _REMOTE_TIMEOUT_SECONDS,
) -> TomlSnapshot:
    """Fetch and parse a TOML config from an HTTPS URL.

    Args:
        url: HTTPS URL to fetch.
        bearer_token: Optional Bearer token for authentication.
        timeout: Request timeout in seconds.

    Returns:
        Parsed TOML snapshot with health metadata.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        return TomlSnapshot(
            {},
            ProviderStatus(
                f"remote ({_redact_url(url)})",
                None,
                ProviderHealth.UNREADABLE,
                f"URL scheme must be https or http, got {parsed.scheme!r}",
            ),
        )

    headers = {"User-Agent": "dcode-remote-config"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    request = urllib.request.Request(url, headers=headers)  # noqa: S310

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            # Check for redirect to non-https
            final_url = response.geturl()
            if urlparse(final_url).scheme != "https" and parsed.scheme == "https":
                return TomlSnapshot(
                    {},
                    ProviderStatus(
                        f"remote ({_redact_url(url)})",
                        None,
                        ProviderHealth.UNREADABLE,
                        "HTTPS URL redirected to non-HTTPS",
                    ),
                )
            raw = response.read(_MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        return TomlSnapshot(
            {},
            ProviderStatus(
                f"remote ({_redact_url(url)})",
                None,
                ProviderHealth.UNREADABLE,
                f"HTTP {exc.code}: {exc.reason}",
            ),
        )
    except (OSError, urllib.error.URLError) as exc:
        return TomlSnapshot(
            {},
            ProviderStatus(
                f"remote ({_redact_url(url)})",
                None,
                ProviderHealth.UNREADABLE,
                f"{type(exc).__name__}: {exc}",
            ),
        )

    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        detail = (
            "not UTF-8 encoded" if isinstance(exc, UnicodeDecodeError) else str(exc)
        )
        return TomlSnapshot(
            {},
            ProviderStatus(
                f"remote ({_redact_url(url)})",
                None,
                ProviderHealth.CORRUPT,
                detail,
            ),
        )

    if not isinstance(data, dict):
        return TomlSnapshot(
            {},
            ProviderStatus(
                f"remote ({_redact_url(url)})",
                None,
                ProviderHealth.CORRUPT,
                "top-level TOML value is not a table",
            ),
        )

    return TomlSnapshot(
        data,
        ProviderStatus(
            f"remote ({_redact_url(url)})",
            None,
            ProviderHealth.OK,
        ),
    )


def parse_remote_urls(config_data: Mapping[str, Any]) -> tuple[RemoteConfigUrl, ...]:
    """Parse the `[remote]` section from config.toml data.

    Expected format::

        [remote]
        urls = ["https://example.com/dcode-config.toml"]

        [remote.auth."https://example.com/dcode-config.toml"]
        bearer_token = "tok-..."

    Args:
        config_data: Parsed TOML table from config.toml.

    Returns:
        Parsed remote URL entries.
    """
    remote = config_data.get("remote")
    if remote is None:
        return ()
    if not isinstance(remote, dict):
        return ()

    raw_urls = remote.get("urls")
    if not isinstance(raw_urls, list):
        return ()

    auth_table = remote.get("auth", {})
    if not isinstance(auth_table, dict):
        auth_table = {}

    results: list[RemoteConfigUrl] = []
    for raw_url in raw_urls:
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip()
        token: str | None = None
        auth_entry = auth_table.get(url)
        if isinstance(auth_entry, dict):
            raw_token = auth_entry.get("bearer_token")
            if isinstance(raw_token, str) and raw_token.strip():
                token = raw_token.strip()
        results.append(RemoteConfigUrl(url=url, bearer_token=token))
    return tuple(results)


def fetch_all_remote_configs(
    urls: tuple[RemoteConfigUrl, ...],
) -> tuple[RemoteConfigResult, ...]:
    """Fetch all remote config URLs and return their snapshots.

    Args:
        urls: Remote config URLs to fetch.

    Returns:
        Results for each URL, preserving order.
    """
    results: list[RemoteConfigResult] = []
    for entry in urls:
        snapshot = fetch_remote_config(entry.url, bearer_token=entry.bearer_token)
        results.append(RemoteConfigResult(url=entry.url, snapshot=snapshot))
    return tuple(results)


def merge_remote_snapshots(
    results: tuple[RemoteConfigResult, ...],
) -> TomlSnapshot:
    """Merge multiple remote config snapshots into one.

    Later URLs take precedence over earlier ones (deep-merged). Unreadable
    or corrupt snapshots are skipped; their health is logged.

    Args:
        results: Results from fetching all configured URLs.

    Returns:
        Merged snapshot, or an empty OK snapshot when nothing was readable.
    """
    from deepagents_code.configuration.resolver import merge_toml_tables

    usable: list[tuple[str, dict[str, Any]]] = []
    for result in results:
        if result.snapshot.status.usable and result.snapshot.data:
            usable.append((_redact_url(result.url), result.snapshot.data))
        elif not result.snapshot.status.usable:
            logger.warning(
                "Remote config from %s is %s: %s",
                _redact_url(result.url),
                result.snapshot.status.health.value,
                result.snapshot.status.detail or "no detail",
            )

    if not usable:
        if results:
            # All failed — report the worst health
            worst = max(
                results,
                key=lambda r: list(ProviderHealth).index(r.snapshot.status.health),
            )
            return TomlSnapshot(
                {},
                ProviderStatus(
                    f"remote ({_redact_url(worst.url)})",
                    None,
                    worst.snapshot.status.health,
                    worst.snapshot.status.detail,
                ),
            )
        return TomlSnapshot(
            {},
            ProviderStatus("remote", None, ProviderHealth.MISSING),
        )

    merged: dict[str, Any] = usable[0][1]
    for _label, table in usable[1:]:
        merged, _sources = merge_toml_tables(
            merged,
            table,
            lower_source="remote",
            higher_source="remote",
        )

    return TomlSnapshot(
        merged,
        ProviderStatus(
            "remote config",
            None,
            ProviderHealth.OK,
        ),
    )


@dataclass(frozen=True, slots=True)
class RemoteConfigProvider:
    """Ranked provider backed by remote TOML config snapshots."""

    name: str = "remote config"
    rank: int = field(default=250)  # REMOTE_CONFIG_RANK
    durable: bool = False
    _snapshot: TomlSnapshot = field(
        default_factory=lambda: TomlSnapshot.absent("remote config"),
        repr=False,
    )

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read one option from the remote config snapshot.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind
        from deepagents_code.configuration.providers import (
            ranked_theme_toml_value,
            ranked_toml_value,
        )

        snapshot = self._snapshot
        if option.kind is OptionKind.THEME_DELEGATE:
            return ranked_theme_toml_value(
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        return ranked_toml_value(
            option,
            snapshot.data,
            rank=self.rank,
            durable=self.durable,
            status=snapshot.status,
        )

    def status(self) -> ProviderStatus:
        """Return health for the current remote config snapshot."""
        return self._snapshot.status

    def reload(self) -> None:
        """No-op; remote config is re-fetched by the caller, not self-refreshing."""
