"""Synchronous providers and provider-domain option coercion."""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, assert_never, cast, override

from deepagents_code._env_vars import classify_env_bool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future
    from http.client import HTTPResponse
    from pathlib import Path
    from typing import Protocol

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue

    class _TimeoutSocket(Protocol):
        """Socket operation needed to tighten an active HTTP response."""

        def settimeout(self, value: float | None) -> None:
            """Set the maximum wait for the next socket operation."""

    class _RemoteOpener(Protocol):
        """Minimal opener surface used by the remote provider."""

        def open(self, request: object, *, timeout: float) -> HTTPResponse:
            """Open one HTTP response."""


from deepagents_code.configuration.resolver import (
    DEFAULT_RANK,
    ENVIRONMENT_RANK,
    USER_RANK,
)
from deepagents_code.configuration.types import (
    Found,
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    TomlSnapshot,
    Unset,
)

REMOTE_MANAGED_CONFIG_MAX_BYTES = 1024 * 1024
REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS = 5
_HTTP_OK = 200
_HTTP_REDIRECT_MIN = 300
_HTTP_REDIRECT_MAX = 400

SHADOWED_TABLE_SUFFIX = "— every option under it falls back to its next source"
"""Tail of the rejection raised when a scalar shadows a whole TOML table.

`config_manifest._emit_ranked_diagnostics` matches this text to deduplicate the
warning across a full-manifest pass. Both sides must share one constant: a
reworded message that no longer matches would silently restore roughly one
duplicated line per option for a single typo.
"""

UNUSABLE_SOURCE_SUFFIX = "— using defaults for every option it would have set"
"""Tail of the rejection raised when a whole TOML source could not be read.

Deduplicated the same way, and for the same reason: a rejected file affects
every option at once, so the warning belongs to the file, not to each key.
"""

RETAINED_SOURCE_SUFFIX = "— still applying the last readable version of it"
"""Tail of the rejection raised when a failed reload kept the previous values.

Distinct from `UNUSABLE_SOURCE_SUFFIX` because the consequence is different:
nothing fell back to a default, but the file on disk no longer describes what
the process is enforcing, and an edit the user just made is not in effect.
"""


def coerce_environment_value(
    option: ConfigOption, raw: str, name: str
) -> ProviderResult[object]:
    """Coerce one present environment value within the env provider domain.

    The returned reason preserves the established diagnostic text. Resolution
    decides when to emit it so health inspection does not log a rejection as a
    side effect of merely reading provider state.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Present environment string.
        name: Environment variable spelling that supplied `raw`.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    if kind in {OptionKind.BOOL, OptionKind.BOOL_MODE_DEFAULT}:
        classified = classify_env_bool(raw)
        if classified is None:
            return Invalid(f"Ignoring {name}={raw!r} (expected bool)")
        return Found(classified)
    if kind is OptionKind.BOOL_PRESENCE:
        return Found(bool(raw))
    if kind is OptionKind.STR:
        return Found(raw)
    if kind is OptionKind.NON_EMPTY_STR:
        value = raw.strip()
        if value:
            return Found(value)
        return Invalid(f"Ignoring {name}={raw!r} (expected non-empty string)")
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._debug import LOG_LEVELS

        level = raw.strip().upper()
        if level in LOG_LEVELS:
            return Found(level)
        valid = ", ".join(LOG_LEVELS)
        return Invalid(f"Ignoring {name}={raw!r} (expected one of {valid})")
    if kind is OptionKind.INT:
        try:
            return Found(int(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int)")
    if kind is OptionKind.NON_NEGATIVE_INT:
        try:
            value = int(raw.strip())
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
        if value >= 0:
            return Found(value)
        return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
    if kind is OptionKind.FLOAT:
        try:
            return Found(float(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected number)")
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import parse_shell_allow_list

        try:
            return Found(parse_shell_allow_list(raw))
        except ValueError:
            return Invalid(f"Ignoring invalid {name}")
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        from deepagents_code.config import _parse_extra_skills_dirs

        try:
            return Found(_parse_extra_skills_dirs(raw, None))
        except (ValueError, RuntimeError):
            return Invalid(f"Ignoring {name} (could not resolve a path)")
    if kind is OptionKind.THEME_DELEGATE:
        # Theme names are resolved by the theme-aware provider path. Keep this
        # defensive passthrough for the compatibility wrapper.
        return Found(raw)
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if raw in VALID_CURSOR_STYLES:
            return Found(raw)
        return Invalid(f"Ignoring {name}={raw!r} (expected 'block' or 'underline')")
    if kind in {
        OptionKind.MODEL_LIST_DELEGATE,
        OptionKind.PTC_DELEGATE,
        OptionKind.STRUCTURED,
    }:
        # No option of these kinds declares `env_var` *or* `fallback_env_vars`,
        # so the env provider never reaches this branch; only the `_coerce_env`
        # compatibility wrapper, which has no env-var guard, can. If a future
        # option gains either name, reject rather than pass the raw string
        # through: an uncoerced value would bypass the delegate parser's
        # validation. Rejection drops to the next-weaker tier (TOML, then the
        # manifest default) -- note that for a policy option such as
        # `models.allowed` the default is "unrestricted", so what keeps that
        # fallback safe is managed config outranking the environment, not the
        # default itself being conservative.
        return Invalid(f"{option.key} is not env-backed; ignoring {name}={raw!r}")
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if raw in VALID_STARTUP_MODES:
            return Found(raw)
        return Invalid(
            f"Ignoring {name}={raw!r} (expected 'manual', 'auto', or 'yolo')"
        )
    assert_never(kind)


def coerce_toml_value(
    option: ConfigOption, raw: object, *, source: str
) -> ProviderResult[object]:
    """Coerce one present TOML value within the file-provider domain.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Parsed TOML value.
        source: Human-readable provider name used in diagnostic text.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    label = option.toml_path or option.key

    if kind in {
        OptionKind.BOOL,
        OptionKind.BOOL_MODE_DEFAULT,
        OptionKind.BOOL_PRESENCE,
    }:
        if isinstance(raw, bool):
            value = not raw if option.invert_toml_bool else raw
            return Found(value)
    elif kind is OptionKind.INT:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return Found(raw)
    elif kind is OptionKind.NON_NEGATIVE_INT:
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return Found(raw)
    elif kind is OptionKind.FLOAT:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return Found(float(raw))
    elif kind is OptionKind.STR:
        if isinstance(raw, str):
            return Found(raw)
    elif kind is OptionKind.NON_EMPTY_STR:
        if isinstance(raw, str) and (value := raw.strip()):
            return Found(value)
    elif kind is OptionKind.MODEL_LIST_DELEGATE:
        from deepagents_code.model_config import parse_model_allowlist

        try:
            return Found(parse_model_allowlist(raw))
        except (TypeError, ValueError) as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
    elif kind is OptionKind.SKILLS_DIRS_DELEGATE:
        if isinstance(raw, list):
            from deepagents_code.config import _parse_extra_skills_dirs

            try:
                return Found(_parse_extra_skills_dirs(None, cast("list[str]", raw)))
            except (ValueError, RuntimeError):
                return Invalid(
                    f"Ignoring {label} in {source} (could not resolve a path)"
                )
    elif kind is OptionKind.PTC_DELEGATE:
        from deepagents_code.config import _parse_interpreter_ptc

        try:
            return Found(_parse_interpreter_ptc(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
    elif kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if isinstance(raw, str) and raw in VALID_CURSOR_STYLES:
            return Found(raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} (expected 'block' or 'underline')"
        )
    elif kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if isinstance(raw, str) and raw in VALID_STARTUP_MODES:
            return Found(raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} "
            "(expected 'manual', 'auto', or 'yolo')"
        )
    elif kind is OptionKind.STRUCTURED:
        return Found(raw)
    elif kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import (
            parse_shell_allow_list,
            parse_shell_allow_list_items,
        )

        try:
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                return Found(parse_shell_allow_list_items(cast("list[str]", raw)))
            if isinstance(raw, str):
                return Found(parse_shell_allow_list(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")

    return Invalid(f"Ignoring {label}={raw!r} in {source} (expected {option.type})")


def ranked_toml_value(
    option: ConfigOption,
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[object]:
    """Read and coerce one option from a parsed TOML provider.

    Args:
        option: Manifest option to read.
        data: Parsed provider table.
        rank: Numeric precedence rank.
        durable: Whether this tier masks lower-priority ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked `Found`, `Unset`, or `Invalid` provider result.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    if not status.usable or not option.toml_keys:
        result: ProviderResult[object] = Unset()
    else:
        node: object = data
        result = Unset()
        for index, key in enumerate(option.toml_keys):
            if not isinstance(node, dict):
                path = option.toml_keys[:index]
                result = Invalid(
                    f"Ignoring {status.name} [{'.'.join(path)}]; expected a "
                    f"table, got {type(node).__name__} {SHADOWED_TABLE_SUFFIX}"
                )
                break
            if key not in node:
                break
            node = node[key]
        else:
            result = coerce_toml_value(option, node, source=status.name)
    return RankedProviderValue(rank, durable, status, result)


def ranked_environment_value(
    option: ConfigOption,
    environ: Mapping[str, str],
    *,
    rank: int,
) -> RankedProviderValue[object]:
    """Read and coerce one option from the process-environment domain.

    Args:
        option: Manifest option to read.
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric precedence rank.

    Returns:
        Ranked provider result. Fallback names remain one provider tier.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    names: list[str] = []
    if option.env_var:
        canonical = option.env_var
        prefixed = (
            canonical
            if canonical.startswith("DEEPAGENTS_CODE_")
            else f"DEEPAGENTS_CODE_{canonical}"
        )
        names.append(prefixed if prefixed in environ else canonical)
    names.extend(option.fallback_env_vars)

    status = ProviderStatus("environment", None, ProviderHealth.OK)
    last_invalid: Invalid | None = None
    diagnostics: list[str] = []
    for name in names:
        raw = environ.get(name)
        if raw is None:
            continue
        status = replace(status, name=f"env ({name})")
        if not raw.strip():
            if option.empty_env_is_false:
                return RankedProviderValue(rank, False, status, Found(False))
            if raw:
                last_invalid = Invalid(
                    f"Ignoring {name}={raw!r} (whitespace-only; treated as unset)"
                )
                diagnostics.append(last_invalid.reason)
            continue
        result = coerce_environment_value(option, raw, name)
        if isinstance(result, Found):
            return RankedProviderValue(
                rank,
                False,
                status,
                result,
                tuple(diagnostics),
            )
        if isinstance(result, Invalid):
            last_invalid = result
            diagnostics.append(result.reason)
    return RankedProviderValue(
        rank,
        False,
        status,
        last_invalid or Unset(),
        tuple(diagnostics),
    )


def ranked_theme_toml_value(
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[object]:
    """Resolve one file provider's terminal-aware theme preference.

    The terminal mapping and `[ui].theme` fallback are one provider domain:
    they share a durability boundary and source rank. Their internal ordering
    stays inside this provider while precedence between managed, environment,
    user, and default remains the ranked resolver's responsibility.

    Args:
        data: Parsed TOML provider table.
        rank: Numeric provider rank.
        durable: Whether this file tier masks lower ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked theme result with the selected TOML path in its display status.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import (
        resolve_terminal_mapping,
        resolve_theme_name,
    )

    if not status.usable:
        return RankedProviderValue(rank, durable, status, Unset())
    ui = data.get("ui")
    if ui is None:
        return RankedProviderValue(rank, durable, status, Unset())
    if not isinstance(ui, dict):
        result: ProviderResult[object] = Invalid(
            f"[ui] in {status.name} should be a table; got "
            f"{type(ui).__name__} while resolving theme"
        )
        return RankedProviderValue(rank, durable, status, result)

    resolved = resolve_terminal_mapping(ui)
    if resolved is not None:
        import os

        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        selected = replace(
            status,
            name=f"{status.name} [ui.terminal_themes.{term_program}]",
        )
        return RankedProviderValue(rank, durable, selected, Found(resolved))

    saved = ui.get("theme")
    resolved = resolve_theme_name(saved)
    if resolved is not None:
        selected = replace(status, name=f"{status.name} [ui.theme]")
        return RankedProviderValue(rank, durable, selected, Found(resolved))
    if isinstance(saved, str):
        result = Invalid(f"Unknown theme '{saved}' in {status.name}; ignoring it")
        return RankedProviderValue(rank, durable, status, result)
    return RankedProviderValue(rank, durable, status, Unset())


def ranked_theme_environment_value(
    environ: Mapping[str, str], *, rank: int
) -> RankedProviderValue[object]:
    """Resolve the theme environment provider.

    Args:
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric environment rank.

    Returns:
        Ranked theme result with the concrete variable name in its status.
    """
    from deepagents_code._env_vars import THEME
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import resolve_theme_name

    status = ProviderStatus(f"env ({THEME})", None, ProviderHealth.OK)
    raw = environ.get(THEME)
    if raw is None:
        return RankedProviderValue(rank, False, status, Unset())
    resolved = resolve_theme_name(raw)
    if resolved is not None:
        return RankedProviderValue(rank, False, status, Found(resolved))
    return RankedProviderValue(
        rank,
        False,
        status,
        Invalid(f"Unknown theme '{raw}' in {THEME}; falling through"),
    )


def ranked_default_value(
    option: ConfigOption, *, rank: int
) -> RankedProviderValue[object]:
    """Produce an option's typed or mode-dependent default provider result.

    Args:
        option: Manifest option whose default should be produced.
        rank: Numeric precedence rank.

    Returns:
        Durable ranked default result.
    """
    from deepagents_code.config_manifest import OptionKind
    from deepagents_code.configuration.resolver import RankedProviderValue

    if option.kind is OptionKind.BOOL_MODE_DEFAULT:
        from deepagents_code._env_vars import DEBUG, EXPERIMENTAL, is_env_truthy

        value: object = is_env_truthy(DEBUG) or is_env_truthy(EXPERIMENTAL)
    elif option.kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._env_vars import DEBUG, is_env_truthy

        value = "DEBUG" if is_env_truthy(DEBUG) else "INFO"
    elif option.kind is OptionKind.THEME_DELEGATE:
        from deepagents_code import theme

        value = theme.DEFAULT_THEME
    elif option.kind is OptionKind.STRUCTURED:
        status = ProviderStatus("default", None, ProviderHealth.OK)
        return RankedProviderValue(rank, True, status, Unset())
    else:
        value = option.default
    status = ProviderStatus("default", None, ProviderHealth.OK)
    return RankedProviderValue(rank, True, status, Found(value))


def _build_remote_opener() -> _RemoteOpener:
    """Build a direct opener only when a remote descriptor is active.

    Returns:
        An opener that bypasses environment proxies and rejects redirects.
    """
    from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

    class _RejectRedirects(HTTPRedirectHandler):
        @override
        def redirect_request(self, *_args: object, **_kwargs: object) -> None:
            return None

    return cast(
        "_RemoteOpener",
        build_opener(ProxyHandler({}), _RejectRedirects()),
    )


def _remote_status(
    name: str,
    path: Path | None,
    health: ProviderHealth,
    detail: str,
    source: str | None = None,
) -> TomlSnapshot:
    """Build an empty snapshot with safe remote-source diagnostics.

    `source` is the validated URL, or `None` before validation accepted one --
    a rejected source may carry the very query token the detail strings are
    written to keep out of operator-visible text.

    Returns:
        An unhealthy snapshot carrying no policy data.
    """
    return TomlSnapshot(
        {},
        ProviderStatus(name, path, health, detail, remote_source=source),
    )


def _validate_remote_url(source: str) -> str:
    """Validate and normalize one configured remote source.

    Returns:
        The normalized absolute HTTPS URL.

    Raises:
        ValueError: If the URL could redirect trust or leak credentials.
    """
    from urllib.parse import urlsplit

    parsed = urlsplit(source)
    if any(char <= " " or char == "\x7f" for char in source):
        # urllib raises `http.client.InvalidURL` (an `HTTPException`, not a
        # `ValueError`) when `opener.open()` meets control characters, which
        # would escape the failure handling in `RemoteTomlProvider.load()`.
        msg = "remote source must not contain whitespace or control characters"
        raise ValueError(msg)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        msg = "remote source must be an absolute HTTPS URL"
        raise ValueError(msg)
    if parsed.username is not None or parsed.password is not None:
        msg = "remote source must not contain credentials"
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        msg = "remote source must not contain a query string or fragment"
        raise ValueError(msg)
    try:
        port = parsed.port
    except ValueError as exc:
        msg = "remote source has an invalid port"
        raise ValueError(msg) from exc
    host = parsed.hostname.rstrip(".")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return parsed._replace(scheme="https", netloc=netloc).geturl()


_READ_CHUNK_SIZE = 65536


def _fail_if_expired(deadline: float) -> None:
    """Raise once the fetch has run past its end-to-end time boundary.

    Raises:
        TimeoutError: If the current monotonic time is at or past *deadline*.
    """
    if time.monotonic() >= deadline:
        msg = "remote source timed out"
        raise TimeoutError(msg)


def _remaining_timeout(deadline: float) -> float:
    """Return positive seconds left before the end-to-end deadline.

    Returns:
        The remaining timeout in seconds.

    Raises:
        TimeoutError: If the deadline has already passed.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        msg = "remote source timed out"
        raise TimeoutError(msg)
    return remaining


def _close_completed_response(future: Future[HTTPResponse]) -> None:
    """Close a response that arrived after its caller stopped waiting."""
    from http.client import HTTPException

    try:
        response = future.result()
    except (HTTPException, OSError, ValueError):
        return
    response.close()


def _open_response_with_deadline(
    opener: _RemoteOpener,
    request: object,
    *,
    deadline: float,
) -> HTTPResponse:
    """Open a response without letting blocking setup exceed the deadline.

    Returns:
        The opened response.

    Raises:
        TimeoutError: If setup does not finish within the deadline.
    """
    from concurrent.futures import Future
    from http.client import HTTPException
    from threading import Thread

    future: Future[HTTPResponse] = Future()

    def open_response() -> None:
        try:
            response = opener.open(
                request,
                timeout=REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS,
            )
        except (HTTPException, OSError, ValueError) as exc:
            future.set_exception(exc)
        else:
            future.set_result(response)

    Thread(target=open_response, name="managed-config-open", daemon=True).start()
    try:
        response = future.result(timeout=_remaining_timeout(deadline))
    except TimeoutError:
        future.add_done_callback(_close_completed_response)
        raise
    try:
        _fail_if_expired(deadline)
    except TimeoutError:
        response.close()
        raise
    return response


def _set_response_timeout(response: HTTPResponse, timeout: float) -> None:
    """Tighten the active response socket to the remaining timeout.

    Raises:
        OSError: If a live response has no controllable socket.
    """
    if response.fp is not None:
        raw = getattr(response.fp, "raw", None)
        sock = cast("_TimeoutSocket | None", getattr(raw, "_sock", None))
        if sock is None:
            msg = "remote source response timeout could not be enforced"
            raise OSError(msg)
        sock.settimeout(timeout)


def _read_response_chunk(
    response: HTTPResponse,
    size: int,
    *,
    deadline: float,
) -> bytes:
    """Read at most one socket chunk within the remaining deadline.

    Returns:
        The next available body chunk.
    """
    _set_response_timeout(response, _remaining_timeout(deadline))
    return response.read1(size)


def _declared_body_length(response: HTTPResponse) -> int | None:
    """Return the body length this response's HTTP framing promises.

    Returns:
        The declared length, or `None` when chunked framing delimits the body.

    Raises:
        ValueError: If framing cannot show that the body arrived whole, or the
            declared length is over the fixed limit.
    """
    raw_length = response.headers.get("Content-Length")
    if raw_length is None:
        encoding = (response.headers.get("Transfer-Encoding") or "").strip()
        if encoding.lower() == "chunked":
            # Chunked framing detects its own truncation: `http.client` raises
            # `IncompleteRead` when the terminating chunk never arrives.
            return None
        # Connection-close framing cannot distinguish a complete policy from
        # one cut short, and TOML truncated at a line boundary still parses --
        # a deny list can lose its entries and still look healthy.
        msg = "remote source did not delimit the response body"
        raise ValueError(msg)
    try:
        declared = int(raw_length)
    except ValueError as exc:
        msg = "remote source declared an invalid body length"
        raise ValueError(msg) from exc
    if declared < 0:
        msg = "remote source declared an invalid body length"
        raise ValueError(msg)
    if declared > REMOTE_MANAGED_CONFIG_MAX_BYTES:
        msg = "remote source response exceeds the size limit"
        raise ValueError(msg)
    return declared


def _read_limited_response(
    response: HTTPResponse,
    *,
    deadline: float,
) -> bytes:
    """Read one response without crossing the size or time boundary.

    Returns:
        The bounded response body.

    Raises:
        IncompleteRead: If HTTP framing reports a truncated response.
        ValueError: If the declared or actual body exceeds the fixed limit.
    """
    declared_length = _declared_body_length(response)
    chunks: list[bytes] = []
    remaining = REMOTE_MANAGED_CONFIG_MAX_BYTES + 1
    while remaining > 0:
        chunk = _read_response_chunk(
            response,
            min(_READ_CHUNK_SIZE, remaining),
            deadline=deadline,
        )
        _fail_if_expired(deadline)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > REMOTE_MANAGED_CONFIG_MAX_BYTES:
        msg = "remote source response exceeds the size limit"
        raise ValueError(msg)
    if declared_length is not None and len(payload) != declared_length:
        if len(payload) > declared_length:
            msg = "remote source sent more than its declared body length"
            raise ValueError(msg)
        from http.client import IncompleteRead

        raise IncompleteRead(payload, declared_length - len(payload))
    return payload


def _parse_remote_toml(
    payload: bytes,
    *,
    name: str,
    path: Path | None,
    source: str,
) -> TomlSnapshot:
    """Parse one remote policy body into a managed snapshot.

    Returns:
        A readable policy snapshot or a safe corrupt-source status.
    """
    try:
        data = tomllib.loads(payload.decode("utf-8"))
    except UnicodeDecodeError:
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source is not UTF-8",
            source,
        )
    except (tomllib.TOMLDecodeError, RecursionError):
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source contains invalid TOML",
            source,
        )
    if not data:
        # An empty local file is a deliberate administrator act. Here the
        # descriptor asserts that this URL holds the whole managed policy, so a
        # document with no keys contradicts it -- a botched publish, a
        # truncated object store overwrite, or an edge answering while the
        # origin is down. Reading it as "nothing is enforced" would let any of
        # those evict live policy.
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote source declared no policy",
            source,
        )
    if "managed_config" in data:
        return _remote_status(
            name,
            path,
            ProviderHealth.CORRUPT,
            "remote policy must not declare [managed_config]",
            source,
        )
    return TomlSnapshot(
        data,
        ProviderStatus(
            name,
            path,
            ProviderHealth.OK,
            None,
            remote_source=source,
        ),
    )


@dataclass(frozen=True, slots=True)
class RemoteTomlProvider:
    """Managed TOML provider backed by one administrator-selected HTTPS URL."""

    name: str
    source: str
    path: Path | None

    def load(self) -> TomlSnapshot:
        """Fetch, bound, and parse the remote managed policy.

        Returns:
            The parsed policy or a safe provider failure status.
        """
        from http.client import HTTPException
        from urllib.error import HTTPError, URLError
        from urllib.request import Request

        try:
            source = _validate_remote_url(self.source)
        except ValueError as exc:
            return _remote_status(
                self.name,
                self.path,
                ProviderHealth.CORRUPT,
                str(exc),
            )
        request = Request(  # noqa: S310  # `_validate_remote_url` permits HTTPS only
            source,
            headers={"Accept": "application/toml"},
        )
        opener = _build_remote_opener()
        deadline = time.monotonic() + REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS
        try:
            with _open_response_with_deadline(
                opener,
                request,
                deadline=deadline,
            ) as response:
                # `urllib` only raises for a status outside 200..299, so a
                # `204` or a `206` reaches here as a success. A partial policy
                # is the dangerous one: TOML cut at a line boundary parses, so
                # a deny list can silently lose its entries.
                if response.status != _HTTP_OK:
                    return _remote_status(
                        self.name,
                        self.path,
                        ProviderHealth.UNREADABLE,
                        f"remote source returned HTTP {response.status}",
                        source,
                    )
                payload = _read_limited_response(response, deadline=deadline)
        except HTTPError as exc:
            detail = (
                "remote source refused redirects"
                if _HTTP_REDIRECT_MIN <= exc.code < _HTTP_REDIRECT_MAX
                else (f"remote source returned HTTP {exc.code}")
            )
            return _remote_status(
                self.name,
                self.path,
                ProviderHealth.UNREADABLE,
                detail,
                source,
            )
        except TimeoutError:
            return _remote_status(
                self.name,
                self.path,
                ProviderHealth.UNREADABLE,
                "remote source timed out",
                source,
            )
        except (HTTPException, URLError, OSError) as exc:
            # `http.client.HTTPException` derives from `Exception`, not
            # `OSError`, so every wire-level parse failure it covers --
            # `IncompleteRead` from a truncated chunked body, `BadStatusLine`,
            # `LineTooLong` -- escapes the other arms. Catching one subclass
            # leaves the siblings to propagate as a traceback, which bypasses
            # the `ManagedConfigError` exit path and crashes `doctor`.
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                # `urllib` wraps a connect-phase stall as `URLError`, which is
                # not a `TimeoutError`, so a blackholed host would otherwise
                # report the generic read failure.
                return _remote_status(
                    self.name,
                    self.path,
                    ProviderHealth.UNREADABLE,
                    "remote source timed out",
                    source,
                )
            # Name the failure class. One fixed string cannot separate an
            # expired certificate on the policy host -- the most
            # security-relevant failure on this boundary -- from a DNS miss, a
            # refused connection, or a local descriptor exhaustion that has
            # nothing to do with the administrator's server. The class name
            # carries no untrusted text: it is a type, not server output.
            return _remote_status(
                self.name,
                self.path,
                ProviderHealth.UNREADABLE,
                f"remote source could not be read ({type(reason).__name__})",
                source,
            )
        except ValueError as exc:
            return _remote_status(
                self.name,
                self.path,
                ProviderHealth.CORRUPT,
                str(exc),
                source,
            )
        return _parse_remote_toml(
            payload,
            name=self.name,
            path=self.path,
            source=source,
        )


@dataclass(slots=True)
class _TomlSnapshotState:
    """Mutable snapshot cell owned by a frozen provider."""

    value: TomlSnapshot | None = None
    """Last usable snapshot, or the empty failed snapshot from an initial read."""

    failure: ProviderStatus | None = None
    """Status of the most recent failed reload, kept for health reporting."""


@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Ranked provider backed by one local TOML file snapshot."""

    name: str
    path: Path | None
    """File this provider reads, or `None` for a snapshot with no known origin.

    A `None` path is not a filename to guess at. Inventing one would make
    `load` read a relative path against the process working directory, and for
    the managed tier that is a trust boundary, not a cosmetic default.
    """

    rank: int = USER_RANK
    durable: bool = True
    snapshot: TomlSnapshot | None = field(default=None, repr=False, compare=False)
    loader: Callable[[], TomlSnapshot] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _state: _TomlSnapshotState = field(
        default_factory=_TomlSnapshotState,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Seed the mutable snapshot cell when a generation is supplied."""
        self._state.value = self.snapshot

    def load(self) -> TomlSnapshot:
        """Parse the file and classify missing, unreadable, or corrupt states.

        Returns:
            Parsed data and provider health. A provider with no path reports
            `INDETERMINATE`, which is not usable: an empty read proves nothing
            about a file whose location is unknown.
        """
        if self.path is None:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    None,
                    ProviderHealth.INDETERMINATE,
                    "no path is known for this source, so it cannot be re-read",
                ),
            )
        try:
            with self.path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.MISSING,
                ),
            )
        except OSError as exc:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.UNREADABLE,
                    type(exc).__name__,
                ),
            )
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            detail = (
                "not UTF-8 encoded" if isinstance(exc, UnicodeDecodeError) else str(exc)
            )
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    detail,
                ),
            )
        if not isinstance(data, dict):
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    "top-level TOML value is not a table",
                ),
            )
        return TomlSnapshot(
            data,
            ProviderStatus(self.name, self.path, ProviderHealth.OK),
        )

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read one option from the current file snapshot.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind

        snapshot = self.current_snapshot()
        if option.kind is OptionKind.THEME_DELEGATE:
            ranked = ranked_theme_toml_value(
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        else:
            ranked = ranked_toml_value(
                option,
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        # The status of the file on disk, which is not the status of the
        # snapshot resolution just read: a failed reload keeps enforcing the
        # last readable generation while `failure` records why the current
        # contents were refused.
        failure = self._state.failure
        return self._with_rejection_diagnostic(
            ranked,
            failure or snapshot.status,
            # Values were retained only when the generation in hand is itself
            # usable. A first read that fails records a failure too, but there
            # is no earlier generation behind it - those options fall back.
            retained=snapshot.status.usable,
        )

    @staticmethod
    def _with_rejection_diagnostic(
        ranked: RankedProviderValue[object],
        status: ProviderStatus,
        *,
        retained: bool,
    ) -> RankedProviderValue[object]:
        """Attach the reason when the file on disk was rejected as a whole.

        Neither rejection is visible in the result otherwise. A file refused on
        first read coerces to `Unset` for every option, which resolution reads
        as "this source declares nothing" — indistinguishable from a file that
        omits the key. A file refused on *reload* is quieter still: resolution
        keeps returning the previous generation's values, so nothing looks
        wrong at all while the user's latest edit silently fails to apply.

        Args:
            ranked: Result already coerced from the snapshot.
            status: Health of the file on disk.
            retained: Whether resolution is still serving an earlier generation
                rather than falling through to lower tiers.

        Returns:
            The result, with a rejection diagnostic when the source is unusable.
        """
        if status.usable:
            return ranked
        location = f" ({status.path})" if status.path is not None else ""
        detail = f": {status.detail}" if status.detail else ""
        suffix = RETAINED_SOURCE_SUFFIX if retained else UNUSABLE_SOURCE_SUFFIX
        reason = (
            f"Ignoring {status.name}{location} — it is "
            f"{status.health.value}{detail} {suffix}"
        )
        return replace(ranked, diagnostics=(reason, *ranked.diagnostics))

    def status(self) -> ProviderStatus:
        """Return health for the current file snapshot.

        A failed reload reports its own health even though resolution keeps
        using the retained snapshot: diagnostics must describe the file on
        disk, not the generation still being enforced.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        state = self._state
        if state.value is None:
            self.reload()
        if state.failure is not None:
            return state.failure
        snapshot = state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot.status

    def reload(self) -> None:
        """Replace the current snapshot with a fresh file read.

        A reload the source cannot use never replaces the last usable
        snapshot. An unusable candidate carries an empty table, which
        resolution reads as "this source declares nothing"; installing it
        would drop the source's restrictions and let lower ranks win. The
        failed status is still recorded so health surfaces report the file on
        disk.
        """
        snapshot = self.loader() if self.loader is not None else self.load()
        if snapshot.status.usable:
            self._state.value = snapshot
            self._state.failure = None
        else:
            if self._state.value is None:
                self._state.value = snapshot
            self._state.failure = snapshot.status

    def current_snapshot(self) -> TomlSnapshot:
        """Return the cached snapshot, loading it on first access.

        Public so `ConfigResolver.toml_snapshot` can hand the generation this
        provider is serving to a caller building a masked variant resolver.

        Returns:
            Current parsed file snapshot.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        if self._state.value is None:
            self.reload()
        snapshot = self._state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot


@dataclass(frozen=True, slots=True)
class EnvProvider:
    """Live process-environment configuration provider."""

    name: str = "environment"
    rank: int = ENVIRONMENT_RANK
    environ: Mapping[str, str] = field(
        default_factory=lambda: os.environ,
        repr=False,
        compare=False,
    )

    @property
    def durable(self) -> bool:
        """Never durable: the environment does not survive the process.

        A property rather than a field because the coercion helpers below stamp
        durability onto every value they emit. A settable field would
        type-check, enter `__eq__`, and change nothing about masking - a lie in
        the one attribute that decides whether a tier can hide another.
        """
        return False

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read one option from the live environment.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind

        if option.kind is OptionKind.THEME_DELEGATE:
            return ranked_theme_environment_value(self.environ, rank=self.rank)
        return ranked_environment_value(option, self.environ, rank=self.rank)

    def status(self) -> ProviderStatus:
        """Return the always-healthy environment provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Keep reading the live environment without cached state."""


@dataclass(frozen=True, slots=True)
class DefaultProvider:
    """Typed manifest-default configuration provider."""

    name: str = "default"
    rank: int = DEFAULT_RANK

    @property
    def durable(self) -> bool:
        """Always durable: manifest defaults are compiled into the process.

        A property for the same reason as `EnvProvider.durable`: the value the
        helpers stamp on each result is the truth, so the attribute must not be
        able to disagree with it.
        """
        return True

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Return one option's manifest default.

        Args:
            option: Manifest option whose default should be returned.

        Returns:
            Ranked default provider result.
        """
        ranked = ranked_default_value(option, rank=self.rank)
        if isinstance(ranked.result, Unset):
            return replace(ranked, result=Found(option.default))
        return ranked

    def status(self) -> ProviderStatus:
        """Return the always-healthy default provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Retain immutable manifest defaults without cached state."""
