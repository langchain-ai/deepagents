"""Provider policy and pricing helpers for cold prompt-cache warnings."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, assert_never
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping, Set as AbstractSet

logger = logging.getLogger(__name__)

COLD_CACHE_WARNING_KEY = "cold-cache"
"""Suppression key for the cold-cache warning in `[warnings].suppress`.

Named rather than spelled inline at each site: the reader, the writer, and the
`/notifications` settings row must agree exactly, and a typo in any one would
leave the warning firing after the user asked it to stop, with no error
anywhere to explain why.
"""
CacheConfidence = Literal["expired", "may_be_cold"]
"""How firmly a passed retention window implies the cached prefix is gone.

`expired` is used when the window is a documented *maximum* (Anthropic's TTL,
OpenAI's `prompt_cache_retention` ceilings): once it passes, the entry is gone.
`may_be_cold` is used when the window is a documented *minimum* (GPT-5.6+),
where the provider is only guaranteed to have kept the entry that long and may
well have kept it longer.
"""

CacheWriteBucket = Literal["generic", "generic_write", "5m"]
"""Which pricing treatment a cold (cache-writing) request receives.

`5m` prices Anthropic's five-minute ephemeral write premium. `generic_write`
tags the miss as a cache write, which GPT-5.6+ bills above the plain input
rate; omitting the detail would price the miss at plain input. The premium's
current magnitude comes from the pricing catalog, not from here. `generic`
covers misses with no write surcharge, which is how OpenAI priced prompt
caching before GPT-5.6.

`generic_write` is assigned by *model-name version* (see
`_openai_uses_thirty_minute_cache`), not by inspecting the catalog, and that
coupling is not enforced: a future 5.6-family model with no published cache
rates would be tagged `generic_write` and priced at plain input by
`estimate_cost`, which drops detail keys the catalog cannot price. Models
carrying no cache rates at all already exist in the catalog, so this is a
reachable state rather than a theoretical one.
"""

ColdCacheReason = Literal["idle", "identity_changed", "age_unknown"]
"""Why a turn is treated as facing a cold prompt cache.

`idle` means the last request is older than the policy's retention window.
`identity_changed` means the model or its cache-affecting params differ from
the last successful turn, so the cached prefix cannot be reused regardless of
age. `age_unknown` means there is no usable record of when this thread last
reached the model -- a checkpoint written before cold-cache tracking existed,
or one whose timestamp could not be parsed -- so the cache cannot be assumed
warm. Each maps to distinct modal copy; they are not interchangeable, and in
particular `age_unknown` must not be reported as `identity_changed`, which
would claim a model change that never happened.
"""

CACHE_IDENTITY_PARAM_KEYS = frozenset(
    {
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
    }
)
"""Invocation params that select or invalidate a provider cache entry.

The identity check compares only these. Comparing whole `model_params` maps
instead would report a model change for every unrelated knob -- `/effort`
rewrites `reasoning_effort` wholesale, and `temperature` or `max_tokens` are
just as inert for caching -- and the modal would then assert that "the previous
cached prefix cannot be reused" when nothing about the prefix moved. A modal
that fires on a false premise trains users into the permanent suppression.

`cache_control` is deliberately absent: `AnthropicPromptCachingMiddleware`
overwrites `model_settings["cache_control"]` with its own TTL on every
Anthropic request (see `_ANTHROPIC_MIDDLEWARE_TTL_SECONDS`), so a
user-supplied value never reaches the wire. Comparing it would report an
identity change for a setting the effective requests never differed on.
"""

_OPENAI_MODEL_VERSION = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")
_DEFAULT_ENDPOINT_PORTS = {"http": 80, "https": 443}

_ANTHROPIC_MINIMUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 512),
    ("claude-fable-5", 512),
    ("claude-mythos-5", 512),
    ("claude-opus-4-7", 2048),
    ("claude-mythos-preview", 2048),
    ("claude-3-5-haiku", 2048),
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
)
"""Prefixes for Claude models whose cache minimum differs from 1,024 tokens.

From the per-model minimums in Anthropic's prompt-caching docs:
https://platform.claude.com/docs/en/build-with-claude/prompt-caching

Prefixes must match real model ids. Haiku 3.5 predates the family-then-version
naming and ships as `claude-3-5-haiku-*`, so a `claude-haiku-3-5` prefix would
never match and would silently fall through to the 1,024 default. Order
matters: the first matching prefix wins, so a more specific prefix must precede
any shorter prefix of it. No current pair nests; the rule constrains future
additions (`claude-opus-5-mini` would have to precede `claude-opus-5`).
"""

_ANTHROPIC_DEFAULT_MINIMUM_TOKENS = 1024
"""Cache minimum for the Claude models the table above does not name."""

_OPENAI_MINIMUM_TOKENS = 1024
"""Minimum cacheable prefix OpenAI documents, independent of Anthropic's."""

_ANTHROPIC_MIDDLEWARE_TTL_SECONDS = 300
"""Retention implied by the `cache_control` this stack actually sends.

`AnthropicPromptCachingMiddleware` runs *inside* `ConfigurableModelMiddleware`
and rewrites `model_settings["cache_control"]` with its own `ttl` on every
Anthropic request (its only gate is `isinstance(request.model, ChatAnthropic)`;
`min_messages_to_cache` defaults to 0). The `ttl` is 5m, the middleware default
this stack never overrides. A user-supplied
`cache_control.ttl` in `model_params` is therefore overwritten before the
request leaves the process, so honoring it here would promise an hour of
retention the API never agreed to and suppress the warning for 55 minutes of a
dead cache. Read the middleware's effective TTL before reintroducing a longer
window.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptCachePolicy:
    """Prompt-cache behavior needed to decide and price a warning.

    Keyword-only because `window_seconds` and `minimum_tokens` are adjacent
    bare ints: positionally, transposing them builds a plausible-looking policy
    that silently misprices and mis-gates.
    """

    provider_name: str
    """Display name of the provider whose endpoint was validated."""

    window_seconds: int
    """Retention window, past which `confidence` describes what is known."""

    confidence: CacheConfidence
    """Whether `window_seconds` is a documented maximum or minimum."""

    minimum_tokens: int
    """Smallest prefix the provider will cache at all."""

    write_bucket: CacheWriteBucket
    """Pricing treatment for the cold request; see `estimate_rewarm_cost`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RewarmEstimate:
    """Estimated input cost for a cold prefix and its warm-cache delta.

    Both figures are USD, non-negative, and finite; `incremental_cost_usd` is
    the part of `cold_cost_usd` that a cache hit would have avoided, so it
    never exceeds it.

    Keyword-only for the same reason as `PromptCachePolicy`: these are adjacent
    bare floats, and transposing them positionally yields copy that reads fine
    ("may cost up to ~$0.02 ... roughly ~$3.40 more than a warm cache hit")
    while being arithmetically impossible.
    """

    cold_cost_usd: float
    """Input spend to send the prefix uncached."""

    incremental_cost_usd: float
    """How much of `cold_cost_usd` a warm cache would have saved."""

    def __post_init__(self) -> None:
        """Enforce the documented finiteness, ordering, and sign invariants.

        Raises:
            ValueError: When either figure is non-finite or negative, or the
                delta exceeds the total it is a part of.
        """
        # Checked first, and separately: `NaN` satisfies neither comparison
        # below (`nan < 0` and `nan > nan` are both `False`), so it would slide
        # past both guards and reach `format_cost_estimate`, where the
        # magnitude arithmetic raises far from the value's real origin.
        if not math.isfinite(self.cold_cost_usd) or not math.isfinite(
            self.incremental_cost_usd
        ):
            msg = (
                f"RewarmEstimate costs must be finite, got "
                f"cold={self.cold_cost_usd!r}, "
                f"incremental={self.incremental_cost_usd!r}"
            )
            raise ValueError(msg)
        if self.cold_cost_usd < 0 or self.incremental_cost_usd < 0:
            msg = (
                f"RewarmEstimate costs must be non-negative, got "
                f"cold={self.cold_cost_usd!r}, "
                f"incremental={self.incremental_cost_usd!r}"
            )
            raise ValueError(msg)
        if self.incremental_cost_usd > self.cold_cost_usd:
            msg = (
                f"RewarmEstimate incremental cost {self.incremental_cost_usd!r} "
                f"cannot exceed the cold cost {self.cold_cost_usd!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ColdCacheWarning:
    """Validated data needed to render one advisory warning.

    Constructed only after every gate has passed -- a policy resolved, the
    prefix cleared the provider's cache minimum, and the priced delta reached
    the configured threshold -- so the modal renders it without re-deciding
    anything.
    """

    policy: PromptCachePolicy
    estimate: RewarmEstimate
    context_tokens: int

    age_seconds: float | None
    """Idle time since the last successful turn.

    `None` exactly when `reason` is `age_unknown`, which is the only case where
    no usable request time exists. Optional rather than a sentinel so the modal
    cannot render an age it does not have.
    """

    reason: ColdCacheReason
    """Why the cache is treated as cold; selects the modal's copy."""

    def __post_init__(self) -> None:
        """Enforce the documented `age_seconds`/`reason` pairing.

        Raises:
            ValueError: When an age is present for `age_unknown`, or absent for
                any other reason.
        """
        if (self.age_seconds is None) != (self.reason == "age_unknown"):
            msg = (
                f"ColdCacheWarning age_seconds={self.age_seconds!r} does not "
                f"pair with reason={self.reason!r}: an age is required except "
                f"for 'age_unknown', which must have none"
            )
            raise ValueError(msg)


def debug_stand_in_policy() -> PromptCachePolicy:
    """Build the placeholder policy used by `DEEPAGENTS_CODE_DEBUG_COLD_CACHE`.

    Keeps the modal reachable on providers with no documented cache policy.
    Lives here rather than in the caller so the Anthropic window and minimum
    stay tied to `_ANTHROPIC_MIDDLEWARE_TTL_SECONDS` and
    `_ANTHROPIC_DEFAULT_MINIMUM_TOKENS` instead of being re-hardcoded, which
    would silently drift the moment either constant is revised.

    The provider name is deliberately Anthropic's: under the debug flag the
    modal may therefore cite Anthropic retention while a different provider is
    active. The figures are illustrative in that mode, not real estimates.

    Returns:
        Stand-in policy shaped like Anthropic's.
    """
    return PromptCachePolicy(
        provider_name="Anthropic",
        window_seconds=_ANTHROPIC_MIDDLEWARE_TTL_SECONDS,
        confidence="expired",
        minimum_tokens=_ANTHROPIC_DEFAULT_MINIMUM_TOKENS,
        write_bucket="5m",
    )


def _opaque_digest(value: str) -> str:
    """Reduce endpoint text to a stable token, keeping secrets out of it.

    Identities are written to the checkpoint store (`_last_cache_endpoint`) and
    only ever compared for equality, so the two parts of an endpoint most
    likely to carry a secret -- the query, and any value too malformed to parse
    -- are recorded as a digest instead of verbatim text. An endpoint that
    authenticates via `?api-key=`, or a key pasted into the base-URL field,
    must not have that value durably copied into a session database nobody
    thinks to inspect.

    This is narrower than `doctor._sanitize_endpoint`, which keeps only
    `scheme://host[:port]` and so also withholds the path.
    `endpoint_cache_identity` keeps the path verbatim because proxies route on
    it, which means a key embedded in a *path* is still recorded in the clear.
    Digesting the path would make the identity useless for its one job, so the
    residual is accepted rather than closed.

    Truncation to 16 hex characters bounds what the checkpoint carries; a
    collision would merely suppress one warning, so the full digest buys
    nothing here. Note the digest is unsalted, so it confirms a *guessed*
    low-entropy value -- irrelevant in practice, since the real credential
    already sits in the local credential store, but it is not a one-way
    guarantee against a known-plaintext check.

    Returns:
        A short hex digest of *value*.
    """
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _normalized_endpoint_cache_identity(base_url: str) -> str | None:
    """Normalize a valid HTTP endpoint, if possible.

    Returns:
        The normalized endpoint identity, or `None` when `base_url` is not an
        HTTP URL with a hostname.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    scheme = parsed.scheme.lower()
    hostname = host.lower().removesuffix(".")
    port = parsed.port
    # `parsed.hostname` strips IPv6 brackets; without them a trailing port is
    # ambiguous, so distinct endpoints like `https://[::1]:8080` and
    # `https://[::1:8080]` would serialize to the same identity.
    authority = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and port != _DEFAULT_ENDPOINT_PORTS[scheme]:
        authority = f"{authority}:{port}"
    path = parsed.path.rstrip("/")
    # Digested, not dropped: a query can route to a separate backend (so it
    # must stay part of the identity) but can equally carry an auth token (so
    # it must not be stored). Equality is all this value is used for, and a
    # digest preserves that exactly.
    query = f"?{_opaque_digest(parsed.query)}" if parsed.query else ""
    return f"{scheme}://{authority}{path}{query}"


def endpoint_cache_identity(base_url: str | None) -> str:
    """Return a stable identity for the endpoint that owns a prompt cache.

    A missing endpoint means the provider's default API. Only scheme, host,
    port, path and a digest of the query are significant; every other spelling
    detail is normalized away, including host/scheme case, a trailing slash, a
    default port, a trailing root dot, fragments, userinfo, and `;params`.
    Path and query stay significant because proxies can route them to separate
    backends; the query does so via `_opaque_digest`, which keeps a
    credential-bearing query out of the checkpoint. Path case is preserved --
    proxies may route on it -- which also means a credential embedded in a path
    is recorded verbatim; see `_opaque_digest` for why that residual is
    accepted.

    The result is opaque: compare it for equality, never parse it.

    Args:
        base_url: Resolved provider endpoint, or `None` for its default API.

    Returns:
        A checkpoint-safe endpoint identity.
    """
    if base_url is None or not base_url.strip():
        return "default"
    stripped = base_url.strip()
    try:
        normalized = _normalized_endpoint_cache_identity(stripped)
    except ValueError:
        # Malformed values stay distinct identities. This is deliberately
        # conservative: a bad endpoint must never be considered cache-equivalent
        # to the provider default or to a valid endpoint. It is digested rather
        # than echoed because this branch is reached exactly when the field
        # holds something unexpected -- a pasted API key being the case that
        # must not reach disk.
        return f"invalid:{_opaque_digest(stripped)}"
    if normalized is not None:
        return normalized
    return f"invalid:{_opaque_digest(stripped)}"


def _official_endpoint(base_url: str | None, hostname: str) -> bool:
    """Return whether an optional endpoint targets the provider's official API."""
    if not base_url:
        return True
    try:
        parsed = urlparse(base_url)
    except ValueError:
        # Logged here rather than left to the caller's "no documented policy"
        # debug line, which reports only that *a* base URL was configured and
        # so cannot distinguish a deliberate gateway (a correct skip) from a
        # typo that silently disables cost warnings for this provider.
        logger.warning(
            "Could not parse configured base URL %r; treating it as a custom "
            "endpoint, which disables prompt-cache warnings for this provider",
            base_url,
        )
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname == hostname


def _endpoint_hostname(base_url: str) -> str | None:
    """Extract a lowercase hostname from an endpoint URL.

    A trailing root dot is removed. Both sides of the trust comparison come
    through this helper -- configured entries via `_trusted_entry_hostname` and
    the live endpoint via `endpoint_ok` -- so stripping here is what lets the
    fully-qualified spelling (`gw.example.com.`) match an entry written the bare
    way, and the reverse. Without it the two spellings name the same server but
    never compare equal, and trust silently fails to apply.

    Returns:
        The lowercase hostname, or `None` when the scheme is not `http`/`https`
        or the host is empty, whitespace-padded, or only a root dot.
    """
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.hostname
    if not host or host != host.strip():
        return None
    normalized = host.lower().removesuffix(".")
    return normalized or None


_TRUSTED_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
"""Shape each dot-separated label of a trusted-endpoint entry must have.

`urlparse` accepts almost any junk as a host -- `smith.langchain,com` and
`not a url` both survive it -- which would silently populate the trust set
with an entry that can never match a real endpoint. Validating per label
instead means a typo is reported rather than stored, including the two most
likely in a hostname list that a single whole-string pattern lets through: a
doubled dot (`smith..langchain.com`, an empty label) and a label edged with a
hyphen (`a.-b.com`). IPv6 literals are not accepted (IPv4 literals are); a
proxy addressed by raw IPv6 must be trusted by DNS name.
"""


def _trusted_host_shape_ok(host: str) -> bool:
    """Return whether a reduced hostname is shaped like a hostname.

    Args:
        host: Lowercase, root-dot-stripped hostname.

    Returns:
        `True` when every dot-separated label is well formed.
    """
    labels = host.split(".")
    return all(_TRUSTED_LABEL.match(label) for label in labels)


def _trusted_entry_hostname(entry: object) -> str | None:
    """Reduce one configured trust entry to a hostname.

    Entries carrying userinfo are rejected rather than reduced: it would
    otherwise be silently reinterpreted as something the user did not write, as
    `api.anthropic.com@evil.example` parses to the host *after* the `@`, so an
    entry that reads as trusting Anthropic would trust `evil.example` instead.
    Whitespace inside an entry is rejected for the same reason -- `urlsplit`
    strips a tab or newline rather than failing, so an entry whose host carries
    an embedded newline before `evil` would otherwise be stored as the single
    host `gw.example.comevil`.

    A *default* port is accepted and discarded (`https://gw.example.com:443/v1`
    names the same endpoint as the portless spelling, and pasting a full URL is
    the obvious thing to do). Any other port is rejected, because trust is
    matched on the host alone -- `endpoint_ok` compares against
    `_endpoint_hostname`, which discards the port -- so a port-scoped entry
    could not be honored as written and would silently widen to every port on
    that host. Rejecting keeps the module's stated bargain that a typo is
    reported rather than stored.

    Args:
        entry: Raw value from the TOML list; any type, validated here.

    Returns:
        The lowercase hostname, or `None` when the entry is unusable.
    """
    if not isinstance(entry, str) or not entry.strip():
        return None
    candidate = entry.strip()
    # Checked before parsing: `urlsplit` would strip these rather than reject.
    if any(character.isspace() for character in candidate):
        return None
    # Bare hosts are accepted alongside URLs for convenience.
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        parsed = urlparse(candidate)
        authority = parsed.netloc
        port = parsed.port
    except ValueError:
        return None
    if "@" in authority:
        return None
    scheme = parsed.scheme.lower()
    if port is not None and port != _DEFAULT_ENDPOINT_PORTS.get(scheme):
        return None
    host = _endpoint_hostname(candidate)
    if host is None or not _trusted_host_shape_ok(host):
        return None
    return host


_LOGGED_DIAGNOSTICS: set[str] = set()
"""Config-diagnostic messages already emitted, so each fires once per process.

Shared by every record `_log_once` carries, not just the entry rejections:
the accepted-host confirmation, the endpoint/trust-set mismatch warning, and
the cross-format suppression notice all key into this one set. A burst of one
kind can therefore evict the memory of another, which only ever costs a
duplicate log line.

`load_trusted_cache_endpoints` re-reads `config.toml` on every turn so a live
edit takes effect without a restart. Without this, one malformed entry would
warn once per turn, crowding out the other warnings in the bounded debug-log
buffer (which partitions retention per level, so the damage is confined to
that level -- see `_debug_buffer.InMemoryLogBuffer`).

Bounded, because the key embeds the rejected entry: a config edited repeatedly
into new bad states would otherwise accumulate one string per distinct typo
for the life of the process. On overflow the whole set is dropped rather than
evicted one by one -- re-warning about a still-broken entry is the harmless
direction, and it keeps a live-edit session from going permanently quiet.

Mutated from worker threads (`app` and `configurable_model` both reach it via
`asyncio.to_thread`). The check-then-clear-then-add below is not atomic, so a
concurrent overflow can drop a key that was just added; a repeated log line is
the harmless direction, so this is left unlocked deliberately.
"""

_MAX_LOGGED_DIAGNOSTICS = 64
"""Cap on `_LOGGED_DIAGNOSTICS` before it is cleared wholesale."""


def _log_once(level: int, message: str, *args: object) -> None:
    """Emit a config-diagnostic record the first time this process produces it.

    Args:
        level: `logging` level. Kept at `INFO` or above -- these records exist
            to explain a silently disabled warning, and `DEBUG` does not reach
            the Debug Console unless debug logging was enabled at startup.
        message: `%`-style format string, also the deduplication key once
            formatted.
        *args: Format arguments.
    """
    formatted = message % args
    if formatted in _LOGGED_DIAGNOSTICS:
        return
    if len(_LOGGED_DIAGNOSTICS) >= _MAX_LOGGED_DIAGNOSTICS:
        _LOGGED_DIAGNOSTICS.clear()
    _LOGGED_DIAGNOSTICS.add(formatted)
    logger.log(level, "%s", formatted)


def _warn_once(message: str, *args: object) -> None:
    """Emit a rejection warning the first time this process produces it."""
    _log_once(logging.WARNING, message, *args)


def load_trusted_cache_endpoints(
    config: dict[str, Any] | None = None,
) -> frozenset[str]:
    """Read `[warnings].trusted_cache_endpoints` as a set of hostnames.

    Entries declare that an alternate endpoint forwards cache-affecting request
    fields (`cache_control`, `prompt_cache_key`, `prompt_cache_retention`) and
    honors the upstream provider's documented retention.

    Trust is matched on the exact host: trusting `example.com` does not trust
    `gw.example.com`. Each host a request may actually reach must be listed.

    Malformed *content* never raises: an unusable entry is dropped and a value
    that is not a list is ignored wholesale. Because a dropped entry silently
    leaves the warning disabled -- the opposite of what the user edited the
    file to achieve -- the offending entry is logged by value and a non-list
    value by type. Each distinct rejection is logged once per process.

    Args:
        config: Parsed user `config.toml` mapping. When omitted, the user
            configuration is loaded from disk and resolved with managed
            configuration, which takes precedence.

    Returns:
        Lowercase hostnames of configured trusted endpoints (possibly empty).
    """
    if config is None:
        from deepagents_code.config_manifest import (
            get_option,
            load_config_toml,
            resolve_scalar,
        )

        option = get_option("warnings.trusted_cache_endpoints")
        if option is None:
            return frozenset()
        entries, _ = resolve_scalar(option, toml_data=load_config_toml())
        # The manifest has no default for this optional structured setting.
        # Preserve the absent-setting behavior without changing diagnostics
        # for malformed configured values.
        if entries is None:
            entries = []
        config = {"warnings": {"trusted_cache_endpoints": entries}}
    warnings_section = config.get("warnings", {})
    if not isinstance(warnings_section, dict):
        # `warnings = "off"` reads like a plausible toggle, and silently
        # discards every option in the section rather than just this one.
        _warn_once(
            "Ignoring [warnings] in config.toml (expected a table, got %s)",
            type(warnings_section).__name__,
        )
        return frozenset()
    entries = warnings_section.get("trusted_cache_endpoints", [])
    if entries == []:
        return frozenset()
    if not isinstance(entries, list):
        _warn_once(
            "Ignoring [warnings].trusted_cache_endpoints in config.toml "
            "(expected a list of hostnames, got %s)",
            type(entries).__name__,
        )
        return frozenset()
    hosts: set[str] = set()
    for entry in entries:
        host = _trusted_entry_hostname(entry)
        if host is None:
            _warn_once(
                "Ignoring [warnings].trusted_cache_endpoints entry %r in "
                "config.toml (expected a bare hostname or http(s) URL, with no "
                "user:password prefix; trust is matched on the host alone, so "
                "a non-default port cannot be honored -- drop it only if you "
                "mean to trust that host on every port)",
                entry,
            )
            continue
        hosts.add(host)
    if hosts:
        # Deduplicated and emitted at a level that reaches the Debug Console by
        # default, so a user who corrects a bad entry gets confirmation on the
        # surface where they saw the complaint. Once per distinct host set,
        # not once per turn.
        _log_once(
            logging.INFO, "Trusting cache endpoints: %s", ", ".join(sorted(hosts))
        )
    return frozenset(hosts)


def _effective_model_name(
    provider: str,
    model_name: str,
) -> str | None:
    """Resolve the model name a policy lookup and pricing should use.

    Gateways and proxies read a `provider/model` prefix in the model field to
    route a request to a provider other than the one the wire format implies
    (an OpenAI-format request carrying `anthropic/claude-...`). Such a hop is
    translated between API formats, and translation rewrites or drops the very
    fields a policy assumes (`cache_control`, `prompt_cache_*`), so the
    upstream provider's documented retention no longer describes what happens.

    Any prefix that does not match the wire-format provider is therefore
    treated as a crossing, including prefixes for providers this module cannot
    price -- suppressing a warning is safe, whereas pricing a route whose cache
    semantics were rewritten is not. A matching prefix (`openai/gpt-5.6` in
    OpenAI format) is a same-provider route, forwarded untranslated; the prefix
    is stripped so model-family detection and pricing see the bare name they
    expect.

    The rule is applied wherever the prefix appears, not only on the LangSmith
    gateway: `provider/model` is the naming convention of proxies generally
    (LiteLLM-style deployments included), and those are exactly the endpoints
    this module lets users trust. Scoping it to one known host would leave a
    prefixed name on every other trusted proxy resolving a policy that could
    never be priced -- so the warning silently never fires -- while the
    unstripped name also defeats model-family detection and yields the wrong
    cache minimum. On a provider's official API no legitimate model name
    carries a prefix, so the rule is inert there.

    Args:
        provider: Wire-format provider from the `provider:model` spec. Both
            sides of the comparison are normalized here, so an unnormalized
            value cannot read every prefixed route as a crossing.
        model_name: Model portion of the spec, as sent to the endpoint.

    Returns:
        The model name to price with, or `None` when the route crosses formats
        or carries a prefix with an empty remainder (`openai/`). A name with no
        prefix is returned unchanged.
    """
    if "/" not in model_name:
        return model_name
    prefix, remainder = model_name.split("/", 1)
    if prefix.strip().lower() != provider.strip().lower() or not remainder.strip():
        return None
    return remainder.strip()


def _openai_uses_thirty_minute_cache(model_name: str) -> bool:
    """Return whether an OpenAI model belongs to the GPT-5.6-or-newer family."""
    match = _OPENAI_MODEL_VERSION.match(model_name.lower())
    if match is None:
        return False
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    return (major, minor) >= (5, 6)


def _anthropic_minimum_tokens(model_name: str) -> int:
    """Return the documented minimum cacheable prefix for a Claude model."""
    normalized = model_name.lower()
    return next(
        (
            minimum
            for prefix, minimum in _ANTHROPIC_MINIMUM_TOKENS
            if normalized.startswith(prefix)
        ),
        _ANTHROPIC_DEFAULT_MINIMUM_TOKENS,
    )


def resolve_prompt_cache_policy(
    model_spec: str,
    model_params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
    trusted_endpoints: AbstractSet[str] | None = None,
) -> PromptCachePolicy | None:
    """Resolve a documented cache policy for one effective model invocation.

    Policies apply only when the endpoint is the provider's official API or a
    user-declared trusted endpoint (see `load_trusted_cache_endpoints`).

    A route that crosses wire formats (e.g. an OpenAI-format request routed to
    an Anthropic model, spelled `openai:anthropic/claude-...`) resolves nothing,
    because the translation such a hop requires rewrites or drops the caching
    fields the policy assumes. See `_effective_model_name`. Cross-format routing
    that is *not* spelled in the model name cannot be detected -- declaring an
    endpoint trusted asserts that it does not do this.

    Args:
        model_spec: `provider:model` identifier for the invocation.
        model_params: Request params that affect retention, if any.
        base_url: Resolved endpoint, or `None` for the provider default.
        trusted_endpoints: Hostnames the user has declared trusted, as returned
            by `load_trusted_cache_endpoints`. Full http(s) URLs are accepted
            too and reduced to their host; entries that are not a usable
            hostname are dropped, exactly as the loader drops them. Matching is
            per exact host -- a trusted `example.com` does not cover
            `gw.example.com`.

    Returns:
        Matching policy, or `None` when retention cannot be resolved safely.
    """
    if ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not model_name:
        return None
    params = model_params or {}

    # Reduced through the same validator the config loader uses, so a value is
    # never trusted here that `load_trusted_cache_endpoints` would reject --
    # and a URL, which callers reasonably reach for, is not a silent no-op.
    trusted = {
        host
        for entry in trusted_endpoints or ()
        if (host := _trusted_entry_hostname(entry)) is not None
    }
    effective_model = _effective_model_name(provider, model_name)
    if effective_model is None:
        # Surfaced for the same reason the host mismatch below is: the only
        # other symptom is an unexplained absence of warnings, and this branch
        # returns before `endpoint_ok` runs, so the trust-mismatch warning
        # cannot stand in for it. A user who edited `config.toml` specifically
        # to enable these warnings and still gets none is told why. Warned when
        # trust is configured (that edit is evidence they want the warnings),
        # informational otherwise; `DEBUG` would not reach the Debug Console.
        #
        # The two levels carry different text on purpose: `_log_once` keys on
        # the *formatted* message, so a shared string would let an earlier
        # `INFO` swallow the `WARNING` that a user who has since configured
        # trust is the one who needs.
        if trusted:
            _warn_once(
                "No cold-cache policy: model %r does not stay in the %r wire "
                "format, so the provider's documented cache retention cannot "
                "be assumed even on a trusted endpoint",
                model_name,
                provider,
            )
        else:
            _log_once(
                logging.INFO,
                "No cold-cache policy: model %r does not stay in the %r wire "
                "format, so the provider's documented cache retention cannot "
                "be assumed for it",
                model_name,
                provider,
            )
        return None
    model_name = effective_model

    def endpoint_ok(official_hostname: str) -> bool:
        # `official_hostname` gates only the official-API branch. Trust is
        # declared per endpoint, not per provider, so a trusted host serves
        # whichever provider is routed through it.
        if _official_endpoint(base_url, official_hostname):
            return True
        actual_host = _endpoint_hostname(base_url) if base_url else None
        if trusted and actual_host in trusted:
            return True
        if trusted:
            # A configured trust set that never matches is unambiguously a
            # misconfiguration -- typically trusting `smith.langchain.com`
            # while requests go to `api.smith.langchain.com`. It is warned
            # rather than logged at debug because its only other symptom is an
            # unexplained absence of warnings, and debug records do not reach
            # the Debug Console at the default log level. Deduplicated, so a
            # standing mismatch does not repeat every turn. Only the host is
            # named: the full endpoint may carry credentials.
            _warn_once(
                "No cold-cache policy: endpoint host %r is neither %s nor in "
                "the trusted set %s (see [warnings].trusted_cache_endpoints)",
                actual_host or "(unparseable)",
                official_hostname,
                sorted(trusted),
            )
            return False
        logger.debug(
            "No cache policy: endpoint host %r is not %s and no endpoints are trusted",
            actual_host or "(unparseable)",
            official_hostname,
        )
        return False

    if provider == "anthropic":
        if not endpoint_ok("api.anthropic.com"):
            return None
        # Deliberately ignores `params["cache_control"]`: see
        # `_ANTHROPIC_MIDDLEWARE_TTL_SECONDS` for why a user-set `ttl` never
        # reaches the wire.
        return PromptCachePolicy(
            provider_name="Anthropic",
            window_seconds=_ANTHROPIC_MIDDLEWARE_TTL_SECONDS,
            confidence="expired",
            minimum_tokens=_anthropic_minimum_tokens(model_name),
            write_bucket="5m",
        )

    if provider != "openai" or not endpoint_ok("api.openai.com"):
        return None

    # Write pricing follows the model version, independent of retention: only
    # GPT-5.6+ bills a miss as a cache write.
    write_bucket: CacheWriteBucket = (
        "generic_write" if _openai_uses_thirty_minute_cache(model_name) else "generic"
    )

    # `in_memory` and `24h` are documented *maximums* ("up to one hour", "a
    # maximum, not a guarantee"): entries may be evicted earlier, so a warning
    # is only defensible once the maximum has passed -- at which point the
    # entry is gone rather than merely doubtful.
    #
    # Checked before the GPT-5.6+ minimum because the two knobs are
    # independent: `prompt_cache_retention` states a maximum lifetime while the
    # 5.6+ guarantee states a minimum one, and an explicitly configured
    # retention is the later, firmer bound. Warning a user who asked for `24h`
    # at the 30-minute mark would contradict their own configuration.
    # https://platform.openai.com/docs/guides/prompt-caching
    retention = params.get("prompt_cache_retention")
    retention_windows = {"in_memory": 3600, "24h": 86400}
    window = retention_windows.get(retention) if isinstance(retention, str) else None
    if window is not None:
        return PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=window,
            confidence="expired",
            minimum_tokens=_OPENAI_MINIMUM_TOKENS,
            write_bucket=write_bucket,
        )

    if write_bucket == "generic_write":
        # 30 minutes is the documented guaranteed *minimum* for GPT-5.6+ with
        # no explicit retention configured, but OpenAI may retain the prefix
        # longer, so past the window it can only be treated as possibly cold.
        # https://platform.openai.com/docs/guides/prompt-caching
        return PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=1800,
            confidence="may_be_cold",
            minimum_tokens=_OPENAI_MINIMUM_TOKENS,
            write_bucket=write_bucket,
        )
    return None


def cache_identity_params(model_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the params that participate in prompt-cache identity.

    Args:
        model_params: Full invocation params, or `None`.

    Returns:
        Only the `CACHE_IDENTITY_PARAM_KEYS` entries present, so two calls can
            be compared without unrelated knobs reading as a cache change.
    """
    if not model_params:
        return {}
    return {
        key: value
        for key, value in model_params.items()
        if key in CACHE_IDENTITY_PARAM_KEYS
    }


def estimate_rewarm_cost(
    context_tokens: int,
    model_spec: str,
    policy: PromptCachePolicy,
) -> RewarmEstimate | None:
    """Estimate cold input spend and the incremental cost over a cache hit.

    Prices are derived by running two synthetic usage payloads through the
    ordinary `estimate_cost` path -- one billed as a full cache read, one as a
    cold request -- so the catalog stays the single source of truth. Outputs
    are zeroed on both sides so the delta is input-only.

    A `provider/model` prefix is resolved through `_effective_model_name`, the
    same helper `resolve_prompt_cache_policy` uses, so a spec that resolved a
    policy is priced under the name that policy was chosen for. Letting the two
    disagree is what makes a warning silently never fire.

    Args:
        context_tokens: Prefix size to price.
        model_spec: `provider:model` identifier for the invocation.
        policy: Resolved policy, which selects the cache-write price bucket.

    Returns:
        Price estimate, or `None` when the prefix is below the provider's cache
            minimum or the usage cannot be priced defensibly.
    """
    if context_tokens < policy.minimum_tokens or ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    provider = provider.strip()
    model_name = model_name.strip()
    if not provider or not model_name:
        return None
    effective_model = _effective_model_name(provider, model_name)
    if effective_model is None:
        return None
    model_name = effective_model

    warm_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
        "input_token_details": {"cache_read": context_tokens},
    }
    cold_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
    }
    match policy.write_bucket:
        case "5m":
            # Anthropic bills a write premium over the ordinary input rate; the
            # detail key must stay in sync with
            # `cost_tracking._cache_write_counts`.
            cold_usage["input_token_details"] = {
                "ephemeral_5m_input_tokens": context_tokens
            }
        case "generic_write":
            # GPT-5.6+ bills a miss as a cache write at 1.25x the input rate,
            # so the write detail must reach `estimate_cost`; omitting it would
            # price the miss at plain input. `cache_write` is the generic alias
            # `_cache_write_counts` reads -- it is load-bearing here, not a
            # defensive spelling.
            cold_usage["input_token_details"] = {"cache_write": context_tokens}
        case "generic":
            # No cache-write detail at all: pre-5.6 OpenAI bills a miss at the
            # ordinary input rate, so tagging those tokens as a cache write
            # would apply a premium the provider never charges and overstate
            # both the displayed cost and the threshold comparison.
            pass
        case _:  # pragma: no cover - exhaustiveness guard
            assert_never(policy.write_bucket)

    from deepagents_code.cost_tracking import estimate_cost

    warm_cost = estimate_cost(warm_usage, model_name, provider)
    cold_cost = estimate_cost(cold_usage, model_name, provider)
    if warm_cost is None or cold_cost is None:
        return None
    if not math.isfinite(warm_cost) or not math.isfinite(cold_cost):
        # Distinct from "the catalog has no price for this model", which is the
        # ordinary `None` above: a non-finite price is corrupt pricing data,
        # and the caller's shared debug line would file it under the benign
        # case.
        logger.warning(
            "Skipping cold-cache warning: %s priced to a non-finite value "
            "(warm=%r, cold=%r), which is a pricing-catalog defect",
            model_spec,
            warm_cost,
            cold_cost,
        )
        return None
    # Clamp both sides before subtracting. Clamping only the total would let a
    # negative `cold_cost` produce an incremental figure larger than the cold
    # figure it is supposed to be a part of.
    cold = max(cold_cost, 0.0)
    warm = max(warm_cost, 0.0)
    return RewarmEstimate(
        cold_cost_usd=cold,
        incremental_cost_usd=max(cold - warm, 0.0),
    )


def parse_cache_timestamp(value: object) -> datetime | None:
    """Parse a persisted UTC timestamp, rejecting malformed or naive values.

    Returns:
        UTC datetime, or `None` when the value is unusable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def format_cache_age(seconds: float) -> str:
    """Format elapsed cache age for compact modal copy.

    Returns:
        Compact hours/minutes label.
    """
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


def format_cache_window(seconds: int) -> str:
    """Format a provider cache window for compact modal copy.

    Returns:
        Compact hours or minutes label.
    """
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"
