"""Tests for cold prompt-cache policy and pricing helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest

from deepagents_code.cold_cache import (
    _LOGGED_DIAGNOSTICS,
    _MAX_LOGGED_DIAGNOSTICS,
    CacheConfidence,
    CacheWriteBucket,
    ColdCacheReason,
    ColdCacheWarning,
    PromptCachePolicy,
    RewarmEstimate,
    _warn_once,
    cache_identity_params,
    debug_stand_in_policy,
    endpoint_cache_identity,
    estimate_rewarm_cost,
    format_cache_age,
    format_cache_window,
    load_trusted_cache_endpoints,
    parse_cache_timestamp,
    resolve_prompt_cache_policy,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_rejection_log() -> None:
    """Clear the once-per-process rejection log between tests.

    Rejections are deduplicated for the life of the process, so without this a
    test asserting on `caplog` would pass or fail depending on whether an
    earlier test happened to log the same message first.
    """
    _LOGGED_DIAGNOSTICS.clear()


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "default"),
        ("", "default"),
        ("   ", "default"),
        (
            "https://Proxy.EXAMPLE.com:443/v1/#fragment",
            "https://proxy.example.com/v1",
        ),
        ("http://proxy.example.com:8080/v1/", "http://proxy.example.com:8080/v1"),
    ],
)
def test_endpoint_cache_identity_normalizes_endpoint_spelling(
    base_url: str | None, expected: str
) -> None:
    assert endpoint_cache_identity(base_url) == expected


def test_endpoint_cache_identity_preserves_query_routing() -> None:
    """Query changes must not be treated as sharing a prompt cache."""
    tenant_a = endpoint_cache_identity("https://proxy.example.com/v1?tenant=a")
    tenant_b = endpoint_cache_identity("https://proxy.example.com/v1?tenant=b")

    assert tenant_a != tenant_b


@pytest.mark.parametrize(
    "base_url",
    [
        "https://proxy.example.com/v1?api-key=sk-secret-value",
        "sk-proj-secret-value",
        "https://user:sk-secret-value@proxy.example.com:99999/v1",
        "http://[::1/v1?token=sk-secret-value",
    ],
)
def test_endpoint_cache_identity_never_embeds_endpoint_secrets(base_url: str) -> None:
    """Identities reach the checkpoint store, so they must not carry secrets.

    Covers the valid-URL, unparseable, bad-port, and malformed-IPv6 paths --
    the last two raise out of `urlparse`/`.port` and land in the `invalid:`
    branch, which historically echoed its input verbatim.
    """
    identity = endpoint_cache_identity(base_url)

    assert "sk-secret-value" not in identity
    assert "sk-proj-secret-value" not in identity


def test_endpoint_cache_identity_keeps_malformed_endpoints_distinct() -> None:
    """Digesting must not collapse bad endpoints onto each other or a good one.

    A malformed endpoint sharing an identity with the provider default would
    silently mark a switch between them as "no change".
    """
    first = endpoint_cache_identity("not a URL")
    second = endpoint_cache_identity("also not a URL")

    assert first != second
    assert first != endpoint_cache_identity(None)
    assert first != endpoint_cache_identity("https://api.openai.com/v1")
    # Stable across calls: the identity is compared against a checkpointed one.
    assert first == endpoint_cache_identity("not a URL")


def test_endpoint_cache_identity_distinguishes_ipv6_authorities() -> None:
    """IPv6 brackets must survive normalization so the port stays unambiguous."""
    assert endpoint_cache_identity("https://[::1]:8080") == "https://[::1]:8080"
    assert endpoint_cache_identity("https://[::1:8080]") == "https://[::1:8080]"
    assert endpoint_cache_identity("https://[::1]:8080") != endpoint_cache_identity(
        "https://[::1:8080]"
    )


def _policy(
    provider_name: str,
    window_seconds: int,
    confidence: CacheConfidence,
    minimum_tokens: int,
    write_bucket: CacheWriteBucket,
) -> PromptCachePolicy:
    """Build a policy positionally to keep expectations readable in tests."""
    return PromptCachePolicy(
        provider_name=provider_name,
        window_seconds=window_seconds,
        confidence=confidence,
        minimum_tokens=minimum_tokens,
        write_bucket=write_bucket,
    )


def test_anthropic_policy_ignores_user_supplied_cache_control_ttl() -> None:
    """A user `ttl` never reaches the wire, so it must not widen the window.

    `AnthropicPromptCachingMiddleware` runs inside `ConfigurableModelMiddleware`
    and overwrites `model_settings["cache_control"]` with its own 5m TTL.
    Honoring the user's `1h` here would suppress the warning for 55 minutes of
    a cache that died at five.
    """
    default = resolve_prompt_cache_policy("anthropic:claude-sonnet-4-6")
    with_ttl = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6",
        {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
    )

    assert default == _policy("Anthropic", 300, "expired", 1024, "5m")
    assert with_ttl == default


@pytest.mark.parametrize(
    ("model", "minimum"),
    [
        ("claude-opus-5", 512),
        ("claude-fable-5", 512),
        ("claude-mythos-5", 512),
        ("claude-opus-4-8", 1024),
        ("claude-sonnet-5", 1024),
        ("claude-sonnet-4-6", 1024),
        ("claude-opus-4-7", 2048),
        ("claude-mythos-preview", 2048),
        ("claude-opus-4-6", 4096),
        ("claude-opus-4-5", 4096),
        ("claude-haiku-4-5", 4096),
    ],
)
def test_resolves_anthropic_per_model_minimums(model: str, minimum: int) -> None:
    policy = resolve_prompt_cache_policy(f"anthropic:{model}")

    assert policy is not None
    assert policy.minimum_tokens == minimum


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-pro", "gpt-6"])
def test_resolves_current_openai_minimum_retention(model: str) -> None:
    policy = resolve_prompt_cache_policy(f"openai:{model}")

    # 30 minutes is the documented guaranteed minimum, but OpenAI may retain
    # the prefix longer, so past the window it may still be warm. GPT-5.6+
    # bills cache writes at a premium over plain input, hence `generic_write`.
    assert policy == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")


def test_resolves_explicit_older_openai_retention() -> None:
    in_memory = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "in_memory"},
    )
    extended = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "24h"},
    )

    # Both windows are documented maximums ("up to one hour", "a maximum, not
    # a guarantee"), so once the window passes the entry is gone -- unlike the
    # GPT-5.6+ minimum, which the provider may exceed.
    assert in_memory == _policy("OpenAI", 3600, "expired", 1024, "generic")
    assert extended == _policy("OpenAI", 86400, "expired", 1024, "generic")


def test_ignores_non_string_openai_retention() -> None:
    assert (
        resolve_prompt_cache_policy("openai:gpt-5.5", {"prompt_cache_retention": 3600})
        is None
    )


def test_estimate_rewarm_cost_respects_per_model_anthropic_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4,096-token floor rejects 3,000 tokens on Haiku but not on Opus 5."""
    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost", lambda *_args: 1.0
    )
    haiku = resolve_prompt_cache_policy("anthropic:claude-haiku-4-5")
    opus = resolve_prompt_cache_policy("anthropic:claude-opus-5")

    assert haiku is not None
    assert opus is not None
    assert estimate_rewarm_cost(3000, "anthropic:claude-haiku-4-5", haiku) is None
    assert estimate_rewarm_cost(3000, "anthropic:claude-opus-5", opus) is not None


def test_skips_unresolved_or_custom_provider_policies() -> None:
    assert resolve_prompt_cache_policy("openai:gpt-5.5") is None
    assert resolve_prompt_cache_policy("google_genai:gemini-3.6-flash") is None
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://gateway.example.com/v1",
        )
        is None
    )
    assert (
        resolve_prompt_cache_policy(
            "anthropic:claude-sonnet-4-6",
            base_url="https://gateway.example.com",
        )
        is None
    )


def test_trusted_endpoints_enable_policies_on_alternate_hosts() -> None:
    gateway = "https://gateway.example.com/v1"
    trusted = {"gateway.example.com"}

    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url=gateway, trusted_endpoints=trusted
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")
    assert resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", base_url=gateway, trusted_endpoints=trusted
    ) == _policy("Anthropic", 300, "expired", 1024, "5m")
    # A different, untrusted host on the same spec still resolves nothing.
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://other.example.com",
            trusted_endpoints=trusted,
        )
        is None
    )


_ANTHROPIC_5M = _policy("Anthropic", 300, "expired", 1024, "5m")
"""Policy a same-format Anthropic route resolves, used as the positive control.

The cross-format tests below pair each suppressed spec with a spec that *does*
resolve on the same host, so a guard that stopped firing would flip a visible
assertion rather than leave every case at `None`.
"""


@pytest.mark.parametrize(
    "base_url",
    ["https://smith.langchain.com", "https://acme.smith.langchain.com"],
)
def test_langsmith_gateway_same_format_routes_keep_policies(base_url: str) -> None:
    trusted = {"smith.langchain.com", "acme.smith.langchain.com"}

    # Bare model names are served by the wire format's own provider.
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url=base_url, trusted_endpoints=trusted
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")
    # An explicit matching prefix is the same route; the prefix must be
    # stripped before model-family detection rather than defeating it.
    assert resolve_prompt_cache_policy(
        "openai:openai/gpt-5.6", base_url=base_url, trusted_endpoints=trusted
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")
    assert resolve_prompt_cache_policy(
        "anthropic:anthropic/claude-opus-4-5",
        base_url=base_url,
        trusted_endpoints=trusted,
    ) == _policy("Anthropic", 300, "expired", 4096, "5m")


@pytest.mark.parametrize(
    "model_spec",
    [
        # Anthropic wire format routed to an OpenAI model.
        "anthropic:openai/gpt-5.6",
        # Prefixes for providers this module cannot price are crossings too:
        # the gateway still translates, so no policy may be assumed.
        "anthropic:google_genai/gemini-3",
        "anthropic:baseten/some-model",
        "anthropic:myorg/claude-opus-4-5",
    ],
)
def test_langsmith_gateway_cross_format_routes_resolve_no_policy(
    model_spec: str,
) -> None:
    gateway = "https://smith.langchain.com"
    trusted = {"smith.langchain.com"}

    # The same host and trust set resolve a policy for a same-format route,
    # so `None` here can only come from the cross-format guard.
    assert (
        resolve_prompt_cache_policy(
            "anthropic:claude-sonnet-4-6", base_url=gateway, trusted_endpoints=trusted
        )
        == _ANTHROPIC_5M
    )
    assert (
        resolve_prompt_cache_policy(
            model_spec, base_url=gateway, trusted_endpoints=trusted
        )
        is None
    )
    # An untrusted gateway resolves nothing regardless.
    assert resolve_prompt_cache_policy(model_spec, base_url=gateway) is None


@pytest.mark.parametrize(
    "host",
    [
        "smith.langchain.com",
        "smith.langchain.com.",
        "notsmith.langchain.com",
        "smith.langchain.com.evil.example",
        "gateway.example.com",
    ],
)
def test_cross_format_specs_resolve_nothing_on_any_endpoint(host: str) -> None:
    """A cross-format `provider/` prefix suppresses wherever it appears.

    `provider/model` is the routing convention of proxies generally, not of one
    known host, and a translated hop rewrites the caching fields any policy
    would assume. Scoping the guard to a single host left every other trusted
    proxy resolving a policy for the wrong provider -- one that could never be
    priced, so the warning silently never fired.
    """
    assert (
        resolve_prompt_cache_policy(
            "anthropic:openai/gpt-5.6",
            base_url=f"https://{host}/gw",
            trusted_endpoints={host},
        )
        is None
    )


def test_same_format_prefix_resolves_off_gateway() -> None:
    """The control for the suppression above: a matching prefix still works.

    Without this, deleting the prefix handling entirely would leave the
    cross-format assertions green.
    """
    assert resolve_prompt_cache_policy(
        "anthropic:anthropic/claude-opus-4-5",
        base_url="https://gateway.example.com/v1",
        trusted_endpoints={"gateway.example.com"},
    ) == resolve_prompt_cache_policy(
        "anthropic:claude-opus-4-5",
        base_url="https://gateway.example.com/v1",
        trusted_endpoints={"gateway.example.com"},
    )


def test_cross_format_suppression_warns_when_endpoints_are_trusted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A user who configured trust and still gets no warning must be told why.

    This branch returns before `endpoint_ok`, so the trust-mismatch warning
    cannot stand in for it -- without its own record the suppression is
    invisible at the default log level, and the only symptom is an unexplained
    absence of cold-cache warnings.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert (
            resolve_prompt_cache_policy(
                "openai:anthropic/claude-opus-5",
                base_url="https://gateway.example.com/v1",
                trusted_endpoints={"gateway.example.com"},
            )
            is None
        )
    assert "does not stay in the 'openai' wire format" in caplog.text
    assert "anthropic/claude-opus-5" in caplog.text


def test_cross_format_suppression_is_reported_without_trust_configured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The untrusted case is informational, but still not silent.

    The text differs from the trusted case deliberately: `_log_once` keys on the
    formatted message, so a shared string would let this `INFO` swallow the
    `WARNING` a user who later configures trust needs to see.
    """
    with caplog.at_level(logging.INFO, logger="deepagents_code.cold_cache"):
        assert (
            resolve_prompt_cache_policy(
                "openai:anthropic/claude-opus-5",
                base_url="https://gateway.example.com/v1",
            )
            is None
        )
    assert "does not stay in the 'openai' wire format" in caplog.text
    assert "even on a trusted endpoint" not in caplog.text


def test_cross_format_trusted_warning_survives_an_earlier_untrusted_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dedup must not let the informational record hide the warning.

    Same model, same provider, trust added between the two calls -- which is the
    real sequence when a user edits `config.toml` mid-session.
    """
    spec = "openai:anthropic/claude-opus-5"
    with caplog.at_level(logging.INFO, logger="deepagents_code.cold_cache"):
        resolve_prompt_cache_policy(spec, base_url="https://gateway.example.com/v1")
        caplog.clear()
        resolve_prompt_cache_policy(
            spec,
            base_url="https://gateway.example.com/v1",
            trusted_endpoints={"gateway.example.com"},
        )
    assert "even on a trusted endpoint" in caplog.text


def test_trusted_endpoints_accepts_urls_as_well_as_hostnames() -> None:
    """A URL in the trust set must not silently no-op."""
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        base_url="https://gateway.example.com/v1",
        trusted_endpoints={"https://gateway.example.com/v1"},
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")


def test_load_trusted_cache_endpoints_parses_hosts_and_urls() -> None:
    config = {
        "warnings": {
            "trusted_cache_endpoints": [
                "https://smith.langchain.com/gw",
                "gateway.example.com",
                "  spaced.example.com  ",
                "UPPER.example.com",
            ]
        }
    }

    assert load_trusted_cache_endpoints(config) == frozenset(
        {
            "smith.langchain.com",
            "gateway.example.com",
            "spaced.example.com",
            "upper.example.com",
        }
    )


@pytest.mark.parametrize(
    "entry",
    [
        "ported.example.com:8443",
        "https://ported.example.com:8443/v1",
        "api.anthropic.com@evil.example",
        "https://api.anthropic.com@evil.example/v1",
        # `urlsplit` strips these rather than failing, so without an explicit
        # check they would be stored as `gw.example.comevil` / `hostname`.
        "gw.example.com\nevil",
        "host\tname",
        # A whole-string pattern accepts both of these; neither can ever match
        # a real endpoint, so both are typos worth reporting.
        "smith..langchain.com",
        "a.-b.com",
        "-gw.example.com",
    ],
)
def test_load_trusted_cache_endpoints_rejects_reinterpretable_entries(
    entry: str,
) -> None:
    """Entries that would be stored as something else must be refused.

    `urlparse` resolves userinfo to the host *after* the `@`, so an entry
    reading as `api.anthropic.com` would silently trust `evil.example`. A
    non-default port is refused because trust is matched on the host alone
    (`endpoint_ok` compares against `_endpoint_hostname`, which discards the
    port), so honoring `:8443` as written is impossible and accepting it would
    silently widen trust to every port on that host.
    """
    config = {"warnings": {"trusted_cache_endpoints": [entry]}}

    assert load_trusted_cache_endpoints(config) == frozenset()


@pytest.mark.parametrize(
    "entry",
    [
        "https://gw.example.com:443/v1",
        "http://gw.example.com:80/v1",
        "gw.example.com:443",
    ],
)
def test_load_trusted_cache_endpoints_accepts_a_default_port(entry: str) -> None:
    """A default port names the same endpoint, so pasting a full URL works.

    Rejecting it taught users to delete the port, which is the *broader*
    action: trust is host-wide, so `gw.example.com` covers every port. The
    guidance must not push toward the wider grant.
    """
    config = {"warnings": {"trusted_cache_endpoints": [entry]}}

    assert load_trusted_cache_endpoints(config) == frozenset({"gw.example.com"})


@pytest.mark.parametrize(
    ("base_url", "entry"),
    [
        # Root dot on the endpoint, bare entry.
        ("https://gw.example.com./v1", "gw.example.com"),
        # Bare endpoint, root dot on the entry.
        ("https://gw.example.com/v1", "gw.example.com."),
        # Both spellings fully qualified.
        ("https://gw.example.com./v1", "gw.example.com."),
    ],
)
def test_trailing_root_dot_matches_the_bare_spelling(base_url: str, entry: str) -> None:
    """Both sides are root-dot-stripped, so the two spellings compare equal.

    A same-format spec is used deliberately: a cross-format one resolves `None`
    with or without the strip, so it could not fail if the normalization were
    deleted.
    """
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6", base_url=base_url, trusted_endpoints={entry}
        )
        is not None
    )


def test_trust_is_host_wide_across_ports() -> None:
    """Document the actual matching semantics the rejection message now states.

    `endpoint_ok` compares `_endpoint_hostname(base_url)`, which discards the
    port, so a trusted host is trusted on every port it is reached on. This is
    the property that makes a port-scoped entry unhonorable rather than merely
    inconvenient.
    """
    for base_url in (
        "https://gw.example.com/v1",
        "https://gw.example.com:9999/v1",
        "http://gw.example.com:8080/v1",
    ):
        assert (
            resolve_prompt_cache_policy(
                "openai:gpt-5.6",
                base_url=base_url,
                trusted_endpoints={"gw.example.com"},
            )
            is not None
        )


@pytest.mark.parametrize(
    "entry",
    [
        "",
        42,
        None,
        True,
        ["nested"],
        "not a url at all ::",
        "not a url at all",
        # `urlparse` accepts `smith.langchain,com` as a host, so the shape
        # check is what keeps a comma-for-a-dot typo out of the trust set.
        "smith.langchain,com",
        # Wildcards are not supported; storing one would never match.
        "*.example.com",
        "ftp://gateway.example.com",
        "https://",
    ],
)
def test_load_trusted_cache_endpoints_rejects_and_logs_bad_entries(
    entry: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = {"warnings": {"trusted_cache_endpoints": [entry]}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset()

    assert "trusted_cache_endpoints" in caplog.text


def test_load_trusted_cache_endpoints_tolerates_missing_or_malformed() -> None:
    assert load_trusted_cache_endpoints({}) == frozenset()
    assert load_trusted_cache_endpoints({"warnings": []}) == frozenset()
    assert load_trusted_cache_endpoints({"warnings": {}}) == frozenset()


def test_load_trusted_cache_endpoints_logs_a_non_table_section(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`warnings = "off"` discards every option in the section, not just this one."""
    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints({"warnings": "off"}) == frozenset()

    assert "expected a table" in caplog.text


def test_load_trusted_cache_endpoints_confirms_a_good_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A user who fixes a bad entry needs confirmation on the same surface.

    Rejections warn; without a matching success record at a level the Debug
    Console shows by default, a corrected config is indistinguishable from one
    that is still being ignored.
    """
    config = {"warnings": {"trusted_cache_endpoints": ["gateway.example.com"]}}

    with caplog.at_level(logging.INFO, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset(
            {"gateway.example.com"}
        )

    assert "Trusting cache endpoints: gateway.example.com" in caplog.text


def test_trusted_set_that_never_matches_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Trust configured but never matched is a misconfiguration, not a default.

    Its only other symptom is an absence of warnings, which looks exactly like
    a healthy cache -- so it must reach the Debug Console at the default level
    rather than sitting in a debug record nobody sees.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert (
            resolve_prompt_cache_policy(
                "openai:gpt-5.6",
                base_url="https://api.gateway.example.com/v1",
                trusted_endpoints={"gateway.example.com"},
            )
            is None
        )

    assert "api.gateway.example.com" in caplog.text
    assert "trusted set" in caplog.text


def test_untrusted_custom_endpoint_stays_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The control for the report above: no trust configured is not an error.

    A custom endpoint with no trust list is the ordinary opt-out, so it must
    not produce a warning on every turn.
    """
    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert (
            resolve_prompt_cache_policy(
                "openai:gpt-5.6", base_url="https://proxy.example.com/v1"
            )
            is None
        )

    assert caplog.text == ""


def test_log_once_cap_releases_instead_of_growing() -> None:
    """The dedup set is bounded, and overflow re-warns rather than going quiet.

    Keyed by formatted message, it would otherwise accumulate one entry per
    distinct typo for the life of the process.
    """
    for index in range(_MAX_LOGGED_DIAGNOSTICS * 2):
        _warn_once("rejected entry %s", index)

    assert len(_LOGGED_DIAGNOSTICS) <= _MAX_LOGGED_DIAGNOSTICS


def test_load_trusted_cache_endpoints_logs_a_non_list_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare string is the likeliest hand-edit and must not fail silently."""
    config = {"warnings": {"trusted_cache_endpoints": "smith.langchain.com"}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset()

    assert "expected a list" in caplog.text


@pytest.mark.parametrize(
    ("bucket", "cold_details"),
    [
        # Pre-5.6 OpenAI bills a miss at the plain input rate, so the cold side
        # carries no cache-write detail at all -- tagging one applies a write
        # premium the provider never charges.
        ("generic", None),
        # GPT-5.6+ bills a miss as a cache write; the generic `cache_write`
        # alias is what `_cache_write_counts` forwards to the catalog.
        ("generic_write", {"cache_write": 50_000}),
        ("5m", {"ephemeral_5m_input_tokens": 50_000}),
    ],
)
def test_estimate_rewarm_cost_uses_policy_bucket(
    monkeypatch: pytest.MonkeyPatch,
    bucket: CacheWriteBucket,
    cold_details: dict[str, int] | None,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_estimate(
        usage: dict[str, Any],
        model_name: str,
        provider: str,
    ) -> float:
        calls.append(usage)
        assert model_name == "model"
        assert provider == "anthropic"
        details = usage.get("input_token_details", {})
        return 0.1 if "cache_read" in details else 1.25

    monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_estimate)
    policy = _policy("Anthropic", 300, "expired", 1024, bucket)

    estimate = estimate_rewarm_cost(50_000, "anthropic:model", policy)

    assert estimate is not None
    assert estimate.cold_cost_usd == pytest.approx(1.25)
    assert estimate.incremental_cost_usd == pytest.approx(1.15)
    assert calls[0]["input_tokens"] == 50_000
    assert calls[0]["input_token_details"] == {"cache_read": 50_000}
    assert calls[1]["input_tokens"] == 50_000
    assert calls[1].get("input_token_details") == cold_details


def test_generic_write_bucket_prices_a_miss_at_the_catalog_write_rate() -> None:
    """A GPT-5.6+ cold turn must cost the 1.25x cache-write rate, not plain input.

    Guards the real pricing path rather than a fake: omitting the write detail
    prices a 100k-token GPT-5.6 miss at plain input, which can skip the
    warning for a turn that actually costs more than the threshold.
    """
    from deepagents_code.cost_tracking import estimate_cost

    policy = resolve_prompt_cache_policy("openai:gpt-5.6-terra")
    assert policy is not None

    estimate = estimate_rewarm_cost(100_000, "openai:gpt-5.6-terra", policy)
    plain_input = estimate_cost(
        {"input_tokens": 100_000, "output_tokens": 0, "total_tokens": 100_000},
        "gpt-5.6-terra",
        "openai",
    )

    assert estimate is not None
    assert plain_input is not None
    assert estimate.cold_cost_usd == pytest.approx(plain_input * 1.25)


def test_generic_bucket_prices_a_miss_at_the_plain_input_rate() -> None:
    """A pre-5.6 OpenAI cold turn must cost exactly the uncached input price.

    The plain `generic` bucket forwards no cache-write detail; even if the
    catalog grows a write rate for the matched model, the miss must stay at
    the input rate the provider actually charges.
    """
    from deepagents_code.cost_tracking import estimate_cost

    policy = resolve_prompt_cache_policy(
        "openai:gpt-5.5", {"prompt_cache_retention": "24h"}
    )
    assert policy is not None
    assert policy.write_bucket == "generic"

    estimate = estimate_rewarm_cost(100_000, "openai:gpt-5.5", policy)
    plain_input = estimate_cost(
        {"input_tokens": 100_000, "output_tokens": 0, "total_tokens": 100_000},
        "gpt-5.5",
        "openai",
    )

    assert estimate is not None
    assert plain_input is not None
    assert estimate.cold_cost_usd == pytest.approx(plain_input)


def test_estimate_rewarm_cost_requires_cacheable_and_priceable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    assert estimate_rewarm_cost(100, "openai:gpt-5.6", policy) is None

    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost",
        lambda *_args: None,
    )
    assert estimate_rewarm_cost(5000, "openai:gpt-5.6", policy) is None


def test_parse_cache_timestamp_requires_timezone() -> None:
    timestamp = datetime(2026, 8, 11, 12, 30, tzinfo=UTC)

    assert parse_cache_timestamp(timestamp.isoformat()) == timestamp
    assert parse_cache_timestamp("2026-08-11T12:30:00") is None
    assert parse_cache_timestamp("not-a-time") is None
    assert parse_cache_timestamp(None) is None


def test_parse_cache_timestamp_normalizes_non_utc_offsets() -> None:
    """A `+05:00` checkpoint must not read as five hours of extra idle time."""
    offset = timezone(timedelta(hours=5))
    local = datetime(2026, 8, 11, 17, 30, tzinfo=offset)

    parsed = parse_cache_timestamp(local.isoformat())

    assert parsed == datetime(2026, 8, 11, 12, 30, tzinfo=UTC)
    assert parsed is not None
    assert parsed.tzinfo is UTC


def test_explicit_official_endpoints_still_resolve() -> None:
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url="https://api.openai.com/v1"
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")
    anthropic = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", base_url="https://api.anthropic.com"
    )
    assert anthropic == _policy("Anthropic", 300, "expired", 1024, "5m")
    # Non-HTTP schemes are not the official API however they parse.
    assert (
        resolve_prompt_cache_policy("openai:gpt-5.6", base_url="ftp://api.openai.com")
        is None
    )


def test_non_dict_cache_control_falls_back_to_default_window() -> None:
    assert resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", {"cache_control": "ephemeral"}
    ) == _policy("Anthropic", 300, "expired", 1024, "5m")


def test_cache_time_formatting() -> None:
    assert format_cache_age(11_520) == "3h 12m"
    assert format_cache_age(300) == "5m"
    assert format_cache_age(0) == "0m"
    assert format_cache_age(-30) == "0m"
    assert format_cache_window(1800) == "30m"
    assert format_cache_window(3600) == "1h"
    assert format_cache_window(86400) == "24h"


@pytest.mark.parametrize(
    "endpoint",
    ["https://smith.langchain.com/gw", "https://gateway.example.com/v1"],
)
def test_prefixed_same_format_route_can_be_priced(endpoint: str) -> None:
    """A same-format route must price, not just resolve a policy.

    `resolve_prompt_cache_policy` strips the `provider/` prefix before model
    family detection; if that stripping does not also reach pricing, the route
    resolves a policy whose estimate is always `None` and the warning silently
    never fires -- indistinguishable, to the user, from having nothing to warn
    about. The bare-name control pins that the two spellings agree, and the
    non-gateway endpoint pins that this holds for any trusted proxy.
    """
    host = endpoint.split("/")[2]
    trusted = {host}

    for prefixed, bare in (
        ("openai:openai/gpt-5.6", "openai:gpt-5.6"),
        ("anthropic:anthropic/claude-opus-4-5", "anthropic:claude-opus-4-5"),
    ):
        policy = resolve_prompt_cache_policy(
            prefixed, base_url=endpoint, trusted_endpoints=trusted
        )
        assert policy is not None
        # The prefix must not defeat family detection either: an unstripped
        # name falls back to the default minimum rather than the model's own.
        assert policy == resolve_prompt_cache_policy(
            bare, base_url=endpoint, trusted_endpoints=trusted
        )
        estimate = estimate_rewarm_cost(150_000, prefixed, policy)
        assert estimate is not None
        assert estimate.incremental_cost_usd > 0
        assert estimate == estimate_rewarm_cost(150_000, bare, policy)


def test_estimate_rewarm_cost_suppresses_cross_format_routes() -> None:
    """Pricing applies the same crossing guard as policy resolution."""
    policy = _policy("Anthropic", 300, "expired", 1024, "5m")

    assert estimate_rewarm_cost(150_000, "anthropic:openai/gpt-5.6", policy) is None


def test_trusted_endpoints_reject_what_the_loader_rejects() -> None:
    """The resolver and the config loader must agree on what is trustable.

    A value the loader drops and logs must not become trusted just because a
    caller passed it directly -- otherwise the validation regex is a config
    lint rather than the invariant of the trust set.
    """
    for bad in ("my_gw.example.com", "*.example.com", "smith.langchain,com", "::1"):
        assert (
            load_trusted_cache_endpoints(
                {"warnings": {"trusted_cache_endpoints": [bad]}}
            )
            == frozenset()
        )
        assert (
            resolve_prompt_cache_policy(
                "openai:gpt-5.6",
                base_url=f"https://{bad}",
                trusted_endpoints={bad},
            )
            is None
        )


def test_trusted_endpoints_tolerate_non_string_entries() -> None:
    """A non-str entry is dropped, not raised on, as the loader drops it.

    The annotation rules this out statically; the check matters because the
    values ultimately originate in a hand-edited TOML file, and raising here
    would surface as an unrelated "could not evaluate the warning" error.
    """
    mixed = {None, 42, True, "gw.example.com"}

    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        base_url="https://gw.example.com",
        trusted_endpoints=mixed,  # ty: ignore
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")


def test_trust_does_not_extend_to_subdomains() -> None:
    """Trust is per exact host, unlike `is_langsmith_gateway_host`.

    Widening this to a suffix match would silently broaden a security boundary,
    so the non-inheritance is pinned rather than left as an implementation
    detail.
    """
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://gw.example.com",
            trusted_endpoints={"example.com"},
        )
        is None
    )


def test_gateway_prefix_with_empty_remainder_is_suppressed() -> None:
    """`openai/` names no model, so nothing can be priced defensibly."""
    assert (
        resolve_prompt_cache_policy(
            "openai:openai/",
            base_url="https://smith.langchain.com",
            trusted_endpoints={"smith.langchain.com"},
        )
        is None
    )


def test_gateway_prefix_match_is_case_insensitive() -> None:
    """A capitalized prefix is the same route, not a crossing."""
    assert resolve_prompt_cache_policy(
        "openai:OpenAI/gpt-5.6",
        base_url="https://smith.langchain.com",
        trusted_endpoints={"smith.langchain.com"},
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")


def test_load_trusted_cache_endpoints_keeps_good_entries_beside_bad(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One typo must not discard the entries the user got right."""
    config = {
        "warnings": {
            "trusted_cache_endpoints": [
                "good.example.com",
                "smith.langchain,com",
                42,
                "https://other.example.com/v1",
            ]
        }
    }

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset(
            {"good.example.com", "other.example.com"}
        )

    assert "smith.langchain,com" in caplog.text
    assert "42" in caplog.text


def test_load_trusted_cache_endpoints_logs_each_rejection_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The config is re-read every turn; the warning must not repeat forever.

    The debug log buffer is bounded, so a per-turn warning would evict every
    other record over a long session -- degrading the only surface on which
    these warnings are visible.
    """
    config = {"warnings": {"trusted_cache_endpoints": ["smith.langchain,com"]}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        for _ in range(5):
            assert load_trusted_cache_endpoints(config) == frozenset()

    assert caplog.text.count("smith.langchain,com") == 1


def test_load_trusted_cache_endpoints_reads_config_from_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The no-argument path is the one production uses and must stay wired.

    Every other test supplies `config` explicitly, so a rename or move of
    `load_config_toml` would ship green and fail only inside the TUI.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[warnings]\ntrusted_cache_endpoints = ["smith.langchain.com"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path)

    assert load_trusted_cache_endpoints() == frozenset({"smith.langchain.com"})


def test_load_trusted_cache_endpoints_prefers_managed_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed endpoint trust replaces a conflicting user configuration."""
    monkeypatch.setattr(
        "deepagents_code.config_manifest.load_config_toml",
        lambda: {"warnings": {"trusted_cache_endpoints": ["user.example.com"]}},
    )
    monkeypatch.setattr(
        "deepagents_code.config_manifest.load_managed_config_toml",
        lambda: {"warnings": {"trusted_cache_endpoints": ["managed.example.com"]}},
    )

    assert load_trusted_cache_endpoints() == frozenset({"managed.example.com"})


def test_load_trusted_cache_endpoints_survives_an_undecodable_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mis-encoded config falls back to defaults instead of raising.

    `UnicodeDecodeError` subclasses `ValueError`, not `OSError` or
    `TOMLDecodeError`, so it needs its own handling for the documented
    "malformed content never raises" contract to hold.
    """
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(b"\xff\xfe[warnings]\n")
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path)

    assert load_trusted_cache_endpoints() == frozenset()

    assert load_trusted_cache_endpoints() == frozenset()


@pytest.mark.parametrize(
    "spec",
    ["anthropic", "", "anthropic:", "anthropic:   ", ":claude-opus-5"],
    ids=["no-colon", "empty", "empty-model", "blank-model", "empty-provider"],
)
def test_malformed_model_specs_resolve_no_policy(spec: str) -> None:
    """An unparseable spec must not resolve a policy to price against."""
    assert resolve_prompt_cache_policy(spec) is None


def test_haiku_35_minimum_matches_its_real_model_id() -> None:
    """Haiku 3.5 ships as `claude-3-5-haiku-*`, not `claude-haiku-3-5`.

    A prefix in the family-then-version style would never match and would
    silently fall through to the 1,024-token default.
    """
    policy = resolve_prompt_cache_policy("anthropic:claude-3-5-haiku-latest")

    assert policy is not None
    assert policy.minimum_tokens == 2048  # Documented minimum.


def test_rewarm_estimate_rejects_impossible_figures() -> None:
    """The delta is part of the total, so it can never exceed it."""
    with pytest.raises(ValueError, match="cannot exceed"):
        RewarmEstimate(cold_cost_usd=0.10, incremental_cost_usd=0.50)
    with pytest.raises(ValueError, match="non-negative"):
        RewarmEstimate(cold_cost_usd=-1.0, incremental_cost_usd=0.0)


def test_debug_stand_in_policy_tracks_the_anthropic_constants() -> None:
    """The stand-in must not re-hardcode values that live as constants."""
    real = resolve_prompt_cache_policy("anthropic:claude-sonnet-4-6")
    stand_in = debug_stand_in_policy()

    assert real is not None
    assert stand_in.window_seconds == real.window_seconds
    assert stand_in.minimum_tokens == real.minimum_tokens
    assert stand_in.write_bucket == real.write_bucket


def test_explicit_retention_widens_the_gpt_56_minimum_window() -> None:
    """An explicitly configured retention outranks the 30-minute guarantee.

    The two knobs are independent: `prompt_cache_retention` states a maximum
    lifetime while the GPT-5.6+ guarantee states a minimum one. Warning a user
    who configured `24h` at the 30-minute mark would contradict their own
    configuration, and the modal would tell them they had exceeded a window
    they explicitly widened.
    """
    policy = resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        {"prompt_cache_retention": "24h"},
    )

    # Write pricing still follows the model version, not the retention knob.
    assert policy == _policy("OpenAI", 86400, "expired", 1024, "generic_write")


def test_explicit_in_memory_retention_applies_to_gpt_56() -> None:
    policy = resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        {"prompt_cache_retention": "in_memory"},
    )

    assert policy == _policy("OpenAI", 3600, "expired", 1024, "generic_write")


def test_unparseable_gpt_56_retention_falls_back_to_the_minimum() -> None:
    """A non-string retention is ignored, leaving the documented guarantee."""
    policy = resolve_prompt_cache_policy(
        "openai:gpt-5.6",
        {"prompt_cache_retention": 3600},
    )

    assert policy == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic_write")


def test_cache_identity_params_ignores_non_cache_knobs() -> None:
    """Only cache-participating params take part in the identity comparison.

    `/effort` rewrites `reasoning_effort` wholesale on every change. Comparing
    whole param maps would report a model change for a knob that does not move
    the cached prefix, and the modal would assert that the prefix "cannot be
    reused" on a false premise.
    """
    before = {"reasoning_effort": "low", "temperature": 0.2}
    after = {"reasoning_effort": "high", "temperature": 0.9, "max_tokens": 512}

    assert cache_identity_params(before) == cache_identity_params(after) == {}


def test_cache_identity_params_ignores_overwritten_cache_control() -> None:
    """`cache_control` is not part of the effective cache identity.

    `AnthropicPromptCachingMiddleware` overwrites `model_settings["cache_control"]`
    with its own 5-minute TTL on every Anthropic request, so a user-supplied
    value never reaches the wire. Treating it as identity would report a model
    change for requests whose cache settings are identical on the wire.
    """
    before = {"cache_control": {"type": "ephemeral", "ttl": "5m"}}
    after = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}

    assert cache_identity_params(before) == cache_identity_params(after) == {}


def test_cache_identity_params_keeps_cache_affecting_knobs() -> None:
    params = {
        "reasoning_effort": "high",
        "prompt_cache_retention": "24h",
        "prompt_cache_key": "thread-1",
    }

    assert cache_identity_params(params) == {
        "prompt_cache_retention": "24h",
        "prompt_cache_key": "thread-1",
    }
    # A change to one of them must still register.
    assert cache_identity_params(params) != cache_identity_params(
        {**params, "prompt_cache_retention": "in_memory"}
    )


def test_cache_identity_params_handles_absent_params() -> None:
    assert cache_identity_params(None) == {}
    assert cache_identity_params({}) == {}


@pytest.mark.parametrize(
    ("cold", "incremental"),
    [
        (float("nan"), float("nan")),
        (float("inf"), 1.0),
        (1.0, float("nan")),
    ],
)
def test_rewarm_estimate_rejects_non_finite_costs(
    cold: float, incremental: float
) -> None:
    """The type enforces the finiteness its docstring promises.

    `NaN` satisfies neither the sign nor the ordering guard (`nan < 0` and
    `nan > nan` are both `False`), so without an explicit check it would
    construct cleanly and only fail later inside cost formatting.
    """
    with pytest.raises(ValueError, match="must be finite"):
        RewarmEstimate(cold_cost_usd=cold, incremental_cost_usd=incremental)


def test_non_finite_price_is_reported_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt pricing data warns instead of reading as "no price published"."""
    from deepagents_code import cost_tracking

    monkeypatch.setattr(cost_tracking, "estimate_cost", lambda *_a, **_k: float("inf"))

    with caplog.at_level("WARNING"):
        estimate = estimate_rewarm_cost(
            50_000,
            "anthropic:claude-opus-4-5",
            _policy("Anthropic", 300, "expired", 1024, "5m"),
        )

    assert estimate is None
    assert "non-finite" in caplog.text


@pytest.mark.parametrize(
    ("age_seconds", "reason"),
    [
        (None, "idle"),
        (None, "identity_changed"),
        (60.0, "age_unknown"),
    ],
)
def test_cold_cache_warning_enforces_the_age_reason_pairing(
    age_seconds: float | None, reason: ColdCacheReason
) -> None:
    """An age and its reason cannot be split apart.

    Without this guard an `age_unknown` warning carrying a defaulted reason
    rendered "idle for 0m" -- an idle duration the caller had explicitly
    determined it did not know.
    """
    with pytest.raises(ValueError, match="does not pair with"):
        ColdCacheWarning(
            policy=_policy("Anthropic", 300, "expired", 1024, "5m"),
            estimate=RewarmEstimate(cold_cost_usd=1.0, incremental_cost_usd=0.9),
            context_tokens=50_000,
            age_seconds=age_seconds,
            reason=reason,
        )


def test_unparseable_base_url_warns_before_disabling_the_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed base URL disables cost warnings, so it cannot be silent.

    The caller's skip line reports only that *a* base URL was configured, so
    it cannot distinguish a deliberate gateway from a typo that switches the
    protection off.
    """
    with caplog.at_level("WARNING"):
        policy = resolve_prompt_cache_policy(
            "anthropic:claude-opus-4-5",
            base_url="http://[",
        )

    assert policy is None
    assert "Could not parse configured base URL" in caplog.text
