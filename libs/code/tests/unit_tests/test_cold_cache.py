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


def test_load_trusted_cache_endpoints_logs_a_non_list_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bare string is the likeliest hand-edit and must not fail silently."""
    config = {"warnings": {"trusted_cache_endpoints": "smith.langchain.com"}}

    with caplog.at_level(logging.WARNING, logger="deepagents_code.cold_cache"):
        assert load_trusted_cache_endpoints(config) == frozenset()

    assert "expected a list" in caplog.text


def test_parse_cache_timestamp_normalizes_non_utc_offsets() -> None:
    """A `+05:00` checkpoint must not read as five hours of extra idle time."""
    offset = timezone(timedelta(hours=5))
    local = datetime(2026, 8, 11, 17, 30, tzinfo=offset)

    parsed = parse_cache_timestamp(local.isoformat())

    assert parsed == datetime(2026, 8, 11, 12, 30, tzinfo=UTC)
    assert parsed is not None
    assert parsed.tzinfo is UTC


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
    tmp_path: Path,
) -> None:
    """Managed endpoint trust replaces a conflicting user configuration."""
    from unit_tests.conftest import redirect_managed_config

    (tmp_path / "config.toml").write_text(
        '[warnings]\ntrusted_cache_endpoints = ["user.example.com"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed_config.toml"
    managed.write_text(
        '[warnings]\ntrusted_cache_endpoints = ["managed.example.com"]\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

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
