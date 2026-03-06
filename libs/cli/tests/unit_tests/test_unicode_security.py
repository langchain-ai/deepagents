"""Unit tests for Unicode security helpers."""

import pytest

from deepagents_cli.unicode_security import (
    CONFUSABLES,
    UnicodeIssue,
    UrlSafetyResult,
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    render_with_unicode_markers,
    strip_dangerous_unicode,
    summarize_issues,
)


def test_detect_dangerous_unicode_empty_for_safe_text() -> None:
    """Clean text should not produce Unicode issues."""
    assert detect_dangerous_unicode("hello world") == []


def test_detect_dangerous_unicode_finds_bidi_and_zero_width() -> None:
    """BiDi and zero-width controls should be identified with code points."""
    text = "a\u202eb\u200bc"
    issues = detect_dangerous_unicode(text)
    assert len(issues) == 2
    assert issues[0].codepoint == "U+202E"
    assert issues[1].codepoint == "U+200B"


def test_strip_dangerous_unicode_removes_hidden_chars() -> None:
    """Sanitizer should remove hidden controls and preserve visible text."""
    assert strip_dangerous_unicode("ap\u200bple") == "apple"


def test_render_with_unicode_markers_makes_hidden_chars_visible() -> None:
    """Marker rendering should expose hidden Unicode controls."""
    rendered = render_with_unicode_markers("a\u202eb")
    assert "U+202E" in rendered
    assert "RIGHT-TO-LEFT OVERRIDE" in rendered


def test_check_url_safety_plain_ascii_domain_is_safe() -> None:
    """A normal ASCII URL should be considered safe."""
    result = check_url_safety("https://apple.com")
    assert result.safe is True
    assert result.warnings == ()


def test_check_url_safety_cyrillic_homograph_is_unsafe() -> None:
    """Mixed-script homograph should be considered unsafe."""
    result = check_url_safety("https://аpple.com")
    assert result.safe is False
    assert any("mixes scripts" in warning for warning in result.warnings)


def test_check_url_safety_punycode_mixed_script_is_unsafe() -> None:
    """Punycode domain that decodes to mixed script should be unsafe."""
    result = check_url_safety("https://xn--pple-43d.com")
    assert result.safe is False
    assert result.decoded_domain is not None


def test_check_url_safety_non_latin_single_script_domain_is_safe() -> None:
    """A non-Latin domain without dangerous patterns should remain safe."""
    result = check_url_safety("https://例え.jp")
    assert result.safe is True


def test_check_url_safety_localhost_and_ip_are_safe() -> None:
    """Local hostnames and IP literals should not be flagged by default."""
    assert check_url_safety("https://localhost:8080").safe is True
    assert check_url_safety("https://192.168.1.1").safe is True


def test_check_url_safety_detects_hidden_unicode_in_url() -> None:
    """Hidden characters anywhere in URL should be flagged as unsafe."""
    result = check_url_safety("https://example.com/\u200badmin")
    assert result.safe is False
    assert result.issues


def test_confusables_contains_expected_script_entries() -> None:
    """Confusable table should include key entries across targeted scripts."""
    assert "\u0430" in CONFUSABLES  # Cyrillic a
    assert "\u03b1" in CONFUSABLES  # Greek alpha
    assert "\u0570" in CONFUSABLES  # Armenian ho
    assert "\uff41" in CONFUSABLES  # Fullwidth a
    assert len(CONFUSABLES) == len(set(CONFUSABLES))


# --- UnicodeIssue __post_init__ validation ---


def test_unicode_issue_rejects_multi_char() -> None:
    """UnicodeIssue should reject character with length != 1."""
    with pytest.raises(ValueError, match="single code point"):
        UnicodeIssue(position=0, character="ab", codepoint="U+0061", name="TEST")


def test_unicode_issue_rejects_mismatched_codepoint() -> None:
    """UnicodeIssue should reject codepoint that doesn't match character."""
    with pytest.raises(ValueError, match="does not match"):
        UnicodeIssue(position=0, character="a", codepoint="U+0062", name="TEST")


# --- UrlSafetyResult tuple immutability ---


def test_url_safety_result_warnings_are_tuple() -> None:
    """UrlSafetyResult.warnings should be a tuple."""
    result = check_url_safety("https://example.com")
    assert isinstance(result.warnings, tuple)
    assert isinstance(result.issues, tuple)


# --- summarize_issues truncation ---


def test_summarize_issues_within_limit() -> None:
    """When <= max_items unique issues, all should be shown."""
    issues = detect_dangerous_unicode("a\u200bb\u200cc")
    summary = summarize_issues(issues)
    assert "U+200B" in summary
    assert "U+200C" in summary
    assert "more" not in summary


def test_summarize_issues_truncates_with_overflow() -> None:
    """When > max_items unique issues, overflow suffix should appear."""
    text = "a\u200b\u200c\u200d\u200e\u200fb"
    issues = detect_dangerous_unicode(text)
    summary = summarize_issues(issues, max_items=2)
    assert "+3 more entries" in summary


def test_summarize_issues_singular_overflow() -> None:
    """Overflow of exactly 1 should use singular 'entry'."""
    text = "a\u200b\u200c\u200db"
    issues = detect_dangerous_unicode(text)
    summary = summarize_issues(issues, max_items=2)
    assert "+1 more entry" in summary


def test_summarize_issues_deduplicates() -> None:
    """Repeated codepoints should be deduplicated."""
    text = "\u200b\u200b\u200b"
    issues = detect_dangerous_unicode(text)
    summary = summarize_issues(issues)
    assert summary.count("U+200B") == 1


# --- format_warning_detail ---


def test_format_warning_detail_within_limit() -> None:
    """When warnings fit max_shown, no overflow indicator."""
    detail = format_warning_detail(("warn1", "warn2"))
    assert detail == "warn1; warn2"
    assert "more" not in detail


def test_format_warning_detail_with_overflow() -> None:
    """When warnings exceed max_shown, overflow indicator appears."""
    detail = format_warning_detail(("a", "b", "c", "d"), max_shown=2)
    assert detail == "a; b; +2 more"


# --- Punycode decode failure ---


def test_check_url_safety_invalid_punycode_is_suspicious() -> None:
    """A malformed punycode label should be flagged as suspicious."""
    result = check_url_safety("https://xn--invalid!!!.com")
    assert result.safe is False
    assert any("could not be decoded" in w for w in result.warnings)


# --- Fullwidth single-script false positive fix ---


def test_check_url_safety_pure_fullwidth_domain_is_safe() -> None:
    """A pure fullwidth Latin domain should not be flagged as confusable."""
    result = check_url_safety("https://\uff41\uff45\uff4f.com")
    # Single-script fullwidth is not a confusable mix
    assert not any("confusable" in w for w in result.warnings)


# --- No-hostname URLs ---


def test_check_url_safety_data_uri_with_hidden_unicode() -> None:
    """Hidden Unicode in a data: URI should still be flagged."""
    result = check_url_safety("data:text/html,\u200bhello")
    assert result.safe is False
    assert result.issues


# --- iter_string_values ---


def test_iter_string_values_flat_dict() -> None:
    """Flat dict should yield top-level string values."""
    result = iter_string_values({"a": "hello", "b": 42})
    assert result == [("a", "hello")]


def test_iter_string_values_nested() -> None:
    """Nested dicts and lists should be traversed."""
    data = {"outer": {"inner": "val"}, "items": ["x", {"deep": "y"}]}
    result = iter_string_values(data)
    paths = {path for path, _ in result}
    assert "outer.inner" in paths
    assert "items[0]" in paths
    assert "items[1].deep" in paths


# --- looks_like_url_key ---


def test_looks_like_url_key_simple() -> None:
    """Simple URL key names should match."""
    assert looks_like_url_key("url") is True
    assert looks_like_url_key("endpoint") is True
    assert looks_like_url_key("command") is False


def test_looks_like_url_key_dotted_path() -> None:
    """Nested key paths should match on the leaf key."""
    assert looks_like_url_key("nested.url") is True
    assert looks_like_url_key("nested.command") is False


def test_looks_like_url_key_array_indexed() -> None:
    """Array-indexed key paths should strip the index."""
    assert looks_like_url_key("urls[0]") is False  # 'urls' not in set
    assert looks_like_url_key("items[0].url") is True


def test_looks_like_url_key_case_insensitive() -> None:
    """Key matching should be case-insensitive."""
    assert looks_like_url_key("URL") is True
    assert looks_like_url_key("Base_URL") is True
