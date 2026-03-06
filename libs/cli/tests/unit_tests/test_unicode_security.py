"""Unit tests for Unicode security helpers."""

from deepagents_cli.unicode_security import (
    CONFUSABLES,
    check_url_safety,
    detect_dangerous_unicode,
    render_with_unicode_markers,
    strip_dangerous_unicode,
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
    assert result.warnings == []


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
