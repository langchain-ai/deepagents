"""Unicode security helpers for deceptive text and URL checks.

This module is intentionally lightweight so it can be imported in display and
approval paths without affecting startup performance.
"""

from __future__ import annotations

import ipaddress
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

_DANGEROUS_CODEPOINTS: frozenset[int] = frozenset(
    {
        # BiDi embedding/override controls
        *range(0x202A, 0x202F),
        # BiDi isolates
        *range(0x2066, 0x206A),
        # Zero-width and invisible formatting controls
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        0x2060,  # WORD JOINER
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        # Other commonly abused invisible controls
        0x00AD,  # SOFT HYPHEN
        0x034F,  # COMBINING GRAPHEME JOINER
        0x115F,  # HANGUL CHOSEONG FILLER
        0x1160,  # HANGUL JUNGSEONG FILLER
    }
)
"""Code points that should be treated as deceptive/invisible for CLI safety."""

_DANGEROUS_CHARACTERS: frozenset[str] = frozenset(
    chr(codepoint) for codepoint in _DANGEROUS_CODEPOINTS
)

# Minimal high-risk confusables for warn-level detection.
CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "\u0430": "a",  # CYRILLIC SMALL LETTER A
    "\u0435": "e",  # CYRILLIC SMALL LETTER IE
    "\u043e": "o",  # CYRILLIC SMALL LETTER O
    "\u0440": "p",  # CYRILLIC SMALL LETTER ER
    "\u0441": "c",  # CYRILLIC SMALL LETTER ES
    "\u0443": "y",  # CYRILLIC SMALL LETTER U
    "\u0445": "x",  # CYRILLIC SMALL LETTER HA
    "\u043d": "h",  # CYRILLIC SMALL LETTER EN
    "\u0456": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0458": "j",  # CYRILLIC SMALL LETTER JE
    "\u043a": "k",  # CYRILLIC SMALL LETTER KA
    "\u0455": "s",  # CYRILLIC SMALL LETTER DZE
    # Greek
    "\u03b1": "a",  # GREEK SMALL LETTER ALPHA
    "\u03b5": "e",  # GREEK SMALL LETTER EPSILON
    "\u03bf": "o",  # GREEK SMALL LETTER OMICRON
    "\u03c1": "p",  # GREEK SMALL LETTER RHO
    "\u03c7": "x",  # GREEK SMALL LETTER CHI
    "\u03ba": "k",  # GREEK SMALL LETTER KAPPA
    "\u03bd": "v",  # GREEK SMALL LETTER NU
    "\u03c4": "t",  # GREEK SMALL LETTER TAU
    # Armenian
    "\u0570": "h",  # ARMENIAN SMALL LETTER HO
    "\u0578": "n",  # ARMENIAN SMALL LETTER VO
    "\u057d": "u",  # ARMENIAN SMALL LETTER SEH
    # Fullwidth Latin examples
    "\uff41": "a",  # Fullwidth a
    "\uff45": "e",  # Fullwidth e
    "\uff4f": "o",  # Fullwidth o
}

_URL_SAFE_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost"})


@dataclass(frozen=True, slots=True)
class UnicodeIssue:
    """A dangerous Unicode character found in text.

    Attributes:
        position: Zero-based index in the original string.
        character: The raw character found in the input.
        codepoint: Uppercase code point string like `U+202E`.
        name: Unicode character name.
    """

    position: int
    character: str
    codepoint: str
    name: str


@dataclass(frozen=True, slots=True)
class UrlSafetyResult:
    """Safety analysis output for a URL string.

    Attributes:
        safe: `True` if no suspicious patterns were found.
        decoded_domain: Punycode-decoded hostname when it differs from the
            original hostname. `None` when unchanged or no hostname exists.
        warnings: Human-readable warning strings.
        issues: Dangerous Unicode issues found in the full URL.
    """

    safe: bool
    decoded_domain: str | None
    warnings: list[str]
    issues: list[UnicodeIssue]


def detect_dangerous_unicode(text: str) -> list[UnicodeIssue]:
    """Detect deceptive or hidden Unicode code points in text.

    Args:
        text: Input text to inspect.

    Returns:
        A list of `UnicodeIssue` entries in source order.
    """
    issues: list[UnicodeIssue] = []
    for position, character in enumerate(text):
        if character not in _DANGEROUS_CHARACTERS:
            continue
        issues.append(
            UnicodeIssue(
                position=position,
                character=character,
                codepoint=_format_codepoint(character),
                name=_unicode_name(character),
            )
        )
    return issues


def strip_dangerous_unicode(text: str) -> str:
    """Remove known dangerous/invisible Unicode characters from text.

    Args:
        text: Input text to sanitize.

    Returns:
        Sanitized text with dangerous characters removed.
    """
    return "".join(ch for ch in text if ch not in _DANGEROUS_CHARACTERS)


def render_with_unicode_markers(text: str) -> str:
    """Render hidden Unicode characters as explicit markers.

    Example output: `abc<U+202E RIGHT-TO-LEFT OVERRIDE>def`.

    Args:
        text: Input text to render.

    Returns:
        Text where dangerous characters are replaced with visible markers.
    """
    rendered_parts: list[str] = []
    for character in text:
        if character not in _DANGEROUS_CHARACTERS:
            rendered_parts.append(character)
            continue
        rendered_parts.append(
            f"<{_format_codepoint(character)} {_unicode_name(character)}>"
        )
    return "".join(rendered_parts)


def summarize_issues(issues: list[UnicodeIssue], *, max_items: int = 3) -> str:
    """Summarize Unicode issues for warning messages.

    Args:
        issues: A list of detected issues.
        max_items: Max unique code points to include in output.

    Returns:
        Comma-separated summary like `U+202E RIGHT-TO-LEFT OVERRIDE, ...`.
    """
    unique_entries: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        entry = f"{issue.codepoint} {issue.name}"
        if entry in seen:
            continue
        seen.add(entry)
        unique_entries.append(entry)

    if len(unique_entries) <= max_items:
        return ", ".join(unique_entries)

    displayed = ", ".join(unique_entries[:max_items])
    remainder = len(unique_entries) - max_items
    suffix = "entry" if remainder == 1 else "entries"
    return f"{displayed}, +{remainder} more {suffix}"


def check_url_safety(url: str) -> UrlSafetyResult:
    """Check a URL for suspicious Unicode and domain spoofing patterns.

    Args:
        url: URL string to inspect.

    Returns:
        `UrlSafetyResult` including decoded domain and warning details.
    """
    warnings: list[str] = []
    suspicious = False

    issues = detect_dangerous_unicode(url)
    if issues:
        suspicious = True
        warnings.append(
            f"URL contains hidden Unicode characters ({summarize_issues(issues)})"
        )

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return UrlSafetyResult(
            safe=not suspicious,
            decoded_domain=None,
            warnings=warnings,
            issues=issues,
        )

    decoded_hostname = _decode_hostname(hostname)
    decoded_domain = decoded_hostname if decoded_hostname != hostname else None
    if decoded_domain:
        warnings.append(f"Punycode domain decodes to '{decoded_domain}'")

    if _is_local_or_ip_hostname(decoded_hostname):
        return UrlSafetyResult(
            safe=not suspicious,
            decoded_domain=decoded_domain,
            warnings=warnings,
            issues=issues,
        )

    for label in _split_hostname_labels(decoded_hostname):
        scripts = _scripts_in_label(label)
        if len(scripts) > 1:
            suspicious = True
            script_names = ", ".join(sorted(scripts))
            warnings.append(f"Domain label '{label}' mixes scripts ({script_names})")

        if _label_has_suspicious_confusable_mix(label):
            suspicious = True
            warnings.append(
                f"Domain label '{label}' contains confusable Unicode characters"
            )

    return UrlSafetyResult(
        safe=not suspicious,
        decoded_domain=decoded_domain,
        warnings=warnings,
        issues=issues,
    )


def _decode_hostname(hostname: str) -> str:
    """Decode `xn--` punycode labels into Unicode labels when possible.

    Returns:
        Hostname with decodable punycode labels replaced by Unicode labels.
    """
    decoded_labels: list[str] = []
    for label in _split_hostname_labels(hostname):
        if label.startswith("xn--"):
            try:
                decoded_labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                decoded_labels.append(label)
            continue
        decoded_labels.append(label)
    return ".".join(decoded_labels)


def _split_hostname_labels(hostname: str) -> list[str]:
    """Split a hostname into non-empty labels.

    Returns:
        Hostname labels without empty entries.
    """
    return [label for label in hostname.split(".") if label]


def _is_local_or_ip_hostname(hostname: str) -> bool:
    """Return whether hostname is localhost or an IP address literal.

    Returns:
        `True` when hostname is localhost or an IP literal, else `False`.
    """
    host = hostname.strip().rstrip(".")
    if not host:
        return False

    if host.lower() in _URL_SAFE_LOCAL_HOSTS:
        return True

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _scripts_in_label(label: str) -> set[str]:
    """Collect non-common scripts used by a domain label.

    Returns:
        Set of script names used by the label, excluding common/inherited.
    """
    scripts: set[str] = set()
    for character in label:
        script = _char_script(character)
        if script in {"Common", "Inherited"}:
            continue
        scripts.add(script)
    return scripts


def _label_has_suspicious_confusable_mix(label: str) -> bool:
    """Return whether a label has likely deceptive confusable characters.

    Returns:
        `True` when the label contains risky confusable composition.
    """
    if not any(character in CONFUSABLES for character in label):
        return False

    scripts = _scripts_in_label(label)
    if len(scripts) > 1:
        return True

    return "Latin" in scripts or "Fullwidth" in scripts


def _char_script(character: str) -> str:
    """Classify a character into a coarse Unicode script bucket.

    Returns:
        One of the known script bucket labels.
    """
    name = unicodedata.name(character, "")
    category = unicodedata.category(character)

    if "FULLWIDTH LATIN" in name:
        return "Fullwidth"
    if "LATIN" in name:
        return "Latin"
    if "CYRILLIC" in name:
        return "Cyrillic"
    if "GREEK" in name:
        return "Greek"
    if "ARMENIAN" in name:
        return "Armenian"
    if any(
        token in name
        for token in (
            "CJK",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "BOPOMOFO",
            "IDEOGRAPHIC",
        )
    ):
        return "EastAsian"

    if category.startswith("M"):
        return "Inherited"
    if category[0] in {"N", "P", "S", "Z", "C"}:
        return "Common"

    return "Other"


def _format_codepoint(character: str) -> str:
    """Format character code point in `U+XXXX` uppercase form.

    Returns:
        Uppercase `U+XXXX` codepoint string.
    """
    return f"U+{ord(character):04X}"


def _unicode_name(character: str) -> str:
    """Return a stable Unicode name with a fallback for unknown code points.

    Returns:
        Unicode name string for the character.
    """
    return unicodedata.name(character, "UNKNOWN CHARACTER")
