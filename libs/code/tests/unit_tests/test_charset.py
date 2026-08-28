"""Tests for charset mode configuration and glyph selection."""

from __future__ import annotations

import os
import sys
from dataclasses import fields
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepagents_code._env_vars import HIDE_SPLASH_VERSION
from deepagents_code.config import (
    _ASCII_BANNER,
    _UNICODE_BANNER,
    ASCII_GLYPHS,
    UNICODE_GLYPHS,
    CharsetMode,
    Glyphs,
    __version__,
    _detect_charset_mode,
    get_banner,
    get_glyphs,
    is_ascii_mode,
    reset_glyphs_cache,
)


@pytest.fixture(autouse=True)
def _restore_glyphs_cache() -> Iterator[None]:
    """Keep the process-global glyph cache from leaking across tests.

    Several tests force the charset to ASCII or Unicode and leave the detected
    mode in `config`'s module-level caches; xdist scheduling then decides which
    unrelated test inherits that state. Reset before *and* after each test so
    the ambient mode is always recomputed from the environment.
    """
    reset_glyphs_cache()
    yield
    reset_glyphs_cache()


class TestCharsetMode:
    """Tests for CharsetMode enum."""


class TestGlyphs:
    """Tests for Glyphs dataclass."""

    def test_unicode_glyphs_are_unicode(self) -> None:
        """Test that UNICODE_GLYPHS contains non-ASCII characters."""
        # These should all be non-ASCII Unicode characters
        assert ord(UNICODE_GLYPHS.tool_prefix) > 127
        assert ord(UNICODE_GLYPHS.ellipsis) > 127
        assert ord(UNICODE_GLYPHS.checkmark) > 127
        assert ord(UNICODE_GLYPHS.error) > 127
        assert ord(UNICODE_GLYPHS.circle_empty) > 127
        assert ord(UNICODE_GLYPHS.circle_filled) > 127
        assert UNICODE_GLYPHS.square_filled == "■"
        assert ord(UNICODE_GLYPHS.checkbox_empty) > 127
        assert ord(UNICODE_GLYPHS.checkbox_checked) > 127
        assert ord(UNICODE_GLYPHS.output_prefix) > 127
        assert ord(UNICODE_GLYPHS.pause) > 127
        assert ord(UNICODE_GLYPHS.newline) > 127
        assert ord(UNICODE_GLYPHS.warning) > 127
        assert ord(UNICODE_GLYPHS.arrow_up) > 127
        assert ord(UNICODE_GLYPHS.arrow_down) > 127
        assert ord(UNICODE_GLYPHS.bullet) > 127
        assert ord(UNICODE_GLYPHS.cursor) > 127
        # Spinner frames are braille characters
        for frame in UNICODE_GLYPHS.spinner_frames:
            assert ord(frame) > 127
        # Box-drawing characters
        assert ord(UNICODE_GLYPHS.box_horizontal) > 127
        assert ord(UNICODE_GLYPHS.hunk_break) > 127

    def test_ascii_glyphs_are_ascii(self) -> None:
        """Test that every ASCII glyph field contains only ASCII characters."""
        for field in fields(Glyphs):
            value = getattr(ASCII_GLYPHS, field.name)
            values = value if isinstance(value, tuple) else (value,)
            assert all(text.isascii() for text in values)

    def test_legacy_ascii_glyphs_are_ascii(self) -> None:
        """Keep explicit coverage of the established ASCII glyph values."""
        for char in ASCII_GLYPHS.tool_prefix:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.ellipsis:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.checkmark:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.error:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.circle_empty:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.circle_filled:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.square_filled:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.checkbox_empty:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.checkbox_checked:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.output_prefix:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.pause:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.newline:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.warning:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.arrow_up:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.arrow_down:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.bullet:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.cursor:
            assert ord(char) < 128
        for frame in ASCII_GLYPHS.spinner_frames:
            for char in frame:
                assert ord(char) < 128
        for char in ASCII_GLYPHS.box_horizontal:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.hunk_break:
            assert ord(char) < 128


class TestDetectCharsetMode:
    """Tests for _detect_charset_mode function."""

    @patch.dict("os.environ", {"DEEPAGENTS_CODE_UI_CHARSET_MODE": "ascii"}, clear=False)
    def test_prefixed_env_var_is_honored(self) -> None:
        """The `DEEPAGENTS_CODE_` prefixed variant resolves the mode."""
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("UI_CHARSET_MODE", None)
            assert _detect_charset_mode() == CharsetMode.ASCII

    @patch.dict(
        "os.environ",
        {"UI_CHARSET_MODE": "ascii", "DEEPAGENTS_CODE_UI_CHARSET_MODE": "unicode"},
        clear=False,
    )
    def test_prefixed_env_var_overrides_canonical(self) -> None:
        """The prefixed variant wins over the canonical one, matching runtime."""
        assert _detect_charset_mode() == CharsetMode.UNICODE


class TestGetGlyphs:
    """Tests for get_glyphs function."""


class TestGlyphUsability:
    """Tests to verify glyph values are usable in context."""


class TestGetBanner:
    """Tests for get_banner function."""

    def test_unicode_banner_contains_box_drawing_chars(self) -> None:
        """Test that Unicode banner contains non-ASCII box drawing characters."""
        # Unicode banner uses box-drawing characters like ╔ ╗ ║ etc
        has_unicode = any(ord(c) > 127 for c in _UNICODE_BANNER)
        assert has_unicode

    def test_ascii_banner_is_pure_ascii(self) -> None:
        """Test that ASCII banner contains only ASCII characters."""
        for char in _ASCII_BANNER:
            assert ord(char) < 128, f"Non-ASCII character found: {char!r}"


class TestIsAsciiMode:
    """Tests for is_ascii_mode helper."""
