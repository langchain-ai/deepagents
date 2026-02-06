"""Tests for charset mode configuration and glyph selection."""

import sys
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.config import (
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
    reset_glyphs_cache,
)


class TestCharsetMode:
    """Tests for CharsetMode enum."""

    def test_charset_mode_values(self) -> None:
        """Test that CharsetMode has expected values."""
        assert CharsetMode.UNICODE.value == "unicode"
        assert CharsetMode.ASCII.value == "ascii"
        assert CharsetMode.AUTO.value == "auto"

    def test_charset_mode_is_str_enum(self) -> None:
        """Test that CharsetMode values are strings."""
        assert isinstance(CharsetMode.UNICODE, str)
        assert CharsetMode.ASCII == "ascii"


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
        assert ord(UNICODE_GLYPHS.box_vertical) > 127
        assert ord(UNICODE_GLYPHS.box_horizontal) > 127
        assert ord(UNICODE_GLYPHS.box_double_horizontal) > 127
        assert ord(UNICODE_GLYPHS.gutter_bar) > 127
        # Tree connectors (check first character of each)
        assert ord(UNICODE_GLYPHS.tree_branch[0]) > 127
        assert ord(UNICODE_GLYPHS.tree_last[0]) > 127
        assert ord(UNICODE_GLYPHS.tree_vertical[0]) > 127

    def test_ascii_glyphs_are_ascii(self) -> None:
        """Test that ASCII_GLYPHS contains only ASCII characters."""
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
        # Spinner frames should all be ASCII
        for frame in ASCII_GLYPHS.spinner_frames:
            for char in frame:
                assert ord(char) < 128
        # Box-drawing characters
        for char in ASCII_GLYPHS.box_vertical:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.box_horizontal:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.box_double_horizontal:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.gutter_bar:
            assert ord(char) < 128
        # Tree connectors
        for char in ASCII_GLYPHS.tree_branch:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.tree_last:
            assert ord(char) < 128
        for char in ASCII_GLYPHS.tree_vertical:
            assert ord(char) < 128

    def test_glyphs_frozen(self) -> None:
        """Test that Glyphs instances are immutable."""
        with pytest.raises(AttributeError):
            UNICODE_GLYPHS.tool_prefix = "changed"  # type: ignore[misc]

    def test_glyphs_all_fields_present(self) -> None:
        """Test that both glyph sets have all required fields."""
        required_fields = [
            "tool_prefix",
            "ellipsis",
            "checkmark",
            "error",
            "circle_empty",
            "circle_filled",
            "output_prefix",
            "spinner_frames",
            "pause",
            "newline",
            "warning",
            "arrow_up",
            "arrow_down",
            "bullet",
            "cursor",
            # Box-drawing characters
            "box_vertical",
            "box_horizontal",
            "box_double_horizontal",
            "gutter_bar",
            # Tree connectors
            "tree_branch",
            "tree_last",
            "tree_vertical",
        ]
        for field in required_fields:
            assert hasattr(UNICODE_GLYPHS, field)
            assert hasattr(ASCII_GLYPHS, field)
            assert getattr(UNICODE_GLYPHS, field) is not None
            assert getattr(ASCII_GLYPHS, field) is not None


class TestDetectCharsetMode:
    """Tests for _detect_charset_mode function."""

    def setup_method(self) -> None:
        """Reset glyphs cache before each test."""
        reset_glyphs_cache()

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}, clear=False)
    def test_explicit_unicode_mode(self) -> None:
        """Test explicit unicode mode via env var."""
        mode = _detect_charset_mode()
        assert mode == CharsetMode.UNICODE

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "ascii"}, clear=False)
    def test_explicit_ascii_mode(self) -> None:
        """Test explicit ascii mode via env var."""
        mode = _detect_charset_mode()
        assert mode == CharsetMode.ASCII

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "UNICODE"}, clear=False)
    def test_case_insensitive_mode(self) -> None:
        """Test that mode parsing is case-insensitive."""
        mode = _detect_charset_mode()
        assert mode == CharsetMode.UNICODE

    @patch.dict(
        "os.environ", {"UI_CHARSET_MODE": "auto", "LANG": "en_US.UTF-8"}, clear=False
    )
    def test_auto_mode_with_utf_lang(self) -> None:
        """Test auto mode detects UTF from LANG env var."""
        # Mock stdout without utf encoding
        mock_stdout = Mock()
        mock_stdout.encoding = "ascii"
        with patch.object(sys, "stdout", mock_stdout):
            mode = _detect_charset_mode()
        assert mode == CharsetMode.UNICODE

    @patch.dict("os.environ", {"LANG": "C", "LC_ALL": ""}, clear=False)
    def test_auto_mode_with_c_locale_falls_back_to_ascii(self) -> None:
        """Test auto mode falls back to ASCII with C locale."""
        # Remove UI_CHARSET_MODE if set
        with patch.dict("os.environ", {"UI_CHARSET_MODE": "auto"}, clear=False):
            mock_stdout = Mock()
            mock_stdout.encoding = "ascii"
            with patch.object(sys, "stdout", mock_stdout):
                mode = _detect_charset_mode()
        assert mode == CharsetMode.ASCII

    def test_auto_mode_with_utf_stdout_encoding(self) -> None:
        """Test auto mode detects UTF from stdout encoding."""
        with patch.dict(
            "os.environ", {"UI_CHARSET_MODE": "auto", "LANG": "C", "LC_ALL": ""}
        ):
            mock_stdout = Mock()
            mock_stdout.encoding = "utf-8"
            with patch.object(sys, "stdout", mock_stdout):
                mode = _detect_charset_mode()
        assert mode == CharsetMode.UNICODE


class TestGetGlyphs:
    """Tests for get_glyphs function."""

    def setup_method(self) -> None:
        """Reset glyphs cache before each test."""
        reset_glyphs_cache()

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}, clear=False)
    def test_get_glyphs_returns_unicode_for_unicode_mode(self) -> None:
        """Test get_glyphs returns UNICODE_GLYPHS for unicode mode."""
        glyphs = get_glyphs()
        assert glyphs is UNICODE_GLYPHS

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "ascii"}, clear=False)
    def test_get_glyphs_returns_ascii_for_ascii_mode(self) -> None:
        """Test get_glyphs returns ASCII_GLYPHS for ascii mode."""
        glyphs = get_glyphs()
        assert glyphs is ASCII_GLYPHS

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}, clear=False)
    def test_get_glyphs_caches_result(self) -> None:
        """Test that get_glyphs caches the result."""
        glyphs1 = get_glyphs()
        glyphs2 = get_glyphs()
        assert glyphs1 is glyphs2

    def test_reset_glyphs_cache_works(self) -> None:
        """Test that reset_glyphs_cache clears the cache."""
        with patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}):
            glyphs1 = get_glyphs()
            assert glyphs1 is UNICODE_GLYPHS

        reset_glyphs_cache()

        with patch.dict("os.environ", {"UI_CHARSET_MODE": "ascii"}):
            glyphs2 = get_glyphs()
            assert glyphs2 is ASCII_GLYPHS


class TestGlyphUsability:
    """Tests to verify glyph values are usable in context."""

    def test_spinner_frames_not_empty(self) -> None:
        """Test that spinner frames have multiple frames for animation."""
        assert len(UNICODE_GLYPHS.spinner_frames) > 1
        assert len(ASCII_GLYPHS.spinner_frames) > 1

    def test_ascii_ellipsis_is_three_dots(self) -> None:
        """Test that ASCII ellipsis is standard three dots."""
        assert ASCII_GLYPHS.ellipsis == "..."

    def test_unicode_ellipsis_is_single_char(self) -> None:
        """Test that Unicode ellipsis is single character."""
        assert len(UNICODE_GLYPHS.ellipsis) == 1

    def test_ascii_spinner_classic_frames(self) -> None:
        """Test ASCII spinner uses parenthesized frames for consistent width."""
        assert set(ASCII_GLYPHS.spinner_frames) == {"(-)", "(\\)", "(|)", "(/)"}

    def test_unicode_box_drawing_characters(self) -> None:
        """Test Unicode box-drawing characters are the expected characters."""
        assert UNICODE_GLYPHS.box_vertical == "│"
        assert UNICODE_GLYPHS.box_horizontal == "─"
        assert UNICODE_GLYPHS.box_double_horizontal == "═"
        assert UNICODE_GLYPHS.gutter_bar == "▌"

    def test_ascii_box_drawing_characters(self) -> None:
        """Test ASCII box-drawing alternatives are simple ASCII."""
        assert ASCII_GLYPHS.box_vertical == "|"
        assert ASCII_GLYPHS.box_horizontal == "-"
        assert ASCII_GLYPHS.box_double_horizontal == "="
        assert ASCII_GLYPHS.gutter_bar == "|"

    def test_unicode_tree_connectors(self) -> None:
        """Test Unicode tree connectors are the expected strings."""
        assert UNICODE_GLYPHS.tree_branch == "├── "
        assert UNICODE_GLYPHS.tree_last == "└── "
        assert UNICODE_GLYPHS.tree_vertical == "│   "

    def test_ascii_tree_connectors(self) -> None:
        """Test ASCII tree connectors are simple ASCII."""
        assert ASCII_GLYPHS.tree_branch == "+-- "
        assert ASCII_GLYPHS.tree_last == "`-- "
        assert ASCII_GLYPHS.tree_vertical == "|   "

    def test_tree_connectors_consistent_width(self) -> None:
        """Test that tree connectors have consistent width for alignment."""
        # All tree connectors should be exactly 4 characters
        assert len(UNICODE_GLYPHS.tree_branch) == 4
        assert len(UNICODE_GLYPHS.tree_last) == 4
        assert len(UNICODE_GLYPHS.tree_vertical) == 4
        assert len(ASCII_GLYPHS.tree_branch) == 4
        assert len(ASCII_GLYPHS.tree_last) == 4
        assert len(ASCII_GLYPHS.tree_vertical) == 4


class TestGetBanner:
    """Tests for get_banner function."""

    def setup_method(self) -> None:
        """Reset glyphs cache before each test."""
        reset_glyphs_cache()

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}, clear=False)
    def test_get_banner_returns_unicode_for_unicode_mode(self) -> None:
        """Test get_banner returns Unicode banner for unicode mode."""
        with patch("deepagents_cli.config._is_editable_install", return_value=False):
            banner = get_banner()
        assert banner is _UNICODE_BANNER

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "ascii"}, clear=False)
    def test_get_banner_returns_ascii_for_ascii_mode(self) -> None:
        """Test get_banner returns ASCII banner for ascii mode."""
        with patch("deepagents_cli.config._is_editable_install", return_value=False):
            banner = get_banner()
        assert banner is _ASCII_BANNER

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "unicode"}, clear=False)
    def test_get_banner_adds_local_install_suffix_for_editable(self) -> None:
        """Test get_banner adds (local) suffix for editable installs."""
        with patch("deepagents_cli.config._is_editable_install", return_value=True):
            banner = get_banner()
        assert "(local)" in banner
        assert f"v{__version__} (local)" in banner

    @patch.dict("os.environ", {"UI_CHARSET_MODE": "ascii"}, clear=False)
    def test_get_banner_adds_local_install_suffix_for_editable_ascii(self) -> None:
        """Test get_banner adds (local) suffix in ASCII mode."""
        with patch("deepagents_cli.config._is_editable_install", return_value=True):
            banner = get_banner()
        assert "(local)" in banner
        assert f"v{__version__} (local)" in banner

    def test_unicode_banner_contains_box_drawing_chars(self) -> None:
        """Test that Unicode banner contains non-ASCII box drawing characters."""
        # Unicode banner uses box-drawing characters like ╔ ╗ ║ etc
        has_unicode = any(ord(c) > 127 for c in _UNICODE_BANNER)
        assert has_unicode

    def test_ascii_banner_is_pure_ascii(self) -> None:
        """Test that ASCII banner contains only ASCII characters."""
        for char in _ASCII_BANNER:
            assert ord(char) < 128, f"Non-ASCII character found: {char!r}"
