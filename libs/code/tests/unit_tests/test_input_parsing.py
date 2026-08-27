"""Unit tests for input parsing utilities."""

from pathlib import Path

import pytest

from deepagents_code.input import (
    ParsedPastedPathPayload,
    dropped_payload_paths,
    extract_leading_pasted_file_path,
    normalize_pasted_path,
    parse_file_mentions,
    parse_pasted_file_paths,
    parse_pasted_path_payload,
    parse_single_pasted_file_path,
)


def test_parse_file_mentions_with_escaped_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure escaped spaces in paths are handled correctly."""
    spaced_dir = tmp_path / "my folder"
    spaced_dir.mkdir()
    file_path = spaced_dir / "test.py"
    file_path.write_text("content")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("@my\\ folder/test.py")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_handles_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure path traversal sequences are resolved to actual paths."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file_path = tmp_path / "test.txt"
    file_path.write_text("content")
    monkeypatch.chdir(subdir)

    _, files = parse_file_mentions("@../test.txt")

    assert files == [file_path.resolve()]


def test_dropped_payload_paths_resolves_non_media(tmp_path: Path) -> None:
    """Classification is left to the caller, so non-media paths resolve too."""
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")

    assert dropped_payload_paths(str(doc)) == [doc.resolve()]


def test_dropped_payload_paths_resolves_multiple(tmp_path: Path) -> None:
    """A multi-file drop returns every resolved path."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")

    assert dropped_payload_paths(f"{img} {doc}") == [img.resolve(), doc.resolve()]


def test_dropped_payload_paths_resolves_file_url(tmp_path: Path) -> None:
    """A `file://` drop payload resolves like a plain path."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(img.as_uri()) == [img.resolve()]


@pytest.mark.parametrize("wrap", ["'{}'", '"{}"', "<{}>"])
def test_dropped_payload_paths_resolves_quoted_payload(
    tmp_path: Path, wrap: str
) -> None:
    """Quoted and bracketed drops resolve, since terminals wrap paths that way.

    The shape guard strips leading `<`, `'`, and `"` for exactly this reason;
    without that strip every quoted drop would look like typed text.
    """
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(wrap.format(img)) == [img.resolve()]


@pytest.mark.parametrize("template", ["{}", "'{}'", '"{}"'])
def test_dropped_payload_paths_resolves_space_bearing_filename(
    tmp_path: Path, template: str
) -> None:
    """A filename containing spaces resolves escaped or quoted.

    This is the modal real-world drop: macOS screenshots are named
    `Screenshot ... at ....png`.
    """
    img = tmp_path / "my shot.png"
    img.write_bytes(b"img")
    raw = str(img)
    payload = template.format(raw if template != "{}" else raw.replace(" ", r"\ "))

    assert dropped_payload_paths(payload) == [img.resolve()]


def test_dropped_payload_paths_resolves_home_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `~/`-shaped drop resolves, matching the shape guard's `~/` arm."""
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths("~/shot.png") == [img.resolve()]


@pytest.mark.parametrize(
    "payload",
    [
        "/usr/local is where it lives",
        "/ is the root directory",
        "~/ is my home",
    ],
)
def test_dropped_payload_paths_ignores_prose_that_passes_shape_guard(
    payload: str,
) -> None:
    """Prose starting with a path-shaped token is still text, not a drop.

    The shape guard only inspects the leading token, so rejection here depends
    on the parser refusing directories and multi-token text. Free-text prompts
    rely on this: a swallowed answer is worse than an inserted path.
    """
    assert dropped_payload_paths(payload) == []


def test_dropped_payload_paths_accepts_windows_drive_shape(mocker) -> None:
    """Windows drive-letter drops pass the shape guard and get parsed."""
    resolved = Path(r"C:\Users\Alice\shot.png")
    mocker.patch(
        "deepagents_code.input.parse_pasted_path_payload",
        return_value=ParsedPastedPathPayload(paths=[resolved]),
    )

    assert dropped_payload_paths(r"C:\Users\Alice\shot.png") == [resolved]


def test_dropped_payload_paths_accepts_windows_unc_shape(mocker) -> None:
    """Windows UNC drops pass the shape guard and get parsed."""
    resolved = Path(r"\\server\share\shot.png")
    mocker.patch(
        "deepagents_code.input.parse_pasted_path_payload",
        return_value=ParsedPastedPathPayload(paths=[resolved]),
    )

    assert dropped_payload_paths(r"\\server\share\shot.png") == [resolved]


def test_dropped_payload_paths_ignores_plain_text() -> None:
    """Ordinary typed text that is not an existing path yields nothing."""
    assert dropped_payload_paths("just some words") == []


def test_dropped_payload_paths_ignores_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path is typed text, not a drop, so it is not resolved.

    Terminals deliver a dragged file as an absolute path, so accepting relative
    tokens here only misfires: it would swallow a hand-typed `assets/logo.png`
    that resolves against the working directory.
    """
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.png").write_bytes(b"img")
    monkeypatch.chdir(tmp_path)

    assert dropped_payload_paths("assets/logo.png") == []


def test_dropped_payload_paths_ignores_leading_path_with_suffix(
    tmp_path: Path,
) -> None:
    """`<path> <question>` is out of scope, matching drop-time chat-input calls."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(f"{img} what's in this image?") == []


@pytest.mark.parametrize(
    "payload",
    [
        "/tmp/a\x00b.png",
        "file://[::1/x.png",
        "file://[bad",
    ],
)
def test_dropped_payload_paths_tolerates_unparseable_payloads(payload: str) -> None:
    """Payloads the OS or URL parser rejects fall back to text, never raise.

    `Path.resolve` raises `ValueError` on an embedded NUL and `urlparse` raises
    on a malformed authority, neither of which is an `OSError`.
    """
    assert dropped_payload_paths(payload) == []


@pytest.mark.parametrize(
    "payload",
    [
        "/tmp/a\x00b.png",
        "file://[::1/x.png",
    ],
)
def test_parse_pasted_file_paths_tolerates_unparseable_payloads(payload: str) -> None:
    """The strict parser's documented "returns an empty list" contract holds."""
    assert parse_pasted_file_paths(payload) == []


def test_parse_pasted_file_paths_handles_overlong_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-long path component must not crash path probing.

    Regression test: holding a key floods the input with a single long token.
    Resolving it against the cwd produces a path whose component exceeds the
    filesystem name limit, so `os.stat` raises `OSError` (`ENAMETOOLONG`). On
    Python <=3.13 `pathlib` lets that propagate (the original crash); on 3.14
    it is swallowed, so this asserts the no-match contract on every version.
    The version-independent guard is `*_handles_oserror_on_probe` below.
    """
    monkeypatch.chdir(tmp_path)
    overlong = "a" * 5000

    assert parse_pasted_file_paths(overlong) == []


def test_parse_pasted_path_payload_handles_overlong_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dropped-path entrypoint must not crash on an over-long token.

    This mirrors the exact path that crashed the TUI: `on_text_area_changed`
    routes freshly typed text through `parse_pasted_path_payload`.
    """
    monkeypatch.chdir(tmp_path)
    overlong = "a" * 5000

    assert parse_pasted_path_payload(overlong) is None


def test_parse_pasted_file_paths_handles_oserror_on_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """`OSError` raised by an `is_file` probe must be swallowed, not propagated.

    Guards against platforms/filesystems that surface the limit at a different
    probe than `resolve`, ensuring the fix does not rely on a specific errno.
    """
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.txt"
    target.write_text("hi")
    mocker.patch("pathlib.Path.is_file", side_effect=OSError(63, "File name too long"))

    assert parse_pasted_file_paths(str(target)) == []


def test_parse_pasted_file_paths_handles_oserror_in_unicode_variant(
    tmp_path: Path, mocker
) -> None:
    """An `OSError` probe inside the Unicode-space fallback must not propagate.

    Exercises the `_resolve_with_unicode_space_variants` traversal: the on-disk
    name carries a narrow no-break space while the paste uses an ASCII space,
    forcing the `iterdir`-match branch where component `is_file`/`is_dir` probes
    run. The path is quoted so it stays a single token instead of being split.
    """
    unicode_name = "Screenshot 2026-02-26 at 2.02.42 AM.png"
    img = tmp_path / unicode_name
    img.write_bytes(b"img")
    ascii_name = unicode_name.replace(chr(0x202F), " ")
    pasted = f"'{str(img).replace(unicode_name, ascii_name)}'"
    mocker.patch("pathlib.Path.is_file", side_effect=OSError(63, "File name too long"))

    assert parse_pasted_file_paths(pasted) == []
