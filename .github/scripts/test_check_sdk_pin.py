"""Tests for check_sdk_pin pin/version comparison."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_sdk_pin import compare_versions, is_prerelease, main


def _write_repo(tmp_path: Path, sdk_version: str, pin: str) -> Path:
    return _write_repo_raw(
        tmp_path,
        sdk_section=f'[project]\nname = "deepagents"\nversion = "{sdk_version}"\n',
        code_deps=f'"deepagents=={pin}", "deepagents-acp>=0.0.8", "rich>=15"',
    )


def _write_repo_raw(tmp_path: Path, sdk_section: str, code_deps: str) -> Path:
    (tmp_path / "libs" / "deepagents").mkdir(parents=True)
    (tmp_path / "libs" / "code").mkdir(parents=True)
    (tmp_path / "libs" / "deepagents" / "pyproject.toml").write_text(sdk_section)
    (tmp_path / "libs" / "code" / "pyproject.toml").write_text(
        "[project]\n"
        'name = "deepagents-code"\n'
        'version = "0.1.0"\n'
        f"dependencies = [{code_deps}]\n"
    )
    return tmp_path


def test_pin_matches(tmp_path) -> None:
    """Matching pin and SDK version returns 0."""
    assert main(_write_repo(tmp_path, "0.6.10", "0.6.10")) == 0


def test_stale_pin(tmp_path) -> None:
    """A pin that lags the SDK version returns 1."""
    assert main(_write_repo(tmp_path, "0.6.11", "0.6.10")) == 1


def test_ahead_pin(tmp_path) -> None:
    """A pin ahead of the workspace SDK version is intentionally allowed."""
    assert main(_write_repo(tmp_path, "0.6.11", "0.7.0a2")) == 0


def test_acp_dependency_not_mistaken_for_sdk(tmp_path) -> None:
    """`deepagents-acp` must not be parsed as the `deepagents` pin.

    The SDK and the (real) `deepagents` pin agree at 0.6.10 while
    `deepagents-acp` sits at a different version; a matcher that grabbed the
    acp line would read 0.0.8 and report a false stale pin.
    """
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents-acp>=0.0.8", "deepagents==0.6.10", "rich>=15"',
    )
    assert main(repo) == 0


def test_missing_pin_raises(tmp_path) -> None:
    """A non-`==` deepagents dependency yields a clear ValueError."""
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents>=0.6.0", "rich>=15"',
    )
    with pytest.raises(ValueError, match="No `deepagents==<version>` pin"):
        main(repo)


def test_missing_sdk_version_raises(tmp_path) -> None:
    """A SDK pyproject without project.version yields a ValueError."""
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\n',
        code_deps='"deepagents==0.6.10"',
    )
    with pytest.raises(ValueError, match="project.version"):
        main(repo)


def test_prerelease_pin_in_sync(tmp_path) -> None:
    """A PEP 440 prerelease pin matching the SDK returns 0.

    The extractor must capture the full `0.7.0a2` token; a strict `X.Y.Z`
    pattern would truncate it to `0.7.0` and falsely report a stale pin
    against an in-sync prerelease SDK.
    """
    assert main(_write_repo(tmp_path, "0.7.0a2", "0.7.0a2")) == 0


def test_stale_prerelease_pin(tmp_path) -> None:
    """Two distinct prereleases must be compared by precedence, not truncated."""
    assert main(_write_repo(tmp_path, "0.7.0a3", "0.7.0a2")) == 1


def test_ahead_prerelease_pin(tmp_path) -> None:
    """A newer prerelease pin is allowed when the workspace SDK is older."""
    assert main(_write_repo(tmp_path, "0.7.0a2", "0.7.0a3")) == 0


def test_dev_release_pin(tmp_path) -> None:
    """A dev release pin compares ahead of an older final release."""
    assert main(_write_repo(tmp_path, "0.6.11", "0.7.0.dev1")) == 0


def test_prerelease_detection() -> None:
    """Prerelease and dev pins are flagged for advisory comments."""
    assert is_prerelease("0.7.0a2")
    assert is_prerelease("0.7.0b1")
    assert is_prerelease("0.7.0rc1")
    assert is_prerelease("0.7.0.dev1")
    assert not is_prerelease("0.7.0")
    assert not is_prerelease("0.7.0.post1")


@pytest.mark.parametrize(
    "version",
    ["0.7.0a2.dev1", "0.7.0.post1.dev1", "1!1.0.0", "1.0.0+local"],
)
def test_is_prerelease_rejects_unsupported(version: str) -> None:
    """`is_prerelease` fails closed on unsupported formats.

    The `_parse_version` guard is the function's load-bearing contract: a
    malformed pin must surface as `ValueError` (exit 2 in the workflow), not
    be silently classified as a non-prerelease. Guards against a refactor
    that drops the validating call as apparently dead code.
    """
    with pytest.raises(ValueError, match="Unsupported version format"):
        is_prerelease(version)


def test_two_segment_pin_recognized(tmp_path) -> None:
    """Any numeric `==` token is accepted and compared.

    The extractor no longer requires three numeric segments. A two-segment
    `0.6` pin is therefore recognized and reported as stale against the SDK's
    `0.6.10` (not silently rejected).
    """
    repo = _write_repo_raw(
        tmp_path,
        sdk_section='[project]\nname = "deepagents"\nversion = "0.6.10"\n',
        code_deps='"deepagents==0.6"',
    )
    assert main(repo) == 1


# --- compare_versions: direct engine tests -------------------------------
#
# main() only ever reports 0 vs 1, so a comparison that errs but lands on the
# correct side of zero would pass the main()-level tests above. These assert
# the ordering engine directly to guard the precedence constants.


def test_compare_cross_dimension_ordering() -> None:
    """Qualifiers sort dev < a < b < rc < final < post (PEP 440)."""
    assert compare_versions("1.0.dev1", "1.0a1") == -1
    assert compare_versions("1.0a1", "1.0b1") == -1
    assert compare_versions("1.0b1", "1.0rc1") == -1
    assert compare_versions("1.0rc1", "1.0") == -1
    assert compare_versions("1.0", "1.0.post1") == -1


def test_compare_release_is_numeric_not_lexical() -> None:
    """Release segments compare numerically: `0.10.0` is newer than `0.6.0`."""
    assert compare_versions("0.10.0", "0.6.0") == 1


def test_compare_zero_pad_equivalence() -> None:
    """A missing trailing segment is zero-padded: `1.0` equals `1`."""
    assert compare_versions("1.0", "1") == 0


def test_zero_pad_equivalence_in_sync(tmp_path) -> None:
    """A `1.0` pin against a `1` SDK reports in sync, not stale."""
    assert main(_write_repo(tmp_path, "1", "1.0")) == 0


@pytest.mark.parametrize(
    "version",
    ["0.7.0a2.dev1", "0.7.0a2.post1", "0.7.0.post1.dev1", "1!1.0.0", "1.0.0+local"],
)
def test_unsupported_version_rejected(version: str) -> None:
    """Combined qualifiers, epochs, and local versions fail closed.

    A single `(release, qualifier)` key cannot order these faithfully, so the
    parser rejects them rather than silently dropping a segment (e.g. treating
    `0.7.0a2.dev1` as equal to `0.7.0a2`). The repo never pins them.
    """
    with pytest.raises(ValueError, match="Unsupported version format"):
        compare_versions(version, "0.7.0a2")


def test_unsupported_pin_propagates_from_main(tmp_path) -> None:
    """An unsupported pin surfaces as ValueError out of main (not a pass)."""
    repo = _write_repo(tmp_path, "1.0.0", "1.0.0+local")
    with pytest.raises(ValueError, match="Unsupported version format"):
        main(repo)


def test_script_exits_2_on_unsupported_version(tmp_path) -> None:
    """The `__main__` wrapper maps a parser ValueError to exit code 2.

    Locks the documented exit-code contract: 2 ("could not determine") must
    stay distinct from 1 (stale, advisory) so callers cannot launder a
    malformed pin into an in-sync pass.
    """
    _write_repo(tmp_path, "1.0.0", "1.0.0+local")
    scripts = tmp_path / ".github" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(Path(__file__).with_name("check_sdk_pin.py"), scripts)
    result = subprocess.run(
        [sys.executable, str(scripts / "check_sdk_pin.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Could not determine SDK pin status" in result.stdout


def test_workflow_warning_comments_link_pin_release() -> None:
    """SDK pin warning comments link the pinned version to its GitHub release.

    Asserts the *contract*, not the exact source spelling: a release-tag URL is
    built for the `deepagents==<pin>` tag (the scheme GitHub releases actually
    use — a wrong scheme would 404), the pin is rendered as a markdown link, and
    that link is interpolated into the warning bodies rather than only defined.
    The comment logic lives in a `github-script` block embedded in YAML, so it
    is not importable from pytest; matching the workflow text is the only
    in-harness option, but these checks tolerate reformatting and renames.
    """
    workflow = Path(__file__).parents[1] / "workflows" / "check_sdk_pin.yml"
    text = workflow.read_text()

    # Release URL targets the `deepagents==<pin>` tag, URL-encoded so `==`
    # survives as a path segment. The tag scheme is load-bearing: get it wrong
    # and every rendered link 404s.
    url_match = re.search(
        r"releases/tag/\$\{encodeURIComponent\(`deepagents==\$\{(\w+)\}`\)\}",
        text,
    )
    assert url_match, "release URL must encode the `deepagents==<pin>` tag"
    pin_var = url_match.group(1)

    # The pin is rendered as a markdown link wrapping that release URL.
    link_match = re.search(
        rf"const (\w+) = `\[deepagents==\$\{{{pin_var}}}\]\(\$\{{\w+}}\)`;",
        text,
    )
    assert link_match, "pin must render as a markdown link to its release"
    link_var = link_match.group(1)

    # The link reaches the warning bodies (stale-pin table + prerelease
    # sentence), not just its own definition line.
    assert text.count(f"${{{link_var}}}") >= 2, (
        "pin release link must be interpolated into the warning comment bodies"
    )
