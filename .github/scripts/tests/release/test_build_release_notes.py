"""Test the shared release-notes builder."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts" / "release"
SCRIPT_PATH = SCRIPT_DIR / "build_release_notes.py"
sys.path.insert(0, str(SCRIPT_DIR))

import build_release_notes as brn

PACKAGE_PATH = Path("libs/example")
REPOSITORY = "langchain-ai/deepagents"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.email", "release-test@example.com")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "commit.gpgSign", "false")


def _commit(repo: Path, path: Path, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", str(path))
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _create_history(repo: Path) -> dict[str, str]:
    """Create a standard package history for testing.

    Returns a dict of commit SHAs keyed by label.
    """
    _init_repo(repo)

    config = {
        "packages": {
            str(PACKAGE_PATH): {
                "exclude-paths": [str(PACKAGE_PATH / "tests")],
            }
        }
    }
    (repo / "release-please-config.json").write_text(json.dumps(config))
    (repo / PACKAGE_PATH / "tests").mkdir(parents=True)
    (repo / PACKAGE_PATH / "module.py").write_text("BASE = 1\n")
    (repo / PACKAGE_PATH / "tests" / "test_module.py").write_text("baseline\n")
    _git(repo, "add", "release-please-config.json", str(PACKAGE_PATH))
    _git(repo, "commit", "-m", "feat(example): initial package")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "example==1.0.0")

    feature = _commit(
        repo,
        PACKAGE_PATH / "module.py",
        "BASE = 1\nFEATURE = 2\n",
        "feat(example): add feature",
    )
    unrelated = _commit(
        repo,
        Path("libs/other/module.py"),
        "OTHER = 1\n",
        "feat(other): unrelated change",
    )
    test_only = _commit(
        repo,
        PACKAGE_PATH / "tests" / "test_module.py",
        "baseline\nnew test\n",
        "test(example): add coverage",
    )
    fix = _commit(
        repo,
        PACKAGE_PATH / "module.py",
        "BASE = 1\nFEATURE = 3\n",
        "fix(example): correct feature",
    )
    release = _commit(
        repo,
        PACKAGE_PATH / "CHANGELOG.md",
        "## 1.0.1\n",
        "release(example): 1.0.1",
    )
    hotfix = _commit(
        repo,
        PACKAGE_PATH / "module.py",
        "BASE = 1\nFEATURE = 4\n",
        "hotfix(example): repair release",
    )
    return {
        "baseline": baseline,
        "feature": feature,
        "unrelated": unrelated,
        "test_only": test_only,
        "fix": fix,
        "release": release,
        "hotfix": hotfix,
    }


def _create_sibling_prerelease_tag(
    repo: Path,
    *,
    base: str,
    version: str,
    index: int,
) -> None:
    branch = f"prerelease-{index}"
    _git(repo, "checkout", "-b", branch, base)
    _commit(
        repo,
        PACKAGE_PATH / "module.py",
        f"VERSION = {index}\n",
        f"hotfix(example): prerelease {version}",
    )
    _git(repo, "tag", f"example=={version}")
    _git(repo, "checkout", "main")


def _entry(sha: str, subject: str) -> str:
    return f"- [`{sha[:7]}`](https://github.com/{REPOSITORY}/commit/{sha}) {subject}"


def _release_please_header(version: str, previous: str) -> str:
    """A production-shaped release-please header.

    The compare URL embeds the *previous* version, which is what makes a
    substring match against the whole line select the wrong section.
    """
    return (
        f"## [{version}](https://github.com/{REPOSITORY}/compare/"
        f"example=={previous}...example=={version}) (2026-07-27)"
    )


# ── Version / prerelease detection ──────────────────────────────────────────


class TestVersionDetection:
    def test_stable_version_is_not_prerelease(self) -> None:
        assert not brn._is_prerelease("1.0.0")
        assert not brn._is_prerelease("0.0.42")
        assert not brn._is_prerelease("2.1.3")

    def test_pep440_suffixes_are_prerelease(self) -> None:
        assert brn._is_prerelease("1.0.0a1")
        assert brn._is_prerelease("1.0.0b2")
        assert brn._is_prerelease("1.0.0rc1")
        assert brn._is_prerelease("1.0.0.dev1")

    def test_dash_separator_is_prerelease(self) -> None:
        assert brn._is_prerelease("1.0.0-rc.1")
        assert brn._is_prerelease("1.0.0-alpha.1")

    @pytest.mark.parametrize(
        "version",
        ["1.0.0", "1.0.0a1", "1.0.0b2", "1.0.0rc1", "1.0.0.dev1", "1.0.0-rc.1",
         "1.0.0.b2", "1.0.0.rc1", "1.0.0-alpha.1", "0.0.42", "2.1.3", "10.20.30"],
    )
    def test_matches_workflow_prerelease_detection(self, version: str) -> None:
        """The script and release.yml must agree on what is a pre-release.

        release.yml's `check-version` step drives the GitHub release's
        `prerelease:` flag. If the two detectors disagree, a release can be
        flagged as a pre-release while its body omits the banner. The
        workflow's own regex is read out of release.yml so the two cannot
        drift apart silently.
        """
        pattern = re.search(
            r'is_pre = bool\(re\.search\(r"([^"]+)", version\) or "-" in version\)',
            _release_yml_text(),
        )
        assert pattern, "could not locate the is_pre expression in release.yml"
        workflow_result = bool(re.search(pattern.group(1), version)) or "-" in version
        assert brn._is_prerelease(version) == workflow_result, version

    def test_parse_prerelease_recognizes_all_forms(self) -> None:
        assert brn._parse_prerelease("1.1.0a1") == ("1.1.0", 1, 1)
        assert brn._parse_prerelease("1.1.0b2") == ("1.1.0", 2, 2)
        assert brn._parse_prerelease("1.1.0rc1") == ("1.1.0", 3, 1)
        assert brn._parse_prerelease("1.1.0.dev3") == ("1.1.0", 0, 3)
        assert brn._parse_prerelease("1.1.0-rc.7") == ("1.1.0", 3, 7)
        assert brn._parse_prerelease("1.1.0-alpha.1") == ("1.1.0", 1, 1)
        assert brn._parse_prerelease("1.1.0-beta.2") == ("1.1.0", 2, 2)
        assert brn._parse_prerelease("1.1.0-preview.1") == ("1.1.0", 3, 1)

    def test_parse_prerelease_fields_are_named(self) -> None:
        """Rank and serial are distinct fields, not positional ints."""
        parsed = brn._parse_prerelease("1.1.0b7")
        assert parsed is not None
        assert parsed.base == "1.1.0"
        assert parsed.rank == 2
        assert parsed.serial == 7

    def test_parse_prerelease_rejects_stable(self) -> None:
        assert brn._parse_prerelease("1.0.0") is None
        assert brn._parse_prerelease("1.0.0-canary") is None

    def test_base_version_strips_suffixes(self) -> None:
        assert brn._base_version("1.0.1a1") == "1.0.1"
        assert brn._base_version("1.0.1rc1") == "1.0.1"
        assert brn._base_version("1.0.1-rc.1") == "1.0.1"
        assert brn._base_version("1.0.1.dev3") == "1.0.1"
        assert brn._base_version("1.0.1") == "1.0.1"


def _release_yml_text() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()


# ── Changelog extraction ────────────────────────────────────────────────────


class TestChangelogExtraction:
    def test_extracts_version_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            "## 1.0.1\n\n"
            "### Bug Fixes\n\n"
            "* fix something\n\n"
            "## 1.0.0\n\n"
            "* initial\n"
        )
        body, reason = brn.extract_changelog_section(changelog, "1.0.1")
        assert reason is None
        assert "### Bug Fixes" in body
        assert "* fix something" in body
        assert "* initial" not in body

    def test_extracts_bracketed_version_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            f"{_release_please_header('1.0.1', '1.0.0')}\n\n"
            "* new feature\n\n"
            f"{_release_please_header('1.0.0', '0.9.9')}\n\n"
            "* old\n"
        )
        body, reason = brn.extract_changelog_section(changelog, "1.0.1")
        assert reason is None
        assert "* new feature" in body
        assert "* old" not in body

    def test_compare_url_previous_version_does_not_match(self, tmp_path: Path) -> None:
        """A header's compare URL names the previous version; it must not match.

        release-please writes `## [0.1.49](.../compare/pkg==0.1.48...pkg==0.1.49)`.
        A substring test for "0.1.48" hits 0.1.49's header and publishes the
        wrong release's notes — the exact scenario this script exists to
        recover from.
        """
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            f"{_release_please_header('0.1.49', '0.1.48')}\n\n"
            "* newest only\n\n"
            f"{_release_please_header('0.1.48', '0.1.47')}\n\n"
            "* the one we asked for\n"
        )
        body, reason = brn.extract_changelog_section(changelog, "0.1.48")
        assert reason is None
        assert body == "* the one we asked for"
        assert "newest only" not in body

    def test_prefix_version_does_not_match_longer_version(self, tmp_path: Path) -> None:
        """`1.0.1` must not match a `1.0.10` header."""
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## 1.0.10\n\n* ten\n\n## 1.0.1\n\n* one\n"
        )
        body, _ = brn.extract_changelog_section(changelog, "1.0.1")
        assert body == "* one"

    def test_reports_missing_version(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## 1.0.0\n\n* initial\n")
        body, reason = brn.extract_changelog_section(changelog, "2.0.0")
        assert body == ""
        assert reason is not None
        assert "Could not find version 2.0.0" in reason

    def test_reports_missing_file(self, tmp_path: Path) -> None:
        body, reason = brn.extract_changelog_section(tmp_path / "NOPE.md", "1.0.0")
        assert body == ""
        assert reason is not None
        assert "No CHANGELOG.md found" in reason

    def test_reports_empty_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## 1.0.0\n\n\n## 0.9.0\n\n* old\n")
        body, reason = brn.extract_changelog_section(changelog, "1.0.0")
        assert body == ""
        assert reason is not None
        assert "is empty" in reason

    def test_last_version_section_extracted(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            f"{_release_please_header('2.0.0', '1.0.0')}\n\n"
            "* latest\n\n"
            f"{_release_please_header('1.0.0', '0.9.0')}\n\n"
            "* oldest\n"
        )
        body, _ = brn.extract_changelog_section(changelog, "1.0.0")
        assert body == "* oldest"


# ── Git log generation ───────────────────────────────────────────────────────


class TestGitLogGeneration:
    def test_stable_release_git_log_newest_first(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            commits["hotfix"],
            REPOSITORY,
        )

        assert "<summary>Git log since example==1.0.0</summary>" in details
        assert "newest first" in details
        assert details.index(commits["hotfix"][:7]) < details.index(
            commits["feature"][:7]
        )
        assert commits["unrelated"] not in details

    def test_initial_release_git_log(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "",
            commits["release"],
            REPOSITORY,
        )
        assert "<summary>Git log for initial release</summary>" in details
        assert _entry(commits["baseline"], "feat(example): initial package") in details
        assert _entry(commits["release"], "release(example): 1.0.1") in details

    def test_release_sha_bounds_the_log(self, tmp_path: Path) -> None:
        """The log ends at release_sha, not at HEAD.

        On workflow_dispatch HEAD may be ahead of the release; commits after
        the release must not be attributed to it.
        """
        commits = _create_history(tmp_path)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            commits["fix"],
            REPOSITORY,
        )
        assert _entry(commits["fix"], "fix(example): correct feature") in details
        assert "hotfix(example): repair release" not in details

    def test_git_log_reports_no_commits_for_empty_range(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        _git(tmp_path, "tag", "example==2.0.0", commits["hotfix"])

        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==2.0.0",
            commits["hotfix"],
            REPOSITORY,
        )
        assert "No commits found." in details
        assert "Git log unavailable" not in details

    def test_git_log_reports_failure_for_bad_range(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        warnings: list[str] = []
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==9.9.9",
            _git(tmp_path, "rev-parse", "HEAD"),
            REPOSITORY,
            warnings,
        )
        assert "Git log unavailable" in details
        assert "No commits found." not in details
        assert any("git log failed" in w for w in warnings)

    def test_git_log_escapes_html(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        subject = "fix(example): escape </details><!--"
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "ESCAPED = 1\n", subject)
        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert "&lt;/details&gt;&lt;!--" in details
        assert "escape </details><!--" not in details

    def test_git_log_truncates_long_subjects(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        long_subject = f"fix(example): {'x' * 250}"
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "LONG = 1\n", long_subject)
        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert "…" in details

    def test_git_log_truncates_before_escaping(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        subject = "x" * 199 + "&" + "x" * 60
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "ORDER = 1\n", subject)
        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert "&amp;…" in details
        assert "&…" not in details

    def test_git_log_limits_large_history(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        tip = commits["hotfix"]
        for index in range(101):
            tip = _commit(
                tmp_path,
                PACKAGE_PATH / "module.py",
                f"VALUE = {index}\n",
                f"fix(example): generated {index}",
            )

        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert "truncated to the newest" in details
        assert details.count("https://github.com/") == brn.MAX_COMMITS

    def test_git_log_includes_exactly_max_commits_without_truncation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The boundary case: exactly MAX_COMMITS commits must not truncate."""
        max_commits = 5
        monkeypatch.setattr(brn, "MAX_COMMITS", max_commits)
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        tip = ""
        for index in range(max_commits):
            tip = _commit(
                tmp_path,
                PACKAGE_PATH / "module.py",
                f"V = {index + 1}\n",
                f"fix(example): generated {index}",
            )

        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert details.count("https://github.com/") == max_commits
        assert "truncated to the newest" not in details

    def test_git_log_applies_byte_limit_after_html_escaping(
        self, tmp_path: Path
    ) -> None:
        """The byte budget counts escaped bytes, so `&` costs 5, not 1."""
        commits = _create_history(tmp_path)
        tip = commits["hotfix"]
        for index in range(30):
            tip = _commit(
                tmp_path,
                PACKAGE_PATH / "module.py",
                f"VALUE = {index}\n",
                f"fix(example): {index} {'&' * 250}",
            )

        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert len(details.encode()) < 26_000
        assert details.count(f"https://github.com/{REPOSITORY}/commit/") < 30
        assert "The log is truncated to the newest" in details

    def test_git_log_respects_byte_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Truncation can be driven by bytes even well under MAX_COMMITS."""
        monkeypatch.setattr(brn, "MAX_GIT_LOG_BYTES", 400)
        commits = _create_history(tmp_path)
        tip = commits["hotfix"]
        for index in range(10):
            tip = _commit(
                tmp_path,
                PACKAGE_PATH / "module.py",
                f"VALUE = {index}\n",
                f"fix(example): generated {index}",
            )

        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY
        )
        assert "truncated to the newest" in details
        assert details.count("https://github.com/") < 10


# ── Predecessor tag resolution ───────────────────────────────────────────────


class TestPreviousTagResolution:
    def test_stable_patch_uses_previous_patch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.1", commits["hotfix"], is_prerelease=False
        )
        assert prev == "example==1.0.0"

    def test_previous_tag_must_be_in_release_history(self, tmp_path: Path) -> None:
        """A tag on a divergent branch is not a valid predecessor."""
        _init_repo(tmp_path)
        _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.0\n",
            "release(example): 1.0.0",
        )
        _git(tmp_path, "tag", "example==1.0.0")

        _git(tmp_path, "checkout", "-b", "newer-version-line")
        _commit(
            tmp_path, PACKAGE_PATH / "module.py", "NEWER = 1\n", "fix(example): newer"
        )
        _git(tmp_path, "tag", "example==1.0.1")

        _git(tmp_path, "checkout", "main")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "MAIN = 1\n", "fix(example): main"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.2", head, is_prerelease=False
        )
        assert prev == "example==1.0.0"

    def test_stable_x_y_0_falls_back_to_latest_stable(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "fix(example): ten")
        _git(tmp_path, "tag", "example==1.0.10")
        # Tagged after 1.0.10 and lexically greater, so neither creation order
        # nor a lexical sort would pick 1.0.10 — only a version sort does.
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "fix(example): nine")
        _git(tmp_path, "tag", "example==1.0.9")
        head = _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 2.0.0\n",
            "release(example): 2.0.0",
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "2.0.0", head, is_prerelease=False
        )
        assert prev == "example==1.0.10"

    def test_initial_release_has_no_predecessor(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.0.0",
            head,
            is_prerelease=False,
            warnings=warnings,
        )
        assert prev == ""
        # A genuine initial release must not warn.
        assert warnings == []

    def test_warns_when_stable_tag_exists_but_is_unreachable(
        self, tmp_path: Path
    ) -> None:
        """Prior tags that exist but aren't reachable are an anomaly.

        Without this warning the log silently spans the package's entire
        history under an "initial release" heading.
        """
        _init_repo(tmp_path)
        _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "checkout", "-b", "orphan-line")
        _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "ORPHAN = 1\n",
            "feat(example): orphan",
        )
        _git(tmp_path, "tag", "example==1.0.0")
        _git(tmp_path, "checkout", "main")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "MAIN = 1\n", "feat(example): main"
        )

        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.1.0",
            head,
            is_prerelease=False,
            warnings=warnings,
        )
        assert prev == ""
        assert any("No prior stable tag reachable" in w for w in warnings)

    @pytest.mark.parametrize(
        ("version", "tags", "expected"),
        [
            ("1.1.0a7", ("1.1.0a1", "1.1.0a6", "1.1.0a8"), "1.1.0a6"),
            ("1.1.0b2", ("1.1.0a9", "1.1.0b1", "1.1.0b3"), "1.1.0b1"),
            ("1.1.0rc2", ("1.1.0b9", "1.1.0rc1", "1.1.0rc3"), "1.1.0rc1"),
            ("1.1.0.dev3", ("1.1.0.dev1", "1.1.0.dev2", "1.1.0.dev4"), "1.1.0.dev2"),
            ("1.1.0-rc.7", ("1.1.0-rc.6", "1.1.0-rc.8"), "1.1.0-rc.6"),
            # Serials compare numerically, not lexically: a10 must beat a6.
            ("1.1.0a11", ("1.1.0a6", "1.1.0a10"), "1.1.0a10"),
            # Zero-padded serials parse as base-10 (a08 -> 8), never octal.
            ("1.1.0a9", ("1.1.0a1", "1.1.0a08"), "1.1.0a08"),
            # dev sorts before a/b/rc, so the latest dev precedes a1.
            ("1.1.0a1", ("1.1.0.dev1", "1.1.0.dev5"), "1.1.0.dev5"),
            # dev does not outrank a later phase: b1 wins over dev9.
            ("1.1.0rc1", ("1.1.0.dev9", "1.1.0b1"), "1.1.0b1"),
            # Dash-form alias: alpha maps to the `a` rank.
            ("1.1.0-beta.1", ("1.1.0-alpha.9",), "1.1.0-alpha.9"),
            # Dash-form alias: preview maps to the `rc` rank.
            ("1.1.0-rc.2", ("1.1.0-preview.1",), "1.1.0-preview.1"),
            # Optional separator: 1.1.0-rc7 (no dot) parses like 1.1.0-rc.7.
            ("1.1.0-rc7", ("1.1.0-rc6",), "1.1.0-rc6"),
        ],
    )
    def test_prerelease_prefers_latest_earlier_sibling(
        self,
        tmp_path: Path,
        version: str,
        tags: tuple[str, ...],
        expected: str,
    ) -> None:
        _init_repo(tmp_path)
        base = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        for index, tag in enumerate(tags):
            _create_sibling_prerelease_tag(
                tmp_path, base=base, version=tag, index=index
            )

        _git(tmp_path, "checkout", "-b", "current-prerelease", base)
        release = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "CURRENT = 1\n",
            f"hotfix(example): prerelease {version}",
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", version, release, is_prerelease=True
        )
        assert prev == f"example=={expected}"

    def test_prerelease_ignores_different_base_sibling(self, tmp_path: Path) -> None:
        """A pre-release of a different base version is not a predecessor."""
        _init_repo(tmp_path)
        base = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        _create_sibling_prerelease_tag(
            tmp_path, base=base, version="1.0.5a1", index=0
        )

        _git(tmp_path, "checkout", "-b", "current", base)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "CURRENT = 1\n", "hotfix(example): rc"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0rc1", head, is_prerelease=True
        )
        assert prev == "example==1.0.0"

    def test_prerelease_ignores_later_phase_sibling(self, tmp_path: Path) -> None:
        """An rc is not a predecessor of a b from the same base."""
        _init_repo(tmp_path)
        base = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        _create_sibling_prerelease_tag(
            tmp_path, base=base, version="1.1.0rc1", index=0
        )

        _git(tmp_path, "checkout", "-b", "current", base)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "CURRENT = 1\n", "hotfix(example): b1"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0b1", head, is_prerelease=True
        )
        assert prev == "example==1.0.0"

    def test_prerelease_ignores_equal_serial_sibling(self, tmp_path: Path) -> None:
        """An equal-precedence tag is the same release, not a predecessor."""
        _init_repo(tmp_path)
        base = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        # 1.1.0-a.1 and 1.1.0a1 are the same precedence, spelled differently.
        _create_sibling_prerelease_tag(
            tmp_path, base=base, version="1.1.0-a.1", index=0
        )

        _git(tmp_path, "checkout", "-b", "current", base)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "CURRENT = 1\n", "hotfix(example): a1"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0a1", head, is_prerelease=True
        )
        assert prev == "example==1.0.0"

    def test_warns_on_unparseable_prerelease(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2"
        )

        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.1.0-canary",
            head,
            is_prerelease=True,
            warnings=warnings,
        )
        assert prev == "example==1.0.0"
        assert any(
            "does not match a recognized pre-release format" in w for w in warnings
        )

    def test_prerelease_falls_back_to_base_version_tag(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.1")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.1a1", head, is_prerelease=True
        )
        assert prev == "example==1.0.1"

    def test_prerelease_falls_back_to_latest_stable(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2"
        )

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0a1", head, is_prerelease=True
        )
        assert prev == "example==1.0.0"

    def test_prerelease_rejects_tag_ahead_of_release(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        release = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "CURRENT = 1\n",
            "hotfix(example): prerelease 1.1.0a7",
        )
        _git(tmp_path, "checkout", "-b", "future-prerelease")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "FUTURE = 1\n", "hotfix: a6")
        _git(tmp_path, "tag", "example==1.1.0a6")
        _git(tmp_path, "checkout", "main")

        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0a7", release, is_prerelease=True
        )
        assert prev == "example==1.0.0"

    def test_placeholder_zero_tag_rejected_as_previous_patch(
        self, tmp_path: Path
    ) -> None:
        """`pkg==0.0.0` is release-please's "never released" placeholder.

        It is rejected by the previous-patch tier. It remains eligible for the
        latest-stable tier, matching the bash this ports.
        """
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==0.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): one"
        )

        assert not brn._valid_previous_tag(
            tmp_path, "example==0.0.0", "example", head
        )
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "0.0.1", head, is_prerelease=False
        )
        assert prev == "example==0.0.0"


# ── Release commit resolution ───────────────────────────────────────────────


class TestReleaseCommit:
    def test_stable_uses_changelog_touch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path, str(PACKAGE_PATH), is_prerelease=False
        )
        assert commit == commits["release"]

    def test_prerelease_uses_head(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path, str(PACKAGE_PATH), is_prerelease=True
        )
        assert commit == commits["hotfix"]

    def test_stable_falls_back_to_head_without_changelog(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        warnings: list[str] = []
        commit = brn.resolve_release_commit(
            tmp_path, str(PACKAGE_PATH), is_prerelease=False, warnings=warnings
        )
        assert commit == head
        assert any("falling back to HEAD" in w for w in warnings)


# ── gh helpers ──────────────────────────────────────────────────────────────


def _gh_stub(handler):
    """Build a `_run_gh` replacement from a callable taking the arg list."""

    def _run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        return handler(args)

    return _run


def _ok(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=0, stdout=stdout, stderr=""
    )


def _fail(stderr: str, code: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=code, stdout="", stderr=stderr
    )


class TestCollectContributors:
    def _repo(self, tmp_path: Path) -> str:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        return _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): one"
        )

    def _run(
        self,
        tmp_path: Path,
        head: str,
        pr_payload: dict,
        monkeypatch: pytest.MonkeyPatch,
        warnings: list[str] | None = None,
    ):
        def handler(args: list[str]):
            if args[0] == "api":
                return _ok("7")
            return _ok(json.dumps(pr_payload))

        monkeypatch.setattr(brn, "_run_gh", _gh_stub(handler))
        return brn.collect_contributors(
            tmp_path,
            str(PACKAGE_PATH),
            head,
            "example==1.0.0",
            REPOSITORY,
            warnings=warnings if warnings is not None else [],
        )

    def test_collects_community_contributor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)
        community, internal = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "alice", "is_bot": False},
                "body": "Twitter: @alicetw\nLinkedIn: https://linkedin.com/in/alice",
                "labels": [],
            },
            monkeypatch,
        )
        assert internal == []
        assert community == [
            brn.Contributor("alice", "alicetw", "https://linkedin.com/in/alice")
        ]

    def test_linkedin_requires_prefixed_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A LinkedIn URL in prose is not the author's own profile."""
        head = self._repo(tmp_path)
        community, _ = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "alice", "is_bot": False},
                "body": (
                    "Thanks to https://www.linkedin.com/in/someone-else"
                    " for the idea!"
                ),
                "labels": [],
            },
            monkeypatch,
        )
        assert community == [brn.Contributor("alice", "", "")]

    def test_linkedin_extracted_from_prefixed_line_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)
        community, _ = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "alice", "is_bot": False},
                "body": (
                    "Credit to https://linkedin.com/in/bystander for reporting.\n"
                    "LinkedIn: https://www.linkedin.com/in/alice\n"
                ),
                "labels": [],
            },
            monkeypatch,
        )
        assert community[0].linkedin == "https://www.linkedin.com/in/alice"

    def test_skips_bot_by_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)
        community, internal = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "dependabot", "is_bot": True},
                "body": "",
                "labels": [],
            },
            monkeypatch,
        )
        assert community == []
        assert internal == []

    def test_skips_bot_by_login_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing or renamed is_bot field must not promote a bot."""
        head = self._repo(tmp_path)
        community, _ = self._run(
            tmp_path,
            head,
            {"author": {"login": "renovate[bot]"}, "body": "", "labels": []},
            monkeypatch,
        )
        assert community == []

    def test_internal_label_routes_to_maintainers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)
        community, internal = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "maint", "is_bot": False},
                "body": "Twitter: @maint",
                "labels": [{"name": "internal"}],
            },
            monkeypatch,
        )
        assert community == []
        assert internal == ["maint"]

    def test_warns_when_labels_field_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)
        warnings: list[str] = []
        self._run(
            tmp_path,
            head,
            {"author": {"login": "alice", "is_bot": False}, "body": ""},
            monkeypatch,
            warnings,
        )
        assert any("no labels field" in w for w in warnings)

    def test_internal_wins_over_community_for_same_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): one")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): two"
        )

        payloads = {
            "1": {
                "author": {"login": "dual", "is_bot": False},
                "body": "",
                "labels": [],
            },
            "2": {
                "author": {"login": "dual", "is_bot": False},
                "body": "",
                "labels": [{"name": "internal"}],
            },
        }
        calls = {"n": 0}

        def handler(args: list[str]):
            if args[0] == "api":
                calls["n"] += 1
                return _ok(str(calls["n"]))
            return _ok(json.dumps(payloads[args[2]]))

        monkeypatch.setattr(brn, "_run_gh", _gh_stub(handler))
        community, internal = brn.collect_contributors(
            tmp_path,
            str(PACKAGE_PATH),
            head,
            "example==1.0.0",
            REPOSITORY,
            warnings=[],
        )
        assert community == []
        assert internal == ["dual"]

    def test_counts_and_warns_on_incomplete_lookups(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated contributor list must not look like a complete one."""
        head = self._repo(tmp_path)
        monkeypatch.setattr(
            brn, "_run_gh", _gh_stub(lambda _args: _fail("HTTP 403 rate limit", 1))
        )
        warnings: list[str] = []
        community, internal = brn.collect_contributors(
            tmp_path,
            str(PACKAGE_PATH),
            head,
            "example==1.0.0",
            REPOSITORY,
            warnings=warnings,
        )
        assert community == []
        assert internal == []
        assert any("INCOMPLETE" in w for w in warnings)

    def test_offline_skips_lookups(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        head = self._repo(tmp_path)

        def explode(_args):
            raise AssertionError("gh must not run in offline mode")

        monkeypatch.setattr(brn, "_run_gh", explode)
        assert brn.collect_contributors(
            tmp_path,
            str(PACKAGE_PATH),
            head,
            "example==1.0.0",
            REPOSITORY,
            offline=True,
        ) == ([], [])


# ── Releaser resolution ─────────────────────────────────────────────────


class TestResolveReleaser:
    def test_warns_when_commit_pulls_lookup_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(brn, "_run_gh", _gh_stub(lambda _a: _fail("HTTP 500")))
        warnings: list[str] = []

        releaser = brn.resolve_releaser(
            REPOSITORY, "abc123", "dispatcher", warnings=warnings
        )

        assert releaser == "dispatcher"
        assert warnings == [
            "releaser lookup: commit-pulls API failed: HTTP 500 — falling back to actor"
        ]

    def test_no_warning_when_commit_has_no_pull_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(brn, "_run_gh", _gh_stub(lambda _a: _ok("")))
        warnings: list[str] = []

        releaser = brn.resolve_releaser(
            REPOSITORY, "abc123", "dispatcher", warnings=warnings
        )

        assert releaser == "dispatcher"
        assert warnings == []

    def test_uses_merger_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(args: list[str]):
            return _ok("7") if args[0] == "api" else _ok("merger")

        monkeypatch.setattr(brn, "_run_gh", _gh_stub(handler))
        assert (
            brn.resolve_releaser(REPOSITORY, "abc123", "dispatcher") == "merger"
        )

    def test_warns_when_pr_view_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(args: list[str]):
            return _ok("7") if args[0] == "api" else _fail("HTTP 502")

        monkeypatch.setattr(brn, "_run_gh", _gh_stub(handler))
        warnings: list[str] = []
        assert (
            brn.resolve_releaser(
                REPOSITORY, "abc123", "dispatcher", warnings=warnings
            )
            == "dispatcher"
        )
        assert any("gh pr view #7 failed" in w for w in warnings)

    @pytest.mark.parametrize(
        "actor", ["github-actions[bot]", "github-actions", "dependabot[bot]", ""]
    )
    def test_drops_bot_accounts(
        self, monkeypatch: pytest.MonkeyPatch, actor: str
    ) -> None:
        monkeypatch.setattr(brn, "_run_gh", _gh_stub(lambda _a: _ok("")))
        assert brn.resolve_releaser(REPOSITORY, "abc123", actor) == ""

    def test_offline_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(_args):
            raise AssertionError("gh must not run in offline mode")

        monkeypatch.setattr(brn, "_run_gh", explode)
        assert brn.resolve_releaser(REPOSITORY, "abc", "actor", offline=True) == ""

    def test_missing_gh_falls_back_to_actor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing gh is not the same as --offline: the actor still counts."""
        monkeypatch.setattr(brn, "_run_gh", _gh_stub(lambda _a: None))
        warnings: list[str] = []
        assert (
            brn.resolve_releaser(REPOSITORY, "abc", "actor", warnings=warnings)
            == "actor"
        )
        assert any("gh executable not found" in w for w in warnings)


# ── Body finalization / size budget ─────────────────────────────────────────


class TestFinalizeBody:
    def test_normal_append(self) -> None:
        body, warnings = brn.finalize_body(
            "Release notes", "<details>Git log</details>"
        )
        assert body == "Release notes\n\n<details>Git log</details>"
        assert warnings == []

    def test_empty_base_returns_details_alone(self) -> None:
        body, warnings = brn.finalize_body("", "<details>Git log</details>")
        assert body == "<details>Git log</details>"
        assert warnings == []

    def test_compact_form_when_full_log_does_not_fit(self) -> None:
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES - 200)
        body, warnings = brn.finalize_body(base, "y" * 1_000)
        assert "Git log omitted because" in body
        assert len(body.encode()) <= brn.MAX_RELEASE_BODY_BYTES
        assert any("Full Git log omitted" in w for w in warnings)

    def test_drops_log_when_no_room(self) -> None:
        base = "x" * 119_990
        body, warnings = brn.finalize_body(base, "y" * 1_000)
        assert body == base
        assert "Git log omitted because" not in body
        assert any("dropped entirely" in w for w in warnings)

    def test_warns_on_oversized_base(self) -> None:
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES + 1)
        body, warnings = brn.finalize_body(base, "<details>Git log</details>")
        assert body == base
        assert any("Base release body" in w for w in warnings)


# ── Body assembly ────────────────────────────────────────────────────────────


class TestBuildBaseBody:
    def _body(self, **overrides) -> str:
        kwargs = {
            "pkg_name": "example",
            "version": "1.0.1",
            "changelog_body": "* fix",
            "is_prerelease": False,
            "community": [],
            "internal": [],
            "releaser": "",
            "base_branch": "main",
            "default_branch": "main",
            "release_commit": "abc1234567890",
            "repository": REPOSITORY,
        }
        kwargs.update(overrides)
        return brn.build_base_body(**kwargs)

    def test_changelog_only(self) -> None:
        assert self._body(changelog_body="* fix something") == "* fix something"

    def test_prerelease_banner(self) -> None:
        body = self._body(version="1.0.1a1", is_prerelease=True)
        assert "> [!WARNING]" in body
        assert "pip install example==1.0.1a1" in body

    def test_contributors_and_releaser(self) -> None:
        body = self._body(
            community=[
                brn.Contributor("user1", "user1tw", ""),
                brn.Contributor("user2", "", "linkedin.com/in/user2"),
            ],
            internal=["maintainer1"],
            releaser="releaser1",
        )
        assert "Thanks to our community contributors: @user1" in body
        assert "[Twitter](https://x.com/user1tw)" in body
        assert "[LinkedIn](https://linkedin.com/in/user2)" in body
        assert "Internal maintainers: @maintainer1" in body
        assert "Released by: @releaser1" in body

    def test_separator_added_once_for_internal_only(self) -> None:
        body = self._body(internal=["maint"])
        assert body.count("---") == 1
        assert "Internal maintainers: @maint" in body

    def test_separator_added_once_for_releaser_only(self) -> None:
        body = self._body(releaser="shipper")
        assert body.count("---") == 1
        assert "Released by: @shipper" in body

    def test_releaser_shown_even_when_also_a_contributor(self) -> None:
        """Shipping is a distinct role from contributing; both are credited."""
        body = self._body(internal=["dual"], releaser="dual")
        assert "Internal maintainers: @dual" in body
        assert "Released by: @dual" in body

    def test_contributor_without_socials(self) -> None:
        body = self._body(community=[brn.Contributor("plain")])
        assert "contributors: @plain" in body
        assert "[Twitter]" not in body

    def test_linkedin_already_absolute_is_not_double_prefixed(self) -> None:
        body = self._body(
            community=[brn.Contributor("u", "", "https://linkedin.com/in/u")]
        )
        assert "https://https://" not in body

    def test_branch_annotation(self) -> None:
        body = self._body(base_branch="v0.7")
        assert "Released from `v0.7`" in body
        assert "abc1234" in body
        # Links to the commit, never tree/<branch>: alpha branches get deleted.
        assert "/commit/abc1234567890" in body
        assert "/tree/" not in body

    def test_no_branch_annotation_on_main(self) -> None:
        assert "Released from" not in self._body(base_branch="main")

    def test_empty_default_branch_is_treated_as_main(self) -> None:
        """An empty default_branch must not annotate every main release."""
        assert "Released from" not in self._body(base_branch="main", default_branch="")


# ── Full integration (offline mode) ─────────────────────────────────────────


class TestBuildReleaseNotes:
    def test_offline_stable_release(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        body, warnings = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=commits["hotfix"],
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert "<details>" in body
        assert "Git log" in body
        assert "community contributors" not in body
        assert "Released by:" not in body
        assert any("Offline mode" in w for w in warnings)

    def test_offline_with_changelog(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.0")
        _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.1\n\n### Bug Fixes\n\n* fix something\n",
            "release(example): 1.0.1",
        )
        head = _git(tmp_path, "rev-parse", "HEAD")

        body, _ = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=head,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert "### Bug Fixes" in body
        assert "* fix something" in body
        assert "<details>" in body

    def test_offline_prerelease(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2"
        )

        body, _ = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.1.0a1",
            release_sha=head,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert "> [!WARNING]" in body
        assert "pip install example==1.1.0a1" in body

    def test_is_prerelease_override_forces_banner(self, tmp_path: Path) -> None:
        """The caller's flag wins over version sniffing.

        CI passes the same value that drives the GitHub release's prerelease
        flag, so the banner cannot disagree with it.
        """
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1"
        )

        body, _ = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.0.b2",
            release_sha=head,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
            is_prerelease=True,
        )
        assert "> [!WARNING]" in body

    def test_unknown_package_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown package"):
            brn.build_release_notes(
                tmp_path,
                package="nonexistent",
                version="1.0.0",
                release_sha="abc123",
                repository=REPOSITORY,
            )

    def test_missing_package_dir_raises(self, tmp_path: Path) -> None:
        """A wrong --repo-root must fail, not emit a plausible empty body."""
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        with pytest.raises(ValueError, match="does not exist"):
            brn.build_release_notes(
                tmp_path,
                package="example",
                version="1.0.0",
                release_sha=_git(tmp_path, "rev-parse", "HEAD"),
                repository=REPOSITORY,
                offline=True,
                working_dir="libs/not-here",
            )

    def test_unresolvable_sha_raises(self, tmp_path: Path) -> None:
        """A bad SHA must fail loudly, not be mislabeled an initial release."""
        _create_history(tmp_path)
        with pytest.raises(ValueError, match="does not resolve to a commit"):
            brn.build_release_notes(
                tmp_path,
                package="example",
                version="1.0.1",
                release_sha="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                repository=REPOSITORY,
                offline=True,
                working_dir=str(PACKAGE_PATH),
            )


# ── CLI ──────────────────────────────────────────────────────────────────────


def _run_cli(repo: Path, *args: str, env: dict | None = None):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env=env,
    )


class TestCli:
    def _base_args(self, repo: Path, head: str) -> list[str]:
        return [
            "--package", "example",
            "--version", "1.0.1",
            "--sha", head,
            "--repo", REPOSITORY,
            "--working-dir", str(PACKAGE_PATH),
            "--offline",
            "--repo-root", str(repo),
        ]

    def test_writes_github_output_heredoc(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        output_file = tmp_path / "gh_output"
        output_file.touch()
        import os

        env = {**os.environ, "GITHUB_OUTPUT": str(output_file)}
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--github-output",
            env=env,
        )
        assert result.returncode == 0, result.stderr
        written = output_file.read_text()
        match = re.match(r"release-body<<(\S+)\n(.*)\n\1\n\Z", written, re.DOTALL)
        assert match, f"unexpected framing: {written[:200]!r}"
        assert "<details>" in match.group(2)

    def test_github_output_delimiter_is_not_bare_eof(self, tmp_path: Path) -> None:
        """A body line of exactly `EOF` must not terminate the heredoc."""
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.1\n\nBefore\nEOF\nAfter\n",
            "release(example): 1.0.1",
        )
        head = _git(tmp_path, "rev-parse", "HEAD")
        output_file = tmp_path / "gh_output"
        output_file.touch()
        import os

        env = {**os.environ, "GITHUB_OUTPUT": str(output_file)}
        result = _run_cli(
            tmp_path, *self._base_args(tmp_path, head), "--github-output", env=env
        )
        assert result.returncode == 0, result.stderr
        written = output_file.read_text()
        match = re.match(r"release-body<<(\S+)\n(.*)\n\1\n\Z", written, re.DOTALL)
        assert match, f"unexpected framing: {written[:200]!r}"
        assert match.group(1) != "EOF"
        body = match.group(2)
        assert "Before" in body
        assert "After" in body

    def test_github_output_unset_is_an_error(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        import os

        env = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--github-output",
            env=env,
        )
        assert result.returncode == 1
        assert "GITHUB_OUTPUT not set" in result.stderr

    def test_out_file(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        target = tmp_path / "body.md"
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--out",
            str(target),
        )
        assert result.returncode == 0, result.stderr
        assert "<details>" in target.read_text()
        assert not (tmp_path / "body.md.tmp").exists()

    def test_out_and_github_output_are_mutually_exclusive(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--out",
            str(tmp_path / "body.md"),
            "--github-output",
        )
        assert result.returncode == 2
        assert "not allowed with" in result.stderr

    def test_stdout_by_default(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        result = _run_cli(tmp_path, *self._base_args(tmp_path, commits["hotfix"]))
        assert result.returncode == 0, result.stderr
        assert "<details>" in result.stdout

    def test_bad_sha_exits_with_error_annotation(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
        )
        assert result.returncode == 1
        assert "::error::" in result.stderr
        assert "does not resolve to a commit" in result.stderr

    def test_warnings_are_emitted_as_annotations(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        result = _run_cli(tmp_path, *self._base_args(tmp_path, commits["hotfix"]))
        assert result.returncode == 0, result.stderr
        assert "::warning::" in result.stderr


# ── Package map ──────────────────────────────────────────────────────────────


class TestPackageMap:
    def test_all_packages_have_valid_paths(self) -> None:
        for package, path in brn.PACKAGE_MAP.items():
            assert (REPO_ROOT / path).is_dir(), f"{package}: {path} does not exist"

    def test_workflow_dropdown_matches(self) -> None:
        """The script's package map must cover every workflow option."""
        import yaml

        workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
        with open(workflow_path) as f:
            data = yaml.safe_load(f)
        options = data[True]["workflow_dispatch"]["inputs"]["package"]["options"]
        script_packages = set(brn.PACKAGE_MAP.keys())
        workflow_packages = set(options)
        missing = workflow_packages - script_packages
        extra = script_packages - workflow_packages
        assert not missing, f"In workflow but not in script: {missing}"
        assert not extra, f"In script but not in workflow: {extra}"

    def test_workflow_working_dirs_match(self) -> None:
        """Paths must match too, not just package names.

        A wrong-but-existing path (deepagents-code -> libs/cli) would otherwise
        pass every other check while scoping the git log to the wrong package.
        """
        text = _release_yml_text()
        case_arms = dict(
            re.findall(
                r"^\s{12}([a-z0-9-]+)\)\n\s+echo \"working-dir=(\S+)\"",
                text,
                re.MULTILINE,
            )
        )
        assert case_arms, "could not parse the working-dir case statement"
        assert case_arms == brn.PACKAGE_MAP, (
            "PACKAGE_MAP disagrees with release.yml's working-dir case statement"
        )
