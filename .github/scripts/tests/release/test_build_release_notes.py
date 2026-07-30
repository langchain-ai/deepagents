"""Test the shared release-notes builder."""

import json
import os
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


def _release_yml_text() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()


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
            warnings=[],
        )

        assert "<summary>Git log since example==1.0.0</summary>" in details
        assert "newest first" in details
        assert details.index(commits["hotfix"][:7]) < details.index(
            commits["feature"][:7]
        )
        assert commits["unrelated"] not in details
        # Package-scoped means the whole package path, including tests/. The
        # fixture's release-please `exclude-paths` governs version bumping, not
        # the git log, so a tests-only commit must still be listed.
        assert _entry(commits["test_only"], "test(example): add coverage") in details

    def test_initial_release_git_log(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "",
            commits["release"],
            REPOSITORY,
            warnings=[],
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
            warnings=[],
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
            warnings=[],
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
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
        )
        assert "&lt;/details&gt;&lt;!--" in details
        assert "escape </details><!--" not in details

    def test_git_log_truncates_long_subjects(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        long_subject = f"fix(example): {'x' * 250}"
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "LONG = 1\n", long_subject)
        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
        )
        assert "…" in details

    def test_git_log_truncates_before_escaping(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        subject = "x" * 199 + "&" + "x" * 60
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "ORDER = 1\n", subject)
        details = brn.generate_git_log(
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
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
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
        )
        assert "truncated to the newest" in details
        # The literal, not brn.MAX_COMMITS: asserting against the constant would
        # keep passing if the constant were lowered.
        assert details.count("https://github.com/") == 100
        assert "truncated to the newest 100 commits" in details

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
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
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
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
        )
        # Scaffolding (the <details> wrapper and description) sits outside the
        # entry budget, so allow a small margin over MAX_GIT_LOG_BYTES.
        assert len(details.encode()) < brn.MAX_GIT_LOG_BYTES + 1_000
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
            tmp_path, str(PACKAGE_PATH), "example==1.0.0", tip, REPOSITORY,
            warnings=[],
        )
        assert "truncated to the newest" in details
        assert details.count("https://github.com/") < 10


# ── Predecessor tag resolution ───────────────────────────────────────────────


class TestPreviousTagResolution:
    def test_stable_patch_uses_previous_patch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.1", commits["hotfix"], is_prerelease=False,
            warnings=[],
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
            tmp_path, "example", "1.0.2", head, is_prerelease=False,
            warnings=[],
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
            tmp_path, "example", "2.0.0", head, is_prerelease=False,
            warnings=[],
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
            tmp_path, "example", version, release, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.1.0rc1", head, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.1.0b1", head, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.1.0a1", head, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.0.1a1", head, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.1.0a1", head, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "1.1.0a7", release, is_prerelease=True,
            warnings=[],
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
            tmp_path, "example", "0.0.1", head, is_prerelease=False,
            warnings=[],
        )
        assert prev == "example==0.0.0"


class TestLatestStableTag:
    """Direct coverage of the two guards inside `_latest_stable_tag`."""

    def test_excludes_the_release_its_own_tag(self, tmp_path: Path) -> None:
        """A re-run whose own tag is already pushed must not select it.

        On notes recovery the tag exists at the release commit. Selecting it
        would make the range `tag..release_sha` empty and publish a "No commits
        found." log — the exact empty-body symptom recovery exists to fix.
        """
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): base"
        )
        _git(tmp_path, "tag", "example==1.0.1")

        assert (
            brn._latest_stable_tag(tmp_path, "example", head, "example==1.0.1") == ""
        )
        # Without the exclusion the tag would come back and empty the range.
        assert (
            brn._latest_stable_tag(tmp_path, "example", head, "") == "example==1.0.1"
        )
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.1", head, is_prerelease=False, warnings=[]
        )
        assert prev == ""

    def test_skips_prereleases_that_version_sort_above_stable(
        self, tmp_path: Path
    ) -> None:
        """`--sort=-version:refname` ranks 1.1.0rc1 above 1.0.10.

        Only the stable-only filter stops a stable release from taking a
        pre-release as its predecessor.
        """
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.10")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): rc"
        )
        _git(tmp_path, "tag", "example==1.1.0rc1")

        merged = _git(
            tmp_path, "tag", "--merged", head, "--sort=-version:refname",
            "--list", "example==*",
        ).splitlines()
        assert merged[0] == "example==1.1.0rc1", "precondition: pre-release sorts first"

        assert (
            brn._latest_stable_tag(tmp_path, "example", head, "example==2.0.0")
            == "example==1.0.10"
        )


class TestUnreachablePredecessorWarnings:
    def _repo_with_unreachable_tag(self, tmp_path: Path) -> str:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "checkout", "-b", "other")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 9\n", "feat(example): other")
        _git(tmp_path, "tag", "example==1.0.0")
        _git(tmp_path, "checkout", "main")
        return _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): main"
        )

    def test_prerelease_warns_like_stable_does(self, tmp_path: Path) -> None:
        """The anomaly warning must not be stable-only.

        Pre-releases are cut from throwaway branches and partial clones more
        often than stable releases, so they need this warning most.
        """
        head = self._repo_with_unreachable_tag(tmp_path)
        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0rc1", head, is_prerelease=True,
            warnings=warnings,
        )
        assert prev == ""
        assert any("No prior stable tag reachable" in w for w in warnings)

    def test_stable_still_warns(self, tmp_path: Path) -> None:
        head = self._repo_with_unreachable_tag(tmp_path)
        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.1.0", head, is_prerelease=False,
            warnings=warnings,
        )
        assert prev == ""
        assert any("No prior stable tag reachable" in w for w in warnings)

    @pytest.mark.parametrize("version", ["1.0.1", "1.1.0"])
    def test_warns_for_both_patch_and_minor_bumps(
        self, tmp_path: Path, version: str
    ) -> None:
        """Must fire regardless of whether the patch component is zero."""
        head = self._repo_with_unreachable_tag(tmp_path)
        warnings: list[str] = []
        brn.resolve_previous_tag(
            tmp_path, "example", version, head, is_prerelease=False,
            warnings=warnings,
        )
        assert any("No prior stable tag reachable" in w for w in warnings)

    def test_silent_when_a_predecessor_is_found(self, tmp_path: Path) -> None:
        """The anomaly warning must not fire on the happy path."""
        commits = _create_history(tmp_path)
        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.1", commits["hotfix"], is_prerelease=False,
            warnings=warnings,
        )
        assert prev == "example==1.0.0"
        assert not any("No prior stable tag reachable" in w for w in warnings)

    def test_non_numeric_patch_warns_and_falls_back(self, tmp_path: Path) -> None:
        """A PEP 440 post-release takes the stable path and must not crash."""
        assert not brn._is_prerelease("1.0.0.post1")
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): post"
        )
        warnings: list[str] = []
        prev = brn.resolve_previous_tag(
            tmp_path, "example", "1.0.0.post1", head, is_prerelease=False,
            warnings=warnings,
        )
        assert prev == "example==1.0.0"
        assert any("does not end in a numeric patch" in w for w in warnings)

    def test_is_ancestor_error_skips_candidate_and_warns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A git error must not be read as "not an ancestor" and accepted."""
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): a1"
        )
        _git(tmp_path, "tag", "example==1.1.0a1")
        tip = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): a2"
        )

        real_git_run = brn._git_run

        def fake(repo, *args):
            if args[:2] == ("merge-base", "--is-ancestor"):
                return subprocess.CompletedProcess(args=list(args), returncode=128)
            return real_git_run(repo, *args)

        monkeypatch.setattr(brn, "_git_run", fake)
        warnings: list[str] = []
        prev = brn._latest_earlier_prerelease_tag(
            tmp_path, "example", "1.1.0a2", tip, warnings
        )
        assert prev == ""
        assert any("--is-ancestor exited 128" in w for w in warnings)


# ── Release commit resolution ───────────────────────────────────────────────


class TestReleaseCommit:
    def test_stable_uses_changelog_touch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            commits["hotfix"],
            is_prerelease=False,
            warnings=[],
        )
        assert commit == commits["release"]

    def test_prerelease_uses_the_release_sha(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            commits["fix"],
            is_prerelease=True,
            warnings=[],
        )
        assert commit == commits["fix"]

    def test_stable_is_bounded_by_the_release_sha(self, tmp_path: Path) -> None:
        """A later release's commit must not be attributed to this one.

        Searching from HEAD would find the newest CHANGELOG.md touch in the
        whole clone, so a local rebuild run from an up-to-date checkout would
        credit a *later* release's contributors and merger to this release.
        """
        commits = _create_history(tmp_path)
        later_release = _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.2\n\n## 1.0.1\n",
            "release(example): 1.0.2",
        )
        assert _git(tmp_path, "rev-parse", "HEAD") == later_release

        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            commits["hotfix"],
            is_prerelease=False,
            warnings=[],
        )
        assert commit == commits["release"]

    def test_stable_falls_back_to_release_sha_without_changelog(
        self, tmp_path: Path
    ) -> None:
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base"
        )
        warnings: list[str] = []
        commit = brn.resolve_release_commit(
            tmp_path, str(PACKAGE_PATH), head, is_prerelease=False, warnings=warnings
        )
        assert commit == head
        assert any("falling back to the release SHA" in w for w in warnings)


# ── gh helpers ──────────────────────────────────────────────────────────────


def _ok(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=0, stdout=stdout, stderr=""
    )


def _fail(stderr: str, code: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=code, stdout="", stderr=stderr
    )


def _pr_handler(pr_number: str, payload: str):
    """A `_run_gh` replacement for the common two-call contributor lookup.

    The commit-to-PR `gh api` call answers *pr_number*; the follow-up
    `gh pr view` answers *payload* verbatim, so a caller can hand over either
    JSON or the raw string a `--jq` query would print.
    """

    def handler(args: list[str]) -> subprocess.CompletedProcess[str]:
        return _ok(pr_number) if args[0] == "api" else _ok(payload)

    return handler


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
        monkeypatch.setattr(brn, "_run_gh", _pr_handler("7", json.dumps(pr_payload)))
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
        assert community[0].linkedin_url == "https://www.linkedin.com/in/alice"

    def test_linkedin_scheme_is_added_at_parse_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A schemeless URL is normalized before it is stored.

        Storing one spelling keeps the gap-filling merge across PRs from making
        the published link depend on commit order.
        """
        head = self._repo(tmp_path)
        community, _ = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "alice", "is_bot": False},
                "body": "LinkedIn: linkedin.com/in/alice\n",
                "labels": [],
            },
            monkeypatch,
        )
        assert community[0].linkedin_url == "https://linkedin.com/in/alice"

    @pytest.mark.parametrize(
        "url",
        ["HTTPS://linkedin.com/in/alice", "Https://LinkedIn.com/in/alice"],
    )
    def test_an_uppercase_scheme_is_not_prefixed_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        """`_LINKEDIN_RE` is IGNORECASE, so the scheme check must be too.

        A case-sensitive check turns `HTTPS://...` into `https://HTTPS://...`,
        publishing a dead link under a real contributor's name.
        """
        head = self._repo(tmp_path)
        community, _ = self._run(
            tmp_path,
            head,
            {
                "author": {"login": "alice", "is_bot": False},
                "body": f"LinkedIn: {url}\n",
                "labels": [],
            },
            monkeypatch,
        )
        assert community[0].linkedin_url == url
        assert "https://https://" not in community[0].linkedin_url.lower()

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

        monkeypatch.setattr(brn, "_run_gh", handler)
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
            brn, "_run_gh", lambda _args: _fail("HTTP 403 rate limit", 1)
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
            warnings=[],
        ) == ([], [])


class TestContributorEdgeCases:
    def _history(self, tmp_path: Path, count: int) -> str:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        tip = ""
        for index in range(count):
            tip = _commit(
                tmp_path,
                PACKAGE_PATH / "module.py",
                f"V = {index + 1}\n",
                f"fix(example): generated {index}",
            )
        return tip

    def _collect(
        self, tmp_path: Path, tip: str, warnings: list[str]
    ) -> brn.Contributors:
        return brn.collect_contributors(
            tmp_path, str(PACKAGE_PATH), tip, "example==1.0.0", REPOSITORY,
            warnings=warnings,
        )

    def test_warns_when_the_range_exceeds_the_commit_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silently truncated contributor list looks complete in the body.

        The git log announces its own truncation in the rendered notes; this
        path has no such surface, so the warning is the only signal.
        """
        monkeypatch.setattr(brn, "MAX_CONTRIBUTOR_COMMITS", 3)
        tip = self._history(tmp_path, 5)

        seen: list[str] = []

        def handler(args: list[str]):
            if args[0] == "api":
                seen.append(args[1])
                return _ok(str(len(seen)))
            return _ok(
                json.dumps(
                    {
                        "author": {"login": f"user{len(seen)}", "is_bot": False},
                        "body": "",
                        "labels": [],
                    }
                )
            )

        monkeypatch.setattr(brn, "_run_gh", handler)
        warnings: list[str] = []
        community, _ = self._collect(tmp_path, tip, warnings)
        assert len(community) == 3
        assert any(
            "only the newest 3 were scanned" in w and "INCOMPLETE" in w
            for w in warnings
        )

    def test_abandons_the_walk_after_consecutive_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanent auth fault must not be retried once per commit."""
        monkeypatch.setattr(brn, "MAX_CONSECUTIVE_GH_FAILURES", 3)
        tip = self._history(tmp_path, 20)
        calls = 0

        def handler(args: list[str]):
            nonlocal calls
            calls += 1
            return _fail("HTTP 403: Resource not accessible by integration")

        monkeypatch.setattr(brn, "_run_gh", handler)
        warnings: list[str] = []
        community, internal = self._collect(tmp_path, tip, warnings)
        assert community == [] and internal == []
        assert calls == 3, "must stop at the failure threshold, not walk 20 commits"
        assert any("abandoned after 3 consecutive failures" in w for w in warnings)

    def test_null_labels_warns_like_a_missing_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`labels: null` hides an `internal` label just as an absent field does."""
        tip = self._history(tmp_path, 1)

        payload = json.dumps(
            {"author": {"login": "alice", "is_bot": False}, "body": "", "labels": None}
        )
        monkeypatch.setattr(brn, "_run_gh", _pr_handler("7", payload))
        warnings: list[str] = []
        self._collect(tmp_path, tip, warnings)
        assert any("returned no labels field" in w for w in warnings)

    def test_one_pr_spanning_two_commits_is_viewed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tip = self._history(tmp_path, 2)
        views = 0

        def handler(args: list[str]):
            nonlocal views
            if args[0] == "api":
                return _ok("42")
            views += 1
            return _ok(
                json.dumps(
                    {
                        "author": {"login": "alice", "is_bot": False},
                        "body": "",
                        "labels": [],
                    }
                )
            )

        monkeypatch.setattr(brn, "_run_gh", handler)
        community, _ = self._collect(tmp_path, tip, [])
        assert views == 1
        assert [c.login for c in community] == ["alice"]

    def test_socials_are_merged_across_prs_first_value_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Later PRs fill gaps only; an earlier non-empty value is kept."""
        tip = self._history(tmp_path, 2)
        bodies = [
            "Twitter: @firsthandle\n",
            "Twitter: @secondhandle\nLinkedIn: https://linkedin.com/in/alice\n",
        ]
        pr_index = 0

        def handler(args: list[str]):
            nonlocal pr_index
            if args[0] == "api":
                pr_index += 1
                return _ok(str(pr_index))
            return _ok(
                json.dumps(
                    {
                        "author": {"login": "alice", "is_bot": False},
                        "body": bodies[pr_index - 1],
                        "labels": [],
                    }
                )
            )

        monkeypatch.setattr(brn, "_run_gh", handler)
        community, _ = self._collect(tmp_path, tip, [])
        assert community == [
            brn.Contributor("alice", "firsthandle", "https://linkedin.com/in/alice")
        ]

    def test_rev_list_failure_warns_and_credits_nobody(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._history(tmp_path, 1)
        monkeypatch.setattr(brn, "_run_gh", lambda args: _ok("1"))
        warnings: list[str] = []
        community, internal = self._collect(tmp_path, "nope-nope-nope", warnings)
        assert community == [] and internal == []
        assert any("git rev-list failed" in w for w in warnings)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("", "gh returned empty output"),
            ("not json", "unparseable gh output"),
            ("[1, 2]", "expected a JSON object"),
        ],
    )
    def test_malformed_pr_payloads_are_reported_distinctly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        payload: str,
        expected: str,
    ) -> None:
        tip = self._history(tmp_path, 1)

        monkeypatch.setattr(brn, "_run_gh", _pr_handler("7", payload))
        warnings: list[str] = []
        community, _ = self._collect(tmp_path, tip, warnings)
        assert community == []
        assert any(expected in w for w in warnings)

    @pytest.mark.parametrize("author", [None, "alice", {"is_bot": False}])
    def test_unusable_author_payloads_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, author: object
    ) -> None:
        tip = self._history(tmp_path, 1)

        payload = json.dumps({"author": author, "body": "", "labels": []})
        monkeypatch.setattr(brn, "_run_gh", _pr_handler("7", payload))
        warnings: list[str] = []
        community, _ = self._collect(tmp_path, tip, warnings)
        assert community == []
        assert any("PR #7" in w for w in warnings)

    def test_incomplete_warning_names_the_first_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare count cannot be debugged; the reason must survive."""
        tip = self._history(tmp_path, 1)
        monkeypatch.setattr(
            brn, "_run_gh", lambda args: _fail("HTTP 404: no such repo")
        )
        warnings: list[str] = []
        self._collect(tmp_path, tip, warnings)
        assert any("no such repo" in w and "INCOMPLETE" in w for w in warnings)


# ── Releaser resolution ─────────────────────────────────────────────────


class TestResolveReleaser:
    def test_warns_when_merged_by_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unmerged PR returns "" on a *successful* call.

        Without this warning the actor is credited as the releaser with nothing
        recorded about why the real lookup came back empty.
        """

        monkeypatch.setattr(brn, "_run_gh", _pr_handler("42", ""))
        warnings: list[str] = []
        releaser = brn.resolve_releaser(
            REPOSITORY, "abc123", "dispatcher", warnings=warnings
        )
        assert releaser == "dispatcher"
        assert any("has no mergedBy" in w for w in warnings)

    def test_warns_when_commit_pulls_lookup_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(brn, "_run_gh", lambda _a: _fail("HTTP 500"))
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
        monkeypatch.setattr(brn, "_run_gh", lambda _a: _ok(""))
        warnings: list[str] = []

        releaser = brn.resolve_releaser(
            REPOSITORY, "abc123", "dispatcher", warnings=warnings
        )

        assert releaser == "dispatcher"
        assert warnings == []

    def test_uses_merger_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(brn, "_run_gh", _pr_handler("7", "merger"))
        assert (
            brn.resolve_releaser(REPOSITORY, "abc123", "dispatcher", warnings=[]) == "merger"
        )

    def test_warns_when_pr_view_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(args: list[str]):
            return _ok("7") if args[0] == "api" else _fail("HTTP 502")

        monkeypatch.setattr(brn, "_run_gh", handler)
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
        monkeypatch.setattr(brn, "_run_gh", lambda _a: _ok(""))
        assert brn.resolve_releaser(REPOSITORY, "abc123", actor, warnings=[]) == ""

    def test_offline_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(_args):
            raise AssertionError("gh must not run in offline mode")

        monkeypatch.setattr(brn, "_run_gh", explode)
        assert brn.resolve_releaser(REPOSITORY, "abc", "actor", offline=True, warnings=[]) == ""

    def test_missing_gh_falls_back_to_actor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing gh is not the same as --offline: the actor still counts."""
        monkeypatch.setattr(brn, "_run_gh", lambda _a: None)
        warnings: list[str] = []
        assert (
            brn.resolve_releaser(REPOSITORY, "abc", "actor", warnings=warnings)
            == "actor"
        )
        assert any("gh executable not found" in w for w in warnings)


class TestRunGh:
    """Exercise the real `_run_gh`, which every other gh test stubs out."""

    def test_missing_gh_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None means "no gh here", which callers treat differently from a
        failed call — it is what keeps `--offline` distinguishable."""

        def boom(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'gh'")

        monkeypatch.setattr(brn.subprocess, "run", boom)
        assert brn._run_gh(["api", "/x"]) is None
        assert brn._gh_result_detail(None) == "gh executable not found"

    def test_timeout_becomes_a_failed_result_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="gh", timeout=brn.GH_TIMEOUT_SECONDS)

        monkeypatch.setattr(brn.subprocess, "run", boom)
        result = brn._run_gh(["api", "/x"])
        assert result is not None
        assert result.returncode == 124
        assert "timed out" in result.stderr

    def test_oserror_becomes_a_failed_result_not_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permission or exec error is not "gh is missing"."""

        def boom(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(brn.subprocess, "run", boom)
        result = brn._run_gh(["api", "/x"])
        assert result is not None
        assert result.returncode == 126

    def test_rate_limit_is_retried_with_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0
        sleeps: list[float] = []

        def fake_run(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return subprocess.CompletedProcess(
                args=["gh"],
                returncode=1,
                stdout="",
                stderr="API rate limit exceeded for user",
            )

        monkeypatch.setattr(brn.subprocess, "run", fake_run)
        monkeypatch.setattr(brn.time, "sleep", sleeps.append)
        brn._run_gh(["api", "/x"])
        assert attempts == brn.GH_MAX_ATTEMPTS
        assert sleeps == [2, 4]

    def test_non_rate_limit_failure_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0

        def fake_run(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return subprocess.CompletedProcess(
                args=["gh"], returncode=1, stdout="", stderr="could not resolve host"
            )

        monkeypatch.setattr(brn.subprocess, "run", fake_run)
        monkeypatch.setattr(brn.time, "sleep", lambda _: pytest.fail("must not sleep"))
        brn._run_gh(["api", "/x"])
        assert attempts == 1

    def test_a_sha_containing_403_is_not_treated_as_throttling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare status codes are valid hex and match inside commit SHAs.

        Matching "403" as a substring would turn a permanent failure into the
        full retry budget on every commit in the range.
        """
        attempts = 0

        def fake_run(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            return subprocess.CompletedProcess(
                args=["gh"],
                returncode=1,
                stdout="",
                stderr="no pull request found for 403bead1c0ffee0000000000000000000000000",
            )

        monkeypatch.setattr(brn.subprocess, "run", fake_run)
        monkeypatch.setattr(brn.time, "sleep", lambda _: pytest.fail("must not sleep"))
        brn._run_gh(["api", "/x"])
        assert attempts == 1

    def test_detail_falls_back_to_the_exit_status(self) -> None:
        result = subprocess.CompletedProcess(
            args=["gh"], returncode=7, stdout="", stderr=""
        )
        assert brn._gh_result_detail(result) == "exit status 7"


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

    def test_oversized_base_does_not_also_blame_the_git_log(self) -> None:
        """Two warnings would defeat the point of distinguishing the causes.

        Nothing here can shrink the base body, so reporting the Git log as the
        thing that did not fit would send a maintainer after the wrong cause.
        """
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES + 1)
        _, warnings = brn.finalize_body(base, "<details>Git log</details>")
        assert not any("Full Git log omitted" in w for w in warnings)

    def test_exact_budget_boundary_appends_cleanly(self) -> None:
        """The budget is inclusive: a body landing exactly on it still fits."""
        details = "<details>Git log</details>"
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES - len(details) - 2)
        body, warnings = brn.finalize_body(base, details)
        assert len(body.encode()) == brn.MAX_RELEASE_BODY_BYTES
        assert body.endswith(details)
        assert warnings == []

    def test_one_byte_over_the_budget_falls_back_to_compact(self) -> None:
        details = "<details>Git log</details>"
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES - len(details) - 1)
        body, warnings = brn.finalize_body(base, details)
        assert not body.endswith(details)
        assert any("Full Git log omitted" in w for w in warnings)


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

    def test_prerelease_banner_is_separated_from_the_changelog(self) -> None:
        body = self._body(
            version="1.0.1a1", is_prerelease=True, changelog_body="* fix something"
        )
        assert body.endswith("`\n\n* fix something")

    def test_empty_changelog_leaves_no_blank_line_run(self) -> None:
        """Pre-releases have no CHANGELOG.md section, so this is the normal case.

        An unconditional separator after the banner stacks with the one every
        later section prepends, leaving three blank lines in the published body.
        """
        body = self._body(
            version="1.0.1a1",
            is_prerelease=True,
            changelog_body="",
            releaser="shipper",
        )
        assert "\n\n\n" not in body
        assert body.count("---") == 1
        assert "Released by: @shipper" in body

    def test_contributors_and_releaser(self) -> None:
        body = self._body(
            community=[
                brn.Contributor("user1", "user1tw", ""),
                brn.Contributor("user2", "", "https://linkedin.com/in/user2"),
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
        body = self._body(community=[brn.Contributor("plain", "", "")])
        assert "contributors: @plain" in body
        assert "[Twitter]" not in body

    def test_linkedin_url_is_rendered_verbatim(self) -> None:
        body = self._body(
            community=[brn.Contributor("u", "", "https://linkedin.com/in/u")]
        )
        assert "[LinkedIn](https://linkedin.com/in/u)" in body
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


class TestBuildReleaseNotesOnline:
    """The non-offline path, where contributor and releaser lookups run.

    Every other integration test passes `offline=True`, which skips this seam
    entirely — including the argument order and which commit bounds the
    contributor range.
    """

    def _stub(self, monkeypatch: pytest.MonkeyPatch, seen_ranges: list[str]) -> None:
        real_git_run = brn._git_run

        def spy(repo, *args):
            if args and args[0] == "rev-list":
                seen_ranges.append(args[1])
            return real_git_run(repo, *args)

        def handler(args: list[str]):
            if args[0] == "api":
                return _ok("11")
            if "mergedBy" in args:
                return _ok("merger-person")
            return _ok(
                json.dumps(
                    {
                        "author": {"login": "outside-dev", "is_bot": False},
                        "body": "Twitter: @outsidedev\n",
                        "labels": [],
                    }
                )
            )

        monkeypatch.setattr(brn, "_git_run", spy)
        monkeypatch.setattr(brn, "_run_gh", handler)

    def test_contributors_and_releaser_reach_the_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commits = _create_history(tmp_path)
        ranges: list[str] = []
        self._stub(monkeypatch, ranges)

        body, _ = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=commits["hotfix"],
            repository=REPOSITORY,
            working_dir=str(PACKAGE_PATH),
            actor="dispatcher",
        )
        assert "Thanks to our community contributors: @outside-dev" in body
        assert "[Twitter](https://x.com/outsidedev)" in body
        assert "Released by: @merger-person" in body

    def test_contributor_range_ends_at_the_release_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attribution stops at the release commit, not the release SHA.

        A transposed or HEAD-anchored range yields an empty list at exit 0 with
        no warning, so the range itself has to be pinned.
        """
        commits = _create_history(tmp_path)
        ranges: list[str] = []
        self._stub(monkeypatch, ranges)

        brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=commits["hotfix"],
            repository=REPOSITORY,
            working_dir=str(PACKAGE_PATH),
        )
        assert ranges == [f"example==1.0.0..{commits['release']}"]


class TestMissingTagsDetection:
    def test_tagless_clone_warns_instead_of_claiming_initial_release(
        self, tmp_path: Path
    ) -> None:
        """No tags at all looks identical to a genuine first release.

        The changelog documenting earlier versions is the evidence that tells
        the two apart, and RELEASING.md names this as the case to watch for.
        """
        _init_repo(tmp_path)
        head = _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.1\n\n* second release\n\n## 1.0.0\n\n* first release\n",
            "release(example): 1.0.1",
        )
        assert _git(tmp_path, "tag", "--list") == "", "precondition: no tags"

        body, warnings = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=head,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert "<summary>Git log for initial release</summary>" in body
        assert any("probably missing tags" in w for w in warnings)

    def test_genuine_initial_release_does_not_warn(self, tmp_path: Path) -> None:
        """A changelog with only this version is a real first release."""
        _init_repo(tmp_path)
        head = _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.0\n\n* first release\n",
            "release(example): 1.0.0",
        )
        _, warnings = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.0",
            release_sha=head,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert not any("missing tags" in w for w in warnings)


class TestChangelogIsReadAtTheReleaseSha:
    def test_later_changelog_edits_do_not_leak_into_the_body(
        self, tmp_path: Path
    ) -> None:
        """The section comes from the release commit, not the working tree."""
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        release = _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.1\n\n* as released\n",
            "release(example): 1.0.1",
        )
        _commit(
            tmp_path,
            PACKAGE_PATH / "CHANGELOG.md",
            "## 1.0.1\n\n* edited after the fact\n",
            "docs(example): rewrite changelog",
        )

        body, _ = brn.build_release_notes(
            tmp_path,
            package="example",
            version="1.0.1",
            release_sha=release,
            repository=REPOSITORY,
            offline=True,
            working_dir=str(PACKAGE_PATH),
        )
        assert "* as released" in body
        assert "edited after the fact" not in body


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


def _parse_heredoc(written: str) -> re.Match[str]:
    """Parse the ``key<<DELIM\\n...\\nDELIM\\n`` framing written to $GITHUB_OUTPUT.

    Group 1 is the delimiter, group 2 the body.
    """
    match = re.match(r"release-body<<(\S+)\n(.*)\n\1\n\Z", written, re.DOTALL)
    assert match, f"unexpected framing: {written[:200]!r}"
    return match


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
        env = {**os.environ, "GITHUB_OUTPUT": str(output_file)}
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--github-output",
            env=env,
        )
        assert result.returncode == 0, result.stderr
        match = _parse_heredoc(output_file.read_text())
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
        env = {**os.environ, "GITHUB_OUTPUT": str(output_file)}
        result = _run_cli(
            tmp_path, *self._base_args(tmp_path, head), "--github-output", env=env
        )
        assert result.returncode == 0, result.stderr
        match = _parse_heredoc(output_file.read_text())
        assert match.group(1) != "EOF"
        body = match.group(2)
        assert "Before" in body
        assert "After" in body

    def test_github_output_unset_is_an_error(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
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
        # stderr, not stdout: the body itself goes to stdout by default, so an
        # annotation there would corrupt it. The runner parses both streams.
        assert "::warning::" in result.stderr

    @pytest.mark.parametrize(
        ("flag_value", "expect_banner"),
        [("true", True), ("false", False)],
    )
    def test_is_prerelease_flag_decides_the_banner(
        self, tmp_path: Path, flag_value: str, expect_banner: bool
    ) -> None:
        """The workflow's authoritative flag must win over version sniffing.

        A regression to a plain truthiness check would put the pre-release
        banner on every stable release.
        """
        commits = _create_history(tmp_path)
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--is-prerelease",
            flag_value,
        )
        assert result.returncode == 0, result.stderr
        assert ("> [!WARNING]" in result.stdout) is expect_banner

    def test_empty_is_prerelease_falls_back_to_sniffing(self, tmp_path: Path) -> None:
        """An unset CI expression must degrade, not fail the step."""
        _init_repo(tmp_path)
        head = _commit(
            tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1"
        )
        result = _run_cli(
            tmp_path,
            "--package", "example",
            "--version", "1.0.1a1",
            "--sha", head,
            "--repo", REPOSITORY,
            "--working-dir", str(PACKAGE_PATH),
            "--offline",
            "--repo-root", str(tmp_path),
            "--is-prerelease", "",
        )
        assert result.returncode == 0, result.stderr
        assert "> [!WARNING]" in result.stdout

    def test_unwritable_out_path_echoes_the_body_to_stdout(
        self, tmp_path: Path
    ) -> None:
        """The body costs up to 200 API calls; a write failure must not lose it."""
        commits = _create_history(tmp_path)
        result = _run_cli(
            tmp_path,
            *self._base_args(tmp_path, commits["hotfix"]),
            "--out",
            str(tmp_path / "no-such-dir" / "body.md"),
        )
        assert result.returncode == 1
        assert "::error::" in result.stderr
        assert "body follows on stdout" in result.stderr
        assert "<details>" in result.stdout

    def test_warnings_are_written_to_the_step_summary(self, tmp_path: Path) -> None:
        """A green fail-open job needs a surface a reviewer actually sees."""
        commits = _create_history(tmp_path)
        summary = tmp_path / "summary.md"
        summary.touch()
        env = {**os.environ, "GITHUB_STEP_SUMMARY": str(summary)}
        result = _run_cli(
            tmp_path, *self._base_args(tmp_path, commits["hotfix"]), env=env
        )
        assert result.returncode == 0, result.stderr
        written = summary.read_text()
        assert "Release notes built with warnings" in written
        assert "gh release edit" in written

    def test_step_summary_is_skipped_when_there_are_no_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        brn._write_step_summary("example", "1.0.1", [])
        assert not summary.exists()


class TestWorkflowWiring:
    """Pin the coupling between the script and the workflow that calls it.

    Renaming the step id or the output key publishes an empty release body, and
    the fail-open design means nothing else would fail.
    """

    def test_step_id_and_output_key_match_the_script(self) -> None:
        text = _release_yml_text()
        assert "id: finalize-release-body" in text
        assert (
            "release-body: ${{ steps.finalize-release-body.outputs.release-body }}"
            in text
        )
        # The key the script writes into $GITHUB_OUTPUT.
        assert 'handle.write(f"release-body<<' in SCRIPT_PATH.read_text()

    def test_workflow_invokes_the_script_with_the_authoritative_flags(self) -> None:
        text = _release_yml_text()
        assert "build_release_notes.py" in text
        for flag in ("--working-dir", "--is-prerelease", "--github-output"):
            assert flag in text, f"workflow no longer passes {flag}"

    def test_release_notes_job_stays_fail_open(self) -> None:
        """`mark-release` must not gate on the notes job.

        Gating would trade a cosmetic empty body for a blocked release that has
        already been published to PyPI.
        """
        # Comments are stripped deliberately: release.yml carries a "do NOT add
        # needs.release-notes.result" note, which is the string being banned.
        live_lines = [
            line
            for line in _release_yml_text().splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert "needs.release-notes.result" not in "\n".join(live_lines)


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
