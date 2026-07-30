"""Test the shared release-notes builder."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts" / "release"
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

    def test_parse_prerelease_recognizes_all_forms(self) -> None:
        assert brn._parse_prerelease("1.1.0a1") == ("1.1.0", 1, 1)
        assert brn._parse_prerelease("1.1.0b2") == ("1.1.0", 2, 2)
        assert brn._parse_prerelease("1.1.0rc1") == ("1.1.0", 3, 1)
        assert brn._parse_prerelease("1.1.0.dev3") == ("1.1.0", 0, 3)
        assert brn._parse_prerelease("1.1.0-rc.7") == ("1.1.0", 3, 7)
        assert brn._parse_prerelease("1.1.0-alpha.1") == ("1.1.0", 1, 1)
        assert brn._parse_prerelease("1.1.0-beta.2") == ("1.1.0", 2, 2)
        assert brn._parse_prerelease("1.1.0-preview.1") == ("1.1.0", 3, 1)

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
        body = brn.extract_changelog_section(changelog, "1.0.1")
        assert "### Bug Fixes" in body
        assert "* fix something" in body
        assert "* initial" not in body

    def test_extracts_bracketed_version_section(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            "## [1.0.1](https://github.com/compare) (2024-01-01)\n\n"
            "* new feature\n\n"
            "## [1.0.0](https://github.com/compare) (2024-01-01)\n\n"
            "* old\n"
        )
        body = brn.extract_changelog_section(changelog, "1.0.1")
        assert "* new feature" in body
        assert "* old" not in body

    def test_returns_empty_for_missing_version(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## 1.0.0\n\n* initial\n")
        body = brn.extract_changelog_section(changelog, "2.0.0")
        assert body == ""

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        body = brn.extract_changelog_section(tmp_path / "NOPE.md", "1.0.0")
        assert body == ""

    def test_last_version_section_extracted(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n"
            "## 2.0.0\n\n"
            "* latest\n\n"
            "## 1.0.0\n\n"
            "* oldest\n"
        )
        body = brn.extract_changelog_section(changelog, "1.0.0")
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
        # Verify ordering: hotfix should appear before feature
        hotfix_pos = details.index(commits["hotfix"][:7])
        feature_pos = details.index(commits["feature"][:7])
        assert hotfix_pos < feature_pos
        # Unrelated commit should not be present
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

    def test_git_log_reports_failure_for_bad_range(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==9.9.9",
            _git(tmp_path, "rev-parse", "HEAD"),
            REPOSITORY,
        )
        assert "Git log unavailable" in details

    def test_git_log_escapes_html(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        subject = 'fix(example): escape </details><!--'
        tip = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "ESCAPED = 1\n",
            subject,
        )
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            tip,
            REPOSITORY,
        )
        assert "&lt;/details&gt;&lt;!--" in details
        assert "escape </details><!--" not in details

    def test_git_log_truncates_long_subjects(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        long_subject = f"fix(example): {'x' * 250}"
        tip = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "LONG = 1\n",
            long_subject,
        )
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            tip,
            REPOSITORY,
        )
        assert "…" in details

    def test_git_log_truncates_before_escaping(self, tmp_path: Path) -> None:
        _create_history(tmp_path)
        subject = "x" * 199 + "&" + "x" * 60
        tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "ORDER = 1\n", subject)
        details = brn.generate_git_log(
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            tip,
            REPOSITORY,
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
            tmp_path,
            str(PACKAGE_PATH),
            "example==1.0.0",
            tip,
            REPOSITORY,
        )
        assert "truncated to the newest" in details
        assert details.count("https://github.com/") <= brn.MAX_COMMITS


# ── Predecessor tag resolution ───────────────────────────────────────────────


class TestPreviousTagResolution:
    def test_stable_patch_uses_previous_patch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.0.1",
            commits["hotfix"],
            is_prerelease=False,
        )
        assert prev == "example==1.0.0"

    def test_stable_x_y_0_falls_back_to_latest_stable(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "fix(example): one")
        _git(tmp_path, "tag", "example==1.0.10")
        head = _commit(tmp_path, PACKAGE_PATH / "CHANGELOG.md", "## 2.0.0\n", "release(example): 2.0.0")

        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "2.0.0",
            head,
            is_prerelease=False,
        )
        assert prev == "example==1.0.10"

    def test_initial_release_has_no_predecessor(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.0.0",
            head,
            is_prerelease=False,
        )
        assert prev == ""

    def test_prerelease_uses_earlier_sibling(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        base = _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        _create_sibling_prerelease_tag(tmp_path, base=base, version="1.1.0a1", index=0)

        _git(tmp_path, "checkout", "-b", "current", base)
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "CURRENT = 1\n", "hotfix(example): a2")

        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.1.0a2",
            head,
            is_prerelease=True,
        )
        assert prev == "example==1.1.0a1"

    def test_prerelease_falls_back_to_base_version_tag(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.1")
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2")

        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.0.1a1",
            head,
            is_prerelease=True,
        )
        assert prev == "example==1.0.1"

    def test_prerelease_falls_back_to_latest_stable(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2")

        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.1.0a1",
            head,
            is_prerelease=True,
        )
        assert prev == "example==1.0.0"

    def test_prerelease_rejects_tag_ahead_of_release(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
        _git(tmp_path, "tag", "example==1.0.0")
        release = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            "CURRENT = 1\n",
            "hotfix(example): prerelease 1.1.0a7",
        )
        _git(tmp_path, "checkout", "-b", "future-prerelease")
        _commit(tmp_path, PACKAGE_PATH / "module.py", "FUTURE = 1\n", "hotfix(example): a6")
        _git(tmp_path, "tag", "example==1.1.0a6")
        _git(tmp_path, "checkout", "main")

        prev = brn.resolve_previous_tag(
            tmp_path,
            "example",
            "1.1.0a7",
            release,
            is_prerelease=True,
        )
        # a6 is ahead of release, so it should be rejected; falls back to latest stable
        assert prev == "example==1.0.0"


# ── Release commit resolution ───────────────────────────────────────────────


class TestReleaseCommit:
    def test_stable_uses_changelog_touch(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            is_prerelease=False,
        )
        assert commit == commits["release"]

    def test_prerelease_uses_head(self, tmp_path: Path) -> None:
        commits = _create_history(tmp_path)
        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            is_prerelease=True,
        )
        assert commit == commits["hotfix"]

    def test_stable_falls_back_to_head_without_changelog(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
        commit = brn.resolve_release_commit(
            tmp_path,
            str(PACKAGE_PATH),
            is_prerelease=False,
        )
        assert commit == head


# ── Releaser resolution ─────────────────────────────────────────────────


class TestResolveReleaser:
    def test_warns_when_commit_pulls_lookup_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failure = subprocess.CompletedProcess(
            args=["gh"],
            returncode=1,
            stdout="",
            stderr="HTTP 500",
        )
        monkeypatch.setattr(brn, "_run_gh", lambda _args: failure)
        warnings: list[str] = []

        releaser = brn.resolve_releaser(
            REPOSITORY,
            "abc123",
            "dispatcher",
            warnings=warnings,
        )

        assert releaser == "dispatcher"
        assert warnings == [
            "releaser lookup: commit-pulls API failed: HTTP 500 — falling back to actor"
        ]

    def test_no_warning_when_commit_has_no_pull_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        success = subprocess.CompletedProcess(
            args=["gh"],
            returncode=0,
            stdout="",
            stderr="",
        )
        monkeypatch.setattr(brn, "_run_gh", lambda _args: success)
        warnings: list[str] = []

        releaser = brn.resolve_releaser(
            REPOSITORY,
            "abc123",
            "dispatcher",
            warnings=warnings,
        )

        assert releaser == "dispatcher"
        assert warnings == []


# ── Body finalization / size budget ─────────────────────────────────────────


class TestFinalizeBody:
    def test_normal_append(self) -> None:
        body, warning = brn.finalize_body("Release notes", "<details>Git log</details>")
        assert body == "Release notes\n\n<details>Git log</details>"
        assert warning is None

    def test_empty_base_returns_details_alone(self) -> None:
        body, warning = brn.finalize_body("", "<details>Git log</details>")
        assert body == "<details>Git log</details>"
        assert warning is None

    def test_respects_size_limit(self) -> None:
        base = "x" * 119_800
        details = "y" * 1_000
        body, _ = brn.finalize_body(base, details)
        assert len(body.encode()) <= brn.MAX_RELEASE_BODY_BYTES

    def test_drops_log_when_no_room(self) -> None:
        base = "x" * 119_990
        details = "y" * 1_000
        body, _ = brn.finalize_body(base, details)
        assert body == base
        assert "Git log omitted because" not in body

    def test_warns_on_oversized_base(self) -> None:
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES + 1)
        body, warning = brn.finalize_body(base, "<details>Git log</details>")
        assert body == base
        assert "Base release body" in warning

    def test_compact_form_when_full_log_does_not_fit(self) -> None:
        # Base is large enough that the full log doesn't fit but the compact form does.
        base = "x" * (brn.MAX_RELEASE_BODY_BYTES - 200)
        details = "y" * 1_000
        body, _ = brn.finalize_body(base, details)
        assert "Git log omitted because" in body


# ── Body assembly ────────────────────────────────────────────────────────────


class TestBuildBaseBody:
    def test_changelog_only(self) -> None:
        body = brn.build_base_body(
            pkg_name="example",
            version="1.0.1",
            changelog_body="* fix something",
            is_prerelease=False,
            community=[],
            internal=[],
            releaser="",
            base_branch="main",
            default_branch="main",
            release_commit="abc123",
            repository=REPOSITORY,
        )
        assert body == "* fix something"

    def test_prerelease_banner(self) -> None:
        body = brn.build_base_body(
            pkg_name="example",
            version="1.0.1a1",
            changelog_body="* fix",
            is_prerelease=True,
            community=[],
            internal=[],
            releaser="",
            base_branch="alpha/test",
            default_branch="main",
            release_commit="abc123",
            repository=REPOSITORY,
        )
        assert "> [!WARNING]" in body
        assert "pip install example==1.0.1a1" in body

    def test_contributors_and_releaser(self) -> None:
        body = brn.build_base_body(
            pkg_name="example",
            version="1.0.1",
            changelog_body="* fix",
            is_prerelease=False,
            community=[
                {"login": "user1", "twitter": "user1tw", "linkedin": ""},
                {"login": "user2", "twitter": "", "linkedin": "linkedin.com/in/user2"},
            ],
            internal=["maintainer1"],
            releaser="releaser1",
            base_branch="main",
            default_branch="main",
            release_commit="abc123",
            repository=REPOSITORY,
        )
        assert "Thanks to our community contributors: @user1" in body
        assert "[Twitter](https://x.com/user1tw)" in body
        assert "[LinkedIn](https://linkedin.com/in/user2)" in body
        assert "Internal maintainers: @maintainer1" in body
        assert "Released by: @releaser1" in body

    def test_branch_annotation(self) -> None:
        body = brn.build_base_body(
            pkg_name="example",
            version="1.0.1",
            changelog_body="* fix",
            is_prerelease=False,
            community=[],
            internal=[],
            releaser="",
            base_branch="v0.7",
            default_branch="main",
            release_commit="abc1234567890",
            repository=REPOSITORY,
        )
        assert "Released from `v0.7`" in body
        assert "abc1234" in body

    def test_no_branch_annotation_on_main(self) -> None:
        body = brn.build_base_body(
            pkg_name="example",
            version="1.0.1",
            changelog_body="* fix",
            is_prerelease=False,
            community=[],
            internal=[],
            releaser="",
            base_branch="main",
            default_branch="main",
            release_commit="abc123",
            repository=REPOSITORY,
        )
        assert "Released from" not in body


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
        # Should contain git log (no changelog section for 1.0.1 in this fixture)
        assert "<details>" in body
        assert "Git log" in body
        # No contributor section in offline mode
        assert "community contributors" not in body
        assert "Released by:" not in body
        assert any("Offline mode" in w for w in warnings)

    def test_offline_with_changelog(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
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
        # Git log appended after changelog
        assert "<details>" in body

    def test_offline_prerelease(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
        _git(tmp_path, "tag", "example==1.0.0")
        head = _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2")

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

    def test_unknown_package_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown package"):
            brn.build_release_notes(
                tmp_path,
                package="nonexistent",
                version="1.0.0",
                release_sha="abc123",
                repository=REPOSITORY,
            )


# ── Package map ──────────────────────────────────────────────────────────────


class TestPackageMap:
    def test_all_packages_have_valid_paths(self) -> None:
        for package, path in brn.PACKAGE_MAP.items():
            full_path = REPO_ROOT / path
            assert full_path.is_dir(), f"{package}: {path} does not exist"

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
