"""Test the package Git log included in GitHub release notes."""

import json
import os
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
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


def _workflow_step(step_id: str) -> str:
    with WORKFLOW_PATH.open() as file:
        workflow = yaml.safe_load(file)
    steps = workflow["jobs"]["release-notes"]["steps"]
    return next(step["run"] for step in steps if step.get("id") == step_id)


def _run_git_log_step(
    repo: Path,
    *,
    previous_tag: str,
    release_sha: str,
) -> str:
    output = repo / "github-output.txt"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "PREV_TAG": previous_tag,
        "RELEASE_SHA": release_sha,
        "REPOSITORY": REPOSITORY,
        "WORKING_DIR": str(PACKAGE_PATH),
    }
    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _workflow_step("generate-git-log")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return _read_multiline_output(output, "details")


def _read_multiline_output(output: Path, name: str) -> str:
    contents = output.read_text()
    prefix = f"{name}<<EOF\n"
    suffix = "\nEOF\n"
    assert contents.startswith(prefix)
    assert contents.endswith(suffix)
    return contents.removeprefix(prefix).removesuffix(suffix)


class _ResolveRefs(NamedTuple):
    """Result of the resolve-refs step.

    ``outputs`` holds the parsed ``$GITHUB_OUTPUT`` values; ``stdout`` holds the
    step's stdout, where GitHub ``::warning::`` workflow commands are emitted so
    tests can assert on them.
    """

    outputs: dict[str, str]
    stdout: str


def _run_resolve_refs_step(repo: Path, *, version: str) -> _ResolveRefs:
    output = repo / "resolve-refs-output.txt"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "PKG_NAME": "example",
        "RELEASE_SHA": _git(repo, "rev-parse", "HEAD"),
        "VERSION": version,
        "WORKING_DIR": str(PACKAGE_PATH),
    }
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _workflow_step("resolve-refs")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    outputs = dict(
        line.split("=", maxsplit=1) for line in output.read_text().splitlines()
    )
    return _ResolveRefs(outputs=outputs, stdout=result.stdout)


class _FinalizeBody(NamedTuple):
    """Result of the finalize-release-body step.

    ``release_body`` is the finalized ``$GITHUB_OUTPUT`` value; ``stdout`` holds
    the step's stdout, where GitHub ``::warning::`` workflow commands are emitted.
    """

    release_body: str
    stdout: str


def _run_finalize_release_body_step(
    directory: Path,
    *,
    base_release_body: str,
    git_log_details: str,
) -> _FinalizeBody:
    output = directory / "finalize-output.txt"
    output.unlink(missing_ok=True)
    env = {
        **os.environ,
        "BASE_RELEASE_BODY": base_release_body,
        "GITHUB_OUTPUT": str(output),
        "GIT_LOG_DETAILS": git_log_details,
    }
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _workflow_step("finalize-release-body")],
        cwd=directory,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return _FinalizeBody(
        release_body=_read_multiline_output(output, "release-body"),
        stdout=result.stdout,
    )


def _create_history(repo: Path) -> dict[str, str]:
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


def _entry(sha: str, subject: str) -> str:
    return f"- [`{sha[:7]}`](https://github.com/{REPOSITORY}/commit/{sha}) {subject}"


def test_previous_tag_must_be_in_release_history(tmp_path: Path) -> None:
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
        tmp_path,
        PACKAGE_PATH / "module.py",
        "NEWER = 1\n",
        "fix(example): newer line",
    )
    _git(tmp_path, "tag", "example==1.0.1")

    _git(tmp_path, "checkout", "main")
    _commit(
        tmp_path,
        PACKAGE_PATH / "module.py",
        "CURRENT = 1\n",
        "fix(example): current line",
    )
    release = _commit(
        tmp_path,
        PACKAGE_PATH / "CHANGELOG.md",
        "## 1.0.2\n",
        "release(example): 1.0.2",
    )

    outputs = _run_resolve_refs_step(tmp_path, version="1.0.2").outputs

    assert outputs["prev-tag"] == "example==1.0.0"
    assert outputs["release-commit"] == release


def test_temporary_repo_disables_inherited_commit_signing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("[commit]\n\tgpgSign = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    repo = tmp_path / "repo"
    repo.mkdir()

    commits = _create_history(repo)

    assert commits["hotfix"] == _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "config", "--local", "--bool", "commit.gpgSign") == "false"


def test_finalize_release_body_respects_github_size_limit(tmp_path: Path) -> None:
    body = _run_finalize_release_body_step(
        tmp_path,
        base_release_body="Release notes",
        git_log_details="<details>Git log</details>",
    ).release_body
    assert body == "Release notes\n\n<details>Git log</details>"

    body = _run_finalize_release_body_step(
        tmp_path,
        base_release_body="x" * 119_800,
        git_log_details="y" * 1_000,
    ).release_body
    assert len(body.encode()) <= 120_000
    assert "Git log omitted because" in body
    assert "y" * 1_000 not in body


def test_release_without_changelog_includes_git_log_once(tmp_path: Path) -> None:
    base_step = _workflow_step("generate-release-body")
    assert "Falling back to git log for release notes" not in base_step

    details = "<details>\n<summary>Git log</summary>\n\nOne commit\n\n</details>"
    body = _run_finalize_release_body_step(
        tmp_path,
        base_release_body="",
        git_log_details=details,
    ).release_body

    assert body == details
    assert body.count("One commit") == 1


def test_stable_release_git_log_includes_package_history_newest_first(
    tmp_path: Path,
) -> None:
    commits = _create_history(tmp_path)

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=commits["hotfix"],
    )

    hotfix = _entry(commits["hotfix"], "hotfix(example): repair release")
    release = _entry(commits["release"], "release(example): 1.0.1")
    fix = _entry(commits["fix"], "fix(example): correct feature")
    test_only = _entry(commits["test_only"], "test(example): add coverage")
    feature = _entry(commits["feature"], "feat(example): add feature")
    assert details == (
        "<details>\n"
        "<summary>Git log since example==1.0.0</summary>\n\n"
        "This commit history includes changes to this package. Commits are "
        "listed newest first.\n\n"
        f"{hotfix}\n{release}\n{fix}\n{test_only}\n{feature}\n\n"
        "</details>"
    )
    assert commits["unrelated"] not in details


def test_git_log_escapes_and_limits_commit_subjects(tmp_path: Path) -> None:
    _create_history(tmp_path)
    subject = f"fix(example): escape </details><!--{'x' * 250}"
    tip = _commit(
        tmp_path,
        PACKAGE_PATH / "module.py",
        "ESCAPED = 1\n",
        subject,
    )

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=tip,
    )

    assert "escape &lt;/details&gt;&lt;!--" in details
    assert "escape </details><!--" not in details
    assert "…" in details


def test_git_log_limits_large_release_history(tmp_path: Path) -> None:
    commits = _create_history(tmp_path)
    tip = commits["hotfix"]
    for index in range(101):
        tip = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            f"VALUE = {index}\n",
            f"fix(example): generated {index}",
        )

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=tip,
    )

    assert details.count(f"https://github.com/{REPOSITORY}/commit/") == 100
    assert (
        "The log is truncated to the newest 100 commits to keep the release "
        "notes a reasonable size."
    ) in details
    assert "fix(example): generated 100" in details
    assert "fix(example): generated 0\n" not in details


def test_git_log_applies_byte_limit_after_html_escaping(tmp_path: Path) -> None:
    commits = _create_history(tmp_path)
    tip = commits["hotfix"]
    for index in range(30):
        tip = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            f"VALUE = {index}\n",
            f"fix(example): {index} {'&' * 250}",
        )

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=tip,
    )

    assert len(details.encode()) < 26_000
    assert details.count(f"https://github.com/{REPOSITORY}/commit/") < 30
    assert "The log is truncated to the newest" in details


def test_initial_release_git_log_includes_package_root_commit(tmp_path: Path) -> None:
    commits = _create_history(tmp_path)

    details = _run_git_log_step(
        tmp_path,
        previous_tag="",
        release_sha=commits["release"],
    )

    assert "<summary>Git log for initial release</summary>" in details
    assert _entry(commits["baseline"], "feat(example): initial package") in details
    assert _entry(commits["release"], "release(example): 1.0.1") in details


def test_manual_release_includes_release_sha_tip(tmp_path: Path) -> None:
    commits = _create_history(tmp_path)

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=commits["fix"],
    )

    assert _entry(commits["fix"], "fix(example): correct feature") in details


def test_git_log_reports_no_commits_for_empty_range(tmp_path: Path) -> None:
    commits = _create_history(tmp_path)
    # A tag on the release tip yields an empty PREV_TAG..RELEASE_SHA range.
    _git(tmp_path, "tag", "example==2.0.0", commits["hotfix"])

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==2.0.0",
        release_sha=commits["hotfix"],
    )

    assert "No commits found." in details
    assert "Git log unavailable" not in details


def test_git_log_reports_failure_for_unresolvable_range(tmp_path: Path) -> None:
    _create_history(tmp_path)

    # An unresolvable previous tag makes `git log` fail. The step must surface the
    # failure with a distinct message and still exit 0 (a swallowed error would be
    # indistinguishable from a genuinely empty history).
    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==9.9.9",
        release_sha=_git(tmp_path, "rev-parse", "HEAD"),
    )

    assert "Git log unavailable" in details
    assert "No commits found." not in details


def test_git_log_truncates_before_escaping_html(tmp_path: Path) -> None:
    _create_history(tmp_path)
    # Place a literal ampersand at index 199 so it is the last character kept by
    # the 200-char truncation. Truncating before escaping keeps the entity intact
    # ("&amp;"); escaping first would shift it past the limit and cut it mid-entity
    # (leaving a bare "&" immediately before the ellipsis).
    subject = "x" * 199 + "&" + "x" * 60
    tip = _commit(tmp_path, PACKAGE_PATH / "module.py", "ORDER = 1\n", subject)

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=tip,
    )

    assert "&amp;…" in details
    assert "&…" not in details


def test_finalize_release_body_drops_git_log_when_no_room(tmp_path: Path) -> None:
    # Base body so large that neither the full log nor the compact notice fits;
    # the step must fall back to the bare base body and stay under the byte cap.
    base = "x" * 119_990
    body = _run_finalize_release_body_step(
        tmp_path,
        base_release_body=base,
        git_log_details="y" * 1_000,
    ).release_body

    assert body == base
    assert len(body.encode()) <= 120_000
    assert "Git log omitted because" not in body


def test_finalize_release_body_flags_oversized_base_body(tmp_path: Path) -> None:
    # When the base body alone exceeds the budget, nothing here can shrink it: the
    # step must forward it unchanged, append no Git log, and warn about the *base
    # body* rather than misattributing the size to the Git log.
    base = "x" * 120_001
    result = _run_finalize_release_body_step(
        tmp_path,
        base_release_body=base,
        git_log_details="<details>Git log</details>",
    )

    assert result.release_body == base
    assert "<details>Git log</details>" not in result.release_body
    assert "Base release body" in result.stdout
    assert "Full Git log omitted" not in result.stdout


def test_prerelease_resolve_refs_uses_base_version_predecessor(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 1\n", "feat(example): v1")
    _git(tmp_path, "tag", "example==1.0.1")
    head = _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 2\n", "feat(example): v2")

    # Every PEP 440 pre-release form must strip to the same base version, be
    # detected as a pre-release, and use HEAD as the release commit.
    for version in ("1.0.1a1", "1.0.1rc1", "1.0.1b2", "1.0.1.dev3", "1.0.1-rc.1"):
        outputs = _run_resolve_refs_step(tmp_path, version=version).outputs
        assert outputs["is-prerelease"] == "true", version
        assert outputs["prev-tag"] == "example==1.0.1", version
        assert outputs["release-commit"] == head, version


def test_latest_stable_tag_selects_highest_reachable_version(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 0\n", "feat(example): base")
    _git(tmp_path, "tag", "example==1.0.0")
    _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 10\n", "fix(example): ten")
    _git(tmp_path, "tag", "example==1.0.10")
    # Tag 1.0.9 last so neither creation order nor lexical order matches version
    # order; only a correct `--sort=-version:refname` picks 1.0.10 as the highest
    # (a lexical sort would rank "1.0.9" above "1.0.10").
    _commit(tmp_path, PACKAGE_PATH / "module.py", "V = 9\n", "fix(example): nine")
    _git(tmp_path, "tag", "example==1.0.9")
    # 2.0.0 is X.Y.0, so there is no computed predecessor tag and resolve-refs
    # must fall back to the highest reachable stable tag.
    _commit(
        tmp_path,
        PACKAGE_PATH / "CHANGELOG.md",
        "## 2.0.0\n",
        "release(example): 2.0.0",
    )

    result = _run_resolve_refs_step(tmp_path, version="2.0.0")

    assert result.outputs["prev-tag"] == "example==1.0.10"
    assert "No prior stable tag reachable" not in result.stdout


@pytest.mark.parametrize("version", ["1.0.1", "1.1.0"])
def test_resolve_refs_warns_when_stable_tag_exists_but_is_unreachable(
    tmp_path: Path,
    version: str,
) -> None:
    # The only stable tag lives on a divergent branch, so it is not reachable from
    # the release commit on main. This is anomalous for both a patch (1.0.1) and a
    # minor bump (1.1.0) — the warning must fire in both cases, not only when the
    # patch component is non-zero.
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
    _git(tmp_path, "checkout", "-b", "divergent")
    _commit(tmp_path, PACKAGE_PATH / "module.py", "OLD = 1\n", "fix(example): old")
    _git(tmp_path, "tag", "example==1.0.0")
    _git(tmp_path, "checkout", "main")
    _commit(tmp_path, PACKAGE_PATH / "module.py", "CUR = 1\n", "fix(example): current")
    _commit(
        tmp_path,
        PACKAGE_PATH / "CHANGELOG.md",
        f"## {version}\n",
        f"release(example): {version}",
    )

    result = _run_resolve_refs_step(tmp_path, version=version)

    assert result.outputs["prev-tag"] == ""
    assert "No prior stable tag reachable" in result.stdout


def test_resolve_refs_initial_release_has_no_predecessor_or_warning(
    tmp_path: Path,
) -> None:
    # A genuine first release has no prior stable tag at all, so the empty
    # predecessor is expected and the anomaly warning must stay silent.
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
    _commit(
        tmp_path,
        PACKAGE_PATH / "CHANGELOG.md",
        "## 1.0.0\n",
        "release(example): 1.0.0",
    )

    result = _run_resolve_refs_step(tmp_path, version="1.0.0")

    assert result.outputs["prev-tag"] == ""
    assert "No prior stable tag reachable" not in result.stdout


def test_git_log_includes_exactly_max_commits_without_truncation(
    tmp_path: Path,
) -> None:
    # Exactly MAX_COMMITS (100) in-range package commits must yield 100 entries and
    # no truncation banner — the boundary where the `--max-count=N+1` fetch and the
    # `-ge`/`-gt` checks could hide an off-by-one.
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 0\n", "feat(example): base")
    _git(tmp_path, "tag", "example==1.0.0")
    tip = ""
    for index in range(100):
        tip = _commit(
            tmp_path,
            PACKAGE_PATH / "module.py",
            f"VALUE = {index}\n",
            f"fix(example): generated {index}",
        )

    details = _run_git_log_step(
        tmp_path,
        previous_tag="example==1.0.0",
        release_sha=tip,
    )

    assert details.count(f"https://github.com/{REPOSITORY}/commit/") == 100
    assert "The log is truncated to the newest" not in details
