"""Build a production-shaped GitHub release body locally.

Reconstructs the same release body that the ``release.yml`` ``release-notes``
job publishes, so a maintainer can rebuild and apply it manually when the CI
job fails or produces an empty body after a successful publish.

The script is intentionally self-contained (no external dependencies) so it
runs in any environment with Python 3.11+ and ``git`` available.

Usage:
    python .github/scripts/release/build_release_notes.py \
        --package deepagents-code \
        --version 0.0.42 \
        --sha <release-sha> \
        --repo langchain-ai/deepagents \
        --out /tmp/release-body.md
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ── Package registry ─────────────────────────────────────────────────────────

PACKAGE_MAP = {
    "deepagents": "libs/deepagents",
    "deepagents-cli": "libs/cli",
    "deepagents-acp": "libs/acp",
    "deepagents-code": "libs/code",
    "deepagents-talon": "libs/talon",
    "deepagents-evals": "libs/evals",
    "langchain-daytona": "libs/partners/daytona",
    "langchain-modal": "libs/partners/modal",
    "langchain-quickjs": "libs/partners/quickjs",
    "langchain-runloop": "libs/partners/runloop",
    "langchain-vercel-sandbox": "libs/partners/vercel",
}

# ── Git helpers ──────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in *repo* and return stripped stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def _git_ok(repo: Path, *args: str) -> bool:
    """Return True when the git command exits 0."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _git_lines(repo: Path, *args: str) -> list[str]:
    """Run a git command and return non-empty lines."""
    out = _git(repo, *args)
    return [line for line in out.splitlines() if line.strip()]


# ── Version / tag logic ─────────────────────────────────────────────────────

_PRE_RE_DASH = re.compile(
    r"^(\d+\.\d+\.\d+)-(dev|a|alpha|b|beta|rc|pre|preview)[.-]?(\d+)$"
)
_PRE_RE_DEV = re.compile(r"^(\d+\.\d+\.\d+)\.dev(\d+)$")
_PRE_RE_ABRC = re.compile(r"^(\d+\.\d+\.\d+)(a|b|rc)(\d+)$")

_RANK = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2, "rc": 3, "pre": 3, "preview": 3}


def _is_prerelease(version: str) -> bool:
    """Detect PEP 440 pre-release versions (mirrors the bash logic)."""
    return (
        "-" in version
        or bool(re.search(r"\d(a|b|rc)\d", version))
        or bool(re.search(r"\.dev\d", version))
    )


def _parse_prerelease(version: str) -> tuple[str, int, int] | None:
    """Parse a pre-release version into (base, rank, serial).

    Returns None when the version does not match a recognized format.
    """
    # .devN form: 2 groups (base, serial)
    m = _PRE_RE_DEV.match(version)
    if m:
        return m.group(1), 0, int(m.group(2))

    # a/b/rc forms: 3 groups (base, kind, serial)
    for pattern in (_PRE_RE_ABRC, _PRE_RE_DASH):
        m = pattern.match(version)
        if m:
            base = m.group(1)
            kind = m.group(2)
            serial = int(m.group(3))
            return base, _RANK[kind], serial

    return None


def _base_version(version: str) -> str:
    """Strip pre-release suffixes to get the base version."""
    base = re.sub(r"-.*$", "", version)
    base = re.sub(r"\.dev\d+$", "", base)
    base = re.sub(r"(a|b|rc)\d+$", "", base)
    return base


def _version_sort_key(tag: str, pkg_name: str) -> tuple[int, ...]:
    """Extract a sortable version tuple from a tag like ``pkg==1.2.3``."""
    version = tag.removeprefix(f"{pkg_name}==")
    parts = []
    for piece in version.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _is_stable_version_tag(tag: str, pkg_name: str) -> bool:
    """Check whether a tag looks like ``pkg==X.Y.Z`` (no pre-release suffix)."""
    version = tag.removeprefix(f"{pkg_name}==")
    return bool(re.match(r"^\d+\.\d+\.\d+$", version))


# ── Predecessor tag resolution ───────────────────────────────────────────────


def _valid_previous_tag(repo: Path, tag: str, pkg_name: str, release_sha: str) -> bool:
    """Check whether *tag* is a usable predecessor for the release."""
    if not tag or tag == f"{pkg_name}==0.0.0":
        return False
    return (
        _git_ok(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}")
        and _git_ok(repo, "merge-base", "--is-ancestor", tag, release_sha)
    )


def _latest_stable_tag(repo: Path, pkg_name: str, release_sha: str, exclude: str) -> str:
    """Return the highest stable tag reachable from *release_sha*."""
    tags = _git_lines(
        repo,
        "tag",
        "--merged",
        release_sha,
        "--sort=-version:refname",
        "--list",
        f"{pkg_name}==*",
    )
    for tag in tags:
        if tag == exclude:
            continue
        if _is_stable_version_tag(tag, pkg_name):
            return tag
    return ""


def _latest_earlier_prerelease_tag(
    repo: Path,
    pkg_name: str,
    version: str,
    release_sha: str,
) -> str:
    """Find the latest earlier same-base pre-release tag."""
    parsed = _parse_prerelease(version)
    if parsed is None:
        return ""
    current_base, current_rank, current_serial = parsed

    tags_output = subprocess.run(
        ["git", "-C", str(repo), "tag", "--list", f"{pkg_name}==*"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tags_output.returncode != 0:
        return ""

    best_rank = -1
    best_serial = -1
    best_tag = ""

    for tag in tags_output.stdout.splitlines():
        tag = tag.strip()
        if not tag or tag == f"{pkg_name}=={version}":
            continue
        tag_version = tag.removeprefix(f"{pkg_name}==")
        candidate = _parse_prerelease(tag_version)
        if candidate is None:
            continue
        candidate_base, candidate_rank, candidate_serial = candidate

        if (
            candidate_base != current_base
            or candidate_rank > current_rank
            or (candidate_rank == current_rank and candidate_serial >= current_serial)
        ):
            continue

        if not _git_ok(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}"):
            continue
        if not _git_ok(repo, "merge-base", tag, release_sha):
            continue

        # Skip future siblings (tags that descend from the release commit).
        ahead = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", release_sha, tag],
            capture_output=True,
            check=False,
        )
        if ahead.returncode != 1:
            continue

        if (candidate_rank > best_rank) or (
            candidate_rank == best_rank and candidate_serial > best_serial
        ):
            best_tag = tag
            best_rank = candidate_rank
            best_serial = candidate_serial

    return best_tag


def resolve_previous_tag(
    repo: Path,
    pkg_name: str,
    version: str,
    release_sha: str,
    *,
    is_prerelease: bool,
) -> str:
    """Resolve the predecessor tag for the git log range.

    Mirrors the three-tier fallback from ``release.yml``:
    pre-releases try (1) latest earlier sibling, (2) base-version tag,
    (3) latest stable. Stable releases try (1) previous patch, (2) latest stable.
    """
    if is_prerelease:
        base = _base_version(version)
        prev = _latest_earlier_prerelease_tag(repo, pkg_name, version, release_sha)
        if prev:
            return prev
        candidate = f"{pkg_name}=={base}"
        if _valid_previous_tag(repo, candidate, pkg_name, release_sha):
            return candidate
        return _latest_stable_tag(repo, pkg_name, release_sha, f"{pkg_name}=={version}")

    # Stable release: try previous patch.
    parts = version.split(".")
    patch = int(parts[-1])
    if patch == 0:
        prev_tag = ""
    else:
        prev_version = ".".join(parts[:-1]) + f".{patch - 1}"
        prev_tag = f"{pkg_name}=={prev_version}"
    if _valid_previous_tag(repo, prev_tag, pkg_name, release_sha):
        return prev_tag
    return _latest_stable_tag(repo, pkg_name, release_sha, f"{pkg_name}=={version}")


# ── Release commit resolution ───────────────────────────────────────────────


def resolve_release_commit(
    repo: Path,
    working_dir: str,
    *,
    is_prerelease: bool,
) -> str:
    """Resolve the commit to attribute the release to.

    For stable releases, uses the last CHANGELOG.md touch (the release-please
    commit). For pre-releases, uses HEAD.
    """
    if is_prerelease:
        return _git(repo, "rev-parse", "HEAD")
    commit = _git(
        repo, "log", "-1", "--format=%H", "--", f"{working_dir}/CHANGELOG.md",
        check=False,
    )
    if not commit:
        commit = _git(repo, "rev-parse", "HEAD")
    return commit


# ── Changelog extraction ────────────────────────────────────────────────────


def extract_changelog_section(changelog_path: Path, version: str) -> str:
    """Extract the section for *version* from a CHANGELOG.md file.

    Mirrors the awk logic from ``release.yml``: matches ``## [X.Y.Z]`` or
    ``## X.Y.Z`` headers and extracts the body between the matching header
    and the next version header.
    """
    if not changelog_path.is_file():
        return ""

    lines = changelog_path.read_text().splitlines(keepends=False)
    body_lines: list[str] = []
    printing = False

    for line in lines:
        if re.match(r"^## \[?\d+\.\d+\.\d+", line):
            if printing:
                break
            if version in line:
                printing = True
                continue
        elif printing:
            body_lines.append(line)

    return "\n".join(body_lines).strip()


# ── Git log generation ───────────────────────────────────────────────────────

MAX_COMMITS = 100
MAX_GIT_LOG_BYTES = 25000
MAX_SUBJECT_LENGTH = 200


def generate_git_log(
    repo: Path,
    working_dir: str,
    prev_tag: str,
    release_sha: str,
    repository: str,
) -> str:
    """Generate the collapsible package-scoped git log section.

    Returns the ``<details>`` block as a string.
    """
    if prev_tag:
        range_spec = f"{prev_tag}..{release_sha}"
        summary = f"Git log since {prev_tag}"
    else:
        range_spec = release_sha
        summary = "Git log for initial release"

    commits_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "--date-order",
            f"--max-count={MAX_COMMITS + 1}",
            "--format=%H",
            range_spec,
            "--",
            working_dir,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    git_log_failed = commits_result.returncode != 0
    commits = [
        line.strip()
        for line in commits_result.stdout.splitlines()
        if line.strip()
    ] if not git_log_failed else []

    entries: list[str] = []
    commit_count = 0
    git_log_bytes = 0
    truncated = False

    for sha in commits:
        subject = _git(repo, "show", "-s", "--format=%s", sha)

        if commit_count >= MAX_COMMITS:
            truncated = True
            break
        if len(subject) > MAX_SUBJECT_LENGTH:
            subject = subject[:MAX_SUBJECT_LENGTH] + "…"
        subject = html.escape(subject, quote=False)

        short_sha = _git(repo, "rev-parse", "--short=7", sha)
        entry = (
            f"- [`{short_sha}`](https://github.com/{repository}/commit/{sha})"
            f" {subject}"
        )
        entry_bytes = len(entry.encode())
        separator_bytes = 1 if entries else 0

        if git_log_bytes + separator_bytes + entry_bytes > MAX_GIT_LOG_BYTES:
            truncated = True
            break

        entries.append(entry)
        git_log_bytes += separator_bytes + entry_bytes
        commit_count += 1

    if git_log_failed:
        log_text = f"Git log unavailable: `git log` failed for range `{range_spec}`."
    elif not entries:
        log_text = "No commits found."
    else:
        log_text = "\n".join(entries)

    description = (
        "This commit history includes changes to this package. "
        "Commits are listed newest first."
    )
    if truncated:
        description += (
            f" The log is truncated to the newest {commit_count} commits"
            " to keep the release notes a reasonable size."
        )

    return (
        f"<details>\n"
        f"<summary>{summary}</summary>\n\n"
        f"{description}\n\n"
        f"{log_text}\n\n"
        f"</details>"
    )


# ── Contributor collection ───────────────────────────────────────────────────


def _gh_api(endpoint: str, *, repo: str = "", jq: str = "") -> str:
    """Run ``gh api`` and return stdout, or empty string on failure."""
    cmd = ["gh", "api", endpoint]
    if jq:
        cmd.extend(["--jq", jq])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _gh_pr_view(pr_num: str, repo: str, fields: str) -> dict | None:
    """Run ``gh pr view`` and return parsed JSON, or None on failure."""
    result = subprocess.run(
        [
            "gh", "pr", "view", pr_num,
            "--repo", repo,
            "--json", fields,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def collect_contributors(
    repo: Path,
    working_dir: str,
    release_commit: str,
    prev_tag: str,
    repository: str,
    *,
    offline: bool = False,
) -> tuple[list[dict[str, str]], list[str]]:
    """Collect community and internal contributors from merged PRs.

    Returns (community_contributors, internal_maintainers) where each
    community contributor is a dict with keys ``login``, ``twitter``,
    ``linkedin``, and each internal maintainer is a login string.
    """
    if offline:
        return [], []

    # Get commits in range for this package.
    if prev_tag:
        range_spec = f"{prev_tag}..{release_commit}"
    else:
        range_spec = release_commit
    commits_output = subprocess.run(
        ["git", "-C", str(repo), "rev-list", range_spec, "--", working_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    if commits_output.returncode != 0:
        return [], []
    shas = [line.strip() for line in commits_output.stdout.splitlines() if line.strip()][
        :100
    ]

    twitter_handles: dict[str, str] = {}
    linkedin_urls: dict[str, str] = {}
    internal_users: set[str] = set()
    seen_prs: set[str] = set()

    for sha in shas:
        pr_num = _gh_api(
            f"/repos/{repository}/commits/{sha}/pulls",
            jq=". [0].number // empty",
        )
        if not pr_num or pr_num in seen_prs:
            continue
        seen_prs.add(pr_num)

        pr_data = _gh_pr_view(pr_num, repository, "author,body,labels")
        if not pr_data:
            continue

        gh_user = (pr_data.get("author") or {}).get("login", "")
        pr_body = pr_data.get("body") or ""

        if (pr_data.get("author") or {}).get("is_bot", False):
            continue

        labels = [label.get("name", "") for label in (pr_data.get("labels") or [])]
        is_internal = "internal" in labels

        if is_internal:
            if gh_user:
                internal_users.add(gh_user)
            continue

        if gh_user:
            twitter = ""
            twitter_match = re.search(
                r"^\s*Twitter:\s*@?([a-zA-Z0-9_]+)", pr_body, re.IGNORECASE | re.MULTILINE
            )
            if twitter_match:
                twitter = twitter_match.group(1)

            linkedin = ""
            linkedin_match = re.search(
                r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?",
                pr_body,
            )
            if linkedin_match:
                linkedin = linkedin_match.group(0)

            if gh_user not in twitter_handles:
                twitter_handles[gh_user] = twitter
                linkedin_urls[gh_user] = linkedin
            else:
                if twitter and not twitter_handles[gh_user]:
                    twitter_handles[gh_user] = twitter
                if linkedin and not linkedin_urls[gh_user]:
                    linkedin_urls[gh_user] = linkedin

    # Dedup: internal users are removed from community list.
    for user in internal_users:
        twitter_handles.pop(user, None)
        linkedin_urls.pop(user, None)

    community = []
    for user in twitter_handles:
        community.append(
            {
                "login": user,
                "twitter": twitter_handles[user],
                "linkedin": linkedin_urls.get(user, ""),
            }
        )

    return community, sorted(internal_users)


def resolve_releaser(
    repository: str,
    release_commit: str,
    actor: str,
    *,
    offline: bool = False,
) -> str:
    """Resolve who shipped the release. Mirrors the bash logic."""
    if offline:
        return ""

    releaser = ""
    if release_commit:
        pr_num = _gh_api(
            f"/repos/{repository}/commits/{release_commit}/pulls",
            jq=". [0].number // empty",
        )
        if pr_num:
            result = subprocess.run(
                [
                    "gh", "pr", "view", pr_num,
                    "--repo", repository,
                    "--json", "mergedBy",
                    "--jq", ".mergedBy.login // empty",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                releaser = result.stdout.strip()

    if not releaser:
        releaser = actor

    # Drop bot accounts.
    if not releaser or releaser.endswith("[bot]") or releaser == "github-actions":
        releaser = ""

    return releaser


# ── Body assembly ────────────────────────────────────────────────────────────

MAX_RELEASE_BODY_BYTES = 120000


def _build_contributor_entry(contributor: dict[str, str]) -> str:
    """Format a single contributor entry."""
    login = contributor["login"]
    twitter = contributor.get("twitter", "")
    linkedin = contributor.get("linkedin", "")

    socials = ""
    if twitter:
        socials = f"[Twitter](https://x.com/{twitter})"
    if linkedin:
        if not linkedin.startswith(("http://", "https://")):
            linkedin = f"https://{linkedin}"
        if socials:
            socials += f", [LinkedIn]({linkedin})"
        else:
            socials = f"[LinkedIn]({linkedin})"

    if socials:
        return f"@{login} ({socials})"
    return f"@{login}"


def build_base_body(
    *,
    pkg_name: str,
    version: str,
    changelog_body: str,
    is_prerelease: bool,
    community: list[dict[str, str]],
    internal: list[str],
    releaser: str,
    base_branch: str,
    default_branch: str,
    release_commit: str,
    repository: str,
) -> str:
    """Assemble the base release body (before git log is appended)."""
    body = changelog_body

    # Prepend pre-release banner.
    if is_prerelease:
        banner = (
            f"> [!WARNING]\n"
            f"> This is a pre-release version. Install with:"
            f" `pip install {pkg_name}=={version}`\n\n"
        )
        body = banner + body

    # Append contributor shoutouts.
    separator_added = False
    if community:
        entries = ", ".join(_build_contributor_entry(c) for c in community)
        body += f"\n\n---\n\nThanks to our community contributors: {entries}"
        separator_added = True

    if internal:
        if not separator_added:
            body += "\n\n---"
            separator_added = True
        entries = ", ".join(f"@{u}" for u in internal)
        body += f"\n\nInternal maintainers: {entries}"

    if releaser:
        if not separator_added:
            body += "\n\n---"
            separator_added = True
        body += f"\n\nReleased by: @{releaser}"

    # Annotate non-default-branch releases.
    if base_branch and base_branch != default_branch:
        short_commit = release_commit[:7]
        body += (
            f"\n\nReleased from `{base_branch}` at commit"
            f" [`{short_commit}`](https://github.com/{repository}/commit/{release_commit})"
        )

    return body


def finalize_body(base_body: str, git_log_details: str) -> tuple[str, str | None]:
    """Append the git log details block, respecting the size budget.

    Returns (final_body, warning_or_none).
    """
    body_bytes = len(base_body.encode())
    details_bytes = len(git_log_details.encode())
    separator_bytes = 2 if base_body else 0

    if body_bytes > MAX_RELEASE_BODY_BYTES:
        return base_body, (
            f"Base release body ({body_bytes} bytes) already exceeds"
            f" the {MAX_RELEASE_BODY_BYTES}-byte budget before the Git log is"
            " added; GitHub may reject it"
        )

    if body_bytes + details_bytes + separator_bytes <= MAX_RELEASE_BODY_BYTES:
        if base_body:
            return f"{base_body}\n\n{git_log_details}", None
        return git_log_details, None

    # Try compact form.
    compact = (
        "<details>\n"
        "<summary>Git log</summary>\n\n"
        "Git log omitted because the rest of these release notes is near"
        " GitHub's size limit.\n\n"
        "</details>"
    )
    compact_bytes = len(compact.encode())

    if body_bytes + compact_bytes + separator_bytes <= MAX_RELEASE_BODY_BYTES:
        if base_body:
            return f"{base_body}\n\n{compact}", None
        return compact, None

    return base_body, None


# ── Main entry point ────────────────────────────────────────────────────────


def build_release_notes(
    repo_root: Path,
    *,
    package: str,
    version: str,
    release_sha: str,
    repository: str,
    offline: bool = False,
    actor: str = "",
    base_branch: str = "",
    default_branch: str = "main",
    working_dir: str = "",
) -> tuple[str, list[str]]:
    """Build a production-shaped GitHub release body.

    Returns (body, warnings).
    """
    warnings: list[str] = []

    if not working_dir:
        working_dir = PACKAGE_MAP.get(package, "")
        if not working_dir:
            valid = ", ".join(sorted(PACKAGE_MAP))
            msg = f"Unknown package '{package}'. Valid packages: {valid}"
            raise ValueError(msg)

    is_pre = _is_prerelease(version)

    # Resolve predecessor tag.
    prev_tag = resolve_previous_tag(
        repo_root, package, version, release_sha, is_prerelease=is_pre,
    )

    # Resolve release commit.
    release_commit = resolve_release_commit(
        repo_root, working_dir, is_prerelease=is_pre,
    )

    # Extract changelog section.
    changelog_path = repo_root / working_dir / "CHANGELOG.md"
    changelog_body = extract_changelog_section(changelog_path, version)

    # Generate git log.
    git_log_details = generate_git_log(
        repo_root, working_dir, prev_tag, release_sha, repository,
    )

    # Collect contributors (skip in offline mode).
    if offline:
        warnings.append(
            "Offline mode: skipping contributor and releaser lookups."
        )
        community: list[dict[str, str]] = []
        internal: list[str] = []
        releaser = ""
    else:
        community, internal = collect_contributors(
            repo_root,
            working_dir,
            release_commit,
            prev_tag,
            repository,
        )
        releaser = resolve_releaser(repository, release_commit, actor)

    # Build base body.
    base_body = build_base_body(
        pkg_name=package,
        version=version,
        changelog_body=changelog_body,
        is_prerelease=is_pre,
        community=community,
        internal=internal,
        releaser=releaser,
        base_branch=base_branch,
        default_branch=default_branch,
        release_commit=release_commit,
        repository=repository,
    )

    # Finalize with git log.
    body, size_warning = finalize_body(base_body, git_log_details)
    if size_warning:
        warnings.append(size_warning)

    return body, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a production-shaped GitHub release body locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --package deepagents-code --version 0.0.42"
            " --sha abc123 --repo langchain-ai/deepagents\n"
            "  %(prog)s --package deepagents --version 0.7.0"
            " --sha def456 --offline --out /tmp/body.md\n"
        ),
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Package name (e.g. deepagents-code)",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version string (e.g. 0.0.42)",
    )
    parser.add_argument(
        "--sha",
        required=True,
        help="Release commit SHA",
    )
    parser.add_argument(
        "--repo",
        default="langchain-ai/deepagents",
        help="GitHub repository (default: langchain-ai/deepagents)",
    )
    parser.add_argument(
        "--out",
        help="Write body to file instead of stdout",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write release-body to $GITHUB_OUTPUT (for CI use)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip GitHub API calls (contributors, releaser)",
    )
    parser.add_argument(
        "--actor",
        default="",
        help="GitHub username to use as releaser fallback",
    )
    parser.add_argument(
        "--base-branch",
        default="",
        help="Branch the release was cut from (for provenance annotation)",
    )
    parser.add_argument(
        "--default-branch",
        default="main",
        help="Default branch name (default: main)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )

    args = parser.parse_args()

    body, warnings = build_release_notes(
        args.repo_root.resolve(),
        package=args.package,
        version=args.version,
        release_sha=args.sha,
        repository=args.repo,
        offline=args.offline,
        actor=args.actor,
        base_branch=args.base_branch,
        default_branch=args.default_branch,
    )

    for warning in warnings:
        print(f"::warning::{warning}", file=sys.stderr)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if not github_output:
            print("::error::GITHUB_OUTPUT not set", file=sys.stderr)
            sys.exit(1)
        with open(github_output, "a") as f:
            f.write("release-body<<EOF\n")
            f.write(body)
            f.write("\nEOF\n")
    elif args.out:
        Path(args.out).write_text(body)
        print(f"Release body written to {args.out}", file=sys.stderr)
    else:
        print(body)


if __name__ == "__main__":
    main()
