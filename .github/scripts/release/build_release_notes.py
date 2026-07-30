"""Build a production-shaped GitHub release body locally.

Reconstructs the same release body that the ``release.yml`` ``release-notes``
job publishes, so a maintainer can rebuild and apply it manually when the CI
job fails or produces an empty body after a successful publish.

The script is intentionally self-contained (no external dependencies) so it
runs in any environment with Python 3.11+ and ``git`` available.

The ``gh`` CLI is optional, but a missing ``gh`` is *not* the same as
``--offline``: contributor collection yields nothing either way, but the
releaser still falls back to ``--actor`` (so the body keeps a
``Released by:`` line), whereas ``--offline`` suppresses that line too.

Usage:
    python .github/scripts/release/build_release_notes.py \
        --package deepagents-code \
        --version 0.0.42 \
        --sha <release-sha> \
        --repo langchain-ai/deepagents \
        --out /tmp/release-body.md

Run from the repository root (or pass ``--repo-root``) in a clone with full
history and tags; predecessor-tag resolution needs both.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ── Package registry ─────────────────────────────────────────────────────────

# Mirrors the package -> working-dir `case` statement in release.yml's `parse`
# step. `TestPackageMap` enforces parity on both keys and values; the workflow
# is the source of truth. CI passes --working-dir explicitly, so this map is
# the convenience path for local runs that only supply --package.
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
    """Run a git command in *repo* and return stripped stdout.

    Args:
        repo: Repository root to run in.
        args: Arguments passed to ``git``.
        check: When True (the default), a non-zero exit raises
            ``CalledProcessError`` with git's stderr attached. Callers that
            need to tell "command failed" apart from "empty result" pass
            False and inspect the returncode themselves.

    Raises:
        subprocess.CalledProcessError: If the command fails and *check* is set.

    """
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        # capture_output swallows stderr; re-attach it so the caller sees
        # git's own diagnosis ("malformed object name", "bad revision", ...)
        # instead of a bare exit status.
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout.strip()


def _git_run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the raw result, never raising.

    Used where a failure must stay distinguishable from an empty result — the
    bash original captured output up front for the same reason (a command
    inside `< <(...)` is exempt from `set -e`, so its failure would otherwise
    be swallowed and mistaken for an empty history).
    """
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def _git_ok(repo: Path, *args: str) -> bool:
    """Return True when the git command exits 0."""
    return _git_run(repo, *args).returncode == 0


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

_RANK = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "rc": 3,
    "pre": 3,
    "preview": 3,
}

_STABLE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class PreRelease(NamedTuple):
    """A parsed pre-release version.

    ``rank`` orders the phase (dev < a < b < rc) and ``serial`` orders within
    a phase. Both are compared in :func:`_latest_earlier_prerelease_tag`; the
    named fields keep those comparisons from depending on tuple position.
    """

    base: str
    rank: int
    serial: int


def _is_prerelease(version: str) -> bool:
    """Detect PEP 440 pre-release versions.

    Deliberately identical to the ``check-version`` step in ``release.yml``,
    which drives the GitHub release's ``prerelease:`` flag. The two must agree
    or a release can be flagged as a pre-release while its body omits the
    pre-release banner. CI passes ``--is-prerelease`` so the workflow's answer
    wins outright; this is the fallback for local runs.
    """
    return bool(re.search(r"(a|b|rc|\.dev)\d", version)) or "-" in version


def _parse_prerelease(version: str) -> PreRelease | None:
    """Parse a pre-release version into (base, rank, serial).

    Returns None when the version does not match a recognized format.
    """
    # .devN form: 2 groups (base, serial)
    m = _PRE_RE_DEV.match(version)
    if m:
        return PreRelease(m.group(1), _RANK["dev"], int(m.group(2)))

    # Suffix (1.0.0rc1) and dash (1.0.0-rc.1) forms: 3 groups
    # (base, kind, serial). _PRE_RE_DASH also matches dev/alpha/beta/pre/preview.
    for pattern in (_PRE_RE_ABRC, _PRE_RE_DASH):
        m = pattern.match(version)
        if m:
            kind = m.group(2)
            rank = _RANK.get(kind)
            if rank is None:
                # Only reachable if a pattern alternation gains a kind that
                # _RANK does not list; fail soft rather than KeyError inside
                # the release job.
                return None
            return PreRelease(m.group(1), rank, int(m.group(3)))

    return None


def _base_version(version: str) -> str:
    """Strip pre-release suffixes to get the base version."""
    base = re.sub(r"-.*$", "", version)
    base = re.sub(r"\.dev\d+$", "", base)
    return re.sub(r"(a|b|rc)\d+$", "", base)


def _is_stable_version_tag(tag: str, pkg_name: str) -> bool:
    """Check whether a tag looks like ``pkg==X.Y.Z`` (no pre-release suffix)."""
    return bool(_STABLE_VERSION_RE.match(tag.removeprefix(f"{pkg_name}==")))


# ── Predecessor tag resolution ───────────────────────────────────────────────


def _valid_previous_tag(repo: Path, tag: str, pkg_name: str, release_sha: str) -> bool:
    """Check whether *tag* is a usable predecessor for the release."""
    # `pkg==0.0.0` is release-please's placeholder for "never released", not a
    # real predecessor.
    if not tag or tag == f"{pkg_name}==0.0.0":
        return False
    return _git_ok(
        repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}"
    ) and _git_ok(repo, "merge-base", "--is-ancestor", tag, release_sha)


def _latest_stable_tag(
    repo: Path, pkg_name: str, release_sha: str, exclude: str
) -> str:
    """Return the highest stable tag reachable from *release_sha*.

    Sorting is ``-version:refname`` (not lexical) so ``1.0.10`` outranks
    ``1.0.9``.
    """
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
    warnings: list[str],
) -> str:
    """Find the latest earlier same-base pre-release tag."""
    current = _parse_prerelease(version)
    if current is None:
        warnings.append(
            f"VERSION '{version}' is treated as a pre-release but does not match a"
            " recognized pre-release format; predecessor selection will fall back"
            " to the base-version or latest stable tag"
        )
        return ""

    tags_result = _git_run(repo, "tag", "--list", f"{pkg_name}==*")
    if tags_result.returncode != 0:
        warnings.append(
            "git tag --list failed"
            f" ({tags_result.stderr.strip()}); skipping earlier-prerelease"
            " predecessor detection"
        )
        return ""

    best = PreRelease("", -1, -1)
    best_tag = ""

    for raw_tag in tags_result.stdout.splitlines():
        tag = raw_tag.strip()
        if not tag or tag == f"{pkg_name}=={version}":
            continue
        candidate = _parse_prerelease(tag.removeprefix(f"{pkg_name}=="))
        if candidate is None:
            continue

        # Keep only same-base candidates strictly earlier than *version*. On
        # equal rank, `>=` (not `>`) excludes an equal serial: that is either
        # *version* itself under a different tag spelling, or a same-precedence
        # duplicate — neither is a valid predecessor.
        if (
            candidate.base != current.base
            or candidate.rank > current.rank
            or (candidate.rank == current.rank and candidate.serial >= current.serial)
        ):
            continue

        # Skip tags that don't resolve, or that share no history with the
        # release commit (disjoint/grafted history — nearly always false
        # within one repo, but cheap insurance).
        tag_ref = f"refs/tags/{tag}^{{commit}}"
        if not _git_ok(repo, "rev-parse", "-q", "--verify", tag_ref):
            continue
        if not _git_ok(repo, "merge-base", tag, release_sha):
            continue

        # Skip "future" siblings: a tag that descends from RELEASE_SHA cannot
        # be a predecessor. --is-ancestor exits 0 (ancestor -> skip), 1
        # (not -> keep), or 128 (error); treat any non-1 as skip so a git error
        # never silently accepts a bad candidate. Warn on the error case so it
        # is not silently bucketed with the expected future-sibling skip.
        ahead_status = _git_run(
            repo, "merge-base", "--is-ancestor", release_sha, tag
        ).returncode
        if ahead_status != 1:
            if ahead_status != 0:
                warnings.append(
                    f"git merge-base --is-ancestor exited {ahead_status} for '{tag}';"
                    " skipping as a predecessor candidate"
                )
            continue

        # Track the latest earlier sibling: highest rank, then highest serial.
        if (candidate.rank > best.rank) or (
            candidate.rank == best.rank and candidate.serial > best.serial
        ):
            best_tag = tag
            best = candidate

    return best_tag


def resolve_previous_tag(
    repo: Path,
    pkg_name: str,
    version: str,
    release_sha: str,
    *,
    is_prerelease: bool,
    warnings: list[str] | None = None,
) -> str:
    """Resolve the predecessor tag for the git log range.

    Mirrors the three-tier fallback from ``release.yml``: pre-releases try
    (1) the latest earlier sibling, (2) the base-version tag, (3) the latest
    stable. Stable releases try (1) the previous patch — ``X.Y.0`` has none, so
    that tier is skipped — then (2) the latest stable.

    Returns ``""`` when no predecessor is reachable. That is expected for a
    genuine initial release; when prior stable tags exist but none are
    reachable, a warning is appended because the git log will then wrongly
    span the package's full history and be labeled an initial release.
    """
    if warnings is None:
        warnings = []

    if is_prerelease:
        base = _base_version(version)
        if not _STABLE_VERSION_RE.match(base):
            warnings.append(
                f"Base version '{base}' is not a clean X.Y.Z after stripping"
                f" pre-release suffixes from '{version}'"
            )
        prev = _latest_earlier_prerelease_tag(
            repo, pkg_name, version, release_sha, warnings
        )
        if prev:
            return prev
        candidate = f"{pkg_name}=={base}"
        if _valid_previous_tag(repo, candidate, pkg_name, release_sha):
            return candidate
        return _latest_stable_tag(repo, pkg_name, release_sha, f"{pkg_name}=={version}")

    # Stable release: try previous patch.
    parts = version.split(".")
    try:
        patch = int(parts[-1])
    except ValueError:
        warnings.append(
            f"Version '{version}' does not end in a numeric patch component;"
            " falling back to the latest stable tag"
        )
        patch = 0
    if patch == 0:
        prev_tag = ""
    else:
        prev_tag = f"{pkg_name}==" + ".".join([*parts[:-1], str(patch - 1)])
    if _valid_previous_tag(repo, prev_tag, pkg_name, release_sha):
        return prev_tag

    resolved = _latest_stable_tag(repo, pkg_name, release_sha, f"{pkg_name}=={version}")
    if not resolved:
        # A stable release with no reachable predecessor is anomalous unless
        # this is genuinely the package's first release. Distinguish the two by
        # whether any prior stable tag exists at all: none means a true initial
        # release (the log spanning full history is expected); some, but none
        # reachable from the release commit, means the log would wrongly span
        # the full history and be labeled an initial release — surface that.
        # Covers minor/major bumps (X.Y.0), not just patches.
        all_tags = _git_run(repo, "tag", "--list", f"{pkg_name}==*")
        prior_stable = [
            tag
            for tag in (line.strip() for line in all_tags.stdout.splitlines())
            if tag
            and tag != f"{pkg_name}=={version}"
            and _is_stable_version_tag(tag, pkg_name)
        ]
        if prior_stable:
            warnings.append(
                f"No prior stable tag reachable from {release_sha} for"
                f" {pkg_name}=={version}; Git log will span the package's full"
                " history and be labeled an initial release"
            )
    return resolved


# ── Release commit resolution ───────────────────────────────────────────────


def resolve_release_commit(
    repo: Path,
    working_dir: str,
    *,
    is_prerelease: bool,
    warnings: list[str] | None = None,
) -> str:
    """Resolve the commit to attribute the release to.

    For stable releases, uses the last ``CHANGELOG.md`` touch — release-please
    updates the changelog in the release commit. On ``workflow_dispatch``
    (manual/recovery) HEAD may be ahead, and using it would attribute
    post-release commits to this release's contributor list. Pre-releases do
    not update ``CHANGELOG.md``, so they use HEAD directly.
    """
    if warnings is None:
        warnings = []

    if is_prerelease:
        return _git(repo, "rev-parse", "HEAD")

    result = _git_run(
        repo, "log", "-1", "--format=%H", "--", f"{working_dir}/CHANGELOG.md"
    )
    if result.returncode != 0:
        warnings.append(
            f"CHANGELOG.md commit lookup failed ({result.stderr.strip()});"
            " falling back to HEAD"
        )
    elif not result.stdout.strip():
        warnings.append(
            f"No commit touches {working_dir}/CHANGELOG.md; falling back to HEAD"
        )
    else:
        return result.stdout.strip()
    return _git(repo, "rev-parse", "HEAD")


# ── Changelog extraction ────────────────────────────────────────────────────

# Captures the header's own version token, stopping at `]`, whitespace, or `)`.
# Matching must be anchored to this token rather than searching the whole line:
# release-please headers embed the *previous* version in the compare URL
# (`## [0.1.49](.../compare/pkg==0.1.48...pkg==0.1.49)`), so a substring test
# for "0.1.48" matches 0.1.49's header and extracts the wrong release's notes.
_CHANGELOG_HEADER_RE = re.compile(r"^## \[?(?P<version>\d+\.\d+\.\d+[^\]\s)]*)\]?")


def extract_changelog_section(
    changelog_path: Path, version: str
) -> tuple[str, str | None]:
    """Extract the section for *version* from a CHANGELOG.md file.

    Ports the awk logic from ``release.yml``, with two deliberate differences:
    the version match is exact rather than a substring test (see
    :data:`_CHANGELOG_HEADER_RE`), and leading/trailing blank lines are
    stripped where the awk original preserved a leading one.

    Returns:
        A ``(body, reason)`` pair. *reason* is None on success and otherwise
        explains why the body is empty, so the caller can distinguish "no
        CHANGELOG.md" from "version not found" from a genuinely empty section.

    """
    if not changelog_path.is_file():
        return "", f"No CHANGELOG.md found at {changelog_path}"

    try:
        lines = changelog_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as err:
        return "", f"Could not read {changelog_path}: {err}"

    body_lines: list[str] = []
    printing = False

    for line in lines:
        header = _CHANGELOG_HEADER_RE.match(line)
        if header:
            if printing:
                break
            if header.group("version") == version:
                printing = True
            continue
        if printing:
            body_lines.append(line)

    if not printing:
        return "", f"Could not find version {version} in {changelog_path}"

    body = "\n".join(body_lines).strip()
    if not body:
        return "", f"Section for version {version} in {changelog_path} is empty"
    return body, None


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
    warnings: list[str] | None = None,
) -> str:
    """Generate the collapsible package-scoped git log section.

    Returns the ``<details>`` block as a string.
    """
    if warnings is None:
        warnings = []

    if prev_tag:
        range_spec = f"{prev_tag}..{release_sha}"
        summary = f"Git log since {prev_tag}"
    else:
        range_spec = release_sha
        summary = "Git log for initial release"

    commits_result = _git_run(
        repo,
        "log",
        "--date-order",
        f"--max-count={MAX_COMMITS + 1}",
        "--format=%H",
        range_spec,
        "--",
        working_dir,
    )
    git_log_failed = commits_result.returncode != 0
    if git_log_failed:
        warnings.append(
            f"git log failed for range '{range_spec}'"
            f" ({commits_result.stderr.strip()}); the Git log section will note"
            " the failure"
        )
        commits: list[str] = []
    else:
        commits = [
            line.strip() for line in commits_result.stdout.splitlines() if line.strip()
        ]

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
            f"- [`{short_sha}`](https://github.com/{repository}/commit/{sha}) {subject}"
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

GH_TIMEOUT_SECONDS = 30
GH_MAX_ATTEMPTS = 3
GH_RETRY_BACKOFF_SECONDS = 2
_RATE_LIMIT_MARKERS = ("rate limit", "secondary rate", "429", "403")


class Contributor(NamedTuple):
    """A community contributor credited in the release notes."""

    login: str
    twitter: str = ""
    linkedin: str = ""


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a ``gh`` command.

    Returns None only when ``gh`` is genuinely absent, so callers can tell
    "no GitHub CLI in this environment" apart from a per-call failure. Timeouts
    and exec errors come back as a synthetic non-zero result instead, and
    rate-limited calls are retried with backoff.
    """
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(GH_MAX_ATTEMPTS):
        try:
            result = subprocess.run(
                ["gh", *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=GH_TIMEOUT_SECONDS,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess(
                args=["gh", *args],
                returncode=124,
                stdout="",
                stderr=f"timed out after {GH_TIMEOUT_SECONDS}s",
            )
        except OSError as err:
            # PermissionError, exec-format error, transient fork failure. Not
            # "gh is missing" — surface it as a failed call so it is counted
            # and reported rather than mistaken for a missing executable.
            result = subprocess.CompletedProcess(
                args=["gh", *args], returncode=126, stdout="", stderr=str(err)
            )

        if result.returncode == 0:
            return result
        last = result

        detail = f"{result.stderr} {result.stdout}".lower()
        if not any(marker in detail for marker in _RATE_LIMIT_MARKERS):
            return result
        if attempt < GH_MAX_ATTEMPTS - 1:
            time.sleep(GH_RETRY_BACKOFF_SECONDS * (2**attempt))

    return last


def _gh_result_detail(result: subprocess.CompletedProcess[str] | None) -> str:
    """Describe a failed ``gh`` invocation."""
    if result is None:
        return "gh executable not found"
    detail = result.stderr.strip() or result.stdout.strip()
    return detail or f"exit status {result.returncode}"


def _gh_api(endpoint: str, *, jq: str = "") -> tuple[str, str | None]:
    """Run ``gh api`` and return its output and any failure detail."""
    cmd = ["api", endpoint]
    if jq:
        cmd.extend(["--jq", jq])
    result = _run_gh(cmd)
    if result is None or result.returncode != 0:
        return "", _gh_result_detail(result)
    return result.stdout.strip(), None


def _gh_pr_view(pr_num: str, repo: str, fields: str) -> tuple[dict | None, str | None]:
    """Run ``gh pr view`` and return parsed JSON plus any failure detail."""
    result = _run_gh(["pr", "view", pr_num, "--repo", repo, "--json", fields])
    if result is None or result.returncode != 0:
        return None, _gh_result_detail(result)
    if not result.stdout.strip():
        return None, "gh returned empty output"
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        return None, f"unparseable gh output: {err}"
    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"
    return parsed, None


_TWITTER_RE = re.compile(
    r"^\s*Twitter:\s*@?([a-zA-Z0-9_]+)", re.IGNORECASE | re.MULTILINE
)
# The URL must appear on a `LinkedIn:` line. Without that gate any
# linkedin.com/in/ link anywhere in the PR body — a thanks, a quoted review, a
# linked issue — is published as the author's own profile.
_LINKEDIN_RE = re.compile(
    r"^\s*LinkedIn:\s*.*?((?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?)",
    re.IGNORECASE | re.MULTILINE,
)


def collect_contributors(
    repo: Path,
    working_dir: str,
    release_commit: str,
    prev_tag: str,
    repository: str,
    *,
    offline: bool = False,
    warnings: list[str] | None = None,
) -> tuple[list[Contributor], list[str]]:
    """Collect community and internal contributors from merged PRs.

    Returns ``(community, internal)`` where *community* is a list of
    :class:`Contributor` and *internal* is a sorted list of logins.
    """
    if warnings is None:
        warnings = []
    if offline:
        return [], []

    range_spec = f"{prev_tag}..{release_commit}" if prev_tag else release_commit
    commits_result = _git_run(repo, "rev-list", range_spec, "--", working_dir)
    if commits_result.returncode != 0:
        warnings.append(
            f"contributor lookup: git rev-list failed for range '{range_spec}'"
            f" ({commits_result.stderr.strip()}); no contributors will be credited"
        )
        return [], []
    shas = [
        line.strip() for line in commits_result.stdout.splitlines() if line.strip()
    ][:MAX_COMMITS]

    contributors: dict[str, Contributor] = {}
    internal_users: set[str] = set()
    seen_prs: set[str] = set()
    failed_lookups = 0

    for sha in shas:
        pr_num, api_error = _gh_api(
            f"/repos/{repository}/commits/{sha}/pulls", jq=".[0].number // empty"
        )
        if api_error:
            failed_lookups += 1
            continue
        if not pr_num or pr_num in seen_prs:
            continue
        seen_prs.add(pr_num)

        pr_data, view_error = _gh_pr_view(pr_num, repository, "author,body,labels")
        if pr_data is None:
            failed_lookups += 1
            warnings.append(
                f"contributor lookup: gh pr view #{pr_num} failed: {view_error}"
            )
            continue

        author = pr_data.get("author")
        if not isinstance(author, dict):
            warnings.append(
                f"contributor lookup: PR #{pr_num} has an unexpected author payload;"
                " contributor will not appear in release notes"
            )
            continue

        gh_user = author.get("login", "")
        # Belt and braces: `is_bot` is authoritative, but fall back to the
        # login suffix so a missing or renamed field cannot silently promote a
        # bot into the community shoutouts.
        if author.get("is_bot", False) or gh_user.endswith("[bot]"):
            continue

        if "labels" not in pr_data:
            warnings.append(
                f"contributor lookup: PR #{pr_num} returned no labels field;"
                " the internal/community split may be wrong"
            )
        labels = [
            label.get("name", "")
            for label in (pr_data.get("labels") or [])
            if isinstance(label, dict)
        ]

        if "internal" in labels:
            if gh_user:
                internal_users.add(gh_user)
            else:
                warnings.append(
                    f"contributor lookup: internal PR #{pr_num} has no author login;"
                    " contributor will not appear in release notes"
                )
            continue

        if not gh_user:
            warnings.append(
                f"contributor lookup: PR #{pr_num} has no author login;"
                " contributor will not appear in release notes"
            )
            continue

        pr_body = pr_data.get("body") or ""
        twitter_match = _TWITTER_RE.search(pr_body)
        twitter = twitter_match.group(1) if twitter_match else ""
        linkedin_match = _LINKEDIN_RE.search(pr_body)
        linkedin = linkedin_match.group(1) if linkedin_match else ""

        existing = contributors.get(gh_user)
        if existing is None:
            contributors[gh_user] = Contributor(gh_user, twitter, linkedin)
        else:
            # Later PRs only fill gaps; the first non-empty value wins.
            contributors[gh_user] = Contributor(
                gh_user,
                existing.twitter or twitter,
                existing.linkedin or linkedin,
            )

    if failed_lookups:
        warnings.append(
            f"contributor lookup: {failed_lookups}/{len(shas)} commits could not be"
            " resolved to a PR; the contributor list is INCOMPLETE"
        )

    # Internal contributors are org members regardless of what labels their
    # other PRs carry, so they are credited only in the internal list.
    for user in internal_users:
        contributors.pop(user, None)

    return list(contributors.values()), sorted(internal_users)


def resolve_releaser(
    repository: str,
    release_commit: str,
    actor: str,
    *,
    offline: bool = False,
    warnings: list[str] | None = None,
) -> str:
    """Resolve who shipped the release.

    Primary source is whoever merged the release PR (the release commit is
    that PR's squash-merge); when the commit has no associated merged PR
    (initial release, direct push, manual recovery) it falls back to *actor*.
    Bot accounts resolve to ``""``. Returns ``""`` in *offline* mode. Lookup
    failures are appended to *warnings* — a real API failure must not silently
    fall through to the actor, which on a manual release would credit the
    dispatcher instead of the merger with no breadcrumb.
    """
    if warnings is None:
        warnings = []
    if offline:
        return ""

    releaser = ""
    if release_commit:
        pr_num, api_error = _gh_api(
            f"/repos/{repository}/commits/{release_commit}/pulls",
            jq=".[0].number // empty",
        )
        if api_error:
            warnings.append(
                "releaser lookup: commit-pulls API failed: "
                f"{api_error} — falling back to actor"
            )
        if pr_num:
            result = _run_gh(
                [
                    "pr",
                    "view",
                    pr_num,
                    "--repo",
                    repository,
                    "--json",
                    "mergedBy",
                    "--jq",
                    ".mergedBy.login // empty",
                ]
            )
            if result is not None and result.returncode == 0:
                releaser = result.stdout.strip()
            else:
                warnings.append(
                    f"releaser lookup: gh pr view #{pr_num} failed: "
                    f"{_gh_result_detail(result)} — falling back to actor"
                )

    if not releaser:
        releaser = actor

    # Drop bot accounts — there is no human to credit. The bare
    # "github-actions" arm covers github.actor surfacing without the "[bot]"
    # suffix, which the endswith check alone would not catch.
    if not releaser or releaser.endswith("[bot]") or releaser == "github-actions":
        return ""

    return releaser


# ── Body assembly ────────────────────────────────────────────────────────────

# Conservative margin beneath GitHub's 125,000-character release-body limit.
# Do not raise this to the real limit: the margin absorbs the difference
# between bytes counted here and characters counted by GitHub.
MAX_RELEASE_BODY_BYTES = 120000

_COMPACT_GIT_LOG = (
    "<details>\n"
    "<summary>Git log</summary>\n\n"
    "Git log omitted because the rest of these release notes is near"
    " GitHub's size limit.\n\n"
    "</details>"
)


def _build_contributor_entry(contributor: Contributor) -> str:
    """Format a single contributor entry."""
    socials = ""
    if contributor.twitter:
        socials = f"[Twitter](https://x.com/{contributor.twitter})"
    if contributor.linkedin:
        linkedin = contributor.linkedin
        if not linkedin.startswith(("http://", "https://")):
            linkedin = f"https://{linkedin}"
        link = f"[LinkedIn]({linkedin})"
        socials = f"{socials}, {link}" if socials else link

    if socials:
        return f"@{contributor.login} ({socials})"
    return f"@{contributor.login}"


def build_base_body(
    *,
    pkg_name: str,
    version: str,
    changelog_body: str,
    is_prerelease: bool,
    community: list[Contributor],
    internal: list[str],
    releaser: str,
    base_branch: str,
    default_branch: str,
    release_commit: str,
    repository: str,
) -> str:
    """Assemble the base release body (before git log is appended)."""
    body = changelog_body

    if is_prerelease:
        body = (
            "> [!WARNING]\n"
            "> This is a pre-release version. Install with:"
            f" `pip install {pkg_name}=={version}`\n\n"
        ) + body

    separator_added = False
    if community:
        entries = ", ".join(_build_contributor_entry(c) for c in community)
        body += f"\n\n---\n\nThanks to our community contributors: {entries}"
        separator_added = True

    if internal:
        if not separator_added:
            body += "\n\n---"
            separator_added = True
        body += f"\n\nInternal maintainers: {', '.join(f'@{u}' for u in internal)}"

    # Credit whoever shipped the release. This is a distinct role from
    # contributing release changes, so show it even when the same user also
    # appears as a community contributor or internal maintainer.
    if releaser:
        if not separator_added:
            body += "\n\n---"
        body += f"\n\nReleased by: @{releaser}"

    # Annotate releases cut from a non-default branch (a vX.Y version line or
    # an alpha/* throwaway branch) with the originating branch and the
    # immutable release commit. Placed after attribution so provenance sits
    # with the ship metadata. Skipped for normal default-branch releases, where
    # the branch carries no signal. The branch is rendered as plain text and
    # the link targets the commit SHA, never tree/<branch>: alpha branches are
    # deleted after release, so a branch link would 404, and the commit is the
    # exact code that shipped even when release-sha points somewhere other than
    # the branch tip.
    default_branch = default_branch or "main"
    if base_branch and base_branch != default_branch:
        body += (
            f"\n\nReleased from `{base_branch}` at commit"
            f" [`{release_commit[:7]}`]"
            f"(https://github.com/{repository}/commit/{release_commit})"
        )

    return body


def finalize_body(base_body: str, git_log_details: str) -> tuple[str, list[str]]:
    """Append the git log details block, respecting the size budget.

    Returns ``(final_body, warnings)``.
    """
    body_bytes = len(base_body.encode())
    details_bytes = len(git_log_details.encode())
    separator_bytes = 2 if base_body else 0

    # The base body alone can exceed the budget (e.g. a very large CHANGELOG
    # section). Nothing here can shrink it, so surface that accurately rather
    # than blaming the Git log, and append nothing.
    if body_bytes > MAX_RELEASE_BODY_BYTES:
        return base_body, [
            f"Base release body ({body_bytes} bytes) already exceeds"
            f" the {MAX_RELEASE_BODY_BYTES}-byte budget before the Git log is"
            " added; GitHub may reject it"
        ]

    if body_bytes + details_bytes + separator_bytes <= MAX_RELEASE_BODY_BYTES:
        if base_body:
            return f"{base_body}\n\n{git_log_details}", []
        return git_log_details, []

    warnings = [
        "Full Git log omitted to keep the release body within GitHub's size limit"
    ]
    compact_bytes = len(_COMPACT_GIT_LOG.encode())
    if body_bytes + compact_bytes + separator_bytes <= MAX_RELEASE_BODY_BYTES:
        if base_body:
            return f"{base_body}\n\n{_COMPACT_GIT_LOG}", warnings
        return _COMPACT_GIT_LOG, warnings

    warnings.append(
        "Git log dropped entirely: even the omission notice does not fit within"
        f" the {MAX_RELEASE_BODY_BYTES}-byte budget"
    )
    return base_body, warnings


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
    is_prerelease: bool | None = None,
) -> tuple[str, list[str]]:
    """Build a production-shaped GitHub release body.

    Args:
        repo_root: Repository root to read git history from.
        package: Package name; used to look up *working_dir* when not given.
        version: Version being released.
        release_sha: Commit the release is cut from.
        repository: ``owner/name`` for building commit links.
        offline: Skip all GitHub API lookups.
        actor: Fallback releaser when no merged PR is found.
        base_branch: Branch the release was cut from.
        default_branch: Default branch; empty is treated as ``main``.
        working_dir: Package directory; derived from *package* when empty.
        is_prerelease: Authoritative pre-release flag. When None, it is
            detected from *version*.

    Returns:
        A ``(body, warnings)`` pair.

    Raises:
        ValueError: If the package is unknown, the package directory does not
            exist, or *release_sha* does not resolve to a commit.

    """
    warnings: list[str] = []

    if not working_dir:
        working_dir = PACKAGE_MAP.get(package, "")
        if not working_dir:
            valid = ", ".join(sorted(PACKAGE_MAP))
            msg = f"Unknown package '{package}'. Valid packages: {valid}"
            raise ValueError(msg)

    # Fail closed on a bad repo root. git log/rev-list exit 0 with empty output
    # for a pathspec that matches nothing, so every downstream step would read
    # the emptiness as a legitimate answer and emit a well-formed "No commits
    # found." body at exit 0.
    package_dir = repo_root / working_dir
    if not package_dir.is_dir():
        msg = (
            f"Package directory '{package_dir}' does not exist. Pass --repo-root"
            f" pointing at the repository root (current: {repo_root})."
        )
        raise ValueError(msg)

    # Fail closed on an unresolvable SHA. It is passed to `merge-base
    # --is-ancestor` and `git tag --merged`, where an unresolvable value exits
    # 128 — indistinguishable from "no predecessor" — and would silently
    # mislabel the Git log as an initial release.
    sha_ref = f"{release_sha}^{{commit}}"
    if not _git_ok(repo_root, "rev-parse", "-q", "--verify", sha_ref):
        msg = f"Release SHA '{release_sha}' does not resolve to a commit in {repo_root}"
        raise ValueError(msg)

    is_pre = _is_prerelease(version) if is_prerelease is None else is_prerelease

    prev_tag = resolve_previous_tag(
        repo_root,
        package,
        version,
        release_sha,
        is_prerelease=is_pre,
        warnings=warnings,
    )
    print(f"Previous tag: {prev_tag or '<none>'}", file=sys.stderr)

    release_commit = resolve_release_commit(
        repo_root, working_dir, is_prerelease=is_pre, warnings=warnings
    )
    print(f"Release commit: {release_commit}", file=sys.stderr)

    changelog_path = package_dir / "CHANGELOG.md"
    changelog_body, changelog_reason = extract_changelog_section(
        changelog_path, version
    )
    if changelog_reason:
        warnings.append(changelog_reason)
    else:
        print(f"Extracted changelog for version {version}", file=sys.stderr)

    git_log_details = generate_git_log(
        repo_root, working_dir, prev_tag, release_sha, repository, warnings
    )

    if offline:
        warnings.append("Offline mode: skipping contributor and releaser lookups.")
        community: list[Contributor] = []
        internal: list[str] = []
        releaser = ""
    else:
        community, internal = collect_contributors(
            repo_root,
            working_dir,
            release_commit,
            prev_tag,
            repository,
            warnings=warnings,
        )
        releaser = resolve_releaser(
            repository, release_commit, actor, warnings=warnings
        )
        if community:
            print(
                "Found community contributors: "
                + ", ".join(c.login for c in community),
                file=sys.stderr,
            )
        if internal:
            print(f"Found internal maintainers: {', '.join(internal)}", file=sys.stderr)
        if releaser:
            print(f"Found releaser: @{releaser}", file=sys.stderr)

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

    body, size_warnings = finalize_body(base_body, git_log_details)
    warnings.extend(size_warnings)

    return body, warnings


def _write_github_output(body: str) -> None:
    """Append the release body to ``$GITHUB_OUTPUT`` as a heredoc.

    Raises:
        SystemExit: If ``GITHUB_OUTPUT`` is unset.

    """
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if not github_output:
        print("::error::GITHUB_OUTPUT not set", file=sys.stderr)
        raise SystemExit(1)
    # Random delimiter per GitHub's guidance: a fixed "EOF" is terminated early
    # by any body line that is exactly "EOF", which truncates the notes and
    # leaks the remainder as stray step outputs.
    delimiter = f"EOF_{secrets.token_hex(16)}"
    with Path(github_output).open("a", encoding="utf-8") as handle:
        handle.write(f"release-body<<{delimiter}\n{body}\n{delimiter}\n")


def _write_out_file(path: str, body: str) -> None:
    """Write the body to *path* atomically, echoing it to stdout on failure.

    Raises:
        SystemExit: If the file cannot be written.

    """
    target = Path(path)
    tmp = target.with_name(f"{target.name}.tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(target)
    except OSError as err:
        # Never lose the body: it cost up to 200 API calls to build, and a
        # truncating rewrite would also have destroyed any previous good copy.
        print(
            f"::error::Could not write {path}: {err}; body follows on stdout",
            file=sys.stderr,
        )
        print(body)
        raise SystemExit(1) from err
    print(f"Release body written to {path}", file=sys.stderr)


def main() -> None:
    """Parse arguments, build the release body, and emit it."""
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
        "--package", required=True, help="Package name (e.g. deepagents-code)"
    )
    parser.add_argument("--version", required=True, help="Version string (e.g. 0.0.42)")
    parser.add_argument("--sha", required=True, help="Release commit SHA")
    parser.add_argument(
        "--repo",
        default="langchain-ai/deepagents",
        help="GitHub repository (default: langchain-ai/deepagents)",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--out", help="Write body to file instead of stdout")
    destination.add_argument(
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
        "--actor", default="", help="GitHub username to use as releaser fallback"
    )
    parser.add_argument(
        "--base-branch",
        default="",
        help="Branch the release was cut from (for provenance annotation)",
    )
    parser.add_argument(
        "--default-branch",
        default="main",
        help="Default branch name; empty is treated as 'main' (default: main)",
    )
    parser.add_argument(
        "--working-dir",
        default="",
        help=(
            "Package directory relative to the repo root. CI passes the value"
            " resolved by the workflow; omit to derive it from --package."
        ),
    )
    parser.add_argument(
        "--is-prerelease",
        # "" is accepted so an unset CI expression degrades to version
        # sniffing rather than failing the step with an argparse error.
        choices=("true", "false", ""),
        default="",
        help=(
            "Authoritative pre-release flag. CI passes the same value that"
            " drives the GitHub release's prerelease flag; omit to detect it"
            " from --version."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory)",
    )

    args = parser.parse_args()

    is_prerelease = args.is_prerelease == "true" if args.is_prerelease else None

    try:
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
            working_dir=args.working_dir,
            is_prerelease=is_prerelease,
        )
    except ValueError as err:
        print(f"::error::{err}", file=sys.stderr)
        raise SystemExit(1) from err

    for warning in warnings:
        print(f"::warning::{warning}", file=sys.stderr)

    if args.github_output:
        _write_github_output(body)
    elif args.out:
        _write_out_file(args.out, body)
    else:
        print(body)


if __name__ == "__main__":
    main()
