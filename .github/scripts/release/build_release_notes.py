"""Build a production-shaped GitHub release body locally.

Reconstructs the same release body that the `release.yml` `release-notes`
job publishes, so a maintainer can rebuild and apply it manually when the CI
job fails or produces an empty body after a successful publish.

The script is intentionally self-contained (no external dependencies) so it
runs in any environment with Python 3.11+ and `git` available.

The `gh` CLI is optional, but a missing `gh` is *not* the same as
`--offline`: contributor collection yields nothing either way, but the
releaser still falls back to `--actor` (so the body keeps a
`Released by:` line), whereas `--offline` suppresses that line too.

Usage:
    python .github/scripts/release/build_release_notes.py \
        --package deepagents-code \
        --version 0.0.42 \
        --sha <release-sha> \
        --repo langchain-ai/deepagents \
        --out /tmp/release-body.md

Run from the repository root (or pass `--repo-root`) in a clone with full
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
        args: Arguments passed to `git`.
        check: When True (the default), a non-zero exit raises
            `CalledProcessError` with git's stderr attached. Callers that
            need to tell "command failed" apart from "empty result" use
            :func:`_git_run` instead, which surfaces the returncode.

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

    Used where a failure must stay distinguishable from an empty result: git
    exits 0 with empty output for a pathspec or range that matches nothing, so
    a caller that only sees the stdout cannot tell "no commits" from "the
    command failed".
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

    `rank` orders the phase (dev < a < b < rc) and `serial` orders within
    a phase. Both are compared in :func:`_latest_earlier_prerelease_tag`; the
    named fields keep those comparisons from depending on tuple position.

    Aliases share a rank, so some spellings tie: `alpha` == `a`,
    `beta` == `b`, and `pre` == `preview` == `rc`. A tie plus an
    equal serial is treated as "not a predecessor" (see the `>=` check in
    :func:`_latest_earlier_prerelease_tag`).
    """

    base: str
    rank: int
    serial: int

    @property
    def precedence(self) -> tuple[int, int]:
        """Order two pre-releases *that share a base*; meaningless across bases.

        Structural tuple comparison would put `base` first and compare it as
        a string, ranking `1.0.10` below `1.0.9`. Callers must compare this
        instead, and only after establishing the bases match.
        """
        return (self.rank, self.serial)


def _is_prerelease(version: str) -> bool:
    """Detect PEP 440 pre-release versions.

    Deliberately identical to the `check-version` step in `release.yml`,
    which drives the GitHub release's `prerelease:` flag. The two must agree
    or a release can be flagged as a pre-release while its body omits the
    pre-release banner. CI passes `--is-prerelease` so the workflow's answer
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
    """Check whether a tag looks like `pkg==X.Y.Z` (no pre-release suffix)."""
    return bool(_STABLE_VERSION_RE.match(tag.removeprefix(f"{pkg_name}==")))


# ── Predecessor tag resolution ───────────────────────────────────────────────


def _valid_previous_tag(repo: Path, tag: str, pkg_name: str, release_sha: str) -> bool:
    """Check whether *tag* is a usable predecessor for the release.

    `_git_ok` collapses `merge-base --is-ancestor`'s exit 1 ("not an
    ancestor") and exit 128 (error) into the same `False`. That is safe
    *here*, unlike in :func:`_latest_earlier_prerelease_tag`, because both refs
    are known to resolve: the tag via the `rev-parse --verify` below, and
    *release_sha* via the check in :func:`build_release_notes`. A non-zero exit
    therefore reflects ancestry rather than a bad ref, and falling back to the
    next predecessor tier is the conservative response.
    """
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

    Sorting is `-version:refname` (not lexical) so `1.0.10` outranks
    `1.0.9`.
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

    best: PreRelease | None = None
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
            or candidate.precedence >= current.precedence
        ):
            continue

        # Skip tags that don't resolve, or that share no history with the
        # release commit (disjoint/grafted history — nearly always false
        # within one repo, but cheap insurance).
        tag_ref = f"refs/tags/{tag}^{{commit}}"
        if not _git_ok(repo, "rev-parse", "-q", "--verify", tag_ref):
            continue
        # Same treatment as the --is-ancestor check below: exit 1 is the
        # expected "no shared history" skip, but any other non-zero is a git
        # error. Dropping a candidate silently widens the log range and
        # misattributes contributors, so name it.
        shared_status = _git_run(repo, "merge-base", tag, release_sha).returncode
        if shared_status != 0:
            if shared_status != 1:
                warnings.append(
                    f"git merge-base exited {shared_status} for '{tag}';"
                    " skipping as a predecessor candidate"
                )
            continue

        # Skip "future" siblings: a tag that descends from release_sha cannot
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
        if best is None or candidate.precedence > best.precedence:
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
    warnings: list[str],
) -> str:
    """Resolve the predecessor tag for the git log range.

    Pre-releases try (1) the latest earlier sibling, (2) the base-version tag,
    (3) the latest stable. Stable releases try (1) the previous patch —
    `X.Y.0` has none, so that tier is skipped — then (2) the latest stable.

    Returns `""` when no predecessor is reachable. That is expected for a
    genuine initial release; when prior stable tags exist but none are
    reachable, a warning is appended because the git log will then wrongly
    span the package's full history and be labeled an initial release. The
    check runs on both the stable and the pre-release path — pre-releases are
    the ones most often cut from throwaway branches and partial clones, so
    they need it most.
    """
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
    else:
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
        prev_tag = (
            ""
            if patch == 0
            else f"{pkg_name}==" + ".".join([*parts[:-1], str(patch - 1)])
        )
        if _valid_previous_tag(repo, prev_tag, pkg_name, release_sha):
            return prev_tag

    # Both paths share the same last tier: the latest stable tag, excluding this
    # release's own.
    resolved = _latest_stable_tag(repo, pkg_name, release_sha, f"{pkg_name}=={version}")

    if not resolved:
        # No reachable predecessor is anomalous unless this is genuinely the
        # package's first release. Distinguish the two by whether any prior
        # stable tag exists at all: none means a true initial release (the log
        # spanning full history is expected); some, but none reachable from the
        # release commit, means the log would wrongly span the full history and
        # be labeled an initial release — surface that. Covers minor/major
        # bumps (X.Y.0), not just patches.
        all_tags = _git_run(repo, "tag", "--list", f"{pkg_name}==*")
        if all_tags.returncode != 0:
            # The enumeration that answers "is this a real initial release?"
            # failed, so the question is unanswered — strictly more alarming
            # than "no prior tags", not less. Warn rather than fall through to
            # the empty-list path, which would silently suppress the warning
            # below and label a full-history log an initial release.
            warnings.append(
                f"git tag --list failed ({all_tags.stderr.strip()}); cannot tell"
                f" whether {pkg_name}=={version} is a genuine initial release."
                " The Git log may wrongly span the package's full history"
            )
            return resolved
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
    release_sha: str,
    *,
    is_prerelease: bool,
    warnings: list[str],
) -> str:
    """Resolve the commit to attribute the release to.

    For stable releases, uses the last `CHANGELOG.md` touch reachable from
    *release_sha* — release-please updates the changelog in the release commit.
    Pre-releases do not update `CHANGELOG.md`, so they use *release_sha*
    itself.

    Every lookup is bounded by *release_sha* rather than `HEAD`. In CI the
    checkout is already at the release SHA, so the two agree; locally they do
    not, and searching from `HEAD` would resolve to a *later* release's
    commit and credit its contributors and merger to this release.

    Appends diagnostics to *warnings*.
    """
    resolved_sha = _git(repo, "rev-parse", f"{release_sha}^{{commit}}")
    if is_prerelease:
        return resolved_sha

    result = _git_run(
        repo,
        "log",
        "-1",
        "--format=%H",
        release_sha,
        "--",
        f"{working_dir}/CHANGELOG.md",
    )
    if result.returncode != 0:
        warnings.append(
            f"CHANGELOG.md commit lookup failed ({result.stderr.strip()});"
            " falling back to the release SHA"
        )
    elif not result.stdout.strip():
        warnings.append(
            f"No commit reachable from {release_sha[:7]} touches"
            f" {working_dir}/CHANGELOG.md; falling back to the release SHA"
        )
    else:
        return result.stdout.strip()
    return resolved_sha


# ── Changelog extraction ────────────────────────────────────────────────────

# Captures the header's own version token, stopping at `]`, whitespace, or `)`.
# Matching must be anchored to this token rather than searching the whole line:
# release-please headers embed the *previous* version in the compare URL
# (`## [0.1.49](.../compare/pkg==0.1.48...pkg==0.1.49)`), so a substring test
# for "0.1.48" matches 0.1.49's header and extracts the wrong release's notes.
_CHANGELOG_HEADER_RE = re.compile(r"^## \[?(?P<version>\d+\.\d+\.\d+[^\]\s)]*)\]?")


def extract_changelog_section_text(
    text: str, version: str, source: str
) -> tuple[str, str | None]:
    """Extract the section for *version* out of CHANGELOG.md *text*.

    The version match is anchored to the header's own token rather than being a
    substring test (see :data:`_CHANGELOG_HEADER_RE`), and leading/trailing
    blank lines are stripped. *source* names the origin of *text* for use in
    the failure reason.

    Returns:
        A `(body, reason)` pair. *reason* is None on success and otherwise
        explains why the body is empty, so the caller can distinguish "no
        CHANGELOG.md" from "version not found" from a genuinely empty section.

    """
    body_lines: list[str] = []
    printing = False

    for line in text.splitlines():
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
        return "", f"Could not find version {version} in {source}"

    body = "\n".join(body_lines).strip()
    if not body:
        return "", f"Section for version {version} in {source} is empty"
    return body, None


def extract_changelog_section(
    changelog_path: Path, version: str
) -> tuple[str, str | None]:
    """Extract the section for *version* from a CHANGELOG.md file on disk."""
    if not changelog_path.is_file():
        return "", f"No CHANGELOG.md found at {changelog_path}"
    try:
        text = changelog_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        return "", f"Could not read {changelog_path}: {err}"
    return extract_changelog_section_text(text, version, str(changelog_path))


def read_changelog_at(
    repo: Path, working_dir: str, release_sha: str, warnings: list[str]
) -> tuple[str, str | None]:
    """Read CHANGELOG.md as it stood at *release_sha*.

    Reading the blob out of the commit rather than the working tree keeps the
    notes consistent with the rest of the body, which is all derived from
    *release_sha*. Falls back to the working tree when the file does not exist
    at that commit, so a local run against a checkout that has the file still
    produces notes.

    Returns:
        A `(text, reason)` pair; *reason* is None when *text* was read.

    """
    rel_path = f"{working_dir}/CHANGELOG.md"
    result = _git_run(repo, "show", f"{release_sha}:{rel_path}")
    if result.returncode == 0:
        return result.stdout, None

    # A non-zero `git show` covers more than "the path is absent at this
    # commit" — a corrupt object, an ambiguous ref, or an I/O error land here
    # too. Carry git's own words instead of asserting a cause we never checked.
    detail = result.stderr.strip()
    at_sha = f"{rel_path} at {release_sha[:7]}" + (f" ({detail})" if detail else "")

    worktree_path = repo / working_dir / "CHANGELOG.md"
    if worktree_path.is_file():
        try:
            text = worktree_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as err:
            # Warn only on success: appending before the read leaves a warning
            # claiming the working tree was "read from" when nothing was.
            return "", f"Could not read {worktree_path}: {err}"
        warnings.append(
            f"Could not read {at_sha}; read from the working tree instead,"
            " which may not match the released content"
        )
        return text, None

    return "", f"Could not read {at_sha}, and no CHANGELOG.md in the working tree"


def changelog_lists_other_versions(text: str, version: str) -> bool:
    """Whether *text* documents a release other than *version*.

    Used as independent evidence that the package has shipped before, so a
    clone with no tags at all can be told apart from a genuine first release.
    """
    return any(
        header.group("version") != version
        for header in (
            _CHANGELOG_HEADER_RE.match(line) for line in text.splitlines()
        )
        if header
    )


# ── Git log generation ───────────────────────────────────────────────────────

MAX_COMMITS = 100
MAX_GIT_LOG_BYTES = 25000
MAX_SUBJECT_LENGTH = 200

# Kept separate from MAX_COMMITS even though the values match: this one bounds
# the GitHub API budget (up to 2 calls per commit), so raising the git-log cap
# must not silently multiply the number of API calls.
MAX_CONTRIBUTOR_COMMITS = 100


def generate_git_log(
    repo: Path,
    working_dir: str,
    prev_tag: str,
    release_sha: str,
    repository: str,
    warnings: list[str],
) -> str:
    """Generate the collapsible package-scoped git log section.

    Appends diagnostics to *warnings*.

    Returns the `<details>` block as a string.
    """
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

    # `--max-count` fetched one commit past the cap so this loop can tell
    # "exactly at the cap" from "there is more".
    for sha in commits:
        if commit_count >= MAX_COMMITS:
            truncated = True
            break

        subject = _git(repo, "show", "-s", "--format=%s", sha)
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
    elif entries:
        log_text = "\n".join(entries)
    elif truncated:
        # The first entry alone blew the byte budget, so there are commits but
        # none are shown. Saying "No commits found." here would be false.
        log_text = "Git log omitted: the first commit entry exceeds the size budget."
    else:
        log_text = "No commits found."

    description = (
        "This commit history includes changes to this package. "
        "Commits are listed newest first."
    )
    if truncated and commit_count:
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

# Retry only on throttling, matched against gh's own wording. Bare status codes
# are deliberately NOT listed: "403" and "429" are valid hex, so they match
# inside any commit SHA echoed in an error message, which would turn a
# permanent failure (missing `pull-requests: read`, a token that cannot read
# the repo) into 3 attempts and 6s of backoff on every one of up to 100
# commits. GitHub reports secondary rate limits as 403, so the phrase matters
# rather than the code.
_RATE_LIMIT_MARKERS = (
    "rate limit",
    "secondary rate",
    "too many requests",
    "retry-after",
)

# Abandon the per-commit contributor walk after this many consecutive failures.
# A permanent auth or connectivity fault fails identically for every commit;
# without this the loop pays the full retry budget 100 times over to learn the
# same thing once.
MAX_CONSECUTIVE_GH_FAILURES = 5


class Contributor(NamedTuple):
    """A community contributor credited in the release notes.

    `twitter_handle` is a bare handle with no leading `@`.
    `linkedin_url` is either empty or an absolute URL: :func:`_parse_socials`
    prefixes `https://` when the PR body omits a scheme, and preserves an
    explicit `http://` or `https://` as written. Normalizing at parse time
    rather than render time is what keeps the gap-filling merge in
    :func:`collect_contributors` from depending on commit order. Empty means
    "not supplied".
    """

    login: str
    twitter_handle: str
    linkedin_url: str


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a `gh` command.

    Returns None only when `gh` is genuinely absent, so callers can tell
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
    """Describe a failed `gh` invocation."""
    if result is None:
        return "gh executable not found"
    detail = result.stderr.strip() or result.stdout.strip()
    return detail or f"exit status {result.returncode}"


def _gh_api(endpoint: str, *, jq: str = "") -> tuple[str, str | None]:
    """Run `gh api` and return its output and any failure detail."""
    cmd = ["api", endpoint]
    if jq:
        cmd.extend(["--jq", jq])
    result = _run_gh(cmd)
    if result is None or result.returncode != 0:
        return "", _gh_result_detail(result)
    return result.stdout.strip(), None


def _gh_pr_view(pr_num: str, repo: str, fields: str) -> tuple[dict | None, str | None]:
    """Run `gh pr view` and return parsed JSON plus any failure detail."""
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
# linked issue — is published as the author's own profile. `.` excludes
# newlines by default, which is what confines the match to the `LinkedIn:`
# line; MULTILINE only governs `^`. Do not add re.DOTALL.
_LINKEDIN_RE = re.compile(
    r"^\s*LinkedIn:\s*.*?((?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?)",
    re.IGNORECASE | re.MULTILINE,
)


class Socials(NamedTuple):
    """Social links advertised in a PR body; either may be `""`.

    Named rather than a bare `tuple[str, str]` because both fields are
    strings feeding a positional :class:`Contributor` constructor — a
    transposition would type-check cleanly and publish
    `https://x.com/https://linkedin.com/in/...` into a release body.
    """

    twitter_handle: str
    linkedin_url: str


def _parse_socials(pr_body: str) -> Socials:
    """Extract the Twitter handle and LinkedIn URL from a PR body.

    Either value is `""` when the body does not advertise it.
    """
    twitter_match = _TWITTER_RE.search(pr_body)
    linkedin_match = _LINKEDIN_RE.search(pr_body)
    linkedin = linkedin_match.group(1) if linkedin_match else ""
    # Normalize the scheme here rather than at render time so the stored value
    # has one spelling. Otherwise which spelling survives the gap-filling merge
    # in `collect_contributors` depends on commit order. The check is
    # case-insensitive to match `_LINKEDIN_RE`: a body writing `HTTPS://` would
    # otherwise be prefixed again into `https://HTTPS://...`, a dead link.
    if linkedin and not linkedin.lower().startswith(("http://", "https://")):
        linkedin = f"https://{linkedin}"
    return Socials(twitter_match.group(1) if twitter_match else "", linkedin)


class Contributors(NamedTuple):
    """The community/internal split credited in the release notes."""

    community: list[Contributor]
    internal: list[str]


def collect_contributors(
    repo: Path,
    working_dir: str,
    release_commit: str,
    prev_tag: str,
    repository: str,
    *,
    offline: bool = False,
    warnings: list[str],
) -> Contributors:
    """Collect community and internal contributors from merged PRs.

    *internal* comes back sorted; *community* keeps commit order. The two are
    disjoint: an org member is credited only as internal, regardless of what
    labels their other PRs carry.

    Appends diagnostics to *warnings*.
    """
    if offline:
        return Contributors([], [])

    range_spec = f"{prev_tag}..{release_commit}" if prev_tag else release_commit
    commits_result = _git_run(repo, "rev-list", range_spec, "--", working_dir)
    if commits_result.returncode != 0:
        warnings.append(
            f"contributor lookup: git rev-list failed for range '{range_spec}'"
            f" ({commits_result.stderr.strip()}); no contributors will be credited"
        )
        return Contributors([], [])
    all_shas = [
        line.strip() for line in commits_result.stdout.splitlines() if line.strip()
    ]
    shas = all_shas[:MAX_CONTRIBUTOR_COMMITS]
    if len(all_shas) > MAX_CONTRIBUTOR_COMMITS:
        # The git log announces its own truncation in the rendered body; this
        # path has no such surface, so the only signal is this warning.
        warnings.append(
            f"contributor lookup: range '{range_spec}' has {len(all_shas)} commits;"
            f" only the newest {MAX_CONTRIBUTOR_COMMITS} were scanned — the"
            " contributor list is INCOMPLETE"
        )

    contributors: dict[str, Contributor] = {}
    internal_users: set[str] = set()
    seen_prs: set[str] = set()
    failed_lookups = 0
    consecutive_failures = 0
    first_api_error = ""

    # Counts commits consumed by the loop. `seen_prs` cannot stand in for this:
    # it holds unique PR numbers, several commits can share one PR, and commits
    # with no PR never enter it — so subtracting it from a commit count mixes
    # units and reports a figure an operator cannot act on.
    processed = 0

    for sha in shas:
        processed += 1
        pr_num, api_error = _gh_api(
            f"/repos/{repository}/commits/{sha}/pulls", jq=".[0].number // empty"
        )
        if api_error:
            failed_lookups += 1
            consecutive_failures += 1
            first_api_error = first_api_error or api_error
            if consecutive_failures >= MAX_CONSECUTIVE_GH_FAILURES:
                warnings.append(
                    "contributor lookup: abandoned after"
                    f" {consecutive_failures} consecutive failures"
                    f" ({api_error}); {len(shas) - processed} commits were"
                    " left unchecked"
                )
                break
            continue
        consecutive_failures = 0
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

        # `or ""` rather than a default: a present-but-null login yields None,
        # and `None.endswith` would raise past main()'s handler as a bare
        # traceback. Matches the defensive normalization on the fields below.
        gh_user = author.get("login") or ""
        # Belt and braces: `is_bot` is authoritative, but fall back to the
        # login suffix so a missing or renamed field cannot silently promote a
        # bot into the community shoutouts.
        if author.get("is_bot", False) or gh_user.endswith("[bot]"):
            continue

        # `is None` rather than `not in`: a present-but-null labels field is
        # swallowed by the `or []` below just as an absent one is, so both must
        # warn or an `internal` maintainer lands in the community shoutouts.
        if pr_data.get("labels") is None:
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

        socials = _parse_socials(pr_data.get("body") or "")

        # Commits arrive newest-first from `git rev-list`, so the most recent
        # PR's socials win and older PRs only fill the gaps it left.
        existing = contributors.get(gh_user, Contributor(gh_user, "", ""))
        contributors[gh_user] = Contributor(
            gh_user,
            existing.twitter_handle or socials.twitter_handle,
            existing.linkedin_url or socials.linkedin_url,
        )

    if failed_lookups:
        detail = f" (first error: {first_api_error})" if first_api_error else ""
        # Over `processed`, not `len(shas)`: the loop may have abandoned early,
        # and a failure count rendered over commits never attempted understates
        # how much of the scan actually went wrong.
        warnings.append(
            f"contributor lookup: {failed_lookups} of {processed} scanned commits"
            f" could not be resolved to a PR{detail}; the contributor list is"
            " INCOMPLETE"
        )

    # Internal contributors are org members regardless of what labels their
    # other PRs carry, so they are credited only in the internal list.
    for user in internal_users:
        contributors.pop(user, None)

    return Contributors(list(contributors.values()), sorted(internal_users))


def _merged_by(repository: str, pr_num: str, warnings: list[str]) -> str:
    """Return the login of whoever merged *pr_num*, or `""` with a warning."""
    result = _run_gh(
        ["pr", "view", pr_num, "--repo", repository,
         "--json", "mergedBy", "--jq", ".mergedBy.login // empty"]
    )
    if result is None or result.returncode != 0:
        warnings.append(
            f"releaser lookup: gh pr view #{pr_num} failed: "
            f"{_gh_result_detail(result)} — falling back to actor"
        )
        return ""

    login = result.stdout.strip()
    if not login:
        # `.mergedBy.login // empty` yields "" on a successful call when the PR
        # is not merged (an open PR on the release branch), which would
        # otherwise credit the actor with no trace of why.
        warnings.append(
            f"releaser lookup: PR #{pr_num} has no mergedBy"
            " (not merged?) — falling back to actor"
        )
    return login


def resolve_releaser(
    repository: str,
    release_commit: str,
    actor: str,
    *,
    offline: bool = False,
    warnings: list[str],
) -> str:
    """Resolve who shipped the release.

    Primary source is whoever merged the release PR (the release commit is
    that PR's squash-merge); when the commit has no associated merged PR
    (initial release, direct push, manual recovery) it falls back to *actor*.
    Bot accounts resolve to `""`. Returns `""` in *offline* mode. Lookup
    failures are appended to *warnings* — a real API failure must not silently
    fall through to the actor, which on a manual release would credit the
    dispatcher instead of the merger with no breadcrumb.
    """
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
            releaser = _merged_by(repository, pr_num, warnings)

    releaser = releaser or actor

    # Drop bot accounts — there is no human to credit. The bare
    # "github-actions" arm covers github.actor surfacing without the "[bot]"
    # suffix, which the endswith check alone would not catch.
    if not releaser or releaser.endswith("[bot]") or releaser == "github-actions":
        return ""

    return releaser


# ── Body assembly ────────────────────────────────────────────────────────────

# Arbitrary ~5,000 safety margin beneath GitHub's 125,000-character
# release-body limit. Counting bytes is already the conservative direction
# (UTF-8 bytes >= characters, so this budget can only under-fill), so the
# margin is not compensating for the byte/character mismatch — it is slack for
# anything the release action might add around the body we hand it.
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
    if contributor.twitter_handle:
        socials = f"[Twitter](https://x.com/{contributor.twitter_handle})"
    if contributor.linkedin_url:
        link = f"[LinkedIn]({contributor.linkedin_url})"
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
        banner = (
            "> [!WARNING]\n"
            "> This is a pre-release version. Install with:"
            f" `pip install {pkg_name}=={version}`"
        )
        # Join only when there is a changelog section to separate from.
        # Pre-releases usually have none, and an unconditional "\n\n" would
        # then stack with the "\n\n" every later section prepends.
        body = f"{banner}\n\n{body}" if body else banner

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
    # the branch tip. Unlike the attribution lines this does not introduce a
    # `---` rule of its own — with no contributors, maintainers, or releaser it
    # follows the changelog directly.
    default_branch = default_branch or "main"
    if base_branch and base_branch != default_branch:
        body += (
            f"\n\nReleased from `{base_branch}` at commit"
            f" [`{release_commit[:7]}`]"
            f"(https://github.com/{repository}/commit/{release_commit})"
        )

    return body


def _with_section(base_body: str, section: str) -> str:
    """Join *section* onto *base_body*, skipping the separator when it is empty.

    Must stay in step with the `separator_bytes` arithmetic in
    :func:`finalize_body`.
    """
    return f"{base_body}\n\n{section}" if base_body else section


def finalize_body(base_body: str, git_log_details: str) -> tuple[str, list[str]]:
    """Append the git log details block, respecting the size budget.

    Returns `(final_body, warnings)`.
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
        return _with_section(base_body, git_log_details), []

    warnings = [
        "Full Git log omitted to keep the release body within GitHub's size limit"
    ]
    compact_bytes = len(_COMPACT_GIT_LOG.encode())
    if body_bytes + compact_bytes + separator_bytes <= MAX_RELEASE_BODY_BYTES:
        return _with_section(base_body, _COMPACT_GIT_LOG), warnings

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
        repository: `owner/name` for building commit links.
        offline: Skip all GitHub API lookups.
        actor: Fallback releaser when no merged PR is found.
        base_branch: Branch the release was cut from.
        default_branch: Default branch; empty is treated as `main`.
        working_dir: Package directory; derived from *package* when empty.
        is_prerelease: Authoritative pre-release flag. When None, it is
            detected from *version*.

    Returns:
        A `(body, warnings)` pair.

    Raises:
        ValueError: If the package is unknown, the package directory does not
            exist, or *release_sha* does not resolve to a commit.
        subprocess.CalledProcessError: If a git command run with `check=True`
            fails (see :func:`_git`). Most git calls here go through
            :func:`_git_run` and degrade to a warning instead; the ones that do
            not are treated as unrecoverable.

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
    # --is-ancestor`, where an unresolvable value exits non-zero —
    # indistinguishable from "no predecessor" — and would silently mislabel the
    # Git log as an initial release. (`git tag --merged` rejects it outright and
    # would surface as a CalledProcessError, but a named error here beats a
    # traceback from three frames down.)
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
        repo_root, working_dir, release_sha, is_prerelease=is_pre, warnings=warnings
    )
    print(f"Release commit: {release_commit}", file=sys.stderr)

    # Two distinct failures share this step and must not be conflated:
    # `read_reason` means CHANGELOG.md could not be read at all, while
    # `section_reason` means it was read but holds no section for this version.
    # Only the second is expected for a pre-release.
    changelog_body = ""
    section_reason: str | None = None
    changelog_text, read_reason = read_changelog_at(
        repo_root, working_dir, release_sha, warnings
    )
    if not read_reason:
        changelog_body, section_reason = extract_changelog_section_text(
            changelog_text, version, f"{working_dir}/CHANGELOG.md"
        )

    if read_reason:
        # Always surface: CHANGELOG.md is checked in regardless of whether this
        # version has a section, so an unreadable or absent file is never the
        # expected state — not even for a pre-release. It also blinds the
        # missing-tags heuristic below, which needs this text as its evidence.
        warnings.append(read_reason)
    elif section_reason:
        if is_pre:
            # Pre-releases do not get a CHANGELOG.md section, so a missing
            # section — as opposed to an unreadable file — is the expected
            # state, not something a maintainer can act on.
            print(f"No changelog section: {section_reason}", file=sys.stderr)
        else:
            warnings.append(section_reason)
    else:
        print(f"Extracted changelog for version {version}", file=sys.stderr)

    if not prev_tag and changelog_lists_other_versions(changelog_text, version):
        # No usable predecessor tag, yet the changelog documents earlier
        # releases. `resolve_previous_tag` only warns when tags exist but are
        # unreachable; a clone with no tags at all looks identical to a genuine
        # first release, and this is the evidence that distinguishes them.
        warnings.append(
            f"No predecessor tag was found for {package}=={version}, but"
            f" {working_dir}/CHANGELOG.md documents earlier releases — this clone"
            " is probably missing tags (run 'git fetch --tags'). The Git log will"
            " span the package's full history and be labeled an initial release"
        )
    elif not prev_tag and read_reason:
        # The heuristic above is the only evidence distinguishing a genuine
        # first release from a tag-less clone, and it runs on the changelog
        # text. An unreadable changelog silently disables it, so say so rather
        # than let the absent warning read as "checked, nothing wrong".
        warnings.append(
            f"No predecessor tag was found for {package}=={version} and"
            f" {working_dir}/CHANGELOG.md could not be read, so the missing-tags"
            " check could not run. The Git log will span the package's full"
            " history and be labeled an initial release — confirm tags are"
            " present (run 'git fetch --tags')"
        )

    git_log_details = generate_git_log(
        repo_root, working_dir, prev_tag, release_sha, repository, warnings
    )

    if offline:
        warnings.append("Offline mode: skipping contributor and releaser lookups.")
        community: list[Contributor] = []
        internal: list[str] = []
        releaser = ""
    else:
        # Deliberately bounded by release_commit, not release_sha: the git log
        # above shows everything that shipped, while attribution stops at the
        # release commit so commits merged after it are not credited here. For
        # a stable release the two differ whenever anything landed between the
        # changelog bump and the release SHA.
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
    """Append the release body to `$GITHUB_OUTPUT` as a heredoc.

    Raises:
        SystemExit: If `GITHUB_OUTPUT` is unset.

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


def _write_step_summary(
    package: str,
    version: str,
    repository: str,
    sha: str,
    warnings: list[str],
    *,
    actor: str,
    base_branch: str,
    default_branch: str,
) -> None:
    """Surface warnings on the run summary, where a reviewer will see them.

    The `release-notes` job is deliberately fail-open and nothing gates on it,
    so a green check is not evidence the body is good. Best-effort: a summary
    that cannot be written must not fail a job whose whole point is to not
    block the release.

    *repository* and *sha* are rendered into the recovery snippet literally.
    Referencing CI-only variables like `$GITHUB_REPOSITORY` would produce a
    block that fails when pasted into the operator's shell, which is the only
    place it is ever meant to run.

    *actor*, *base_branch*, and *default_branch* are forwarded so the snippet
    rebuilds the body this run published rather than a subtly different one.
    They are keyword-only and required: every one of them can silently drop an
    attribution line, so a caller must decide rather than inherit a default.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not summary_path or not warnings:
        return
    # Carry the flags that change the body's *content*, so a pasted rebuild
    # reproduces this run. `--actor` supplies the `Released by:` line when the
    # release commit has no merged PR to read the merger from, and
    # `--base-branch` drives the `Released from ...` provenance line. Flags the
    # workflow passes only to avoid re-deriving what the script can work out
    # itself (`--working-dir`, `--is-prerelease`, `--repo-root`) are omitted to
    # keep the snippet short.
    provenance = [f'--actor "{actor}"'] if actor else []
    if base_branch and base_branch != (default_branch or "main"):
        provenance.append(f'--base-branch "{base_branch}"')
        # Only needed when the repository default differs from the script's own
        # `--default-branch` default; passing it verbatim keeps the
        # base-vs-default comparison, and so the provenance line, identical to
        # this run's. Omitted otherwise so the common case stays uncluttered.
        if default_branch and default_branch != "main":
            provenance.append(f'--default-branch "{default_branch}"')
    lines = [
        "### ⚠️ Release notes built with warnings",
        "",
        f"`{package}=={version}` is being released, but the generated body may"
        " be incomplete or wrong. Review it, and rebuild locally if needed"
        " (see RELEASING.md > Release Notes Job Failed or GitHub Release Body"
        " Is Empty):",
        "",
        "```bash",
        "python .github/scripts/release/build_release_notes.py \\",
        f'  --package "{package}" --version "{version}" \\',
        f'  --sha "{sha}" --repo "{repository}" \\',
        *(f"  {flag} \\" for flag in provenance),
        "  --out /tmp/release-body.md",
        "",
        f'gh release edit "{package}=={version}" --repo "{repository}" \\',
        "  --notes-file /tmp/release-body.md",
        "```",
        "",
        *(f"- {warning}" for warning in warnings),
        "",
    ]
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    except OSError as err:
        print(f"::warning::Could not write step summary: {err}", file=sys.stderr)


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
        tmp.unlink(missing_ok=True)
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
    if is_prerelease is None and os.environ.get("GITHUB_ACTIONS"):
        # CI is supposed to pass the authoritative flag. Losing it means the
        # banner is decided by version sniffing and can disagree with the
        # GitHub release's own prerelease flag.
        print(
            "::warning::--is-prerelease was not supplied; pre-release status will be"
            f" sniffed from version '{args.version}' and may disagree with the"
            " GitHub release's prerelease flag",
            file=sys.stderr,
        )

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
    except (ValueError, subprocess.CalledProcessError) as err:
        # CalledProcessError included so an unexpected git failure still names
        # itself on the run summary instead of exiting with a bare traceback.
        # `str(CalledProcessError)` reports only the argv and exit status, so
        # append the stderr that `_git` deliberately re-attaches — otherwise
        # git's own diagnosis ("fatal: bad object", "malformed object name")
        # is collected and then thrown away right here.
        detail = (getattr(err, "stderr", "") or "").strip()
        print(f"::error::{err}{f' {detail}' if detail else ''}", file=sys.stderr)
        raise SystemExit(1) from err

    # GitHub renders only the first 10 warning annotations per step, so emit the
    # aggregate "the output is untrustworthy" lines ahead of the per-item ones
    # they summarize.
    warnings.sort(key=lambda w: "INCOMPLETE" not in w)
    for warning in warnings:
        print(f"::warning::{warning}", file=sys.stderr)
    _write_step_summary(
        args.package,
        args.version,
        args.repo,
        args.sha,
        warnings,
        actor=args.actor,
        base_branch=args.base_branch,
        default_branch=args.default_branch,
    )

    if args.github_output:
        _write_github_output(body)
    elif args.out:
        _write_out_file(args.out, body)
    else:
        print(body)


if __name__ == "__main__":
    main()
