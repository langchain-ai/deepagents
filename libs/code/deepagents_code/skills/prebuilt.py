"""Install prebuilt LangChain/LangSmith skill collections.

The LangChain and LangSmith teams publish curated agent-skill collections at
`langchain-ai/langchain-skills` and `langchain-ai/langsmith-skills`. This module
fetches those collections (as GitHub source tarballs, no `git`/`npx` required)
and installs the individual skills into a local skills directory so they show up
in skill discovery.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pathlib import Path

# Skills live under this prefix inside each collection repo.
_SKILLS_SUBPATH = "config/skills"

# Network timeout (seconds) for the tarball download.
_DOWNLOAD_TIMEOUT = 60.0


@dataclass(frozen=True)
class PrebuiltCollection:
    """A curated, remotely hosted collection of agent skills.

    Attributes:
        key: Short identifier used on the command line (e.g. `langchain`).
        name: Human-readable collection name.
        owner: GitHub owner/org.
        repo: GitHub repository name.
        ref: Git ref (branch or tag) to download.
        description: One-line summary for help/listing output.
    """

    key: str
    name: str
    owner: str
    repo: str
    description: str
    ref: str = "main"

    @property
    def url(self) -> str:
        """GitHub source tarball URL for the configured ref."""
        return (
            f"https://github.com/{self.owner}/{self.repo}"
            f"/archive/refs/heads/{self.ref}.tar.gz"
        )


PREBUILT_COLLECTIONS: dict[str, PrebuiltCollection] = {
    "langchain": PrebuiltCollection(
        key="langchain",
        name="LangChain Skills",
        owner="langchain-ai",
        repo="langchain-skills",
        description="Build agents with LangChain, LangGraph, and Deep Agents",
    ),
    "langsmith": PrebuiltCollection(
        key="langsmith",
        name="LangSmith Skills",
        owner="langchain-ai",
        repo="langsmith-skills",
        description="Observe and evaluate LLM apps with LangSmith",
    ),
}
"""Known prebuilt collections, keyed by their command-line identifier."""


@dataclass
class InstallResult:
    """Outcome of installing a single collection.

    Attributes:
        collection: The collection that was processed.
        installed: Names of skills newly written to disk.
        skipped: Names of skills skipped because they already existed.
    """

    collection: PrebuiltCollection
    installed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class PrebuiltSkillsError(Exception):
    """Raised when a prebuilt collection cannot be downloaded or extracted."""


def _safe_members(
    tar: tarfile.TarFile, skills_prefix: str
) -> dict[str, list[tarfile.TarInfo]]:
    """Group tar members by skill name, rejecting unsafe paths.

    Args:
        tar: Opened tar archive for a collection.
        skills_prefix: Archive path prefix that contains the skills
            (e.g. `repo-main/config/skills/`).

    Returns:
        Mapping of skill name to the member entries that belong to it.

    Raises:
        PrebuiltSkillsError: If a member escapes the skills prefix.
    """
    grouped: dict[str, list[tarfile.TarInfo]] = {}
    for member in tar.getmembers():
        if not member.name.startswith(skills_prefix):
            continue
        relative = member.name[len(skills_prefix) :]
        if not relative:
            continue
        if relative.startswith("/") or ".." in relative.split("/"):
            msg = f"Unsafe path in archive: {member.name}"
            raise PrebuiltSkillsError(msg)
        skill_name = relative.lstrip("/").split("/", 1)[0]
        grouped.setdefault(skill_name, []).append(member)
    return grouped


def install_collection(
    collection: PrebuiltCollection,
    dest_dir: Path,
    *,
    force: bool = False,
) -> InstallResult:
    """Download a collection and install its skills into `dest_dir`.

    Args:
        collection: The collection to install.
        dest_dir: Target skills directory (created if missing).
        force: Overwrite skills that already exist instead of skipping them.

    Returns:
        An `InstallResult` describing what was installed or skipped.

    Raises:
        PrebuiltSkillsError: If the download fails or the archive is malformed.
    """
    try:
        response = httpx.get(
            collection.url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        msg = f"Failed to download {collection.name}: {exc}"
        raise PrebuiltSkillsError(msg) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    result = InstallResult(collection=collection)

    try:
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            root = tar.getmembers()[0].name.split("/", 1)[0]
            skills_prefix = f"{root}/{_SKILLS_SUBPATH}/"
            grouped = _safe_members(tar, skills_prefix)
            if not grouped:
                msg = f"No skills found in {collection.name} ({_SKILLS_SUBPATH})"
                raise PrebuiltSkillsError(msg)

            for skill_name in sorted(grouped):
                skill_dir = dest_dir / skill_name
                if skill_dir.exists() and not force:
                    result.skipped.append(skill_name)
                    continue
                _extract_skill(tar, grouped[skill_name], skills_prefix, dest_dir)
                result.installed.append(skill_name)
    except tarfile.TarError as exc:
        msg = f"Failed to extract {collection.name}: {exc}"
        raise PrebuiltSkillsError(msg) from exc

    return result


def _extract_skill(
    tar: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    skills_prefix: str,
    dest_dir: Path,
) -> None:
    """Write the files of one skill into `dest_dir`, stripping the prefix.

    Args:
        tar: Opened tar archive.
        members: Member entries belonging to a single skill.
        skills_prefix: Archive prefix to strip from member names.
        dest_dir: Target skills directory.
    """
    for member in members:
        relative = member.name[len(skills_prefix) :].lstrip("/")
        target = dest_dir / relative
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with extracted, target.open("wb") as out:
                out.write(extracted.read())


def resolve_collections(keys: list[str]) -> list[PrebuiltCollection]:
    """Resolve user-supplied collection keys to `PrebuiltCollection` objects.

    Args:
        keys: Collection keys, where `all` expands to every known collection.

    Returns:
        Ordered, de-duplicated list of matching collections.

    Raises:
        PrebuiltSkillsError: If a key does not match a known collection.
    """
    if not keys or "all" in keys:
        return list(PREBUILT_COLLECTIONS.values())

    resolved: list[PrebuiltCollection] = []
    for key in keys:
        collection = PREBUILT_COLLECTIONS.get(key.lower())
        if collection is None:
            known = ", ".join([*PREBUILT_COLLECTIONS, "all"])
            msg = f"Unknown skill collection '{key}'. Choose from: {known}"
            raise PrebuiltSkillsError(msg)
        if collection not in resolved:
            resolved.append(collection)
    return resolved
