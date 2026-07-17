"""Build agent payloads and managed directory entries for deployment.

This is a pure function over `Project`; no I/O happens here. The result is
suitable for `ApiClient.create_agent`, `ApiClient.patch_agent`, and the Hub
directory commit API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from deepagents_cli.deploy.project import Project, Skill, Subagent

Mode = Literal["create", "patch"]


def build_payload(project: Project, *, mode: Mode = "create") -> dict[str, Any]:
    """Compose the request body for create_agent / patch_agent."""
    payload = build_metadata_payload(project)

    payload["system_prompt"] = project.system_prompt

    if project.tools is not None:
        payload["tools"] = project.tools

    if project.skills:
        payload["skills"] = [_skill_dict(s) for s in project.skills]

    if project.subagents:
        payload["subagents"] = [_subagent_dict(s) for s in project.subagents]

    extra_files = _collect_extra_files(project.subagents)
    if extra_files:
        payload["files"] = extra_files

    # `mode` remains for compatibility with earlier callers. PATCH callers
    # should use `build_metadata_payload` so file state is handled by Hub
    # directory commits instead of the agent PATCH endpoint.
    _ = mode
    return payload


def build_metadata_payload(project: Project) -> dict[str, Any]:
    """Compose the request body for metadata-only PATCH updates."""
    payload: dict[str, Any] = {"name": project.name}
    if project.description:
        payload["description"] = project.description
    if project.runtime:
        payload["runtime"] = project.runtime
    elif project.model:
        payload["runtime"] = {"model": {"model_id": project.model}}
    if project.backend:
        payload["backend"] = project.backend
    if project.permissions:
        payload["permissions"] = project.permissions
    if project.extras:
        payload["extras"] = project.extras
    return payload


def build_directory_files(project: Project) -> dict[str, str]:
    """Return the desired managed directory tree for the project."""
    files: dict[str, str] = {"AGENTS.md": project.system_prompt}
    if project.tools_text is not None:
        files["tools.json"] = project.tools_text

    for skill in project.skills:
        base = f"skills/{skill.name}"
        files[f"{base}/SKILL.md"] = skill.skill_file
        for rel, content in skill.files.items():
            files[f"{base}/{rel}"] = content

    for subagent in project.subagents:
        base = f"subagents/{subagent.name}"
        files[f"{base}/AGENTS.md"] = _render_subagent_agents_md(subagent)
        if subagent.tools_text is not None:
            files[f"{base}/tools.json"] = subagent.tools_text
        for rel, content in subagent.extra_files.items():
            files[f"{base}/{rel}"] = content
    return files


def build_directory_delta(
    remote_files: Mapping[str, Any],
    local_files: Mapping[str, str],
) -> dict[str, dict[str, str] | None]:
    """Return Hub directory commit entries to make remote match local."""
    delta: dict[str, dict[str, str] | None] = {}
    for path, content in local_files.items():
        if _remote_content(remote_files.get(path)) != content:
            delta[path] = {"type": "file", "content": content}
    for path in sorted(remote_files):
        if path not in local_files and _is_managed_path(path):
            delta[path] = None
    return delta


def _skill_dict(skill: Skill) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "inline",
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
    }
    if skill.files:
        out["files"] = dict(skill.files)
    return out


def _subagent_dict(sa: Subagent) -> dict[str, Any]:
    out: dict[str, Any] = {"name": sa.name}
    if sa.description:
        out["description"] = sa.description
    if sa.model_id:
        out["model_id"] = sa.model_id
    out["instructions"] = sa.instructions
    if sa.tools is not None:
        out["tools"] = sa.tools
    return out


def _collect_extra_files(subagents: list[Subagent]) -> dict[str, dict[str, str]]:
    """Map raw-files entries from subagents into the top-level `files` field."""
    out: dict[str, dict[str, str]] = {}
    for sa in subagents:
        for rel, content in sa.extra_files.items():
            out[f"subagents/{sa.name}/{rel}"] = {"content": content}
    return out


def _render_subagent_agents_md(subagent: Subagent) -> str:
    frontmatter: list[str] = []
    if subagent.description:
        frontmatter.append(f"description: {json.dumps(subagent.description)}")
    if subagent.model_id:
        frontmatter.append(f"model_id: {json.dumps(subagent.model_id)}")
    return "---\n" + "\n".join(frontmatter) + f"\n---\n\n{subagent.instructions}"


def _remote_content(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    data = cast("dict[str, object]", entry)
    if data.get("type") != "file":
        return None
    content = data.get("content")
    return content if isinstance(content, str) else None


def _is_managed_path(path: str) -> bool:
    return path in {"AGENTS.md", "tools.json"} or path.startswith(
        ("skills/", "subagents/")
    )
