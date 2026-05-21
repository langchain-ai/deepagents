"""Build the JSON body POSTed (or PATCHed) to /v1/deepagents/agents.

This is a pure function over `Project`; no I/O happens here. The result is
suitable for `ApiClient.create_agent` or `ApiClient.patch_agent`.
"""

from __future__ import annotations

from typing import Any, Literal

from deepagents_cli.deploy.project import Project, Skill, Subagent


Mode = Literal["create", "patch"]


def build_payload(project: Project, *, mode: Mode = "create") -> dict[str, Any]:
    """Compose the request body for create_agent / patch_agent."""
    payload: dict[str, Any] = {"name": project.name}
    if project.description:
        payload["description"] = project.description
    if project.runtime:
        payload["runtime"] = project.runtime
    if project.permissions:
        payload["permissions"] = project.permissions
    if project.extras:
        payload["extras"] = project.extras

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

    # `mode` is exposed for forward-compat (e.g. emitting `deleted_paths` on
    # patch). Today the body shape is identical; we still keep the literal
    # type to document caller intent.
    _ = mode
    return payload


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
