from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from rlm import create_rlm_agent

_GITHUB_TOKEN_ENV_VARS = (
    "GITHUB_FINE_GRAINED_PAT",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
_CLASSIFIER_MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are an issue-prioritization analyst for a GitHub repository.

Your FIRST action must be to load in the github-triage skill before doing anything else. Do not proceed until you have loaded it.

Core responsibilities:
- Execute repository triage for the requested `{org}/{repo}`.
- Default source scope to open issues, open pull requests, and open discussions unless the user asks otherwise.
- Assume runs may be repairs; reconcile against existing `triage/` outputs and explicitly flag missing or mismatched items.
- Produce decision-oriented product pillars grounded in triaged evidence.

Hard guardrails:
- Keep analysis factual and concise; call out ambiguity instead of guessing.
- Include references back to source issues/discussions/PRs in generated outputs.

If this prompt conflicts with skill instructions, follow the skill.
"""


def _format_fetch_error(err: Exception) -> str:
    """Format HTTP/network failures for fetch-style tools."""

    if isinstance(err, httpx.HTTPStatusError):
        status = err.response.status_code
        req_url = str(err.request.url)
        body = (err.response.text or "").strip()
        body = body[:500] + ("..." if len(body) > 500 else "")
        auth_hint = ""
        if req_url.startswith("https://api.github.com/") and status in (401, 403):
            auth_hint = (
                " GitHub API auth may be missing or invalid. Set one of: "
                "GITHUB_FINE_GRAINED_PAT, GITHUB_TOKEN, GH_TOKEN."
            )
        return (
            f"Fetch failed with HTTP {status} for URL: {req_url}\n"
            f"Response body:\n{body}\n"
            "Adjust request parameters (pagination, auth headers, or query shape) and retry."
            f"{auth_hint}"
        )
    if isinstance(err, httpx.RequestError):
        req_url = str(err.request.url) if err.request else "<unknown>"
        return (
            f"Fetch failed with a network error while requesting {req_url}: {err}\n"
            "Check network/DNS, endpoint validity, and retry with smaller scope if needed."
        )
    return f"Fetch failed: {type(err).__name__}: {err}"


def _http_fetch(url: str, timeout_seconds: float = 20.0) -> str:
    """Fetch a URL and return the response text."""

    headers: dict[str, str] = {"User-Agent": "deepagents-scratchpad/1.0"}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
        for env_name in _GITHUB_TOKEN_ENV_VARS:
            token = os.getenv(env_name)
            if token:
                headers["Authorization"] = f"Bearer {token}"
                break

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=timeout_seconds,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text
    except (httpx.HTTPStatusError, httpx.RequestError) as err:
        raise RuntimeError(_format_fetch_error(err)) from err


def _normalize_json_schema(
    json_schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if not json_schema:
        raise ValueError("classifier requires a json_schema object")

    strict = bool(json_schema.get("strict", True))
    schema_payload = json_schema.get("schema")
    if isinstance(schema_payload, dict):
        normalized = dict(schema_payload)
        title = normalized.get("title")
        if not isinstance(title, str) or not title.strip():
            name = json_schema.get("name")
            normalized["title"] = (
                name.strip()
                if isinstance(name, str) and name.strip()
                else "classifier_output"
            )
        return normalized, strict

    normalized = dict(json_schema)
    title = normalized.get("title")
    if not isinstance(title, str) or not title.strip():
        name = normalized.get("name")
        normalized["title"] = (
            name.strip()
            if isinstance(name, str) and name.strip()
            else "classifier_output"
        )
    return normalized, strict


def _coerce_classifier_output(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        if isinstance(data, dict):
            return data
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("classifier output was not a JSON object")


@tool("fetch")
def fetch(url: str, timeout_seconds: float = 20.0) -> str:
    """Deterministic HTTP fetch tool used by interpreter skills."""

    return _http_fetch(url, timeout_seconds=timeout_seconds)


@tool("classifier")
def classifier(
    prompt: str,
    description: str | None = None,
    json_schema: dict[str, Any] | None = None,
    item: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured classifier tool for strict JSON-schema outputs."""

    del item, record
    schema_payload, strict = _normalize_json_schema(json_schema)
    llm = ChatAnthropic(model=_CLASSIFIER_MODEL, temperature=0)
    runnable = llm.with_structured_output(
        schema=schema_payload,
        method="json_schema",
        strict=strict,
    )
    messages: list[tuple[str, str]] = []
    if description and description.strip():
        messages.append(("system", description.strip()))
    messages.append(("user", prompt))
    result = runnable.invoke(messages)
    return _coerce_classifier_output(result)


def _format_tool_error(tool_name: str, err: Exception) -> str:
    """Format tool failures into guidance that the model can use for retries."""

    return f"Tool `{tool_name}` failed: {type(err).__name__}: {err}"


@wrap_tool_call
async def tool_error_to_message(request, handler):
    """Return tool errors to the model as ToolMessage instead of aborting the run."""

    try:
        return await handler(request)
    except Exception as err:
        tool_name = request.tool_call.get("name", "<unknown_tool>")
        return ToolMessage(
            content=_format_tool_error(tool_name, err),
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def build_agent(model: str | None = None):
    """Build the recursive agent stack used by the CLI."""
    root_dir = Path(__file__).resolve().parent / "output"
    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    raw_skills_backend = FilesystemBackend(
        root_dir=root_dir / "skills",
        virtual_mode=True,
    )
    # Skills metadata paths originate from SkillsMiddleware on `backend` and
    # are rooted at `/skills/...`. Route that prefix to the dedicated skills
    # backend root (`output/skills`).
    skills_backend = CompositeBackend(
        default=raw_skills_backend,
        routes={"/skills/": raw_skills_backend},
    )

    return create_rlm_agent(
        model=model,
        tools=[fetch, classifier],
        max_depth=1,
        middleware=[tool_error_to_message],
        system_prompt=SYSTEM_PROMPT,
        skills=["skills/"],
        backend=backend,
        skills_backend=skills_backend,
    )


@lru_cache(maxsize=4)
def get_agent(model: str | None = None):
    """Memoized agent accessor for callers that want a module-level singleton."""

    return build_agent(model=model)
