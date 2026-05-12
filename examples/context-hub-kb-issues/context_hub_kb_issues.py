"""One-file Context Hub example: deep agent + deploy + LangSmith issues-board wiring.

This module intentionally keeps everything in one place:
- `agent` graph definition using `CompositeBackend` + `ContextHubBackend`
- optional `langgraph deploy` helper
- issues-board create-or-patch wiring for the deployed tracing project
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
DEFAULT_MEMORIES_HUB_IDENTIFIER = "my-agent"
SUCCESS_CODES = {200, 201}
HTTP_CONFLICT = 409


def resolve_model_name() -> str:
    """Return model id from env, falling back to a stable default."""
    return os.getenv("DEEPAGENT_MODEL", DEFAULT_MODEL)


def resolve_memories_identifier(*, project_name: str | None = None) -> str:
    """Return Context Hub identifier from env, project name, or default.

    Identifier format can be `repo`, `owner/repo`, or `-/repo`.
    """
    env_identifier = os.getenv("MEMORIES_HUB_IDENTIFIER")
    if env_identifier:
        return env_identifier
    if project_name:
        return project_name
    return DEFAULT_MEMORIES_HUB_IDENTIFIER


def build_agent() -> CompiledStateGraph:
    """Create a deep agent with durable `/memories/` in Context Hub."""
    try:
        from langchain.chat_models import init_chat_model
        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, ContextHubBackend, StateBackend
    except ImportError as exc:
        msg = (
            "Missing runtime deps for graph construction. Install deepagents graph "
            "dependencies before deploying/running the graph."
        )
        raise RuntimeError(msg) from exc

    memories_identifier = resolve_memories_identifier()

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": ContextHubBackend(memories_identifier),
        },
    )

    return create_deep_agent(
        model=init_chat_model(model=resolve_model_name(), temperature=0),
        backend=backend,
    )


try:
    # Keep a module-level graph object for langgraph import-time resolution.
    agent = build_agent()
except RuntimeError:
    # Allow `--help` and wiring-only flows in lightweight environments.
    agent = None


def resolve_langsmith_api_key() -> str | None:
    """Return LangSmith-compatible API key from environment variables."""
    for key_name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        api_key = os.getenv(key_name)
        if api_key:
            return api_key
    return None


def resolve_langsmith_endpoint() -> str:
    """Return LangSmith API endpoint with a sensible default."""
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    return (endpoint or "https://api.smith.langchain.com").rstrip("/")


def resolve_context_hub_repo_handle(identifier: str) -> str:
    """Extract `repo` from Context Hub identifier.

    Supported input formats:
    - `repo`
    - `owner/repo`
    - `-/repo`
    """
    if "/" not in identifier:
        if not identifier:
            msg = "Invalid MEMORIES_HUB_IDENTIFIER: value cannot be empty."
            raise ValueError(msg)
        return identifier

    owner, sep, repo_handle = identifier.partition("/")
    if not sep or not owner or not repo_handle:
        msg = (
            "Invalid MEMORIES_HUB_IDENTIFIER. Expected `repo`, `owner/repo`, "
            "or `-/repo`, "
            f"got: {identifier!r}"
        )
        raise ValueError(msg)
    return repo_handle


def make_temp_langgraph_config(script_path: Path) -> Path:
    """Create a temporary langgraph config pointing at this file's `agent` object."""
    config = {
        "dependencies": ["."],
        "graphs": {
            "agent": f"./{script_path.name}:agent",
        },
    }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="langgraph-config-",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        json.dump(config, handle)
    return Path(handle.name)


def run_langgraph_deploy(
    *,
    script_path: Path,
    project_name: str,
    memories_identifier: str,
) -> None:
    """Run `langgraph deploy` using a generated config for this one-file example."""
    if shutil.which("langgraph") is None:
        msg = "`langgraph` CLI not found. Install with: pip install 'langgraph-cli[inmem]'"
        raise RuntimeError(msg)

    config_path = make_temp_langgraph_config(script_path)
    cmd = [
        "langgraph",
        "deploy",
        "-c",
        str(config_path),
        "--name",
        project_name,
        "--verbose",
    ]

    env = os.environ.copy()
    env["LANGGRAPH_CLI_ANALYTICS_SOURCE"] = "deepagents"
    env["MEMORIES_HUB_IDENTIFIER"] = memories_identifier

    print(f"Deploying graph with name: {project_name}")
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=False, env=env, cwd=script_path.parent)
        if result.returncode != 0:
            msg = f"`langgraph deploy` failed with exit code {result.returncode}"
            raise RuntimeError(msg)
    finally:
        config_path.unlink(missing_ok=True)


def resolve_tracer_session_id_by_project_name(*, project_name: str, api_key: str) -> str:
    """Resolve tracing project id (session id) by deployed project name."""
    try:
        from langsmith import Client
        from langsmith.utils import LangSmithNotFoundError
    except ImportError as exc:
        msg = (
            "Missing dependency `langsmith`. Install it before running this script: "
            "`uv add langsmith` or `pip install langsmith`."
        )
        raise RuntimeError(msg) from exc

    endpoint = resolve_langsmith_endpoint()
    client = Client(api_url=endpoint, api_key=api_key)

    try:
        project = client.read_project(project_name=project_name)
    except LangSmithNotFoundError as exc:
        msg = (
            "Could not resolve tracing project after deploy. "
            f"Project name: {project_name!r}. Original error: {exc}"
        )
        raise RuntimeError(msg) from exc

    return str(project.id)


def request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, str]:
    """Send JSON request with stdlib HTTP client and return status/body text."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            status = response.getcode()
            text = response.read().decode("utf-8", errors="replace")
            return status, text
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text
    except URLError as exc:
        msg = f"HTTP request failed for {method} {url}: {exc}"
        raise RuntimeError(msg) from exc


def upsert_issues_board_config(
    *,
    session_id: str,
    api_key: str,
    context_hub_repo_handle: str,
) -> None:
    """Create or patch issues-board config for the deployed agent."""
    endpoint = resolve_langsmith_endpoint()
    url = f"{endpoint}/v1/platform/sessions/{session_id}/issues-agent"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    tenant_id = os.getenv("LANGSMITH_TENANT_ID")
    if tenant_id:
        headers["x-tenant-id"] = tenant_id

    create_payload = {
        "cron_schedule": "0 */6 * * *",
        "heavy_model": "anthropic:issues-agent-heavy",
        "light_model": "anthropic:issues-agent-light",
        "context_hub_repo_handle": context_hub_repo_handle,
    }

    create_status, create_text = request_json(
        method="POST",
        url=url,
        headers=headers,
        payload=create_payload,
        timeout_seconds=20,
    )

    if create_status in SUCCESS_CODES:
        print(
            "Issues board auto-wired for tracing project "
            f"{session_id} ({context_hub_repo_handle})."
        )
        return

    if create_status == HTTP_CONFLICT:
        patch_status, patch_text = request_json(
            method="PATCH",
            url=url,
            headers=headers,
            payload={"context_hub_repo_handle": context_hub_repo_handle},
            timeout_seconds=20,
        )
        if patch_status in SUCCESS_CODES:
            print(
                "Issues board already existed; updated context hub handle to "
                f"{context_hub_repo_handle}."
            )
            return

        msg = (
            "Failed to patch existing issues board config. "
            f"HTTP {patch_status}: {patch_text[:300]}"
        )
        raise RuntimeError(msg)

    msg = (
        "Failed to create issues board config. "
        f"HTTP {create_status}: {create_text[:300]}"
    )
    raise RuntimeError(msg)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Deploy this Context Hub-backed deep agent and auto-wire a LangSmith "
            "issues board to the same Context Hub repo handle."
        )
    )
    parser.add_argument(
        "--project-name",
        default="my-agent",
        help=(
            "Deployment/tracing project name (default: my-agent). "
            "Also used as Context Hub memories repo when "
            "MEMORIES_HUB_IDENTIFIER is unset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Deploy example graph and configure issues-board integration."""
    args = parse_args()
    memories_identifier = resolve_memories_identifier(project_name=args.project_name)
    os.environ["MEMORIES_HUB_IDENTIFIER"] = memories_identifier

    script_path = Path(__file__).resolve()
    run_langgraph_deploy(
        script_path=script_path,
        project_name=args.project_name,
        memories_identifier=memories_identifier,
    )

    api_key = resolve_langsmith_api_key()
    if api_key is None:
        msg = "Missing LANGSMITH_API_KEY (or LANGCHAIN_API_KEY)."
        raise RuntimeError(msg)

    repo_handle = resolve_context_hub_repo_handle(memories_identifier)
    session_id = resolve_tracer_session_id_by_project_name(
        project_name=args.project_name,
        api_key=api_key,
    )

    upsert_issues_board_config(
        session_id=session_id,
        api_key=api_key,
        context_hub_repo_handle=repo_handle,
    )


if __name__ == "__main__":
    main()
