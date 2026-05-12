"""Deploy a Context Hub memory-backed deep agent and wire its issues board."""

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
DEFAULT_PROJECT_NAME = "my-agent"
DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
SUCCESS_CODES = {200, 201}
HTTP_CONFLICT = 409


def memories_identifier(*, project_name: str) -> str:
    """Return Context Hub memories identifier from env or project name."""
    return os.getenv("MEMORIES_HUB_IDENTIFIER") or project_name


def langsmith_endpoint() -> str:
    """Return LangSmith API endpoint from env or default."""
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    return (endpoint or DEFAULT_ENDPOINT).rstrip("/")


def langsmith_api_key() -> str:
    """Return LangSmith API key from env variables."""
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        msg = "Missing LANGSMITH_API_KEY (or LANGCHAIN_API_KEY)."
        raise RuntimeError(msg)
    return api_key


def repo_handle(identifier: str) -> str:
    """Return Context Hub repo handle from `repo`, `owner/repo`, or `-/repo`."""
    if "/" not in identifier:
        if not identifier:
            msg = "Invalid MEMORIES_HUB_IDENTIFIER: value cannot be empty."
            raise ValueError(msg)
        return identifier

    owner, sep, handle = identifier.partition("/")
    if not sep or not owner or not handle:
        msg = (
            "Invalid MEMORIES_HUB_IDENTIFIER. Expected `repo`, `owner/repo`, or "
            f"`-/repo`, got: {identifier!r}"
        )
        raise ValueError(msg)
    return handle


def build_agent() -> CompiledStateGraph:
    """Build a deep agent with durable `/memories/` in Context Hub."""
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

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": ContextHubBackend(memories_identifier(project_name=DEFAULT_PROJECT_NAME)),
        },
    )

    return create_deep_agent(
        model=init_chat_model(model=os.getenv("DEEPAGENT_MODEL", DEFAULT_MODEL), temperature=0),
        backend=backend,
    )


try:
    # Required for langgraph module import resolution.
    agent = build_agent()
except RuntimeError:
    # Allow `--help` in environments missing graph deps.
    agent = None


def deploy_graph(*, script_path: Path, project_name: str, memories_id: str) -> None:
    """Deploy this file as a langgraph graph."""
    if shutil.which("langgraph") is None:
        msg = "`langgraph` CLI not found. Install with: pip install 'langgraph-cli[inmem]'"
        raise RuntimeError(msg)

    config = {
        "dependencies": ["."],
        "graphs": {
            "agent": f"./{script_path.name}:agent",
        },
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="langgraph-config-",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle)
        config_path = Path(handle.name)

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
    env["MEMORIES_HUB_IDENTIFIER"] = memories_id

    print(f"Deploying graph with name: {project_name}")
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=False, env=env, cwd=script_path.parent)
        if result.returncode != 0:
            msg = f"`langgraph deploy` failed with exit code {result.returncode}"
            raise RuntimeError(msg)
    finally:
        config_path.unlink(missing_ok=True)


def session_id_for_project(*, project_name: str, api_key: str, endpoint: str) -> str:
    """Resolve LangSmith tracing project/session id by project name."""
    try:
        from langsmith import Client
        from langsmith.utils import LangSmithNotFoundError
    except ImportError as exc:
        msg = "Missing dependency `langsmith`. Install it with `uv add langsmith`."
        raise RuntimeError(msg) from exc

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
) -> tuple[int, str]:
    """Send JSON request and return `(status_code, body_text)`."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310
            status = response.getcode()
            text = response.read().decode("utf-8", errors="replace")
            return status, text
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text
    except URLError as exc:
        msg = f"HTTP request failed for {method} {url}: {exc}"
        raise RuntimeError(msg) from exc


def upsert_issues_board(*, session_id: str, api_key: str, endpoint: str, handle: str) -> None:
    """Create-or-patch LangSmith issues board for this deployed project."""
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
        "context_hub_repo_handle": handle,
    }

    create_status, create_text = request_json(
        method="POST",
        url=url,
        headers=headers,
        payload=create_payload,
    )

    if create_status in SUCCESS_CODES:
        print(f"Issues board wired for tracing project {session_id} ({handle}).")
        return

    if create_status == HTTP_CONFLICT:
        patch_status, patch_text = request_json(
            method="PATCH",
            url=url,
            headers=headers,
            payload={"context_hub_repo_handle": handle},
        )
        if patch_status in SUCCESS_CODES:
            print(f"Issues board existed; updated context hub handle to {handle}.")
            return

        msg = (
            "Failed to patch existing issues board config. "
            f"HTTP {patch_status}: {patch_text[:300]}"
        )
        raise RuntimeError(msg)

    msg = f"Failed to create issues board config. HTTP {create_status}: {create_text[:300]}"
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
        default=DEFAULT_PROJECT_NAME,
        help=(
            "Deployment/tracing project name (default: my-agent). Also used as "
            "Context Hub memories repo when MEMORIES_HUB_IDENTIFIER is unset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run deploy + issues-board wiring flow."""
    args = parse_args()
    endpoint = langsmith_endpoint()
    api_key = langsmith_api_key()

    memories_id = memories_identifier(project_name=args.project_name)
    os.environ["MEMORIES_HUB_IDENTIFIER"] = memories_id

    script_path = Path(__file__).resolve()
    deploy_graph(
        script_path=script_path,
        project_name=args.project_name,
        memories_id=memories_id,
    )

    session_id = session_id_for_project(
        project_name=args.project_name,
        api_key=api_key,
        endpoint=endpoint,
    )
    upsert_issues_board(
        session_id=session_id,
        api_key=api_key,
        endpoint=endpoint,
        handle=repo_handle(memories_id),
    )


if __name__ == "__main__":
    main()
