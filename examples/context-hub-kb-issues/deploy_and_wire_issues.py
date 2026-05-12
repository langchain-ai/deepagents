"""Deploy a Context Hub-backed deep agent and auto-wire its LangSmith issues board.

This script mirrors the issues-board upsert logic used in CLI deploy flow:
- deploys with `langgraph deploy`
- resolves the tracing project id by deployment name
- POSTs `/v1/platform/sessions/{session_id}/issues-agent`
- PATCHes the existing config on HTTP 409 conflict
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUCCESS_CODES = {200, 201}
HTTP_CONFLICT = 409
DEFAULT_MEMORIES_HUB_IDENTIFIER = "-/my-agent"


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
    """Extract `repo` from Context Hub identifier (`owner/repo` or `-/repo`)."""
    owner, sep, repo_handle = identifier.partition("/")
    if not sep or not owner or not repo_handle:
        msg = (
            "Invalid MEMORIES_HUB_IDENTIFIER. Expected `owner/repo` or `-/repo`, "
            f"got: {identifier!r}"
        )
        raise ValueError(msg)
    return repo_handle


def run_langgraph_deploy(config_path: Path, *, project_name: str) -> None:
    """Run `langgraph deploy` for this example graph."""
    if shutil.which("langgraph") is None:
        msg = "`langgraph` CLI not found. Install with: pip install 'langgraph-cli[inmem]'"
        raise RuntimeError(msg)

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

    print(f"Deploying graph with name: {project_name}")
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"`langgraph deploy` failed with exit code {result.returncode}")


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
        required=True,
        help="Deployment/tracing project name (must match deployed graph project).",
    )
    parser.add_argument(
        "--config",
        default="langgraph.json",
        help="Path to langgraph config (default: langgraph.json).",
    )
    parser.add_argument(
        "--memories-identifier",
        default=os.getenv("MEMORIES_HUB_IDENTIFIER", DEFAULT_MEMORIES_HUB_IDENTIFIER),
        help=(
            "Context Hub identifier for /memories/ route, e.g. '-/my-agent' or "
            "'my-org/my-agent'."
        ),
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip `langgraph deploy` and only wire/update the issues board.",
    )
    return parser.parse_args()


def main() -> None:
    """Deploy example graph and configure issues-board integration."""
    args = parse_args()

    if not args.skip_deploy:
        run_langgraph_deploy(Path(args.config), project_name=args.project_name)

    api_key = resolve_langsmith_api_key()
    if api_key is None:
        msg = "Missing LANGSMITH_API_KEY (or LANGCHAIN_API_KEY)."
        raise RuntimeError(msg)

    repo_handle = resolve_context_hub_repo_handle(args.memories_identifier)
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
