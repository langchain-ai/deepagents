"""Create one LangSmith experiment session per Harbor model matrix entry."""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def experiment_name(*, agent_impl: str, model: str, run_id: str, run_attempt: str) -> str:
    """Build a GitHub-run-scoped LangSmith experiment name."""
    return "-".join(
        (
            "deepagents-harbor",
            _safe_name(agent_impl),
            _safe_name(model),
            _safe_name(run_id),
            _safe_name(run_attempt),
        )
    )


def add_experiment_ids(
    matrix: dict[str, Any],
    *,
    agent_impl: str,
    run_id: str,
    run_attempt: str,
    create_session: Callable[[str, dict[str, object]], str],
) -> dict[str, list[dict[str, Any]]]:
    """Add one shared LangSmith experiment ID to every entry for each model."""
    entries = matrix.get("include")
    if not isinstance(entries, list):
        msg = "MODEL_MATRIX must contain an include list"
        raise TypeError(msg)

    include: list[dict[str, Any]] = []
    experiments: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            msg = "Every MODEL_MATRIX include entry must contain a string model"
            raise TypeError(msg)
        model = entry["model"]
        if model not in experiments:
            name = experiment_name(
                agent_impl=agent_impl,
                model=model,
                run_id=run_id,
                run_attempt=run_attempt,
            )
            experiment_id = create_session(
                name,
                {
                    "source": "deepagents-harbor-workflow",
                    "agent_impl": agent_impl,
                    "model": model,
                    "github_run_id": run_id,
                    "github_run_attempt": run_attempt,
                },
            )
            experiments[model] = (name, experiment_id)
        name, experiment_id = experiments[model]
        include.append(
            {
                **entry,
                "langsmith_experiment_name": name,
                "langsmith_experiment_id": experiment_id,
            }
        )
    return {"include": include}


def _create_session(name: str, metadata: dict[str, object]) -> str:
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        msg = "LANGSMITH_API_KEY is required"
        raise RuntimeError(msg)

    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    base_url = endpoint if endpoint.endswith("/api/v1") else f"{endpoint}/api/v1"
    payload = json.dumps(
        {
            "id": str(uuid.uuid4()),
            "name": name,
            "start_time": datetime.datetime.now(datetime.UTC).isoformat(),
            "extra": {"metadata": metadata},
        }
    ).encode()
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    if workspace_id := os.environ.get("LANGSMITH_WORKSPACE_ID"):
        headers["LANGSMITH-WORKSPACE-ID"] = workspace_id
    request = urllib.request.Request(  # noqa: S310
        f"{base_url}/sessions", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        msg = f"LangSmith session creation failed with HTTP status {exc.code}"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        msg = "Could not connect to LangSmith while creating an experiment session"
        raise RuntimeError(msg) from exc

    experiment_id = data.get("id") if isinstance(data, dict) else None
    if not isinstance(experiment_id, str) or not experiment_id:
        msg = "LangSmith session creation returned no experiment id"
        raise RuntimeError(msg)
    return experiment_id


def main() -> None:
    """Create the experiments and write the enriched matrix to GitHub output."""
    matrix = json.loads(os.environ["MODEL_MATRIX"])
    result = add_experiment_ids(
        matrix,
        agent_impl=os.environ["HARBOR_AGENT_IMPL"],
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        create_session=_create_session,
    )
    output = "matrix=" + json.dumps(result, separators=(",", ":"))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as file:
            file.write(output + "\n")
    else:
        print(output)  # noqa: T201


if __name__ == "__main__":
    try:
        main()
    except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
