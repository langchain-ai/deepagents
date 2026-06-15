"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from langchain.chat_models import init_chat_model

_DEFAULT_WORKDIR = Path("/app")

_SYSTEM_PROMPT = """You are running in a Harbor benchmark sandbox.

Complete the task autonomously. There is no human operator available to answer
follow-up questions, so make reasonable assumptions and keep working until the
task is complete.

Use the sandbox working directory for all file and shell operations. In Terminal
Bench-style tasks this is usually `/app`; use `pwd` if you need to confirm the
current directory.

Prefer non-interactive command variants. Do not run commands that wait for
human input.
"""


def _configurable(config: dict[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    value = config.get("configurable")
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`configurable` must be a dictionary"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _model_kwargs(configurable: dict[str, object]) -> dict[str, Any]:
    value = configurable.get("model_kwargs")
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "`configurable.model_kwargs` must be a dictionary"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _model_name(configurable: dict[str, object]) -> str:
    value = configurable.get("model") or os.environ.get("HARBOR_MODEL")
    if not isinstance(value, str) or not value.strip():
        msg = "`configurable.model` or `HARBOR_MODEL` must provide a model name"
        raise ValueError(msg)
    return value


def _workdir(configurable: dict[str, object]) -> Path:
    value = configurable.get("cwd")
    if value is None:
        return _DEFAULT_WORKDIR
    if not isinstance(value, str | Path):
        msg = "`configurable.cwd` must be a string path"
        raise TypeError(msg)
    return Path(value)


def make_graph(config: dict[str, object] | None = None) -> object:
    """Create the Deep Agents Code CLI harness graph Harbor should run.

    Harbor's installed `langgraph` agent loads this factory from
    `langgraph.json` inside each benchmark sandbox. The returned value is the
    LangGraph graph produced by Deep Agents Code's headless constructor.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model` and optional provider kwargs in
            `configurable.model_kwargs`.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    assistant_id = os.environ.get("HARBOR_SESSION_ID") or f"harbor-{uuid.uuid4()}"
    graph, _backend = create_cli_agent(
        model=model,
        assistant_id=assistant_id,
        sandbox=None,
        sandbox_type="harbor",
        system_prompt=_SYSTEM_PROMPT,
        interactive=False,
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=True,
        cwd=_workdir(configurable),
    )
    return graph


def make_bare_graph(config: dict[str, object] | None = None) -> object:
    """Create a Deep Agents SDK graph Harbor should run directly.

    This path avoids the Deep Agents Code CLI harness while still attaching a
    local shell backend rooted at Harbor's sandbox workdir so terminal-bench
    tasks can use filesystem and command execution tools.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model` and optional provider kwargs in
            `configurable.model_kwargs`.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
    )
