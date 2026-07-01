"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from langchain.chat_models import init_chat_model

# This module is loaded by Harbor as a top-level langgraph entrypoint; make sibling
# modules in the project directory importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _text_only_media_middleware import TextOnlyMediaMiddleware  # noqa: E402
from _repeat_tool_call_guard_middleware import (  # noqa: E402
    RepeatToolCallGuardMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_WORKDIR = Path("/app")

_SHELL_ENV_DENYLIST = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "BASETEN_API_KEY",
        "FIREWORKS_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
        "NVIDIA_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    }
)

_SYSTEM_PROMPT = """You are running in a Harbor benchmark sandbox.

Complete the task autonomously. There is no human operator available to answer
follow-up questions, so make reasonable assumptions and keep working until the
task is complete.

Use the sandbox working directory for all file and shell operations. In Terminal
Bench-style tasks this is usually `/app`; use `pwd` if you need to confirm the
current directory.

Prefer non-interactive command variants. Do not run commands that wait for
human input.

This model is text-only. Never read an image, video, audio, or other binary file into
the conversation (for example, do not use read_file on a .png/.jpg/.mp4/.wav) — passing
binary/image content to the model will fail. To work with such files, inspect them
programmatically instead: write and run code (e.g. Python with PIL/OpenCV/numpy, ffmpeg,
tesseract/OCR) or use shell tools (`file`, `xxd`, `identify`), and reason from the
program, data, or rendered output that produced them.
"""


# Appended to the system prompt unless disabled via DEEPAGENTS_SELF_VERIFY=0 (used to
# A/B test the self-verification lever locally and in CI). See IDEAS.md "Iter-SV".
_SELF_VERIFY_BLOCK = """

Your solution is graded by automated programmatic tests, so correctness is checked
mechanically — never by your own assertion that you are done. Before finishing, VERIFY:
re-read the exact requirements (output file paths, formats, and expected values must
match precisely), then run a concrete check that exercises your solution against the
task's own inputs/examples — e.g. run your regex against the provided sample, execute
your script and diff its output against what is required, or run the query and inspect
the rows. Run this check the way the grader will: in a FRESH shell or new process (e.g.
`bash -lc '...'`), not your current session, and make any required setup persistent
(symlink the binary into a standard PATH directory, install it, or write a profile entry)
rather than relying on an `export` that only lives in your current shell. If the check
fails, fix it and re-verify — but if it still fails after two or three focused attempts,
submit your best solution and stop; do not loop indefinitely re-verifying. Cover edge
cases, not just the happy path.
"""


def _system_prompt() -> str:
    """System prompt. Self-verification block is OFF by default (a clean A/B showed it
    nets ~0 and can trigger verify→fix recursion loops); opt back in with
    DEEPAGENTS_SELF_VERIFY=1."""
    if os.environ.get("DEEPAGENTS_SELF_VERIFY", "0").strip() == "1":
        return _SYSTEM_PROMPT + _SELF_VERIFY_BLOCK
    return _SYSTEM_PROMPT


@contextmanager
def _scrub_shell_env() -> Iterator[None]:
    saved = {name: os.environ.pop(name, None) for name in _SHELL_ENV_DENYLIST}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        kwargs: dict[str, Any] = {}
    elif not isinstance(value, dict):
        msg = "`configurable.model_kwargs` must be a dictionary"
        raise TypeError(msg)
    else:
        kwargs = {str(key): item for key, item in value.items()}
    # Retry transient/rate-limit errors (e.g. Baseten 429 throttle) with backoff so a
    # single 429 doesn't fatally crash the agent. Override via configurable.model_kwargs.
    kwargs.setdefault("max_retries", 8)
    # NVIDIA's recommended sampling for Nemotron-3-Ultra. max_tokens=32000 in particular
    # overrides Baseten's low server-default completion cap (~4096) that was truncating
    # deliverables mid-write. All overridable via configurable.model_kwargs.
    kwargs.setdefault("temperature", 0.6)
    kwargs.setdefault("max_tokens", 32000)
    kwargs.setdefault("top_p", 0.95)
    return kwargs


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
    with _scrub_shell_env():
        graph, _backend = create_cli_agent(
            model=model,
            assistant_id=assistant_id,
            sandbox=None,
            sandbox_type="harbor",
            system_prompt=_system_prompt(),
            interactive=False,
            auto_approve=True,
            enable_memory=False,
            enable_skills=False,
            enable_shell=True,
            extra_middleware=[
                TextOnlyMediaMiddleware(),
                RepeatToolCallGuardMiddleware(),
            ],
            cwd=_workdir(configurable),
        )
    # Break runaway identical-tool-call loops earlier. deepagents-code sets
    # recursion_limit=1000; the loop-guard middleware nudges a change of approach, and
    # this lowers the hard backstop so worst-case token waste is bounded.
    graph = graph.with_config({"recursion_limit": 300})
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
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_system_prompt(),
    )
