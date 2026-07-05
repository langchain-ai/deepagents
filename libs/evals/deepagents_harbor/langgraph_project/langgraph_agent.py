"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


# --- perf: de-duplicate wrap-span trace serialization -------------------------
# LangChain wraps every middleware `wrap_model_call`/`wrap_tool_call` in
# `@traceable(process_inputs=_scrub_inputs)`. The stock `_scrub_inputs` keeps the full
# `request` — including the entire, growing `messages` history (and `state`) — in each
# span's recorded inputs. With ~9 wrap layers the whole conversation is re-serialized
# ~9x per model call: O(n) per call -> O(n^2) per run, which dominates wall-clock late
# in long agent runs and drives timeouts. The messages are already recorded on the
# inner model (LLM) span, so keeping them on every wrapper span is pure redundancy.
# We replace `messages` with a lightweight placeholder and drop `state`; trace
# structure, span outputs, the LLM span, and reward feedback are untouched. Guarded so
# a LangChain change that renames/removes the symbol is a no-op, not a break. (Ideally
# upstreamed into `_scrub_inputs` later; local patch to test quickly for now.)
def _install_lean_wrap_span_scrub() -> None:
    try:
        from langchain.agents import factory as _factory  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.warning("lean wrap-span scrub: langchain.agents.factory not importable; skipping")
        return
    orig = getattr(_factory, "_scrub_inputs", None)
    if orig is None or getattr(orig, "_lean_wrap_span", False):
        return

    def _lean_scrub(inputs: dict[str, Any]) -> dict[str, Any]:
        filtered = orig(inputs)
        req = filtered.get("request")
        if isinstance(req, dict):
            req = dict(req)
            msgs = req.get("messages")
            if isinstance(msgs, list):
                req["messages"] = f"<{len(msgs)} messages omitted from wrap-span trace>"
            req.pop("state", None)
            filtered["request"] = req
        return filtered

    _lean_scrub._lean_wrap_span = True  # type: ignore[attr-defined]
    _factory._scrub_inputs = _lean_scrub
    logger.info("Installed lean wrap-span trace scrub (drops redundant messages/state from wrap spans).")


_install_lean_wrap_span_scrub()
# ------------------------------------------------------------------------------

_DEFAULT_WORKDIR = Path("/app")
# Directories to try, in order, when the resolved workdir is absent from the task image.
_WORKDIR_FALLBACKS: tuple[Path, ...] = (Path("/app"), Path("/workspace"), Path("/root"), Path("/"))

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

_SYSTEM_PROMPT = """You are an autonomous coding agent in a sandboxed environment with shell and
filesystem access. No human is available to answer questions: make reasonable
assumptions and keep working until the task is fully complete, rather than
stopping to report what you would do.

Work from the sandbox working directory (run `pwd` if unsure). Prefer
non-interactive command flags; never run a command that waits for human input.

## Match the spec exactly

File paths, filenames, field names, identifiers, and output formats must match the
task wording character-for-character — `value` is not `val`, `/app/result.txt` is
not `/app/results.txt`. If the task defines a schema or names an output file, copy
it verbatim; do not rename or "improve" it.

## Let code do the work, not prose

Use your reply to decide the approach, not to carry it out. You have a limited
output budget — deriving results, simulating logic, or hand-writing file contents
in your reasoning will exhaust it before anything reaches disk. Instead:

- Compute, test, and verify by writing a script and running it in the shell, then
  reading the result. Code is your scratchpad for trial-and-error.
- Generate large or repetitive files (data, generated code, long transformed text)
  with a script that writes them — never by typing the contents out yourself,
  whether in a message or as a `write_file` argument.
- Keep reasoning short and decision-focused, and act early rather than thinking at
  length before your first tool call. When a task needs multiple steps, plan them as
  concrete actions to run ("write and run `encoder.py`"), not thinking steps
  ("analyze", "derive") — a plan should commit you to executing, not deliberating.

## Use the right tool

Prefer dedicated tools over raw shell: `read_file` over `cat`, `write_file` over
`echo`/heredoc, `edit_file` over `sed`/`awk`, `grep`/`glob` over shell equivalents.
Read large files with pagination. Make independent tool calls in parallel.

## Keep durable notes

Maintain a notes file e.g. `/app/Notes.md` of findings, decisions, and results of
experiments, and the exact required output contract; update it as you learn. Re-read
it when confused, when resuming, or after summarization events, when you feel you lack
context about what you're solving.

## Verify against the real check, not your own

Run it the way it will actually be run. If the task ships tests, examples, or sample
input/output, run them; they are your success signal. Otherwise turn a concrete example
from the task into a runnable check and iterate until it passes — the exact command, from
a clean directory. Derive what you check from the task's literal wording, not your own
assumptions: a self-written test that passes proves nothing if it asserts a field, path,
or value you invented.

## Finish with a verified deliverable

If the task asks for a file or on-disk output, that artifact must exist before you
stop — confirm it with the shell (`ls`, `cat`). Never end having only described or
planned the deliverable.
"""


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
        return {}
    if not isinstance(value, dict):
        msg = "`configurable.model_kwargs` must be a dictionary"
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


# Per-model defaults merged UNDER the caller's `configurable.model_kwargs` when
# building the chat model. Keyed on the exact `provider:model` spec so no other
# model is affected. Baseten serves Nemotron 3 Ultra with reasoning OFF by
# default; reasoning is enabled only via
# `extra_body.chat_template_kwargs.enable_thinking` (the Nemotron `system:
# "detailed thinking on"` convention and OpenAI `reasoning_effort` have no effect
# on that deployment). Without this the agent runs the model non-reasoning, which
# is a misconfiguration for an agentic benchmark. It also needs the same NVIDIA
# cookbook sampling as Fireworks (temperature 0.6 / top_p 0.95) plus an explicit
# `max_tokens`: Baseten's server default output cap is 4096, and with thinking ON the
# reasoning trace alone consumes it, so hard turns come back empty (reasoning-starved)
# and even truncate mid-thought. 32000 gives reasoning room and still leaves budget
# for the tool call / answer. `top_p` is a first-class field on the OpenAI-compatible
# Baseten client, so it is passed top-level (unlike Fireworks, where it rides in
# `model_kwargs`).
_SPEC_INIT_DEFAULTS: dict[str, dict[str, Any]] = {
    "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": {
        "temperature": 0.6,
        "max_tokens": 32000,
        "top_p": 0.95,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    # NVIDIA's Nemotron 3 Ultra agentic-coding cookbook recommends these sampling
    # params. Without them the eval sends nothing and Fireworks runs at its server
    # default (~temperature 1.0, unbounded output), which drives exploratory
    # rambling / non-convergence on long agentic tasks. `temperature` and
    # `max_tokens` are first-class ChatFireworks fields; `top_p` is not, so it must
    # ride in `model_kwargs` (the integration would otherwise shunt it there with a
    # warning). Merged UNDER caller model_kwargs, so an explicit override still wins.
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4": {
        "temperature": 0.6,
        "max_tokens": 32000,
        "model_kwargs": {"top_p": 0.95},
    },
    # Fireworks dedicated deployment. Temperature lowered from the NVIDIA cookbook's
    # 0.6 to 0.3 to curb exploratory rambling / non-convergence (thrash) on long
    # agentic tasks; top_p and max_tokens keep the cookbook values.
    "fireworks:accounts/langchain-fireworks/deployments/nemotron-tb-test": {
        "temperature": 0.3,
        "max_tokens": 32000,
        "model_kwargs": {"top_p": 0.95},
    },
}


def _resolve_init_kwargs(model_name: str, caller_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge any per-spec default `init_chat_model` kwargs under the caller's.

    The caller's `configurable.model_kwargs` always win on conflict. `extra_body`
    is merged one level deep so a caller adding an unrelated `extra_body` key does
    not drop the spec default (and an explicit caller `extra_body` entry still
    overrides the default for that key). Does not mutate `caller_kwargs`.
    """
    defaults = _SPEC_INIT_DEFAULTS.get(model_name)
    if not defaults:
        return caller_kwargs
    merged: dict[str, Any] = {**defaults, **caller_kwargs}
    default_body = defaults.get("extra_body")
    caller_body = caller_kwargs.get("extra_body")
    if isinstance(default_body, dict) and isinstance(caller_body, dict):
        merged["extra_body"] = {**default_body, **caller_body}
    return merged


def _model_name(configurable: dict[str, object]) -> str:
    value = configurable.get("model") or os.environ.get("HARBOR_MODEL")
    if not isinstance(value, str) or not value.strip():
        msg = "`configurable.model` or `HARBOR_MODEL` must provide a model name"
        raise ValueError(msg)
    return value


def _workdir(configurable: dict[str, object]) -> Path:
    value = configurable.get("cwd")
    if value is not None and not isinstance(value, str | Path):
        msg = "`configurable.cwd` must be a string path"
        raise TypeError(msg)
    chosen = _DEFAULT_WORKDIR if value is None else Path(value)
    if chosen.is_dir():
        return chosen
    # The resolved workdir is absent from this task image. Harbor passes no cwd, so this is
    # usually the /app default, but some task images use /workspace instead — and spawning
    # shell commands in a missing cwd makes every command fail with FileNotFoundError,
    # stranding the agent. Fall back to the first directory that exists. Warn rather than
    # fail silently so the misconfiguration stays visible in the logs.
    fallback = next((c for c in _WORKDIR_FALLBACKS if c.is_dir()), Path("/"))
    logger.warning("configured workdir %s does not exist; falling back to %s", chosen, fallback)
    return fallback


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
    model_name = _model_name(configurable)
    model = init_chat_model(
        model_name, **_resolve_init_kwargs(model_name, _model_kwargs(configurable))
    )
    _raw_session = os.environ.get("HARBOR_SESSION_ID") or f"harbor-{uuid.uuid4()}"
    # Keep only characters dcode allows in an agent name.
    assistant_id = re.sub(r"[^A-Za-z0-9_\-\s]", "_", _raw_session)
    with _scrub_shell_env():
        graph, _backend = create_cli_agent(
            model=model,
            assistant_id=assistant_id,
            sandbox=None,
            sandbox_type="harbor",
            system_prompt=_SYSTEM_PROMPT,
            interactive=False,
            auto_approve=True,
            # No human is available in this headless eval, so `ask_user` can only
            # ever raise GraphInterrupt and kill the trial with empty output
            # (observed on Nemotron: it asks install-method permission / requests a
            # file it could read itself). Remove the tool entirely; a stray call to
            # a now-absent tool degrades to a recoverable unknown-tool error rather
            # than a run-ending interrupt.
            enable_ask_user=False,
            enable_memory=False,
            enable_skills=False,
            enable_shell=True,
            # Finalize + anti-ramble middleware now come from the GLM-5.2 harness
            # profile (deepagents.profiles.harness), applied for the fireworks
            # glm-5p2 model this harness runs.
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
    model_name = _model_name(configurable)
    model = init_chat_model(
        model_name, **_resolve_init_kwargs(model_name, _model_kwargs(configurable))
    )
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
    )


_TAU3_SYSTEM_PROMPT = """You are a customer-service agent in a Harbor benchmark, \
talking with a simulated user through the `tau3-runtime` MCP tools. Follow the \
task's policy exactly.

Protocol:
- Call `start_conversation` exactly once at the very start to begin (or resume) the
  conversation and read the user's first message.
- Call `send_message_to_user` to say anything to the user; it returns their next
  message.
- Use the domain tools (also on the `tau3-runtime` server) to inspect or modify the
  environment.
- In each step, either talk to the user OR call one domain tool — never both, and
  only one tool call at a time.
- When you are confident the case is resolved, end the conversation by calling
  `end_conversation` (or, if your agent emits stop tokens directly, reply
  `###STOP###`).

Unlike terminal tasks, there IS a user to talk to here: do not try to finish
silently. Keep working with the user until the case is resolved.
"""


def _mcp_connections(configurable: dict[str, object]) -> dict[str, Any]:
    """Build langchain-mcp-adapters connections from Harbor-forwarded servers.

    Harbor's LangGraph agent forwards the task environment's declared MCP servers
    via ``configurable["mcp_servers"]`` (a list of dicts shaped like Harbor's
    ``MCPServerConfig``: ``name``/``transport``/``url``/``command``/``args``). We
    connect only to those environment-declared servers, and only over remote
    transports.

    ``stdio`` servers are rejected on purpose: they carry a local ``command``/
    ``args`` that ``MultiServerMCPClient`` would execute inside the agent sandbox.
    Since the dataset (selectable via the workflow's ``dataset_override``) controls
    this config, honoring ``stdio`` would let an untrusted dataset run arbitrary
    commands in CI. tau3-runtime is a remote ``streamable-http`` server, so only
    ``streamable-http``/``sse`` (URL-based) transports are allowed.

    Args:
        configurable: The graph's ``configurable`` mapping.

    Returns:
        A mapping of server name to a langchain-mcp-adapters connection dict.

    Raises:
        ValueError: If no MCP servers were forwarded, a server uses an
            unsupported (e.g. ``stdio``) transport, or a server lacks a URL.
        TypeError: If ``mcp_servers`` is not a list of mappings.
    """
    servers = configurable.get("mcp_servers")
    if not servers:
        msg = (
            "tau3 graph requires MCP servers forwarded via "
            "`configurable['mcp_servers']`. Harbor's LangGraph agent must forward "
            "the task environment's MCP servers into the graph configurable; the "
            "pinned Harbor release does not yet do this, so run tau3 with a "
            "`harbor_package_override` that includes MCP-server forwarding until it "
            "ships in a release."
        )
        raise ValueError(msg)
    if not isinstance(servers, list):
        msg = "`configurable.mcp_servers` must be a list"
        raise TypeError(msg)

    connections: dict[str, Any] = {}
    for raw in servers:
        if not isinstance(raw, dict):
            msg = "Each entry in `configurable.mcp_servers` must be a mapping"
            raise TypeError(msg)
        server = cast("dict[str, Any]", raw)
        name = str(server["name"])
        transport = server.get("transport", "sse")
        if transport in ("streamable-http", "http"):
            transport = "streamable_http"
        if transport not in ("streamable_http", "sse"):
            msg = (
                f"MCP server {name!r} uses unsupported transport {transport!r}; the "
                "tau3 graph only allows remote transports (streamable-http, sse). "
                "stdio servers are rejected to avoid executing dataset-provided "
                "commands in the agent sandbox."
            )
            raise ValueError(msg)
        url = server.get("url")
        if not url:
            msg = f"MCP server {name!r} must declare a 'url' for transport {transport!r}"
            raise ValueError(msg)
        connections[name] = {"transport": transport, "url": url}
    return connections


async def make_tau3_graph(config: dict[str, object] | None = None) -> object:
    """Create a conversational Deep Agents graph for tau3-bench (and tau2) tasks.

    Unlike the terminal-bench graphs, this attaches the task environment's
    ``tau3-runtime`` MCP tools (``start_conversation``, ``send_message_to_user``,
    domain tools, ...) so the agent can converse with the simulated user. The MCP
    server connection comes from Harbor's forwarded ``configurable["mcp_servers"]``;
    no URL is hardcoded.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            ``configurable.model`` and the task's MCP servers in
            ``configurable.mcp_servers``.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name or MCP servers are provided.
    """
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    client = MultiServerMCPClient(_mcp_connections(configurable))
    tools = await client.get_tools()
    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=_TAU3_SYSTEM_PROMPT,
    )
