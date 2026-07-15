"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import os
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
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
    )


def make_bare_ci_graph(config: dict[str, object] | None = None) -> object:
    """``make_bare_graph`` with the code interpreter (``eval``) tool enabled.

    Identical to the bare graph — same generic system prompt and local shell
    backend — but with ``CodeInterpreterMiddleware`` attached, so the only
    difference from ``bare_deepagent`` is the code interpreter. Nothing about the
    prompt or tools is tailored to any dataset. ``langchain_quickjs`` is imported
    lazily so loading this module never requires QuickJS.

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
    from langchain_quickjs import CodeInterpreterMiddleware  # noqa: PLC0415

    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False)
    # Bound the interpreter's resource use (per-eval timeout + a ceiling on eval
    # calls) so a retry-spiral can't run away.
    middleware: list[Any] = [CodeInterpreterMiddleware(timeout=300.0, max_ptc_calls=12)]
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=_SYSTEM_PROMPT,
        middleware=middleware,
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
