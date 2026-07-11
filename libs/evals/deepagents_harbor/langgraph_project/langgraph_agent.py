"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from deepagents_code.config import detect_provider, settings
from deepagents_code.model_config import ModelSpec
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_WORKDIR = Path("/app")
_MAX_ASSISTANT_ID_LENGTH = 64
_ASSISTANT_ID_HASH_LENGTH = 12
_INVALID_ASSISTANT_ID_RUN = re.compile(r"[^A-Za-z0-9_-]+")

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


def _harbor_assistant_id(session_id: str | None) -> str:
    """Normalize Harbor's session ID for dcode's filesystem-backed agent ID."""
    if not session_id:
        return f"harbor-{uuid.uuid4()}"

    assistant_id = _INVALID_ASSISTANT_ID_RUN.sub("-", session_id)
    if _INVALID_ASSISTANT_ID_RUN.match(session_id):
        assistant_id = assistant_id.removeprefix("-")
    if _INVALID_ASSISTANT_ID_RUN.fullmatch(session_id[-1]):
        assistant_id = assistant_id.removesuffix("-")
    if not assistant_id:
        return f"harbor-{uuid.uuid4()}"
    if assistant_id == session_id and len(assistant_id) <= _MAX_ASSISTANT_ID_LENGTH:
        return assistant_id

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:_ASSISTANT_ID_HASH_LENGTH]
    prefix_length = _MAX_ASSISTANT_ID_LENGTH - _ASSISTANT_ID_HASH_LENGTH - 1
    return f"{assistant_id[:prefix_length]}-{digest}"


def _apply_model_identity(model_spec: str, model: object) -> None:
    """Populate dcode `settings` model identity from the selected model.

    `create_cli_agent` -> `get_system_prompt` builds the prompt's
    `### Model Identity` section from the global dcode `settings` singleton
    (`model_name`, `model_provider`, `model_context_limit`,
    `model_unsupported_modalities`). Harbor builds the model itself via
    `init_chat_model` and never touches those settings, so without this the
    identity section renders empty and the eval agent never learns which model
    it is. We set them here from Harbor's `configurable.model` spec plus the
    model's resolved profile, mirroring the extraction
    `deepagents_code.config.create_model` performs for the real CLI.

    This mutates a process-level singleton; tests must snapshot/restore it (see
    the autouse fixture in the unit tests).

    Args:
        model_spec: The model spec from `configurable.model` / `HARBOR_MODEL`,
            e.g. `"anthropic:claude-sonnet-4-5"` or a bare `"claude-sonnet-4-5"`.
        model: The instantiated chat model (read for its `.profile`).
    """
    parsed = ModelSpec.try_parse(model_spec)
    if parsed is not None:
        provider, name = parsed.provider, parsed.model
    else:
        name = model_spec.lstrip(":")
        provider = detect_provider(name) or ""

    settings.model_name = name
    settings.model_provider = provider

    # Mirror create_model: pull context window + unsupported input modalities
    # from the model profile when the provider exposes one.
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        max_input = profile.get("max_input_tokens")
        settings.model_context_limit = max_input if isinstance(max_input, int) else None
        modality_keys = {
            "image_inputs": "image",
            "audio_inputs": "audio",
            "video_inputs": "video",
            "pdf_inputs": "pdf",
        }
        settings.model_unsupported_modalities = frozenset(
            label for key, label in modality_keys.items() if profile.get(key) is False
        )


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
    model_spec = _model_name(configurable)
    model_kwargs = _model_kwargs(configurable)
    # Experiment (for now): pin GLM-5.2's reasoning effort to "max" for the eval.
    # Fireworks GLM takes this as a nested `model_kwargs={"reasoning_effort": ...}`
    # on the model constructor (see dcode `reasoning_effort._fireworks_model_params`).
    # Gated to GLM-5.2 so the shared harness is unaffected for other models; an
    # explicit `configurable.model_kwargs` reasoning_effort still wins.
    if "glm-5p2" in model_spec or "glm-5.2" in model_spec:
        nested = model_kwargs.setdefault("model_kwargs", {})
        if isinstance(nested, dict):
            nested.setdefault("reasoning_effort", "max")
    model = init_chat_model(model_spec, **model_kwargs)
    # Feed the selected model into dcode's system-prompt `### Model Identity`
    # section (create_cli_agent -> get_system_prompt reads it from `settings`).
    _apply_model_identity(model_spec, model)
    assistant_id = _harbor_assistant_id(os.environ.get("HARBOR_SESSION_ID"))
    with _scrub_shell_env():
        # Do not pass `system_prompt`: leaving it unset makes `create_cli_agent`
        # build the real Deep Agents Code (dcode) production system prompt via
        # `get_system_prompt(interactive=False, cwd=...)`. Overriding it would mean
        # the CLI-harness eval never exercises the dcode system prompt we ship. The
        # sandbox/headless/workdir guidance the old override hand-rolled is already
        # covered by the generated headless prompt.
        #
        # Do not pass `sandbox_type` either. We run locally (`sandbox=None`) on a
        # shell backend rooted at Harbor's `cwd`, so the local-mode prompt (rooted
        # at `cwd`) is the accurate description. A non-None `sandbox_type` would
        # route `get_system_prompt` through `get_default_working_dir(sandbox_type)`,
        # which raises `ValueError` for any provider not in dcode's sandbox registry
        # (e.g. "harbor", which is not a registered provider).
        graph, _backend = create_cli_agent(
            model=model,
            assistant_id=assistant_id,
            sandbox=None,
            interactive=False,
            auto_approve=True,
            enable_ask_user=False,
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
