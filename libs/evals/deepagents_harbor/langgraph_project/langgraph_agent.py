"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code._glm_5p2_profile import _GLM_5P2_MODEL_SPECS
from deepagents_code.agent import create_cli_agent
from deepagents_code.config import detect_provider, settings
from deepagents_code.model_config import ModelSpec
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_DEFAULT_WORKDIR = Path("/app")
_MAX_ASSISTANT_ID_LENGTH = 64
_ASSISTANT_ID_HASH_LENGTH = 12
_INVALID_ASSISTANT_ID_RUN = re.compile(r"[^A-Za-z0-9_-]+")
# Single source of truth for which specs are GLM-5.2: reuse dcode's exact,
# case-sensitive spec set so this eval default and dcode's prompt profile apply
# to precisely the same specs. Matching case-insensitively here would re-bump
# reasoning for a spec dcode does not classify as GLM — the divergence we avoid.
_GLM_5_2_MODEL_SPECS = frozenset(_GLM_5P2_MODEL_SPECS)

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


def _apply_glm_5_2_reasoning_default(model_spec: str, model_kwargs: dict[str, Any]) -> None:
    """Default GLM-5.2's reasoning effort to `"high"` for the eval when unset.

    Experiment (for now). Fireworks GLM takes this as a nested
    `model_kwargs={"reasoning_effort": ...}` on the model constructor (see
    dcode `reasoning_effort._fireworks_model_params`). Gated case-sensitively
    to dcode's exact GLM-5.2 profile specs so the shared harness is unaffected
    for other models and this default fires on precisely the specs dcode also
    guards; an explicit provider-native `reasoning`/`reasoning_effort` still
    wins.
    """
    if model_spec not in _GLM_5_2_MODEL_SPECS:
        return
    if "reasoning_effort" in model_kwargs or "reasoning" in model_kwargs:
        return

    nested = model_kwargs.get("model_kwargs")
    if nested is None and "model_kwargs" not in model_kwargs:
        nested = {}
        model_kwargs["model_kwargs"] = nested
    if not isinstance(nested, dict):
        return
    if "reasoning_effort" in nested or "reasoning" in nested:
        return
    nested["reasoning_effort"] = "high"


def _build_model(configurable: dict[str, object]) -> BaseChatModel:
    """Build the chat model, applying provider-specific eval defaults.

    OpenAI gates `reasoning_effort` + function tools to `/v1/responses` for
    gpt-5.x, and its model profile defaults `reasoning_effort`. The model is
    built here directly via `init_chat_model`, which bypasses the Deep Agents
    OpenAI provider profile that would set `use_responses_api=True`, so set it
    explicitly for `openai:` models.

    The GLM-5.2 eval profiles additionally default `reasoning_effort` to
    `"high"` via `_apply_glm_5_2_reasoning_default`. A caller-supplied
    `model_kwargs` value still wins in both cases.
    """
    name = _model_name(configurable)
    kwargs = _model_kwargs(configurable)
    if name.startswith("openai:") and "use_responses_api" not in kwargs:
        kwargs["use_responses_api"] = True
    _apply_glm_5_2_reasoning_default(name, kwargs)
    return init_chat_model(name, **kwargs)


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
    settings.model_context_limit = None
    settings.model_unsupported_modalities = frozenset()

    # Mirror create_model: pull context window + unsupported input modalities
    # from the model profile when the provider exposes one.
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        # No usable profile: the identity section renders with no context window
        # and no modality restrictions. Warn (not debug) so an eval running with
        # a degraded identity is attributable — harbor configures no logging, so
        # a debug record would be dropped at the default root level and the
        # intended attribution would never reach the operator.
        logger.warning(
            "Model %r exposes no profile dict; Model Identity will omit the "
            "context limit and unsupported modalities",
            name,
        )
        return

    max_input = profile.get("max_input_tokens")
    if isinstance(max_input, int):
        settings.model_context_limit = max_input
    else:
        # A profile that is present but lacks a usable context window is an
        # unexpected shape (e.g. a renamed key); surface it rather than silently
        # coercing to None with no signal.
        logger.warning(
            "Model %r profile has no usable 'max_input_tokens' (%r); Model "
            "Identity will omit the context limit",
            name,
            max_input,
        )
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
    model = _build_model(configurable)
    # Feed the selected model into dcode's system-prompt `### Model Identity`
    # section (create_cli_agent -> get_system_prompt reads it from `settings`).
    _apply_model_identity(_model_name(configurable), model)
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
    model = _build_model(configurable)
    # DIAGNOSTIC (throwaway): pin virtual_mode=False to match canonical (1d1db9c).
    # Main flipped the LocalShellBackend default to True; the adapter never set it,
    # so main silently virtualizes sandbox paths where canonical used them as-is.
    # Testing whether that flip drives the context-retrieval regression.
    backend = LocalShellBackend(root_dir=_workdir(configurable), inherit_env=False, virtual_mode=False)
    # No `system_prompt`: keep the bare agent on `create_deep_agent`'s
    # prompt-free default. The sandbox workdir is already enforced by the shell
    # backend's `root_dir`.
    return create_deep_agent(
        model=model,
        backend=backend,
    )


def _mcp_connections(configurable: dict[str, object]) -> dict[str, Any]:
    """Build langchain-mcp-adapters connections from Harbor-forwarded servers.

    Harbor's LangGraph agent forwards the task environment's declared MCP servers
    via `configurable["mcp_servers"]` (a list of dicts shaped like Harbor's
    `MCPServerConfig`: `name`/`transport`/`url`/`command`/`args`). We
    connect only to those environment-declared servers, and only over remote
    transports.

    `stdio` servers are rejected on purpose: they carry a local `command`/
    `args` that `MultiServerMCPClient` would execute inside the agent sandbox.
    Since the dataset (selectable via the workflow's `dataset_override`) controls
    this config, honoring `stdio` would let an untrusted dataset run arbitrary
    commands in CI. tau3-runtime is a remote `streamable-http` server, so only
    `streamable-http`/`sse` (URL-based) transports are allowed.

    Args:
        configurable: The graph's `configurable` mapping.

    Returns:
        A mapping of server name to a langchain-mcp-adapters connection dict.

    Raises:
        ValueError: If no MCP servers were forwarded, a server uses an
            unsupported (e.g. `stdio`) transport, or a server lacks a URL.
        TypeError: If `mcp_servers` is not a list of mappings.
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
    `tau3-runtime` MCP tools (`start_conversation`, `send_message_to_user`,
    domain tools, ...) so the agent can converse with the simulated user. The MCP
    server connection comes from Harbor's forwarded `configurable["mcp_servers"]`;
    no URL is hardcoded.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            `configurable.model` and the task's MCP servers in
            `configurable.mcp_servers`.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name or MCP servers are provided.
    """
    configurable = _configurable(config)
    model = _build_model(configurable)
    client = MultiServerMCPClient(_mcp_connections(configurable))
    tools = await client.get_tools()
    # No `system_prompt`: the tau3-runtime conversation protocol comes from the
    # MCP tools' server-advertised descriptions, without adding an authored base
    # prompt.
    return create_deep_agent(
        model=model,
        tools=tools,
    )
