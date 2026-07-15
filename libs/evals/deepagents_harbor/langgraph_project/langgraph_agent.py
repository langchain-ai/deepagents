"""LangGraph entrypoint for running Deep Agents under Harbor."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents_code.agent import create_cli_agent
from deepagents_code.config import detect_provider, settings
from deepagents_code.model_config import ModelSpec
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages.utils import get_buffer_string
from langchain_core.tools import BaseTool, tool
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
    model = init_chat_model(model_spec, **_model_kwargs(configurable))
    # Feed the selected model into dcode's system-prompt `### Model Identity`
    # section (create_cli_agent -> get_system_prompt reads it from `settings`).
    _apply_model_identity(model_spec, model)
    assistant_id = os.environ.get("HARBOR_SESSION_ID") or f"harbor-{uuid.uuid4()}"
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


def _head_tail(text: str, limit: int = 10000) -> str:
    """Truncate long tool output to ``limit`` chars, keeping the head and tail.

    Coding output (build logs, disassembly, file dumps) is often huge; capping what
    the agent sees per call keeps context small. Combined with the prompt guidance to
    save expensive output to a file and slice it, this avoids re-injecting large dumps.
    """
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return (
        f"{text[:half]}\n"
        f"[... {omitted} chars omitted; output limited to {limit} chars, "
        f"redirect to a file and read slices ...]\n"
        f"{text[-half:]}"
    )


def _make_execute_tool(backend: LocalShellBackend) -> BaseTool:
    """Build a single shell-execution tool backed by ``backend``.

    The returned tool runs a command in the sandbox and returns the combined
    stdout/stderr (capped head/tail), appending a truncation note and a non-zero exit
    code when present.
    """

    @tool
    def execute(command: str, timeout: int | None = None) -> str:
        """Run a shell command in the sandbox and return its output.

        Chain a sequence with `&&`/`;` to run it in one call. Output is capped, so
        redirect large output to a file and read slices instead of dumping it.
        """
        result = backend.execute(command, timeout=timeout)
        output = _head_tail(result.output)
        if result.truncated:
            output += "\n[output truncated]"
        if result.exit_code:
            output += f"\n[exit code: {result.exit_code}]"
        return output

    return execute


def make_minimal_graph(config: dict[str, object] | None = None) -> object:
    """Create the minimal coding agent.

    A deliberately lean coding agent built directly on LangChain's ``create_agent``,
    bypassing ``create_deep_agent`` and all of its middleware and built-in tools. It
    has a concise coding system prompt (explore, plan, iterate against the task's own
    tests/examples, verify before finishing) and a small tool surface: a one-shot
    ``execute`` plus a background-process set (``run_background``/``poll``/
    ``write_stdin``/``kill``) so long-running or interactive commands are polled
    instead of blocking to timeout. No filesystem or multimodal tools, so it avoids
    that failure surface. This is the base we build up from, kept minimal on purpose.

    Args:
        config: LangGraph runtime config. Harbor passes the selected model in
            ``configurable.model`` and optional provider kwargs in
            ``configurable.model_kwargs``.

    Returns:
        A compiled LangGraph graph invokable by Harbor's LangGraph runner.

    Raises:
        TypeError: If configurable values have unexpected types.
        ValueError: If no model name is provided.
    """
    configurable = _configurable(config)
    model = init_chat_model(_model_name(configurable), **_model_kwargs(configurable))
    workdir = _workdir(configurable)
    backend = LocalShellBackend(root_dir=workdir, inherit_env=False)
    manager = _ProcessManager(cwd=workdir, env=backend._env)  # noqa: SLF001  # reuse execute's scrubbed env
    tools = [_make_execute_tool(backend), *_make_background_tools(manager)]
    # Context management: terminus-2-style retention summarization. It fires proactively
    # at the token trigger and, rather than eliding tool outputs (which caused a re-fetch
    # thrash loop), drafts a summary then critiques and answers what it is missing, so the
    # file contents, errors, and reverse-engineering constants the agent needs survive the
    # compaction. Prompt caching no-ops for non-Anthropic models.
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=CODING_SYSTEM_PROMPT,
        middleware=[
            _TerminusSummarizationMiddleware(
                model,
                trigger=("tokens", 50_000),
                keep=("messages", 20),
                summary_prompt=CODING_SUMMARY_PROMPT,
            ),
            AnthropicPromptCachingMiddleware(),
        ],
    )


CODING_SYSTEM_PROMPT = """You are an autonomous coding agent working in a sandbox at /app. \
There is no human available to answer questions, so make reasonable assumptions and keep working \
until the task is fully solved.

Approach every task like this:
1. Explore first. List and read the task's files, and look for anything you can check yourself \
against: a provided test script, an example input with a known expected output, a reference \
binary or program, or worked examples in the task description. Treat these as your ground truth.
2. Plan, then implement. Prefer writing a script or generator to produce the required artifact \
rather than hand-writing it.
3. Verify before you finish. Run the task's own tests/examples and iterate against concrete \
failures until they pass. Do not declare the task done until you have actually observed your \
solution meeting the acceptance criteria. When a check fails, read the error, fix the root cause, \
and re-run.

Running commands:
- Use `execute` for quick commands that finish within a few seconds.
- For anything long-running or interactive (compiles, builds, test suites, servers, REPLs, a \
program that renders or runs for a while), use `run_background` to start it, then `poll` it in \
reasonable increments (wait 20-60 seconds per poll, not many tiny polls) to watch its output \
instead of blocking on a single command. Use `write_stdin` to feed input to an interactive \
process and `kill` to stop one. This lets you observe partial progress and never hang waiting \
for a command to finish.
- Prefer non-interactive flags; never run a command that silently waits for human input without \
driving it with `write_stdin`.
- Batch a natural sequence into ONE `execute` call with `&&` or `;` (for example \
`gcc -o p p.c && ./p && python check.py`) instead of one command per turn. Fewer, batched turns \
are dramatically faster.
- Run an expensive command once and save its output to a file (for example \
`objdump -d bin > /tmp/d.asm`), then read slices with `grep`/`sed`/`head`. Tool output is capped, \
so do not re-run the same command or re-read the same large file: cache it once and slice it.

If the task forbids reading a file (for example, reproduce an image without reading it), do NOT \
read or even sample that file: it wastes turns and violates the task. Instead commit fully to \
reverse-engineering any provided reference program or binary as your ground truth: disassemble \
it, decode its constants, reconstruct the algorithm, and verify by matching its output exactly. \
Pick one strategy and drive it to completion rather than oscillating between approaches.

Build artifacts incrementally, and never paste large file contents, data tables, or generated \
code into your message. Write files with a heredoc (`cat > f <<'EOF' ... EOF`) or a generator \
script, and build in pieces, compiling and running to check as you go. One-shot mega-outputs are \
slow, hit output limits, and are hard to debug.

Be mindful of your time budget. Do not loop the same failing approach: if something is not \
working after a couple of attempts, step back and commit to a genuinely different strategy rather \
than burning the whole budget repeating a dead end.

Before you declare the task complete, run the task's own tests or verifier ONE more time and \
confirm you have actually observed them pass. Keep going until then; do not stop early.
"""


CODING_SUMMARY_PROMPT = """You are compacting the working context of a coding agent. The \
conversation below will be REPLACED by your output, so preserve exactly what is needed to keep \
coding without redoing work. Be concise but do not drop concrete details.

Fill in each section (write "None" if empty):

## GOAL
The task objective and its acceptance criteria.

## FILES
Every file created or modified: its path plus current contents or a precise description of its \
state, enough to continue editing or recreate it.

## VERIFICATION STATE
Which of the task's tests/examples you have run, the latest result (pass or fail), and the exact \
error messages or diffs from the most recent failure.

## APPROACH
The current strategy, what you have tried, and approaches ruled out and why. Record key facts \
discovered (constants, file locations, commands that work).

## NEXT STEPS
The specific next actions, including the exact commands to run.

Respond ONLY with the sections above.

<messages>
{messages}
</messages>"""


_SUMMARY_CRITIQUE_PROMPT = """A coding agent's full history will be replaced by the draft summary \
below. List 5 to 10 specific questions about concrete details a coding agent needs to continue \
WITHOUT redoing work that the draft may be missing: exact file paths and current contents, the \
precise error from the last failing test, key constants or values discovered, the exact commands \
that work, and the current build/verification state. Output only the numbered questions.

DRAFT SUMMARY:
{summary}"""


_SUMMARY_ANSWER_PROMPT = """Answer each question below concisely and concretely using ONLY the \
conversation history provided. Preserve exact paths, constants, commands, and error text. If the \
history does not contain the answer, write "unknown".

QUESTIONS:
{questions}

<messages>
{messages}
</messages>"""


class _TerminusSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware with a terminus-2-style retention handoff.

    Instead of a single-pass extract, it (1) drafts the base summary, (2) asks the
    model which concrete details the draft is missing, and (3) answers those questions
    from the full history, appending the answers. This preserves the file contents,
    error text, and reverse-engineering constants a coding agent needs so it does not
    have to re-fetch them after a compaction. The extra passes degrade gracefully to
    the draft on any error, and reuse the base trigger/keep/persistence machinery.
    """

    _MAX_SUMMARY_CHARS = 12000

    def _augment(self, draft: str, questions: str, answers: str) -> str:
        return f"{draft}\n\n## ADDITIONAL DETAILS\n{answers}"[: self._MAX_SUMMARY_CHARS]

    def _create_summary(self, messages_to_summarize: list[Any]) -> str:
        draft = super()._create_summary(messages_to_summarize)
        try:
            questions = self.model.invoke(
                _SUMMARY_CRITIQUE_PROMPT.format(summary=draft),
                config={"metadata": {"lc_source": "summarization-critique"}},
            ).text.strip()
            formatted = get_buffer_string(
                self._trim_messages_for_summary(messages_to_summarize), format="xml"
            )
            answers = self.model.invoke(
                _SUMMARY_ANSWER_PROMPT.format(questions=questions, messages=formatted),
                config={"metadata": {"lc_source": "summarization-answer"}},
            ).text.strip()
        except Exception:  # noqa: BLE001  # graceful degradation to the base draft
            return draft
        return self._augment(draft, questions, answers)

    async def _acreate_summary(self, messages_to_summarize: list[Any]) -> str:
        draft = await super()._acreate_summary(messages_to_summarize)
        try:
            questions = (
                await self.model.ainvoke(
                    _SUMMARY_CRITIQUE_PROMPT.format(summary=draft),
                    config={"metadata": {"lc_source": "summarization-critique"}},
                )
            ).text.strip()
            formatted = get_buffer_string(
                self._trim_messages_for_summary(messages_to_summarize), format="xml"
            )
            answers = (
                await self.model.ainvoke(
                    _SUMMARY_ANSWER_PROMPT.format(questions=questions, messages=formatted),
                    config={"metadata": {"lc_source": "summarization-answer"}},
                )
            ).text.strip()
        except Exception:  # noqa: BLE001  # graceful degradation to the base draft
            return draft
        return self._augment(draft, questions, answers)


class _ProcessManager:
    """Track long-running background shell processes for the coding agent.

    Each process runs in the sandbox workdir with the same key-scrubbed env the
    one-shot `execute` tool uses. A daemon thread drains combined stdout/stderr into
    a buffer so `poll` returns incremental output without blocking. Processes are
    daemon-threaded and die with the ephemeral sandbox; `kill` stops one explicitly.
    """

    def __init__(self, cwd: Path, env: dict[str, str]) -> None:
        self._cwd = cwd
        self._env = env
        self._procs: dict[str, dict[str, Any]] = {}
        self._count = 0

    def start(self, command: str) -> str:
        self._count += 1
        handle = f"proc-{self._count}"
        proc = subprocess.Popen(  # noqa: S602  # shell execution is the tool's purpose
            command,
            shell=True,
            cwd=str(self._cwd),
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        rec: dict[str, Any] = {"proc": proc, "buf": [], "read": 0, "lock": threading.Lock()}

        def _drain() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                with rec["lock"]:
                    rec["buf"].append(line)

        threading.Thread(target=_drain, daemon=True).start()
        self._procs[handle] = rec
        return handle

    def poll(self, handle: str, wait_seconds: float) -> str:
        rec = self._procs.get(handle)
        if rec is None:
            return f"[unknown handle {handle!r}]"
        proc = rec["proc"]
        deadline = time.monotonic() + max(0.0, min(wait_seconds, 60.0))
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.5)
        with rec["lock"]:
            new = "".join(rec["buf"][rec["read"] :])
            rec["read"] = len(rec["buf"])
        new = _head_tail(new)
        status = "still running" if proc.poll() is None else f"exited with code {proc.returncode}"
        return f"[{handle}: {status}]\n{new}" if new else f"[{handle}: {status}] (no new output)"

    def write_stdin(self, handle: str, text: str) -> str:
        rec = self._procs.get(handle)
        if rec is None:
            return f"[unknown handle {handle!r}]"
        proc = rec["proc"]
        if proc.stdin is None or proc.poll() is not None:
            return f"[{handle}: process is not accepting input]"
        proc.stdin.write(text if text.endswith("\n") else text + "\n")
        proc.stdin.flush()
        return f"[{handle}: sent input]"

    def kill(self, handle: str) -> str:
        rec = self._procs.get(handle)
        if rec is None:
            return f"[unknown handle {handle!r}]"
        rec["proc"].terminate()
        return f"[{handle}: terminated]"


def _make_background_tools(manager: _ProcessManager) -> list[BaseTool]:
    """Build the background-process tool set (run_background/poll/write_stdin/kill)."""

    @tool
    def run_background(command: str) -> str:
        """Start a long-running or interactive shell command in the background.

        Returns a handle. Use `poll` to read its output and status, `write_stdin` to
        send it input, and `kill` to stop it. Use this instead of `execute` for
        compiles, builds, test suites, servers, or anything that runs more than a few
        seconds, so you can watch progress instead of blocking.
        """
        return manager.start(command)

    @tool
    def poll(handle: str, wait_seconds: int = 30) -> str:
        """Wait up to `wait_seconds` (max 60) for a background process, then return new output and status."""
        return manager.poll(handle, wait_seconds)

    @tool
    def write_stdin(handle: str, text: str) -> str:
        """Send a line of input to a background process's stdin."""
        return manager.write_stdin(handle, text)

    @tool
    def kill(handle: str) -> str:
        """Terminate a background process by handle."""
        return manager.kill(handle)

    return [run_background, poll, write_stdin, kill]


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
