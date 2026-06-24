"""Deep Agents system adapter for continual-learning-bench.

Wraps a LangChain Deep Agent (``deepagents``) as a
:class:`ContinualLearningSystem`, using the agent's **own** memory mechanism as
the continual-learning substrate:

* ``MemoryMiddleware`` (enabled via ``create_deep_agent(memory=...)``) loads
  ``/memory/AGENTS.md`` into the system prompt at the start of every turn,
  wrapped in ``<agent_memory>`` boundary markers and treated as untrusted data.
* The agent itself distils and updates that file with its built-in
  ``edit_file`` / ``write_file`` tools as it learns. There is no separate
  reflection/extraction process — the agent owns its memory.

The memory file lives in the in-state filesystem (``DeepAgentState["files"]``);
this adapter threads it from one ``respond()`` call to the next, which is what
carries learning across instances. ``reset()`` wipes it, so the framework's
stateless baseline is genuinely stateless and ``mean_gain`` measures only what
the agent learned.

Structured output uses ``create_deep_agent(response_format=...)``: the agent
emits the task's per-turn schema natively (read from ``structured_response``),
so there is no separate extraction call either. One model interaction per turn.

The default in-state ``StateBackend`` gives the agent no host shell/filesystem,
so there is no avenue to read provider credentials from the environment.

NOTE: This module targets continual-learning-bench's package layout
(``src.interface`` / ``src.registry`` / ``src.usage``) and runs only once
deployed into a clbench checkout under ``src/systems/deepagents/``. See this
directory's README and ``sync_to_clbench.sh``.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from deepagents import create_deep_agent

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
)
from ...registry import register_system
from ...usage import UsageEvent

# The agent's own durable notes, loaded into the prompt every turn and updated
# by the agent via edit_file. Single source of truth for what it has learned.
_AGENT_MEMORY_PATH = "/memory/AGENTS.md"
_MEMORY_SOURCES = [_AGENT_MEMORY_PATH]

_SEED_AGENTS_MD = "# Strategy notes\n\n(empty - update this as you learn)\n"

_SYSTEM_PROMPT = f"""\
You are being evaluated on a continual-learning benchmark: a sequence of \
related instances in a shared environment. You are scored on how much you \
improve as you learn from earlier instances.

Your durable strategy lives in {_AGENT_MEMORY_PATH}, which is loaded into your \
context every turn. As you discover what works in this environment, keep that \
file up to date with the edit_file/write_file tools: record concise, \
generalizable lessons (tendencies to exploit, what worked, what to avoid) and \
prune anything you find to be wrong. It is the only thing that carries into \
the next instance, so invest in it. Never store secrets or credentials.

When you are given feedback on a previous action, use it to update your notes \
before you act again.\
"""


def _file_data(content: str) -> dict[str, str]:
    """Build an in-state FileData record (see deepagents.backends.protocol)."""
    return {"content": content, "encoding": "utf-8"}


def _read_file_data(files: dict[str, Any], path: str) -> str:
    """Return the text content of an in-state file, or '' if absent."""
    entry = files.get(path)
    if isinstance(entry, dict):
        return str(entry.get("content", ""))
    return ""


@register_system("deepagents")
class DeepAgentsSystem(ContinualLearningSystem):
    """A Deep Agent evaluated as a continual-learning system.

    The agent maintains its own memory (``/memory/AGENTS.md`` in the in-state
    filesystem); the adapter just threads that filesystem across instances.
    """

    supports_baseline = True
    parallel_safe = True  # in-memory state only; no fixed host paths or ports.

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        name: str = "deepagents",
    ) -> None:
        """
        Args:
            model: ``provider:model`` string passed to ``init_chat_model``.
            name: System identifier surfaced in results and viewers.
        """
        self._name = name
        self._model_name = model
        self._model = init_chat_model(model)
        # Agents are cached per response schema (rebuilt only when the task's
        # schema changes). They hold no learned state — memory lives in _files.
        self._agents: dict[type[BaseModel], Any] = {}
        # The agent's persistent memory, threaded across respond() calls.
        self._files: dict[str, Any] = {}
        self._pending_feedback: str | None = None
        self.interaction_count = 0
        self._seed_memory()

    def _seed_memory(self) -> None:
        """Reset memory to empty scaffolding (no learned content)."""
        self._files = {_AGENT_MEMORY_PATH: _file_data(_SEED_AGENTS_MD)}

    def _get_agent(self, schema: type[BaseModel]) -> Any:
        """Return a deep agent that emits ``schema`` as its structured response."""
        agent = self._agents.get(schema)
        if agent is None:
            agent = create_deep_agent(
                model=self._model,
                system_prompt=_SYSTEM_PROMPT,
                memory=_MEMORY_SOURCES,
                response_format=schema,
            )
            self._agents[schema] = agent
        return agent

    def _memory_snapshot(self) -> dict[str, str]:
        """Current memory as a ``{path: content}`` dict (the shape clbench logs)."""
        return {path: _read_file_data(self._files, path) for path in _MEMORY_SOURCES}

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1

        # Surface feedback so the agent can update its own notes before acting.
        feedback = None
        if query.feedback is not None and query.feedback.content.strip():
            feedback = query.feedback.content.strip()
        elif self._pending_feedback:
            feedback = self._pending_feedback
        self._pending_feedback = None

        prompt = query.prompt or "(no content)"
        if feedback:
            prompt = f"Feedback on your previous action:\n{feedback}\n\n{prompt}"

        # One model interaction: the agent reads its notes, optionally updates
        # them via edit_file, and emits the structured action.
        agent = self._get_agent(query.response_schema)
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": prompt}],
                "files": self._files,
            }
        )
        # Thread the (possibly agent-updated) memory filesystem forward.
        self._files = result.get("files", self._files)
        self._record_usage(result.get("messages", []))

        action = result.get("structured_response")
        if not isinstance(action, BaseModel):
            raise RuntimeError(
                "Deep agent did not return a structured response matching "
                f"{query.response_schema.__name__}."
            )

        return Response(
            action=action,
            metadata={
                "system": "deepagents",
                "model": self._model_name,
                "interaction": self.interaction_count,
                # clbench records this per step (path -> content) for the viewer.
                "memory_files": self._memory_snapshot(),
            },
        )

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        """Capture the outcome so the next turn's prompt can surface it.

        No model call and no file write here — distilling the outcome into
        memory is the agent's job on its next turn.
        """
        content = observation.content.strip()
        if content:
            self._pending_feedback = content

    def reset(self) -> None:
        """Wipe learned memory.

        Called once at the start of a stateful rollout, and before *every*
        instance in the stateless baseline.
        """
        self._seed_memory()
        self._pending_feedback = None
        self.interaction_count = 0

    @property
    def name(self) -> str:
        return self._name

    def get_run_artifacts(self) -> dict[str, Any]:
        """Export final memory so the viewer can show what the agent learned.

        ``memory_files`` is the key clbench's trace storage reads (path -> content).
        """
        return {
            "artifact_type": "deepagents",
            "model": self._model_name,
            "interaction_count": self.interaction_count,
            "memory_files": self._memory_snapshot(),
        }

    def _record_usage(self, messages: list[Any]) -> None:
        """Aggregate token usage from message ``usage_metadata`` into a UsageEvent."""
        input_tokens = 0
        output_tokens = 0
        seen = False
        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue
            seen = True
            input_tokens += int(usage.get("input_tokens", 0) or 0)
            output_tokens += int(usage.get("output_tokens", 0) or 0)
        if not seen:
            return
        self.record_usage_event(
            UsageEvent(
                call_type="completion",
                model=self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )
