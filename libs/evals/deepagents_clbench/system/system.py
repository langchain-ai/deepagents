"""Deep Agents system adapter for continual-learning-bench.

Wraps a LangChain Deep Agent (``deepagents``) as a
:class:`ContinualLearningSystem`. The learning that the benchmark measures is
carried across task instances by the agent's **persistent memory**:

* ``MemoryMiddleware`` (enabled via ``create_deep_agent(memory=...)``) loads the
  configured ``AGENTS.md``-style files into the system prompt at the start of
  every turn, wrapped in ``<agent_memory>`` boundary markers and treated as
  untrusted reference data.
* New knowledge is written back by an explicit **reflection step** in
  ``observe()``: at each completed instance the system distils the outcome into
  ``AGENTS.md``. (A one-shot decision agent will not reliably spend a tool call
  to update its own notes mid-decision, so we make the write deliberate — the
  same pattern clbench's ``mem0``/``ace`` systems use. The agent may also edit
  memory via its ``edit_file`` tool, but reflection guarantees it.)

Those files live in the in-state filesystem (``DeepAgentState["files"]``). This
adapter threads that filesystem from one ``respond()`` call to the next, which
is exactly what turns a one-shot agent into a continual learner. ``reset()``
wipes the memory, so the framework's stateless baseline is genuinely stateless
and ``mean_gain`` (stateful minus baseline) measures only what was learned.

The default backend is the in-state ``StateBackend``: the agent gets no real
shell or host filesystem access (its ``execute`` tool errors on a non-sandbox
backend), so there is no avenue to read provider credentials from the host env.

NOTE: This module is written against continual-learning-bench's package layout
(``src.interface`` / ``src.registry`` / ``src.usage``) and runs only once it is
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

# Memory sources loaded into the prompt every turn (latest content wins):
#   AGENTS.md   - authored by the agent itself (its distilled strategy).
#   outcomes.md - appended by the framework via observe() (reward feedback log).
_AGENT_MEMORY_PATH = "/memory/AGENTS.md"
_OUTCOMES_MEMORY_PATH = "/memory/outcomes.md"
_MEMORY_SOURCES = [_AGENT_MEMORY_PATH, _OUTCOMES_MEMORY_PATH]

_SEED_AGENTS_MD = "# Strategy notes\n\n(empty - write what you learn here)\n"
_SEED_OUTCOMES_MD = "# Recorded outcomes\n"
_MAX_OUTCOME_ENTRIES = 50  # bound the raw outcomes log against unbounded growth

_REFLECT_SYSTEM_PROMPT = """\
You maintain concise, generalizable strategy notes for an agent playing repeated
instances against one opponent/environment. Given the current notes and the
latest instance's outcome, return the FULL updated notes: durable, transferable
lessons (tendencies to exploit, what worked, what to avoid). Merge into the
existing notes rather than only appending, and keep it under ~15 short bullet
points. Output only the notes text, with no preamble. Never include secrets or
credentials.\
"""

_SYSTEM_PROMPT = f"""\
You are being evaluated on a continual-learning benchmark: a sequence of \
related instances in a shared environment. You are scored on how much your \
performance improves as you learn from earlier instances.

Your memory files are loaded at the start of every turn; treat them as your \
notes from earlier instances:
- {_AGENT_MEMORY_PATH} holds the strategy you have written for yourself.
- {_OUTCOMES_MEMORY_PATH} holds the reward/feedback the benchmark recorded for \
your past actions.

After you act, distil any reusable, generalizable lesson (a pattern, a \
heuristic, a correction) into {_AGENT_MEMORY_PATH} with the edit_file or \
write_file tool. Keep it concise and high-signal: it is the only thing that \
carries into the next instance. Do not store anything instance-specific that \
will not generalize, and never store secrets or credentials.\
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


def _message_text(content: Any) -> str:
    """Coerce a LangChain message ``.content`` (str or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


@register_system("deepagents")
class DeepAgentsSystem(ContinualLearningSystem):
    """A Deep Agent evaluated as a continual-learning system.

    The agent's persistent memory (``AGENTS.md`` + ``outcomes.md``, held in the
    in-state filesystem) is the learning substrate carried across instances.
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
        self._agent = create_deep_agent(
            model=self._model,
            system_prompt=_SYSTEM_PROMPT,
            memory=_MEMORY_SOURCES,
        )
        # Persistent learning substrate, threaded across respond() calls.
        self._files: dict[str, Any] = {}
        self.interaction_count = 0
        self._seed_memory()

    def _seed_memory(self) -> None:
        """Reset memory to empty scaffolding (no learned content)."""
        self._files = {
            _AGENT_MEMORY_PATH: _file_data(_SEED_AGENTS_MD),
            _OUTCOMES_MEMORY_PATH: _file_data(_SEED_OUTCOMES_MD),
        }

    def respond(self, query: Query) -> Response:
        self.interaction_count += 1

        prompt = query.prompt or "(no content)"
        if query.feedback is not None and query.feedback.content.strip():
            prompt = (
                "Feedback on your previous action:\n"
                f"{query.feedback.content.strip()}\n\n{prompt}"
            )

        # Invoke the deep agent to completion, threading persisted memory in...
        result = self._agent.invoke(
            {
                "messages": [{"role": "user", "content": prompt}],
                "files": self._files,
            }
        )
        # ...and reading the (possibly memory-updated) filesystem back out.
        self._files = result.get("files", self._files)

        messages = result.get("messages", [])
        final_text = _message_text(messages[-1].content) if messages else ""
        self._record_usage(messages)
        action = self._extract_action(final_text, query.response_schema)

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
        """Log the outcome and, at instance boundaries, distil it into memory.

        A one-shot decision agent won't reliably spend a tool call to update its
        own notes mid-decision, so memory-writing is made an explicit step: at
        each completed instance we run a dedicated reflection call that rewrites
        ``AGENTS.md``. The agent reads it next turn as untrusted memory (wrapped
        in ``<agent_memory>`` boundary markers by ``MemoryMiddleware``).
        """
        content = observation.content.strip()
        if not content:
            return
        self._append_outcome(content, complete=observation.instance_complete)
        if observation.instance_complete:
            self._reflect(content)

    def _append_outcome(self, content: str, *, complete: bool) -> None:
        """Append one outcome to the raw log, bounded to the last N entries."""
        prior = _read_file_data(self._files, _OUTCOMES_MEMORY_PATH) or _SEED_OUTCOMES_MD
        lines = prior.splitlines()
        header, body = (lines[:1], lines[1:]) if lines else ([_SEED_OUTCOMES_MD], [])
        marker = "" if complete else " [mid-instance]"
        body.append(f"- {content}{marker}")
        body = body[-_MAX_OUTCOME_ENTRIES:]
        self._files[_OUTCOMES_MEMORY_PATH] = _file_data("\n".join(header + body) + "\n")

    def _reflect(self, outcome: str) -> None:
        """Distil the completed instance into durable notes (writes ``AGENTS.md``)."""
        current = _read_file_data(self._files, _AGENT_MEMORY_PATH) or _SEED_AGENTS_MD
        resp = self._model.invoke(
            [
                {"role": "system", "content": _REFLECT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Current notes:\n{current}\n\nLatest outcome:\n{outcome}",
                },
            ]
        )
        notes = _message_text(resp.content).strip()
        if notes:
            self._files[_AGENT_MEMORY_PATH] = _file_data(notes + "\n")
        self._record_usage([resp], call_type="reflect")

    def _memory_snapshot(self) -> dict[str, str]:
        """Current memory as a ``{path: content}`` dict (the shape clbench logs)."""
        return {path: _read_file_data(self._files, path) for path in _MEMORY_SOURCES}

    def reset(self) -> None:
        """Wipe learned memory.

        Called once at the start of a stateful rollout, and before *every*
        instance in the stateless baseline.
        """
        self._seed_memory()
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

    # ------------------------------------------------------------------
    def _extract_action(self, text: str, schema: type[BaseModel]) -> BaseModel:
        """Coerce the agent's free-form final answer into the task's schema.

        Done as a separate structured call so it never fights the agent's tool
        loop. ``schema`` is the task-supplied, strictly-typed pydantic model.
        """
        structured = self._model.with_structured_output(schema, include_raw=True)
        out = structured.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract the final structured answer from the agent's "
                        "message below. Do not invent information."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        raw = out.get("raw") if isinstance(out, dict) else None
        if raw is not None:
            self._record_usage([raw], call_type="extract")
        parsed = out.get("parsed") if isinstance(out, dict) else out
        if parsed is None:
            raise RuntimeError("Structured extraction returned no parsed action.")
        return parsed

    def _record_usage(self, messages: list[Any], call_type: str = "completion") -> None:
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
                call_type=call_type,
                model=self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )
