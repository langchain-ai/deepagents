"""Middleware: don't let the agent terminate on a non-deliverable.

Deterministic completion gate for two terminal-failure modes seen in LangSmith trace
analysis of this model on Terminal-Bench:

  (A) INTENT-STOP — the final assistant turn ends on a bare intent statement
      ("R is now installed. Let me create the adaptive rejection sampler.") with no tool
      call, and the deliverable was never produced.
  (B) UNSUPPORTED-CLAIM — the final summary claims a file was created
      ("Created `/app/summary.csv`") but the trajectory contains no write to that path.

A system-prompt instruction does not fix these — the model is not exercising judgment
(self-verification was a measured wash). So, before the graph ends, if the terminal
assistant turn matches (A) or (B), this injects a corrective user message and jumps back
to the model to force one more attempt. Bounded by `max_nudges`; `recursion_limit` is the
hard backstop. Conservative by design: it only acts on a terminal (no-tool-call) turn and
on specific ending signals, to avoid interrupting an agent that is finishing correctly.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

# Output-artifact-looking absolute paths (deliverables usually live under these roots).
_PATH = r"/(?:app|root|home|tmp|workspace|srv|data)[\w./+-]*\.\w{1,8}"
_PATH_RE = re.compile(_PATH)
# "…let me/I'll create|write|build|implement…" — a creation-intent phrase.
_INTENT_RE = re.compile(
    r"(?is)(?:let me|i['’]?ll|i will|now,?\s*let me|next,?\s*i(?:['’]?ll| will)?)"
    r"\b[^.\n]{0,80}\b(?:creat|writ|build|implement|generat|assembl|construct|start)"
)
# A creation claim tied to a specific path ("created /app/out.csv").
_CLAIM_RE = re.compile(
    r"(?is)\b(?:creat\w*|wrote|written|sav\w*|generat\w*|produc\w*|output)\b[^.\n]{0,60}?("
    + _PATH
    + r")"
)


def _text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _is_ai(msg: Any) -> bool:
    return getattr(msg, "type", "") == "ai" or msg.__class__.__name__ == "AIMessage"


def _written_paths(messages: list[Any]) -> set[str]:
    """Paths that appear in any tool call's arguments (i.e. plausibly written/touched)."""
    written: set[str] = set()
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call.get("args") if isinstance(call, dict) else None
            if isinstance(args, dict):
                blob = " ".join(str(value) for value in args.values())
                written.update(_PATH_RE.findall(blob))
    return written


class CompletionGateMiddleware(AgentMiddleware):
    """Force one more turn if the agent tries to end without producing the deliverable."""

    def __init__(self, *, max_nudges: int = 2) -> None:
        super().__init__()
        self._max_nudges = max_nudges
        self._nudges = 0

    def _gate(self, state: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        # Only act on a *terminal* assistant turn (no pending tool calls).
        if not _is_ai(last) or getattr(last, "tool_calls", None):
            return None
        if self._nudges >= self._max_nudges:
            return None

        text = _text(last)
        written = _written_paths(messages)

        def unwritten(path: str) -> bool:
            return not any(path == w or path in w or w in path for w in written)

        reason: str | None = None
        claim = _CLAIM_RE.search(text)
        if claim and unwritten(claim.group(1)):
            reason = (
                f"Your final message claims `{claim.group(1)}` was created, but the "
                "trajectory shows no write to that path."
            )
        elif _INTENT_RE.search(text[-400:]):
            reason = (
                "Your final message ends by stating you are about to create the "
                "deliverable, but you did not actually do it."
            )

        if reason is None:
            return None

        self._nudges += 1
        nudge = HumanMessage(
            content=(
                f"[completion-gate] Do not stop yet. {reason} Produce the required output "
                "now: write it to the exact path using your file/shell tools, then confirm "
                "it exists. If you genuinely cannot, state explicitly that you are blocked "
                "and why — do not end on an intent statement or an unverified claim."
            )
        )
        return {"jump_to": "model", "messages": [nudge]}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._gate(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._gate(state)
