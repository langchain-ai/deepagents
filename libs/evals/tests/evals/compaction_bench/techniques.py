"""Summarization technique adapters for the compaction benchmark.

A ``SummarizationTechnique`` is a small adapter that, given a model and
a backend, constructs the middleware stack the runner should wire into
``create_deep_agent``. The Protocol lets the runner iterate techniques
uniformly; adding a new one is a ~50-line file plus a registration
in ``TECHNIQUES``.

### What ships in v1

Two techniques, both under an **aggressive 80k trigger** so compaction
fires 2-3 times per run:

- ``deepagents``: ``SummarizationMiddleware`` with the default prompt
  shipped in ``deepagents``. This is the production baseline.
- ``openai_compact``: the same middleware mechanics with an OpenAI
  "compact"-style summarization prompt and an OpenAI model driving the
  summary generation. Treats OpenAI's Codex-CLI-style compaction as
  *a prompt + summarizer-model combination* layered on top of the
  existing middleware - the fairest apples-to-apples comparison we
  can do without inventing a parallel middleware stack. See the
  ``OPENAI_COMPACT_PROMPT`` docstring for the prompt text.

### Why not a second middleware class?

The design doc assumed a ``CodexCompactionMiddleware`` existed in
``deepagents``; it does not. Rather than fork a second middleware
that differs only in prompt + summarizer-model, we reuse
``SummarizationMiddleware`` and vary only those two axes. If an
actual OpenAI-provided compaction API becomes available later, a
third adapter can wrap it without touching the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tests.evals.compaction_bench.task_spec import (
    AGGRESSIVE_KEEP_MESSAGES,
    AGGRESSIVE_TRIGGER_TOKENS,
)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SummarizationTechnique(Protocol):
    """Adapter for constructing a compaction middleware under test.

    Implementations are typically stateless dataclasses. The runner
    calls ``build_middleware`` once per (instance, seed) run.

    Attributes:
        name: Stable identifier used as the scorecard key (e.g.
            ``"deepagents"``). Must be filesystem-safe and short.
    """

    name: str

    def build_middleware(
        self,
        *,
        consumer_model: BaseChatModel,
        backend: BACKEND_TYPES,
    ) -> list[AgentMiddleware]:
        """Construct the middleware list that wraps the agent.

        The runner passes the ``consumer_model`` (i.e. the model the
        agent is itself using) so techniques that prefer to summarize
        with the same model can do so. Techniques that want a
        different summarizer model instantiate it internally.

        Args:
            consumer_model: The model the agent is using; available for
                techniques that want to reuse it as the summarizer.
            backend: The ``FilesystemBackend`` (or equivalent) the
                summarization middleware should offload history to.

        Returns:
            Middleware list, in the order they should wrap the agent.
        """
        ...


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
#
# Both prompts are pinned in source and versioned in git. Changing either
# requires re-validating any downstream judge / kappa measurements that
# compared outputs against it.


DEEPAGENTS_DEFAULT_PROMPT: str = ""
"""Sentinel: use ``SummarizationMiddleware``'s shipped ``DEFAULT_SUMMARY_PROMPT``.

Stored as an empty string to signal "no override". When the
``deepagents`` adapter runs, it leaves the prompt parameter unset so
the middleware falls back to its own default. Keeping this as a named
constant rather than ``None`` documents the intent.
"""


OPENAI_COMPACT_PROMPT: str = """\
You are summarizing a long coding-assistant session so the conversation
can be compacted. Produce a structured summary that preserves every piece
of load-bearing state a successor assistant would need to continue the
work without re-reading the full history.

Organize the summary into five labeled sections, in this order:

1. **Goals and constraints** — the user's stated objectives and every
   explicit constraint or rule they have set. Preserve the original
   wording where possible. Do not omit constraints even if they seem
   redundant.
2. **Decisions made and alternatives rejected** — for each non-trivial
   decision, record what was chosen, what was considered and rejected,
   and the specific reason for the rejection. The *reason* is as
   important as the decision itself.
3. **Current state** — what files have been created or modified, what
   the canonical file paths are, and what tests exist. Reference files
   by their full path.
4. **Outstanding work and open questions** — anything not yet done
   that a successor needs to know about.
5. **Recent context** — a short paragraph covering the most recent
   exchange, so the successor has immediate context for what the user
   just said.

Be concise but complete. Prefer bullet points over prose. Do not
invent content that was not in the conversation.
"""
"""OpenAI-style structured-compact prompt.

Modeled after the public OpenAI Codex-CLI compaction behavior:
structured slots over free-form prose, explicit retention of decision
rationale, separation of "durable state" from "recent context". If
OpenAI ships a first-party compaction endpoint later, that endpoint
can be wrapped as an additional technique without retiring this one
(it remains a useful "prompt-level compaction" baseline).
"""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepAgentsTechnique:
    """Production-default deepagents summarization under an aggressive trigger.

    Attributes:
        name: ``"deepagents"``.
        trigger_tokens: Token count that fires summarization. Defaults
            to the aggressive 80k value from ``task_spec``.
        keep_messages: Recent messages retained verbatim after
            summarization. Defaults to the aggressive 15-message value.
    """

    name: str = "deepagents"
    trigger_tokens: int = AGGRESSIVE_TRIGGER_TOKENS
    keep_messages: int = AGGRESSIVE_KEEP_MESSAGES

    def build_middleware(
        self,
        *,
        consumer_model: BaseChatModel,
        backend: BACKEND_TYPES,
    ) -> list[AgentMiddleware]:
        """Construct the deepagents summarization middleware stack.

        Uses ``consumer_model`` as the summarizer; this matches current
        production behavior where a single model drives both agent
        execution and summarization.

        Args:
            consumer_model: Agent's model; reused for summarization.
            backend: Backend for history offload.

        Returns:
            Single-element middleware list.
        """
        # Imports are local so the module is importable in unit tests
        # that don't install the full deepagents middleware surface.
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
        )

        middleware = SummarizationMiddleware(
            model=consumer_model,
            backend=backend,
            trigger=("tokens", self.trigger_tokens),
            keep=("messages", self.keep_messages),
            # Leave summary_prompt at its middleware-level default.
        )
        return [middleware]


@dataclass(frozen=True)
class OpenAICompactTechnique:
    """OpenAI-style structured-compact prompt + OpenAI summarizer model.

    The consumer model (whatever the agent is running on) is left
    unchanged. Only the *summarizer* side - the model that generates
    the compact summary and the prompt that instructs it - is swapped.

    Attributes:
        name: ``"openai_compact"``.
        summarizer_model: Model identifier for summary generation.
            Defaults to a current GPT-class model; the actual version
            is pinned at runtime via ``langchain.chat_models.init_chat_model``.
        trigger_tokens: Token count that fires summarization. Same
            aggressive value as ``DeepAgentsTechnique`` for fair
            comparison.
        keep_messages: Recent messages retained verbatim. Same as
            ``DeepAgentsTechnique``.
    """

    name: str = "openai_compact"
    summarizer_model: str = "openai:gpt-5.1"
    trigger_tokens: int = AGGRESSIVE_TRIGGER_TOKENS
    keep_messages: int = AGGRESSIVE_KEEP_MESSAGES

    def build_middleware(
        self,
        *,
        consumer_model: BaseChatModel,
        backend: BACKEND_TYPES,
    ) -> list[AgentMiddleware]:
        """Construct the OpenAI-compact middleware stack.

        Args:
            consumer_model: Agent's model; **not** used for summarization.
            backend: Backend for history offload.

        Returns:
            Single-element middleware list.
        """
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
        )
        from langchain.chat_models import init_chat_model

        # Unused parameter retained for Protocol conformance; techniques
        # that reuse the consumer model need the argument here.
        _ = consumer_model

        summarizer = init_chat_model(self.summarizer_model)

        middleware = SummarizationMiddleware(
            model=summarizer,
            backend=backend,
            trigger=("tokens", self.trigger_tokens),
            keep=("messages", self.keep_messages),
            summary_prompt=OPENAI_COMPACT_PROMPT,
        )
        return [middleware]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TECHNIQUES: dict[str, SummarizationTechnique] = {
    "deepagents": DeepAgentsTechnique(),
    "openai_compact": OpenAICompactTechnique(),
}
"""Registry of techniques the runner will sweep over.

Keyed by ``SummarizationTechnique.name``. Adding a new technique means
(a) implementing the Protocol and (b) registering the instance here —
no other file changes required.
"""


__all__ = [
    "DEEPAGENTS_DEFAULT_PROMPT",
    "OPENAI_COMPACT_PROMPT",
    "TECHNIQUES",
    "DeepAgentsTechnique",
    "OpenAICompactTechnique",
    "SummarizationTechnique",
]
