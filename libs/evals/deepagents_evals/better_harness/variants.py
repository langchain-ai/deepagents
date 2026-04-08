"""Prompt variants and patching helpers for better-harness experiments."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents_evals.better_harness.benchmarks import HarnessVariant

BETTER_HARNESS_VARIANT_FILE_ENV = "BETTER_HARNESS_VARIANT_FILE"

EXAMPLE_BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools.
You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is ambiguous, ask questions before acting.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it.
Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**
- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""


@dataclass(frozen=True)
class PromptHarnessVariant:
    """A materialized base-prompt variant used in eval subprocesses."""

    label: str
    base_agent_prompt: str
    selected_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize the variant to JSON-compatible data."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Persist the variant to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> PromptHarnessVariant:
        """Load a variant from disk."""
        payload = json.loads(path.read_text())
        return cls(
            label=str(payload["label"]),
            base_agent_prompt=str(payload["base_agent_prompt"]),
            selected_modules=tuple(payload.get("selected_modules", [])),
        )


def render_variant_base_prompt(
    *,
    base_prompt: str,
    variant: HarnessVariant,
    module_prompts: dict[str, str],
) -> str:
    """Render the full base prompt for a harness variant."""
    variant_prompt = variant.render_prompt(module_prompts)
    if not variant_prompt:
        return base_prompt
    return base_prompt.rstrip() + "\n\n" + variant_prompt.strip()


def build_prompt_variant(
    *,
    label: str,
    base_prompt: str,
    variant: HarnessVariant,
    module_prompts: dict[str, str],
) -> PromptHarnessVariant:
    """Build a materialized prompt variant from a base prompt and prompt modules."""
    return PromptHarnessVariant(
        label=label,
        base_agent_prompt=render_variant_base_prompt(
            base_prompt=base_prompt,
            variant=variant,
            module_prompts=module_prompts,
        ),
        selected_modules=variant.module_names,
    )


@contextmanager
def patched_base_agent_prompt(prompt: str) -> Iterator[None]:
    """Temporarily patch `deepagents.graph.BASE_AGENT_PROMPT`."""
    import deepagents.graph as graph_module

    original_prompt = graph_module.BASE_AGENT_PROMPT
    graph_module.BASE_AGENT_PROMPT = prompt
    try:
        yield
    finally:
        graph_module.BASE_AGENT_PROMPT = original_prompt


def load_variant_from_env() -> PromptHarnessVariant | None:
    """Load a prompt variant from `BETTER_HARNESS_VARIANT_FILE`, if configured."""
    raw_path = os.environ.get(BETTER_HARNESS_VARIANT_FILE_ENV)
    if not raw_path:
        return None
    return PromptHarnessVariant.load(Path(raw_path))


def patch_deepagents_from_env() -> None:
    """Patch the Deep Agents base prompt from the configured environment file."""
    variant = load_variant_from_env()
    if variant is None:
        return

    import deepagents.graph as graph_module

    graph_module.BASE_AGENT_PROMPT = variant.base_agent_prompt
