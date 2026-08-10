"""Presentation for the `/context` slash command."""

from __future__ import annotations

from deepagents_code._markdown import escape_markdown
from deepagents_code._session_stats import format_token_count


def build_context_usage_markdown(
    *,
    context_tokens: int,
    conversation_tokens: int | None,
    context_limit: int | None,
    model_spec: str | None,
    approximate: bool,
) -> str:
    """Build a context report that distinguishes measured and estimated usage.

    Args:
        context_tokens: Latest total for the active model context.
        conversation_tokens: Approximate tokens in the effective message history.
        context_limit: Active model's input-token limit, when known.
        model_spec: Active model identifier.
        approximate: Whether `context_tokens` is stale or locally estimated.

    Returns:
        Markdown for the context-window report.
    """
    if context_limit is not None and context_limit <= 0:
        context_limit = None
    lines = ["## Context usage"]
    if model_spec:
        lines.append(escape_markdown(model_spec))

    if context_tokens <= 0:
        lines.extend(("", "No usage reported yet."))
        if context_limit is not None:
            lines.append(
                f"**Context window:** {format_token_count(context_limit)} tokens"
            )
        else:
            lines.append("Context window limit unavailable for this model.")
    else:
        count_prefix = "~" if approximate else ""
        current_tokens = format_token_count(context_tokens)
        lines.append("")
        if context_limit is not None:
            percent = context_tokens / context_limit * 100
            remaining = max(0, context_limit - context_tokens)
            limit = format_token_count(context_limit)
            current = f"**Current:** {count_prefix}{current_tokens} / {limit} tokens"
            lines.extend(
                (
                    f"{current} ({percent:.1f}%)",
                    f"**Remaining:** {format_token_count(remaining)} tokens",
                )
            )
        else:
            lines.extend(
                (
                    f"**Current:** {count_prefix}{current_tokens} tokens",
                    "Context window limit unavailable for this model.",
                )
            )

        if conversation_tokens is not None:
            conversation = min(max(0, conversation_tokens), context_tokens)
            fixed = context_tokens - conversation
            lines.extend(("", "### Estimated composition"))
            for label, tokens in (
                ("System prompt + tools", fixed),
                ("Conversation", conversation),
            ):
                percent_suffix = (
                    f" ({tokens / context_limit * 100:.1f}%)" if context_limit else ""
                )
                count = format_token_count(tokens)
                lines.append(f"- **{label}:** ~{count} tokens{percent_suffix}")

    lines.extend(
        ("", "**Automatic offload:** enabled; use `/offload` to compact sooner.")
    )
    if context_tokens > 0:
        lines.append(
            "*Current total is approximate; composition is estimated.*"
            if approximate
            else "*Current total is provider-reported; composition is estimated.*"
        )
    return "\n".join(lines)
