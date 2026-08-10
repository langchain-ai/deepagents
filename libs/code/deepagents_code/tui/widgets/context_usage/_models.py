"""Data model for the context-usage visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_ColorRole = Literal["warning", "primary", "secondary", "accent", "muted"]


@dataclass(frozen=True, slots=True)
class _Category:
    label: str
    tokens: int
    color: _ColorRole


@dataclass(frozen=True, slots=True)
class _Snapshot:
    context_tokens: int | None
    context_limit: int | None
    conversation_tokens: int | None
    model_spec: str | None
    approximate: bool
    categories: tuple[_Category, ...]

    @classmethod
    def from_usage(
        cls,
        *,
        context_tokens: int | None,
        conversation_tokens: int | None,
        context_limit: int | None,
        model_spec: str | None,
        approximate: bool,
    ) -> _Snapshot:
        total = None if context_tokens is None else max(0, context_tokens)
        conversation = (
            None if conversation_tokens is None else max(0, conversation_tokens)
        )
        limit = (
            context_limit if context_limit is not None and context_limit > 0 else None
        )
        categories: list[_Category] = []

        if total is None:
            if conversation:
                categories.append(
                    _Category("Conversation estimate", conversation, "primary")
                )
            if limit is not None:
                categories.append(
                    _Category(
                        "Unreported capacity",
                        max(0, limit - (conversation or 0)),
                        "accent",
                    )
                )
        elif total > 0:
            if conversation is None:
                categories.append(_Category("Used context", total, "secondary"))
            else:
                conversation = min(conversation, total)
                fixed = total - conversation
                if fixed:
                    categories.append(
                        _Category("System prompt + tools", fixed, "warning")
                    )
                if conversation:
                    categories.append(
                        _Category("Conversation", conversation, "primary")
                    )

        if total is not None and limit is not None:
            categories.append(_Category("Free space", max(0, limit - total), "muted"))

        return cls(
            context_tokens=total,
            context_limit=limit,
            conversation_tokens=conversation,
            model_spec=model_spec,
            approximate=approximate,
            categories=tuple(categories),
        )

    @property
    def scale_tokens(self) -> int:
        categorized = sum(category.tokens for category in self.categories)
        return max(self.context_limit or 0, categorized, 1)

    @property
    def displayed_usage(self) -> int:
        return self.context_tokens or self.conversation_tokens or 0
