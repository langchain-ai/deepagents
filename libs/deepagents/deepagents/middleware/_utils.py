"""Utility functions for middleware."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ContentBlock, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain.agents.middleware.types import ModelRequest

_SYSTEM_SECTIONS_KEY = "_deepagents_system_sections"
_BASE_SYSTEM_MESSAGE_KEY = "_deepagents_base_system_message"

# Lower order → assembled earlier in the final system prompt.
# Sections not listed here default to order 100.
BUILTIN_SECTION_ORDER: dict[str, int] = {
    "filesystem": 10,
    "subagents": 20,
    "async_subagents": 25,
    "skills": 30,
    "memory": 40,
}


@dataclass
class SystemSection:
    """A named section of the system prompt contributed by one middleware.

    Sections are assembled in ascending ``order`` order (ties broken by key
    name) into the final ``SystemMessage`` by
    :class:`SystemPromptAssemblerMiddleware`.
    """

    key: str
    content: str
    cache_control: bool = False
    order: int = 100


def set_system_section(
    request: ModelRequest[Any],
    key: str,
    content: str,
    *,
    cache_control: bool = False,
    order: int | None = None,
) -> ModelRequest[Any]:
    """Store a named system-prompt section on the request.

    Replaces any existing section for ``key``.  The actual assembly into
    ``SystemMessage`` is deferred to :class:`SystemPromptAssemblerMiddleware`
    (the innermost tail middleware), so sections set by outer middleware are
    visible to inner middleware via :func:`get_system_section` but do not yet
    appear in ``request.system_message``.

    Args:
        request: The current model request.
        key: Section identifier (e.g. ``"memory"``, ``"skills"``).
        content: Section text.
        cache_control: If ``True`` and the active model is Anthropic, the
            assembler will attach a ``cache_control`` ephemeral breakpoint to
            this content block.
        order: Assembly order.  Defaults to ``BUILTIN_SECTION_ORDER[key]``
            when the key is recognised, otherwise ``100``.

    Returns:
        New ``ModelRequest`` with the section stored in ``model_settings``.
    """
    resolved_order = order if order is not None else BUILTIN_SECTION_ORDER.get(key, 100)
    sections: dict[str, SystemSection] = dict(
        request.model_settings.get(_SYSTEM_SECTIONS_KEY, {})
    )
    sections[key] = SystemSection(
        key=key,
        content=content,
        cache_control=cache_control,
        order=resolved_order,
    )

    # Save the original base system_message on the first set_system_section call so that
    # SystemPromptAssemblerMiddleware can later rebuild from the unmodified base.
    new_settings: dict[str, Any] = {**request.model_settings, _SYSTEM_SECTIONS_KEY: sections}
    if _BASE_SYSTEM_MESSAGE_KEY not in request.model_settings:
        new_settings[_BASE_SYSTEM_MESSAGE_KEY] = request.system_message

    # Also append directly to system_message so that middleware works correctly
    # when SystemPromptAssemblerMiddleware is not in the stack (e.g. vanilla
    # create_agent usage).  The assembler overrides this with a properly-ordered
    # rebuild when it is present.
    existing = request.system_message
    existing_blocks: list[ContentBlock] = list(existing.content_blocks) if existing else []
    text = f"\n\n{content}" if existing_blocks else content
    block: ContentBlock = {"type": "text", "text": text}
    if cache_control:
        try:
            from langchain_anthropic import ChatAnthropic

            if isinstance(request.model, ChatAnthropic):
                block = {**block, "cache_control": {"type": "ephemeral"}}  # ty: ignore[invalid-assignment]
        except ImportError:
            pass
    new_blocks = [*existing_blocks, block]
    new_system_message = SystemMessage(content_blocks=new_blocks)

    return request.override(system_message=new_system_message, model_settings=new_settings)


def get_system_section(request: ModelRequest[Any], key: str) -> str | None:
    """Return the content of a named system-prompt section, or ``None``.

    Args:
        request: The current model request.
        key: Section identifier.

    Returns:
        Section content string, or ``None`` if not set.
    """
    sections: dict[str, SystemSection] = request.model_settings.get(_SYSTEM_SECTIONS_KEY, {})
    section = sections.get(key)
    return section.content if section is not None else None


def assemble_system_message(
    base: SystemMessage | None,
    sections: dict[str, SystemSection],
    model: BaseChatModel | None = None,
) -> SystemMessage | None:
    """Assemble a base ``SystemMessage`` and named sections into a final message.

    The base message's blocks come first (preserving any existing
    ``cache_control`` markers), followed by the sections sorted by
    ``(order, key)``.  Adjacent blocks are separated by a ``\\n\\n`` prefix on
    each section's text — matching the historical behaviour of
    ``append_to_system_message``.

    Args:
        base: Existing system message (the user-supplied system prompt).
        sections: Named sections from :func:`set_system_section` calls.
        model: Active chat model; used to gate ``cache_control`` application
            (only added for Anthropic models).

    Returns:
        A new ``SystemMessage``, or ``None`` when there is nothing to assemble.
    """
    try:
        from langchain_anthropic import ChatAnthropic

        is_anthropic = model is not None and isinstance(model, ChatAnthropic)
    except ImportError:
        is_anthropic = False

    base_blocks: list[ContentBlock] = list(base.content_blocks) if base else []
    sorted_sections = sorted(sections.values(), key=lambda s: (s.order, s.key))

    final_blocks: list[ContentBlock] = list(base_blocks)
    for section in sorted_sections:
        if not section.content:
            continue
        text = f"\n\n{section.content}" if final_blocks else section.content
        block: ContentBlock = {"type": "text", "text": text}
        if section.cache_control and is_anthropic:
            block = {**block, "cache_control": {"type": "ephemeral"}}  # ty: ignore[invalid-assignment]
        final_blocks.append(block)

    if not final_blocks:
        return None
    return SystemMessage(content_blocks=final_blocks)


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    .. deprecated::
        Use :func:`set_system_section` to contribute named, addressable
        sections that assemble in deterministic order.

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New ``SystemMessage`` with the text appended.
    """
    warnings.warn(
        "append_to_system_message is deprecated. "
        "Use set_system_section() to contribute named sections that can be "
        "replaced and assembled in deterministic order.",
        DeprecationWarning,
        stacklevel=2,
    )
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)
