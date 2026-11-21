"""Mid-thread reasoning effort changes via OpenAI `configuration_update` blocks.

On models that support it, OpenAI's Responses API accepts a
`configuration_update` input item that changes `reasoning.effort` from that
point in the conversation onward. Changing the request-level effort instead
rewrites the model-side prefix and invalidates the prompt cache, so `/effort`
on a thread that already reached the model keeps sending the thread's recorded
request-level effort and carries the new one in a block attached to the latest
`HumanMessage`. The block is persisted in `messages` so it keeps its position
in history on later turns; moving it would invalidate the cache just the same.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from deepagents_code.cold_cache import (
    EFFORT_IDENTITY_KEY,
    configuration_update_supported,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import AnyMessage

logger = logging.getLogger(__name__)

CONFIGURATION_UPDATE_TYPE = "configuration_update"
"""Content-block type `langchain-openai` hoists into a Responses input item."""


@dataclass(frozen=True)
class EffortUpdate:
    """A request rewritten to change effort without moving the cache prefix."""

    request: ModelRequest
    """Request carrying the pinned request-level effort and any new block."""

    cache_params: dict[str, Any] | None
    """Cache-identity params describing the request-level effort actually sent."""

    message: HumanMessage | None
    """Updated message to persist in state, or `None` when no block was added."""


def _is_update_block(block: object) -> bool:
    """Return whether a content block is a `configuration_update`."""
    return isinstance(block, Mapping) and block.get("type") == CONFIGURATION_UPDATE_TYPE


def _block_effort(block: object) -> str | None:
    """Return the effort a `configuration_update` content block sets, if any."""
    if not isinstance(block, Mapping) or not _is_update_block(block):
        return None
    reasoning = block.get("reasoning")
    if not isinstance(reasoning, Mapping):
        return None
    effort = reasoning.get("effort")
    return effort if isinstance(effort, str) else None


def _message_blocks(message: object) -> list[object]:
    """Return the content blocks of a `HumanMessage`, or `[]` for other messages."""
    if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
        return []
    return list(message.content)


def latest_block_effort(messages: Sequence[AnyMessage]) -> str | None:
    """Return the effort set by the latest `configuration_update` in history.

    Returns:
        The effort that applies at the end of `messages`, or `None` when no
            block in history sets one.
    """
    for message in reversed(messages):
        for block in reversed(_message_blocks(message)):
            effort = _block_effort(block)
            if effort is not None:
                return effort
    return None


def _with_effort_block(message: HumanMessage, effort: str) -> HumanMessage:
    """Return `message` with a `configuration_update` setting `effort` first.

    An existing block on the same message is updated in place rather than
    duplicated, so an effort changed twice before the message is sent still
    produces a single item.

    Returns:
        A copy of `message` with the same ID and the effort block attached.
    """
    content = message.content
    blocks: list[Any] = (
        list(content)
        if isinstance(content, list)
        else [{"type": "text", "text": content}]
    )
    for index, block in enumerate(blocks):
        if isinstance(block, Mapping) and _is_update_block(block):
            reasoning = block.get("reasoning")
            nested = dict(reasoning) if isinstance(reasoning, Mapping) else {}
            blocks[index] = {**block, "reasoning": {**nested, "effort": effort}}
            break
    else:
        block = {"type": CONFIGURATION_UPDATE_TYPE, "reasoning": {"effort": effort}}
        blocks.insert(0, block)
    return message.model_copy(update={"content": blocks})


def _pin_request_effort(
    model: object, settings: Mapping[str, Any], effort: str | None
) -> dict[str, Any]:
    """Return `settings` with the request-level effort replaced by `effort`.

    A native `reasoning` mapping wins over flat `reasoning_effort` in the
    Responses payload, so the effort is written into whichever shape the
    request (or the model's own `reasoning`) uses.

    Returns:
        New model settings sending `effort` (or no effort when `None`).
    """
    pinned = {
        key: value for key, value in settings.items() if key != "reasoning_effort"
    }
    reasoning = pinned.get("reasoning")
    if not isinstance(reasoning, Mapping):
        reasoning = getattr(model, "reasoning", None)
    if isinstance(reasoning, Mapping):
        nested = {key: value for key, value in reasoning.items() if key != "effort"}
        if effort is not None:
            nested["effort"] = effort
        pinned["reasoning"] = nested
    elif effort is not None:
        pinned["reasoning_effort"] = effort
    return pinned


def _uses_responses_api(request: ModelRequest) -> bool:
    """Return whether the request's model will call OpenAI's Responses API.

    `configuration_update` only exists there; Chat Completions would forward
    the block as an unknown content part.
    """
    check = getattr(request.model, "_use_responses_api", None)
    if not callable(check):
        return False
    defaults = getattr(request.model, "_default_params", None)
    payload = {**(defaults if isinstance(defaults, Mapping) else {})}
    payload.update(request.model_settings)
    if request.tools:
        payload["tools"] = [
            tool if isinstance(tool, Mapping) else {"type": "function"}
            for tool in request.tools
        ]
    try:
        return bool(check(payload))
    except Exception:
        logger.debug("Could not determine the OpenAI API surface", exc_info=True)
        return False


def strip_configuration_updates(request: ModelRequest) -> ModelRequest:
    """Drop `configuration_update` blocks for a model that cannot accept them.

    A thread can switch from a GPT-6 model to another provider or an older
    model mid-conversation; the blocks stay in state but must not reach a
    request that would reject or misread them.

    Returns:
        `request` unchanged when no block is present, otherwise a copy without
            them.
    """
    if not any(
        _is_update_block(block)
        for message in request.messages
        for block in _message_blocks(message)
    ):
        return request
    messages = [_without_blocks(message) for message in request.messages]
    return request.override(messages=messages)


def _without_blocks(message: AnyMessage) -> AnyMessage:
    """Return `message` without `configuration_update` content blocks."""
    blocks = _message_blocks(message)
    kept = [block for block in blocks if not _is_update_block(block)]
    if len(kept) == len(blocks):
        return message
    return message.model_copy(update={"content": kept})


def _latest_human_message(messages: Sequence[AnyMessage]) -> HumanMessage | None:
    """Return the latest `HumanMessage` in `messages`, if any."""
    return next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )


def _recorded_effort(
    state: Mapping[str, Any], model_spec: str
) -> tuple[bool, str | None]:
    """Return whether this model already reached the API, and its request effort.

    Returns:
        `(recorded, effort)`, where `effort` is the request-level effort the
            last successful call on `model_spec` sent (`None` for the default).
    """
    if state.get("_last_cache_model_spec") != model_spec:
        return False, None
    params = state.get("_last_cache_params")
    effort = params.get(EFFORT_IDENTITY_KEY) if isinstance(params, Mapping) else None
    return True, effort if isinstance(effort, str) else None


def _target_effort(model_spec: str, desired: str | None) -> str | None:
    """Resolve the effort a block must name, mapping "default" to its level.

    Returns:
        `desired`, or the profile default when `desired` is `None`.
    """
    if desired is not None:
        return desired
    from deepagents_code.reasoning_effort import default_effort_for_model

    return default_effort_for_model(model_spec)


def prepare_request(
    request: ModelRequest,
    model_spec: str | None,
    cache_params: dict[str, Any] | None,
    *,
    persist: bool,
) -> EffortUpdate:
    """Fit `configuration_update` blocks to the model the request targets.

    Performs a blocking profile read when the desired effort is the model
    default; async callers should offload.

    Args:
        request: The request as resolved by `ConfigurableModelMiddleware`.
        model_spec: Resolved `provider:model` spec of `request.model`.
        cache_params: Cache-identity projection of the effective params; its
            effort is the effort the user wants this turn.
        persist: Whether this middleware owns the thread's checkpointed state.
            Only then can a new block be persisted, so subagents never add one.

    Returns:
        The request to send, its cache identity, and any message to persist.
    """
    unchanged = EffortUpdate(request, cache_params, None)
    if not (
        model_spec
        and configuration_update_supported(model_spec)
        and _uses_responses_api(request)
    ):
        return EffortUpdate(strip_configuration_updates(request), cache_params, None)
    if not persist:
        return unchanged
    return _effort_update(request, model_spec, cache_params) or unchanged


def _effort_update(
    request: ModelRequest, model_spec: str, cache_params: dict[str, Any] | None
) -> EffortUpdate | None:
    """Carry an effort change in a `configuration_update` block.

    Returns:
        The rewritten request, or `None` when it should be sent unchanged
            (nothing recorded yet, no change, or the change cannot be carried).
    """
    recorded, pinned = _recorded_effort(request.state or {}, model_spec)
    desired = (cache_params or {}).get(EFFORT_IDENTITY_KEY)
    request_effort = pinned if recorded else desired
    current = latest_block_effort(request.messages) or request_effort
    message = None
    if current != desired:
        target = _target_effort(model_spec, desired)
        latest = _latest_human_message(request.messages)
        if target is None or latest is None or latest.id is None:
            logger.debug("Cannot carry the effort change in a configuration_update")
            return None
        if target != current:
            message = _with_effort_block(latest, target)
    if message is None and request_effort == desired:
        return None
    return _rewrite(request, cache_params, request_effort, message)


def _rewrite(
    request: ModelRequest,
    cache_params: dict[str, Any] | None,
    request_effort: str | None,
    message: HumanMessage | None,
) -> EffortUpdate:
    """Build the rewritten request and its cache identity.

    Returns:
        The request sending `request_effort` with `message` swapped in.
    """
    overrides: dict[str, Any] = {
        "model_settings": _pin_request_effort(
            request.model, request.model_settings, request_effort
        )
    }
    if message is not None:
        overrides["messages"] = [
            message if existing.id == message.id else existing
            for existing in request.messages
        ]
    params = {
        key: value
        for key, value in (cache_params or {}).items()
        if key != EFFORT_IDENTITY_KEY
    }
    if request_effort is not None:
        params[EFFORT_IDENTITY_KEY] = request_effort
    return EffortUpdate(request.override(**overrides), params or None, message)
