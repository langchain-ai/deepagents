"""Concise GLM-5.2 harness profile for Deep Agents Code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from deepagents._models import get_model_identifier  # noqa: PLC2701
from deepagents.profiles import HarnessProfile, register_harness_profile
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel
    from langgraph.prebuilt.tool_node import ToolCallRequest


_GLM_5P2_MODEL_SPECS: tuple[str, ...] = (
    "fireworks:accounts/fireworks/models/glm-5p2",
    "openrouter:z-ai/glm-5.2",
    "baseten:zai-org/GLM-5.2",
)
"""Exact provider/model specs that receive the GLM-5.2 profile."""

_GLM_5P2_MODEL_IDENTIFIERS: frozenset[str] = frozenset(
    spec.partition(":")[2] for spec in _GLM_5P2_MODEL_SPECS
)
"""Provider-native identifiers recognized as GLM-5.2."""

_SYSTEM_PROMPT_SUFFIX = """\
<glm_5p2_execution>
Execute the task directly. Identify every required output path before acting, and \
translate all must, only, exact, ordered, ranged, and prohibited requirements into \
a short execution checklist. Prefer concrete progress over commentary.

Create a valid, parseable artifact before long-running installs, research, or tuning. \
Keep it valid while refining it, reserve the final part of the run for verification, \
and near the time limit stop exploring and leave the best complete artifact.

Use the exact requested version, date, revision, tokenizer, library, or source, never \
memory or a nearby substitute. Before changing protected inputs, record a checksum \
and use a separate working copy when the task permits. Treat \
supplied or fetched source-of-truth data as authoritative: apply only transformations \
the task explicitly requests, preserve everything else exactly, and compare the final \
artifact against that source or its stated allowlist. Do not strip prefixes or tags, \
repair grammar, normalize, reformat, or add cleanup unless explicitly requested.

Run the artifact with the same interpreter and entrypoint that will be evaluated, \
and confirm dependencies through that interpreter. A successful exit only proves \
that the command ran; checks must assert \
the actual result against the required value and fail nonzero on mismatch. Exercise \
task-stated examples plus relevant values below, at, and above each boundary, \
including negative cases and cleanup behavior.

For optimization tasks, preserve correctness and input bytes first. Inspect algorithm \
and execution-plan structure instead of relying only on timing, then use repeated \
measurements against the requested reference with enough margin for noise.

This is a text-only model. Do not call `read_file` on images, PDFs, audio, or video. \
Extract needed text, metadata, or frames with a shell utility or script. Never place \
binary or encoded media in model context. Do not reopen generated media for visual \
inspection; validate it with task-specific non-visual checks.

If a required dependency or command is unavailable, make one retry after correcting \
the invocation. If it still fails, pivot immediately instead of repeating the same \
approach.

Fix only failures caused by your work. Stop immediately once every requested artifact \
is complete and the assertions pass; do not add speculative extras.
</glm_5p2_execution>"""
"""Text appended to the Deep Agents system prompt for GLM-5.2."""

_MEDIA_READ_ERROR = (
    "Error: this model can consume only text from `read_file`. Use a shell command "
    "or script to extract the needed information into text, then read that text output."
)
"""Fixed error used instead of reflecting unsupported media results."""


def _has_only_text_content(message: ToolMessage) -> bool:
    """Return whether a successful tool message contains only valid text."""
    content = message.content
    if isinstance(content, str):
        return True
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(block, str)
        or (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        for block in content
    )


def _is_glm_5p2_model(model: str | BaseChatModel) -> bool:
    """Return whether a model or spec has an exact GLM-5.2 identifier."""
    if isinstance(model, str):
        _, separator, identifier = model.partition(":")
        model_identifier = identifier if separator else model
    else:
        model_identifier = get_model_identifier(model)
    return model_identifier in _GLM_5P2_MODEL_IDENTIFIERS


def _without_trusted_suffix(prompt: str) -> str:
    """Remove exact, delimited GLM suffixes without changing surrounding text.

    Returns:
        Prompt without trusted GLM suffixes.
    """
    start = prompt.find(_SYSTEM_PROMPT_SUFFIX)
    while start >= 0:
        end = start + len(_SYSTEM_PROMPT_SUFFIX)
        has_left_boundary = start == 0 or prompt[start - 2 : start] == "\n\n"
        has_right_boundary = end == len(prompt) or prompt[end : end + 2] == "\n\n"
        if not (has_left_boundary and has_right_boundary):
            start = prompt.find(_SYSTEM_PROMPT_SUFFIX, end)
            continue

        if start == 0 and end < len(prompt):
            end += 2
        elif start > 0:
            start -= 2
        prompt = prompt[:start] + prompt[end:]
        start = prompt.find(_SYSTEM_PROMPT_SUFFIX, max(0, start - 2))
    return prompt


def _transition_system_prompt(prompt: str | None, *, active: bool) -> str | None:
    """Add, remove, or deduplicate the trusted GLM suffix at the prompt tail.

    Returns:
        Transitioned prompt, or `None` when no prompt was supplied.
    """
    if prompt is None:
        return None
    base = _without_trusted_suffix(prompt)
    if not active:
        return base
    if not base:
        return _SYSTEM_PROMPT_SUFFIX
    return f"{base}\n\n{_SYSTEM_PROMPT_SUFFIX}"


class _GlmReadFileMediaState(AgentState):
    """Private state shared between the GLM model and tool wrappers."""

    _glm_5p2_active: Annotated[NotRequired[bool], PrivateStateAttr]


class _GlmReadFileMediaGuard(AgentMiddleware[_GlmReadFileMediaState]):
    """Keep unsupported `read_file` media out of GLM-5.2 requests."""

    state_schema = _GlmReadFileMediaState

    def __init__(self, model: str | BaseChatModel) -> None:
        """Capture the construction model as a safe tool-state fallback.

        Args:
            model: Model instance or spec used to construct this agent stack.
        """
        super().__init__()
        self._construction_active = _is_glm_5p2_model(model)

    @staticmethod
    def _prepare_model_request(
        request: ModelRequest,
    ) -> tuple[ModelRequest, bool]:
        """Classify the resolved model and transition its trusted prompt suffix.

        Returns:
            The request to send downstream and whether its model is GLM-5.2.
        """
        active = _is_glm_5p2_model(request.model)
        prompt = request.system_prompt
        transitioned = _transition_system_prompt(prompt, active=active)
        if transitioned != prompt:
            request = request.override(system_prompt=transitioned)
        return request, active

    @staticmethod
    def _model_result(
        response: ModelResponse, *, active: bool
    ) -> ExtendedModelResponse:
        """Attach the resolved GLM classification as private state.

        Returns:
            Extended response that updates the private tool-gating state.
        """
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_glm_5p2_active": active}),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        """Classify the resolved model before calling it.

        Returns:
            Model response with the effective GLM state update.
        """
        request, active = self._prepare_model_request(request)
        return self._model_result(handler(request), active=active)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ExtendedModelResponse:
        """Classify and call the resolved model asynchronously.

        Returns:
            Model response with the effective GLM state update.
        """
        request, active = self._prepare_model_request(request)
        return self._model_result(await handler(request), active=active)

    def _active_for_tool(self, request: ToolCallRequest) -> bool:
        """Read validated private state or use the construction fallback.

        Returns:
            Whether media reads should be blocked for this tool call.
        """
        state = request.state
        if isinstance(state, Mapping):
            active = state.get("_glm_5p2_active")
            if isinstance(active, bool):
                return active
        return self._construction_active

    @staticmethod
    def _normalize(
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Replace non-text reads without reflecting untrusted result data.

        Returns:
            The original safe result or a generic text-only error.
        """
        if request.tool_call.get("name") != "read_file":
            return result
        if isinstance(result, ToolMessage) and _has_only_text_content(result):
            return result
        return ToolMessage(
            content=_MEDIA_READ_ERROR,
            name="read_file",
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run the validated tool call, then remove unsupported media output.

        Returns:
            The original safe result or a generic text-only error.
        """
        result = handler(request)
        if not self._active_for_tool(request):
            return result
        return self._normalize(request, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run and normalize an asynchronous tool call.

        Returns:
            The original safe result or a generic text-only error.
        """
        result = await handler(request)
        if not self._active_for_tool(request):
            return result
        return self._normalize(request, result)


_GLM_5P2_PROFILE = HarnessProfile(
    system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
)
"""Harness profile shared by the exact GLM-5.2 registrations."""

_glm_5p2_profile_registered = False
"""Process-wide guard that keeps profile registration idempotent."""


def _ensure_glm_5p2_profile_registered() -> None:
    """Register the GLM-5.2 harness profile exactly once per process."""
    global _glm_5p2_profile_registered  # noqa: PLW0603
    if _glm_5p2_profile_registered:
        return

    for spec in _GLM_5P2_MODEL_SPECS:
        register_harness_profile(spec, _GLM_5P2_PROFILE)
    _glm_5p2_profile_registered = True
