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
from langchain_core.messages import AIMessage, ToolMessage
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

_SPEC_BY_IDENTIFIER: dict[str, str] = {
    spec.partition(":")[2]: spec for spec in _GLM_5P2_MODEL_SPECS
}
"""Provider-native identifier to full registry spec, for ownership lookups."""

_FIREWORKS_GLM_5P2_IDENTIFIER = _GLM_5P2_MODEL_SPECS[0].partition(":")[2]
"""Fireworks model identifier whose terminal output cap was measured."""

_SYSTEM_PROMPT_SUFFIX = """\
<execution>
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
</execution>"""
"""Text appended to the Deep Agents system prompt for GLM-5.2."""

_MEDIA_READ_ERROR = (
    "Error: this model can consume only text from `read_file`. Use a shell command "
    "or script to extract the needed information into text, then read that text output."
)
"""Fixed error used instead of reflecting unsupported media results."""

_TERMINAL_STALL_RECOVERY_SUFFIX = """\
<terminal_stall_recovery>
Your prior attempt exhausted its output budget without taking an action. Stop \
explaining or planning and call a tool now to create or update the requested \
deliverable. Prefer the smallest valid artifact, then run one discriminating check. \
Keep any reasoning brief enough to reach the tool call.
</terminal_stall_recovery>"""
"""One-shot instruction used to recover a capped headless model turn."""


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


def _model_identifier(model: str | BaseChatModel) -> str | None:
    """Return the provider-native identifier for a model or model spec."""
    if isinstance(model, str):
        _, separator, identifier = model.partition(":")
        return identifier if separator else model
    return get_model_identifier(model)


def _is_glm_5p2_model(model: str | BaseChatModel) -> bool:
    """Return whether a model or spec has an exact GLM-5.2 identifier."""
    return _model_identifier(model) in _GLM_5P2_MODEL_IDENTIFIERS


def _is_fireworks_glm_5p2_model(model: str | BaseChatModel) -> bool:
    """Return whether a model resolves to the measured Fireworks GLM-5.2."""
    return _model_identifier(model) == _FIREWORKS_GLM_5P2_IDENTIFIER


def _dcode_owns_suffix(model: str | BaseChatModel) -> bool:
    """Return whether the dcode suffix is the one registered for this spec.

    The harness registry is authoritative. When a user override or built-in has
    claimed the spec with a different suffix (registration deferred to it), the
    guard must leave that suffix alone instead of appending the dcode one, so an
    override actually wins rather than getting the dcode suffix bolted on.
    """
    identifier = _model_identifier(model)
    spec = _SPEC_BY_IDENTIFIER.get(identifier) if identifier is not None else None
    if spec is None:
        return False
    from deepagents.profiles.harness.harness_profiles import (
        _HARNESS_PROFILES,  # noqa: PLC2701
    )

    existing = _HARNESS_PROFILES.get(spec)
    return (
        existing is not None and existing.system_prompt_suffix == _SYSTEM_PROMPT_SUFFIX
    )


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
    """Apply GLM-5.2 model-response and `read_file` media guards."""

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

        The returned flag drives `read_file` media gating and tracks GLM-5.2 by
        model identity, since a text-only model cannot consume media regardless
        of which profile owns the prompt. The suffix transition is narrower: it
        only manages the dcode suffix for a spec the dcode profile actually owns,
        so a user/built-in override that won registration keeps its own suffix.

        Returns:
            The request to send downstream and whether its model is GLM-5.2.
        """
        active = _is_glm_5p2_model(request.model)
        suffix_active = active and _dcode_owns_suffix(request.model)
        prompt = request.system_prompt
        transitioned = _transition_system_prompt(prompt, active=suffix_active)
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
        response = handler(request)
        return self._model_result(response, active=active)

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
        response = await handler(request)
        return self._model_result(response, active=active)

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


class _GlmTerminalStallRecovery(AgentMiddleware):
    """Recover a headless GLM-5.2 turn that hit the output cap without acting.

    A tool-free GLM response truncated by the output cap (`finish_reason
    "length"`) has stalled: the empty turn would otherwise end a headless run
    with no deliverable. This retries such a turn once with a trusted recovery
    instruction, reasoning disabled, and a forced tool call, which by
    construction cannot itself re-stall. It is scoped to headless runs by only
    being registered on the profile there; interactive turns may legitimately be
    tool-free, so they must not be forced into an action.
    """

    @staticmethod
    def _is_terminal_stall(response: ModelResponse) -> bool:
        """Return whether a turn was truncated by the output cap without acting."""
        if response.structured_response is not None or len(response.result) != 1:
            return False
        message = response.result[0]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return False
        metadata = message.response_metadata
        return isinstance(metadata, dict) and metadata.get("finish_reason") == "length"

    @staticmethod
    def _recovery_request(request: ModelRequest) -> ModelRequest:
        """Append a trusted one-shot recovery instruction to a model request.

        Returns:
            Request with the recovery instruction appended to its system prompt.
        """
        prompt = request.system_prompt
        recovery_prompt = (
            _TERMINAL_STALL_RECOVERY_SUFFIX
            if not prompt
            else f"{prompt}\n\n{_TERMINAL_STALL_RECOVERY_SUFFIX}"
        )
        model_settings = {
            **request.model_settings,
            "reasoning_effort": "none",
        }
        return request.override(
            system_prompt=recovery_prompt,
            tool_choice="any",
            model_settings=model_settings,
        )

    @classmethod
    def _should_recover(
        cls,
        response: ModelResponse,
        *,
        model: str | BaseChatModel,
    ) -> bool:
        """Return whether this GLM call should receive one recovery retry."""
        return _is_fireworks_glm_5p2_model(model) and cls._is_terminal_stall(response)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Retry once when a headless GLM turn stalls at the output cap.

        Returns:
            The original response, or the recovered response after one retry.
        """
        response = handler(request)
        if self._should_recover(response, model=request.model):
            response = handler(self._recovery_request(request))
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry once when a headless GLM turn stalls at the output cap.

        Returns:
            The original response, or the recovered response after one retry.
        """
        response = await handler(request)
        if self._should_recover(response, model=request.model):
            response = await handler(self._recovery_request(request))
        return response


_GLM_5P2_PROFILE = HarnessProfile(
    system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
)
"""Harness profile shared by the exact GLM-5.2 registrations.

Kept suffix-only: process-global registration cannot express per-session or
per-model wiring, so `_GlmReadFileMediaGuard` and `_GlmTerminalStallRecovery`
are installed per stack by `create_cli_agent` where the session mode is known
and the runtime model is re-checked on every call.
"""

_glm_5p2_profile_registered = False
"""Process-wide guard that keeps profile registration idempotent."""


def _ensure_glm_5p2_profile_registered() -> None:
    """Register the GLM-5.2 harness profile once per process, per provider spec.

    Registration is additive with the incoming profile winning on scalar
    conflicts, so a spec that already carries a suffix profile (a user override
    or a built-in) is skipped rather than silently replaced. In the common case
    no GLM profile exists yet, so the dcode profile is registered for each spec.
    """
    global _glm_5p2_profile_registered  # noqa: PLW0603
    if _glm_5p2_profile_registered:
        return

    from deepagents.profiles.harness.harness_profiles import (
        _HARNESS_PROFILES,  # noqa: PLC2701
        _ensure_harness_profiles_loaded,  # noqa: PLC2701
    )

    _ensure_harness_profiles_loaded()
    for spec in _GLM_5P2_MODEL_SPECS:
        existing = _HARNESS_PROFILES.get(spec)
        if existing is not None and existing.system_prompt_suffix is not None:
            continue
        register_harness_profile(spec, _GLM_5P2_PROFILE)
    _glm_5p2_profile_registered = True
