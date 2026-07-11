"""Concise GLM-5.2 harness profile for Deep Agents Code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.profiles import HarnessProfile, register_harness_profile
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command


_GLM_5P2_MODEL_SPECS: tuple[str, ...] = (
    "fireworks:accounts/fireworks/models/glm-5p2",
    "openrouter:z-ai/glm-5.2",
    "baseten:zai-org/GLM-5.2",
)
"""Exact provider/model specs that receive the GLM-5.2 profile."""

_SYSTEM_PROMPT_SUFFIX = """\
<glm_5p2_execution>
Execute the task directly and keep your reasoning proportional to the work. Read \
the request carefully, honor every stated constraint, and prefer concrete progress \
over commentary about what you intend to do.

This is a text-only model. Do not call `read_file` on images, PDFs, audio, or video. \
Use a shell utility or a short script to extract relevant text, metadata, or frames \
into a text representation, then inspect that text. Never place binary or encoded \
media in model context. Do not reopen generated media for visual inspection; \
validate it with task-specific non-visual checks.

Create the requested artifact first when the task asks for a file, patch, report, or \
other deliverable. Then inspect and refine the artifact itself; do not substitute a \
long explanation for the requested output. Treat supplied inputs as immutable and \
preserve their fidelity. Write derived outputs separately unless the user explicitly \
requests an in-place change, and retain exact names, values, ordering, and formatting \
that the task makes significant.

If a required dependency or command is unavailable, make one retry after correcting \
the invocation or environment assumption. If it still fails, pivot immediately to \
an available equivalent instead of repeating the same approach.

Run task-named checks and inspect their actual results. Fix only failures caused by \
your work. Stop immediately once the requested artifact is complete and the named \
checks pass; do not add speculative extras or continue polishing beyond the request.
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


class _GlmReadFileMediaGuard(AgentMiddleware):
    """Keep unsupported `read_file` media out of GLM-5.2 requests."""

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
        return self._normalize(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Run and normalize an asynchronous tool call.

        Returns:
            The original safe result or a generic text-only error.
        """
        return self._normalize(request, await handler(request))


_GLM_5P2_PROFILE = HarnessProfile(
    system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
    extra_middleware=lambda: (_GlmReadFileMediaGuard(),),
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
