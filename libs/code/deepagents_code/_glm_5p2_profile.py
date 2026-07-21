"""GLM-5.2 harness profile for Deep Agents Code.

Bundles a concise execution-focused prompt suffix with one-shot recovery for the
measured Fireworks headless terminal-stall failure mode.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

# Private import: the SDK exposes no public helpers to derive the exact
# provider/model spec that terminal-stall recovery needs to classify each call.
from deepagents._models import (  # noqa: PLC2701
    get_model_identifier,
    get_model_provider,
)
from deepagents.profiles import HarnessProfile, register_harness_profile
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)


_FIREWORKS_GLM_5P2_SPEC = "fireworks:accounts/fireworks/models/glm-5p2"
"""Fireworks GLM-5.2 spec. Terminal-stall recovery is scoped to this one spec
because the output cap that produces a tool-free `finish_reason "length"` turn
was measured only here; the other providers are intentionally excluded."""

_GLM_5P2_MODEL_SPECS: tuple[str, ...] = (
    _FIREWORKS_GLM_5P2_SPEC,
    "openrouter:z-ai/glm-5.2",
    "baseten:zai-org/GLM-5.2",
)
"""Exact provider/model specs that receive the GLM-5.2 profile."""

_GLM_5P2_MODEL_SPEC_SET = frozenset(_GLM_5P2_MODEL_SPECS)
"""Exact provider/model specs recognized as GLM-5.2."""

# Enforce the latent invariants the derived structures above rely on, at import
# time rather than by luck of the current literals: every spec is `provider:id`,
# specs are unique, and the measured Fireworks spec is actually registered.
if not all(":" in spec for spec in _GLM_5P2_MODEL_SPECS):
    msg = "every GLM-5.2 spec must be in `provider:identifier` form"
    raise ValueError(msg)
if len(_GLM_5P2_MODEL_SPEC_SET) != len(_GLM_5P2_MODEL_SPECS):
    msg = "GLM-5.2 specs must be unique"
    raise ValueError(msg)
if _FIREWORKS_GLM_5P2_SPEC not in _GLM_5P2_MODEL_SPECS:
    msg = "the measured Fireworks spec must be a registered GLM-5.2 spec"
    raise ValueError(msg)

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

_TERMINAL_STALL_RECOVERY_SUFFIX = """\
<terminal_stall_recovery>
Your prior attempt exhausted its output budget without taking an action. Stop \
explaining or planning and call a tool now to create or update the requested \
deliverable. Prefer the smallest valid artifact, then run one discriminating check. \
Keep any reasoning brief enough to reach the tool call.
</terminal_stall_recovery>"""
"""One-shot instruction used to recover a capped headless model turn."""


def _model_spec(model: str | BaseChatModel) -> str | None:
    """Return the exact provider/model spec for a model or model spec."""
    if isinstance(model, str):
        _, separator, _ = model.partition(":")
        return model if separator else None
    identifier = get_model_identifier(model)
    provider = get_model_provider(model)
    if identifier is None or provider is None:
        return None
    return f"{provider}:{identifier}"


def _is_fireworks_glm_5p2_model(model: str | BaseChatModel) -> bool:
    """Return whether a model resolves to the measured Fireworks GLM-5.2."""
    return _model_spec(model) == _FIREWORKS_GLM_5P2_SPEC


class _GlmTerminalStallRecovery(AgentMiddleware):
    """Recover a headless Fireworks GLM-5.2 turn that hit the output cap.

    A tool-free GLM response truncated by the output cap (`finish_reason
    "length"`) has stalled: the empty turn would otherwise end a headless run
    with no deliverable. This retries such a turn once with a trusted recovery
    instruction, reasoning disabled, and a forced tool call. The retry is issued
    at most once, so it cannot loop even when the provider ignores the forced
    tool call and the retry itself re-stalls (see `_log_if_still_stalled`).

    Two scoping constraints are intentionally kept out of the process-global
    harness profile:

    - Headless-only. `create_cli_agent` installs this middleware only in
      non-interactive mode. Interactive turns may legitimately be tool-free, so
      they must not be forced into an action.
    - Fireworks-only. `_should_recover` gates on `_is_fireworks_glm_5p2_model`,
      so only `_FIREWORKS_GLM_5P2_SPEC` recovers. The output cap that produces
      the stall was measured only there; OpenRouter and Baseten are excluded.
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
        # Recovery is Fireworks-only (`_should_recover`), and Fireworks reads
        # `reasoning_effort` nested under `model_kwargs` (see dcode
        # `reasoning_effort._fireworks_model_params`), not at the top level. This
        # middleware runs inner to `ConfigurableModelMiddleware`, so a top-level
        # key set here would never be translated into the nested form the model
        # actually reads, leaving reasoning at its default. Write the nested form
        # directly so the retry runs with reasoning disabled as documented,
        # preserving any other `model_kwargs` already on the request.
        existing_model_kwargs = request.model_settings.get("model_kwargs")
        model_kwargs = (
            dict(existing_model_kwargs)
            if isinstance(existing_model_kwargs, Mapping)
            else {}
        )
        model_kwargs["reasoning_effort"] = "none"
        model_settings = {**request.model_settings, "model_kwargs": model_kwargs}
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
            logger.info("GLM-5.2 headless turn stalled at output cap; retrying once")
            response = handler(self._recovery_request(request))
            self._log_if_still_stalled(response)
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
            logger.info("GLM-5.2 headless turn stalled at output cap; retrying once")
            response = await handler(self._recovery_request(request))
            self._log_if_still_stalled(response)
        return response

    @classmethod
    def _log_if_still_stalled(cls, response: ModelResponse) -> None:
        """Warn when the forced-tool recovery retry still returned a stalled turn.

        The retry cannot loop (it is issued at most once), so this is only an
        observability signal: it means the provider did not honor the forced
        tool call, and the headless run may end with no deliverable.
        """
        if cls._is_terminal_stall(response):
            logger.warning(
                "GLM-5.2 stall recovery retry still produced no tool call; "
                "the headless run may end without a deliverable"
            )


_GLM_5P2_PROFILE = HarnessProfile(
    system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
)
"""Prompt-only harness profile shared by the exact GLM-5.2 registrations.

Headless terminal-stall recovery remains separate because whether a session is
interactive is known only when `create_cli_agent` assembles its runtime stack.
"""

_glm_5p2_profile_registered = False
"""Process-wide guard that keeps profile registration idempotent."""


def _ensure_glm_5p2_profile_registered() -> None:
    """Register the GLM-5.2 harness profile once per process, per provider spec.

    Because `register_harness_profile` merges with the incoming profile winning
    on scalar conflicts, a spec that already carries a suffix profile (a user
    override or a built-in) would have its suffix clobbered; the explicit skip
    below is a deliberate countermeasure so an existing suffix is left alone
    rather than silently replaced. In the common case no GLM profile exists yet,
    so the dcode profile is registered for each spec.

    This mutates the process-global harness registry without a lock. Concurrent
    first-calls are benign: `register_harness_profile` is additive/idempotent and
    the profile is identical, so a redundant merge is the worst case.
    """
    # PLW0603: intentional module-level flag toggled once to keep registration
    # idempotent for the process; the concurrent-write case is benign (docstring).
    global _glm_5p2_profile_registered  # noqa: PLW0603
    if _glm_5p2_profile_registered:
        return

    # Private imports: the harness registry and its lazy-load hook expose no
    # public accessor, and this profile must reach into it to register/skip per spec.
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
