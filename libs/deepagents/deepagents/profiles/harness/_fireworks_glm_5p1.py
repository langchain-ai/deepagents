"""Built-in Fireworks GLM-5p1 harness profile.

Registers a `HarnessProfile` for `fireworks:accounts/fireworks/models/glm-5p1`
that targets three model-fault clusters surfaced by the deep-agents
eval suite on this model.

The first two are addressed by the system-prompt suffix:

- *Plan / stop discipline* — the model loops on read-only calls, drops
  required final mutations, or repeats a successful mutation. Most
  visible on tau2-airline tasks where actions match the expected
  trajectory but the agent fails to converge.
- *Argument fidelity on mutating tools* — the model selects the right
  tool but routes a wrong target ID, an inverted state value, or an
  empty/defaulted string parameter. Most visible on tau2 db-state
  mismatches, BFCL state mismatches, and HITL tests asserting on
  specific tool-call arguments.

The third is addressed by `FireworksReasoningContentMiddleware`,
attached via `extra_middleware`:

- *Output channel routing* — when `tools` are bound and the model
  produces a final answer without calling a tool, GLM-5p1 served via
  Fireworks routes the answer into `additional_kwargs.reasoning_content`
  with empty `content`. An earlier revision tried to fix this with an
  "Output Channel" suffix section telling the model never to leave
  `content` empty. A local A/B / ablation study (`/tmp/ablation_results.tsv`,
  N=5 per cell, 7 variants, 2 tests, 70 runs) showed that section was
  the *sole* cause of stable regressions on
  `test_single_tool_get_food_calories` and
  `test_single_tool_get_user_email` (0/10 with the rule, 10/10 without
  it) — the rule appeared to prime the model to bifurcate its output
  into reasoning-vs-content channels and land the answer in the wrong
  one. The direct-API success it showed in isolation (8/8) did not
  transfer to the deepagents harness's longer prompt. The middleware
  closes the cluster at the integration layer instead: it inspects the
  response post-call and copies `reasoning_content` → `content` only
  when `content` is empty and there are no tool calls, which is
  deterministic and does not depend on the model obeying any prompt
  rule.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them.

This module exists as the audit anchor for the model: its presence
documents that GLM-5p1 has been profiled against eval data. If a
future GLM revision changes its training or output discipline, add
the new key here (mirroring the Codex `_CODEX_MODEL_SPECS` pattern)
rather than reusing this exact key.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, BaseMessage

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_GLM_5P1_MODEL_SPEC: str = "fireworks:accounts/fireworks/models/glm-5p1"
"""Exact `init_chat_model` spec for the Fireworks GLM-5p1 hosted model.

Matches both registry lookup paths in `_harness_profile_for_model`: the
caller-provided spec string and the `provider:identifier` key derived
from a pre-built `ChatFireworks` instance whose `model_name` is
`accounts/fireworks/models/glm-5p1`.
"""

_SYSTEM_PROMPT_SUFFIX: str = """\
## Tool Execution Discipline

- Before issuing a tool call that mutates state (creates, updates, \
cancels, sends, books, sets), restate in one sentence the target \
object and the intended change. This forces a deliberate check of \
the parameters before they go on the wire.
- Tool arguments must come directly from the conversation. Never \
default a string parameter to "" (empty), "latest", or a value from \
an earlier unrelated turn that the user did not explicitly carry \
forward. When a required argument is ambiguous, ask one targeted \
question instead of guessing.
- Do not re-issue a read-only tool call whose result is already \
visible above. Refer back to the prior result instead.

## Parallel Tool Use

- When tool calls do not depend on each other's outputs, batch them \
into a single response (for example, reading multiple reference \
files at once).
- Do not parallelize tool calls when one depends on the result of \
another. Never use placeholders or guess missing parameters.

## Stop Conditions

- A task is complete when every requested action has succeeded, every \
TODO created via `write_todos` is resolved (done, blocked with a \
one-sentence reason, or cancelled), and you have communicated the \
outcome to the user. Stop the turn when these hold.
- Do not repeat a successful mutation to "double-check" — repeated \
calls accumulate side effects on the underlying system.
- Before finishing, write a brief confirmation of what changed (or \
what did not change) so the user does not have to re-derive it from \
the tool trace."""
"""Text appended to the assembled base system prompt."""


class FireworksReasoningContentMiddleware(AgentMiddleware):
    """Surface `additional_kwargs.reasoning_content` as `content` when content is empty.

    GLM-5p1 served via Fireworks chat-completions occasionally routes a
    short final answer into `additional_kwargs.reasoning_content` with
    `content == ""` and no tool calls. The eval harness (and most
    callers) read the user-visible answer from `content`, so the model
    appears to return nothing. This middleware rewrites those messages
    after the model call returns: if `content` is falsy (empty string
    or empty list), there are no `tool_calls`, and `reasoning_content`
    is a non-empty string, it copies `reasoning_content` into `content`
    while preserving the message id, usage metadata, and the original
    `reasoning_content` entry in `additional_kwargs`.

    The trigger condition is intentionally narrow:

    - `not msg.content` — leaves messages that already have visible
      content alone, so a model that emits both `content` and
      `reasoning_content` keeps the human-authored content untouched.
    - `not msg.tool_calls` — tool-calling responses legitimately have
      empty `content`, and rewriting them would interfere with
      `ToolNode`/`SubAgentMiddleware` dispatch.
    - `reasoning_content` must be a non-empty string — defends against
      providers that set the key to `None`, an empty string, or some
      future structured shape we do not yet handle.

    Stateless. Safe to share a single instance across stacks; a factory
    is used in `register()` only as defensive hygiene against future
    additions to instance state.
    """

    name = "FireworksReasoningContentMiddleware"

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Intercept the synchronous model call and rewrite the response in place."""
        return self._maybe_rewrite(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Intercept the asynchronous model call and rewrite the response in place."""
        return self._maybe_rewrite(await handler(request))

    @classmethod
    def _maybe_rewrite(
        cls,
        response: ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Dispatch on the three legal `wrap_model_call` return shapes.

        `ModelCallResult` is `ModelResponse | AIMessage | ExtendedModelResponse`.
        Each gets rewritten in place; non-AI messages and AI messages
        that do not match the trigger pass through unchanged.
        """
        if isinstance(response, AIMessage):
            return cls._rewrite_message(response)
        if isinstance(response, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=cls._rewrite_response(response.model_response),
                command=response.command,
            )
        return cls._rewrite_response(response)

    @classmethod
    def _rewrite_response(cls, mr: ModelResponse[Any]) -> ModelResponse[Any]:
        """Map the message rewrite over `ModelResponse.result`."""
        rewritten: list[BaseMessage] = [cls._rewrite_message(m) if isinstance(m, AIMessage) else m for m in mr.result]
        return ModelResponse(result=rewritten, structured_response=mr.structured_response)

    @staticmethod
    def _rewrite_message(msg: AIMessage) -> AIMessage:
        """Surface `reasoning_content` as `content` when the trigger condition holds.

        Pass-through when any precondition fails. The rewrite is done
        via `model_copy` so the returned message preserves id,
        usage_metadata, response_metadata, and any other fields the
        caller relies on.
        """
        if msg.content:
            return msg
        if msg.tool_calls:
            return msg
        rc = msg.additional_kwargs.get("reasoning_content")
        if not isinstance(rc, str) or not rc.strip():
            return msg
        return msg.model_copy(update={"content": rc})


def _make_extra_middleware() -> tuple[AgentMiddleware, ...]:
    """Factory for the GLM-5p1 profile's `extra_middleware` slot.

    A factory (rather than a static tuple) is used so each stack the
    profile applies to — main agent, declarative subagents, the
    auto-added general-purpose subagent — gets its own middleware
    instance. The middleware is currently stateless, so this is purely
    defensive hygiene against future state additions, but it costs
    nothing and keeps the contract aligned with `extra_middleware`'s
    documented "use a factory when middleware should not be shared
    across stacks" guidance.
    """
    return (FireworksReasoningContentMiddleware(),)


def register() -> None:
    """Register the built-in Fireworks GLM-5p1 harness profile."""
    _register_harness_profile_impl(
        _GLM_5P1_MODEL_SPEC,
        HarnessProfile(
            system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
            extra_middleware=_make_extra_middleware,
        ),
    )
