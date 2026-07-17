"""Rubric middleware with dcode's bounded model retry policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from deepagents.middleware.rubric import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
    RubricState,
)
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from deepagents.middleware.rubric import RubricEvaluation
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from deepagents_code._cli_context import CLIContextSchema


class ReliableRubricMiddleware(RubricMiddleware):
    """Apply dcode's configured retry policy inside the rubric grader agent."""

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        max_iterations: int = 3,
        on_evaluation: Callable[[RubricEvaluation], None] | None = None,
        model_retry_override: int | None = None,
        model_retry_fallback: int | None = None,
    ) -> None:
        """Initialize rubric grading with an optional explicit CLI retry budget.

        Args:
            model: Model or model identifier used by the grader.
            system_prompt: Custom grading instructions.
            tools: Read-only tools available to the grader.
            max_iterations: Maximum rubric iterations.
            on_evaluation: Optional callback for completed evaluations.
            model_retry_override: Explicit `--max-retries` value inherited from
                the current process, or `None` to use request/checkpoint metadata.
            model_retry_fallback: Caller-provided budget for models without
                attached metadata.

        Raises:
            TypeError: If a retry value is not an integer.
            ValueError: If a retry value is negative.
        """
        retry_values = {
            "model_retry_override": model_retry_override,
            "model_retry_fallback": model_retry_fallback,
        }
        for name, value in retry_values.items():
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                msg = f"`{name}` must be an int or None"
                raise TypeError(msg)
            if value is not None and value < 0:
                msg = f"`{name}` must be non-negative"
                raise ValueError(msg)
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=max_iterations,
            on_evaluation=on_evaluation,
        )
        self._model_retry_override = model_retry_override
        self._model_retry_fallback = model_retry_fallback

    def _ensure_grader(self) -> Any:  # noqa: ANN401
        """Create the grader with model-node retries instead of bare SDK calls.

        Returns:
            The cached or newly constructed grader agent.
        """
        if self._grader is not None:
            return self._grader

        from langchain.agents import create_agent

        from deepagents_code._cli_context import CLIContextSchema
        from deepagents_code.config import (
            CLI_MAX_RETRIES_KEY,
            DEFAULT_MODEL_RETRIES,
            create_model,
            set_model_retry_metadata,
        )
        from deepagents_code.model_retry import CodeModelRetryMiddleware

        retry_fallback = (
            self._model_retry_override
            if self._model_retry_override is not None
            else self._model_retry_fallback
            if self._model_retry_fallback is not None
            else DEFAULT_MODEL_RETRIES
        )
        if isinstance(self._model, str):
            retry_kwargs = (
                {CLI_MAX_RETRIES_KEY: self._model_retry_override}
                if self._model_retry_override is not None
                else None
            )
            grader_model = create_model(
                self._model,
                extra_kwargs=retry_kwargs,
            ).model
            if (
                self._model_retry_override is not None
                or self._model_retry_fallback is not None
            ):
                set_model_retry_metadata(
                    grader_model,
                    retries=retry_fallback,
                    cli_override=self._model_retry_override,
                )
        else:
            grader_model = self._model
        middleware: list[AgentMiddleware[Any, Any]] = [
            CodeModelRetryMiddleware(max_retries=retry_fallback)
        ]
        self._grader = create_agent(
            model=grader_model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=middleware,
            context_schema=CLIContextSchema,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=GraderResponse,
        )
        return self._grader

    def _grader_context(self, state: RubricState) -> CLIContextSchema:
        """Build nested-agent context carrying the effective retry override.

        Returns:
            A `CLIContextSchema` whose private carrier is request-local.
        """
        from deepagents_code._cli_context import CLIContextSchema
        from deepagents_code.config import CLI_MAX_RETRIES_KEY

        retry_override = self._model_retry_override
        if retry_override is None:
            model_params = cast("dict[str, Any]", state).get("_model_params")
            if isinstance(model_params, dict):
                raw = model_params.get(CLI_MAX_RETRIES_KEY)
                if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                    retry_override = raw
        params = (
            {CLI_MAX_RETRIES_KEY: retry_override} if retry_override is not None else {}
        )
        return CLIContextSchema(model_params=params)

    def _grade(self, state: RubricState, iteration: int) -> GraderResponse:
        """Grade synchronously with request-local retry metadata.

        Returns:
            The extracted grader response.
        """
        grader = self._ensure_grader()
        payload = self._build_grader_payload(state, iteration)
        result = grader.invoke(
            {"messages": [HumanMessage(content=payload)]},
            context=self._grader_context(state),
        )
        return self._extract_graded(result)

    async def _agrade(self, state: RubricState, iteration: int) -> GraderResponse:
        """Grade asynchronously with request-local retry metadata.

        Returns:
            The extracted grader response.
        """
        grader = self._ensure_grader()
        payload = self._build_grader_payload(state, iteration)
        result = await grader.ainvoke(
            {"messages": [HumanMessage(content=payload)]},
            context=self._grader_context(state),
        )
        return self._extract_graded(result)
