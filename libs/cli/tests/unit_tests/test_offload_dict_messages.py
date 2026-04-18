"""Tests for #2741 fix: `perform_offload()` handles serialized state payloads.

Remote HTTP state snapshots and checkpointer fallbacks may surface messages as
raw dicts rather than `BaseMessage` objects. `perform_offload()` must
normalize these before passing them to middleware helpers like
`get_buffer_string()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import BaseMessage

from deepagents_cli.offload import OffloadResult, perform_offload

if TYPE_CHECKING:
    from deepagents.middleware.summarization import SummarizationEvent

# Raw dict messages — what remote state snapshots and checkpoint fallbacks
# may return before deserialization.
_DICT_MESSAGES: list[dict[str, Any]] = [
    {
        "content": "Hi!",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": None,
        "id": "human-1",
    },
    {
        "content": "Hello! How can I help?",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "ai",
        "name": None,
        "id": "ai-1",
    },
    {
        "content": "Tell me a joke",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": None,
        "id": "human-2",
    },
    {
        "content": "Why did the chicken cross the road?",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "ai",
        "name": None,
        "id": "ai-2",
    },
    {
        "content": "Why?",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": None,
        "id": "human-3",
    },
    {
        "content": "To get to the other side!",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "ai",
        "name": None,
        "id": "ai-3",
    },
]

# Patch targets
_CREATE_MODEL_PATH = "deepagents_cli.offload.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization.compute_summarization_defaults"
)
_MW_CLASS_PATH = "deepagents.middleware.summarization.SummarizationMiddleware"
_TOKEN_COUNT_PATH = "deepagents_cli.offload.count_tokens_approximately"
_OFFLOAD_BACKEND_PATH = "deepagents_cli.offload.offload_messages_to_backend"


def _mock_perform_deps(*, cutoff: int = 3) -> tuple[MagicMock, MagicMock]:
    """Return (mock_model_result, mock_middleware) for perform_offload tests."""
    mock_model = MagicMock()
    mock_model.profile = {"max_input_tokens": 200_000}
    mock_result = MagicMock()
    mock_result.model = mock_model

    mock_mw = MagicMock()
    mock_mw._apply_event_to_messages.side_effect = lambda msgs, _ev: list(msgs)
    mock_mw._determine_cutoff_index.return_value = cutoff
    mock_mw._partition_messages.side_effect = lambda msgs, idx: (
        msgs[:idx],
        msgs[idx:],
    )
    mock_mw._acreate_summary = AsyncMock(return_value="Summary of conversation.")

    summary_msg = MagicMock()
    summary_msg.content = "Summary of conversation."
    summary_msg.additional_kwargs = {"lc_source": "summarization"}
    mock_mw._build_new_messages_with_path.return_value = [summary_msg]
    mock_mw._compute_state_cutoff.side_effect = lambda _ev, c: c
    mock_mw._filter_summary_messages.side_effect = lambda msgs: msgs

    return mock_result, mock_mw


class TestDictMessageNormalization:
    """Verify `perform_offload()` normalizes serialized state payloads."""

    async def test_dict_messages_converted_before_middleware(self) -> None:
        """Dict messages are converted to BaseMessage before reaching middleware."""
        model_result, mock_mw = _mock_perform_deps(cutoff=3)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(_OFFLOAD_BACKEND_PATH, new_callable=AsyncMock, return_value="/p.md"),
        ):
            result = await perform_offload(
                messages=list(_DICT_MESSAGES),
                prior_event=None,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)
        assert result.messages_offloaded == 3
        assert result.messages_kept == 3

        passed_msgs = mock_mw._apply_event_to_messages.call_args[0][0]
        assert all(isinstance(m, BaseMessage) for m in passed_msgs)

    async def test_dict_prior_event_summary_message_converted(self) -> None:
        """A prior_event with a dict summary_message is normalized."""
        model_result, mock_mw = _mock_perform_deps(cutoff=0)

        dict_summary = {
            "content": "Previous summary.",
            "additional_kwargs": {"lc_source": "summarization"},
            "response_metadata": {},
            "type": "human",
            "name": None,
            "id": "summary-1",
        }
        prior_event = cast(
            "SummarizationEvent",
            {
                "cutoff_index": 2,
                "summary_message": dict_summary,
                "file_path": "/old.md",
            },
        )

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=50),
        ):
            from deepagents_cli.offload import OffloadThresholdNotMet

            result = await perform_offload(
                messages=list(_DICT_MESSAGES),
                prior_event=prior_event,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadThresholdNotMet)

        passed_event = mock_mw._apply_event_to_messages.call_args[0][1]
        assert isinstance(passed_event["summary_message"], BaseMessage)

    async def test_base_message_inputs_unchanged(self) -> None:
        """Already-proper BaseMessage objects pass through without issue."""
        from langchain_core.messages.utils import convert_to_messages

        model_result, mock_mw = _mock_perform_deps(cutoff=3)
        proper_messages = convert_to_messages(_DICT_MESSAGES)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(_OFFLOAD_BACKEND_PATH, new_callable=AsyncMock, return_value="/p.md"),
        ):
            result = await perform_offload(
                messages=proper_messages,
                prior_event=None,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)

    async def test_dict_messages_and_dict_summary_normalized_together(self) -> None:
        """Both flags at once: dicts in `messages` and in `prior_event`."""
        model_result, mock_mw = _mock_perform_deps(cutoff=3)

        dict_summary = {
            "content": "Previous summary.",
            "additional_kwargs": {"lc_source": "summarization"},
            "response_metadata": {},
            "type": "human",
            "name": None,
            "id": "summary-1",
        }
        prior_event = cast(
            "SummarizationEvent",
            {
                "cutoff_index": 1,
                "summary_message": dict_summary,
                "file_path": "/old.md",
            },
        )

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(_OFFLOAD_BACKEND_PATH, new_callable=AsyncMock, return_value="/p.md"),
        ):
            result = await perform_offload(
                messages=list(_DICT_MESSAGES),
                prior_event=prior_event,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)

        passed_msgs = mock_mw._apply_event_to_messages.call_args[0][0]
        passed_event = mock_mw._apply_event_to_messages.call_args[0][1]
        assert all(isinstance(m, BaseMessage) for m in passed_msgs)
        assert isinstance(passed_event["summary_message"], BaseMessage)

    async def test_heterogeneous_messages_fully_normalized(self) -> None:
        """A list mixing BaseMessage and dict entries is normalized end-to-end."""
        from langchain_core.messages.utils import convert_to_messages

        model_result, mock_mw = _mock_perform_deps(cutoff=3)
        # First element is a BaseMessage; rest are dicts. The [0]-only
        # heuristic would have missed this; `any(...)` catches it.
        head = convert_to_messages([_DICT_MESSAGES[0]])
        tail: list[Any] = list(_DICT_MESSAGES[1:])
        mixed: list[Any] = [*head, *tail]

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(_OFFLOAD_BACKEND_PATH, new_callable=AsyncMock, return_value="/p.md"),
        ):
            result = await perform_offload(
                messages=mixed,
                prior_event=None,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)

        passed_msgs = mock_mw._apply_event_to_messages.call_args[0][0]
        assert all(isinstance(m, BaseMessage) for m in passed_msgs)

    async def test_empty_messages_with_dict_summary_still_normalizes(self) -> None:
        """Empty `messages=[]` does not crash the guard; summary still normalized."""
        model_result, mock_mw = _mock_perform_deps(cutoff=0)

        dict_summary = {
            "content": "Previous summary.",
            "additional_kwargs": {"lc_source": "summarization"},
            "response_metadata": {},
            "type": "human",
            "name": None,
            "id": "summary-1",
        }
        prior_event = cast(
            "SummarizationEvent",
            {
                "cutoff_index": 0,
                "summary_message": dict_summary,
                "file_path": "/old.md",
            },
        )

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=0),
        ):
            from deepagents_cli.offload import OffloadThresholdNotMet

            result = await perform_offload(
                messages=[],
                prior_event=prior_event,
                thread_id="test-thread",
                model_spec="anthropic:claude-sonnet-4-20250514",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadThresholdNotMet)
        passed_event = mock_mw._apply_event_to_messages.call_args[0][1]
        assert isinstance(passed_event["summary_message"], BaseMessage)
