"""Check routing requirements using the model request and explicit capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from deepagents_evals.routing.eligibility import ineligible_reasons, request_tokens

if TYPE_CHECKING:
    from langchain_core.language_models.model_profile import ModelProfile


class Answer(BaseModel):
    value: str


def test_prompt_estimate_includes_system_tools_and_response_schema() -> None:
    model = FakeMessagesListChatModel(responses=[])
    request = ModelRequest(model=model, messages=[HumanMessage(content="Help")])
    system = request.override(system_message=SystemMessage(content="Policy " * 100))
    tools = system.override(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find records " * 100,
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    structured = tools.override(response_format=AutoStrategy(Answer))

    assert (
        request_tokens(request)
        < request_tokens(system)
        < request_tokens(tools)
        < request_tokens(structured)
    )


@pytest.mark.parametrize("kind", ["image", "audio", "video", "file"])
def test_unsupported_modalities_are_excluded(kind: str) -> None:
    field = "pdf_inputs" if kind == "file" else f"{kind}_inputs"
    model = FakeMessagesListChatModel(responses=[], profile={field: False})
    request = ModelRequest(
        model=model,
        messages=[
            HumanMessage(
                content=[
                    {
                        "type": kind,
                        "url": "https://example.com/input",
                        "mime_type": "application/pdf",
                    }
                ]
            )
        ],
    )

    assert field in ineligible_reasons(model, request, input_tokens=100, output_tokens=1000)


def test_tool_result_images_need_tool_message_support() -> None:
    model = FakeMessagesListChatModel(
        responses=[], profile={"image_inputs": True, "image_tool_message": False}
    )
    request = ModelRequest(
        model=model,
        messages=[
            ToolMessage(
                content=[{"type": "image", "url": "https://example.com/image.png"}],
                tool_call_id="image",
            )
        ],
    )

    assert ineligible_reasons(model, request, input_tokens=100, output_tokens=1000) == [
        "image_tool_message"
    ]


@pytest.mark.parametrize(
    ("strategy", "profile", "excluded"),
    [
        (ProviderStrategy(Answer), {"structured_output": False, "tool_calling": True}, True),
        (ToolStrategy(Answer), {"structured_output": True, "tool_calling": False}, True),
        (ToolStrategy(Answer), {"tool_calling": True, "tool_choice": False}, True),
        (
            AutoStrategy(Answer),
            {"structured_output": False, "tool_calling": True, "tool_choice": True},
            False,
        ),
        (AutoStrategy(Answer), {"structured_output": False, "tool_calling": False}, True),
        (AutoStrategy(Answer), {"structured_output": True, "tool_choice": False}, False),
    ],
)
def test_structured_output_respects_the_requested_strategy(
    strategy: ProviderStrategy[Answer] | ToolStrategy[Answer] | AutoStrategy[Answer],
    profile: ModelProfile,
    excluded: bool,
) -> None:
    model = FakeMessagesListChatModel(responses=[], profile=profile)
    request = ModelRequest(model=model, messages=[], response_format=strategy)

    assert (
        bool(ineligible_reasons(model, request, input_tokens=100, output_tokens=1000)) is excluded
    )


def test_context_window_reserves_output_and_honors_request_limits() -> None:
    model = FakeMessagesListChatModel(
        responses=[], profile={"max_input_tokens": 4000, "max_output_tokens": 2000}
    )
    request = ModelRequest(model=model, messages=[], model_settings={"max_tokens": 2000})

    assert ineligible_reasons(model, request, input_tokens=2000, output_tokens=1000) == []
    assert ineligible_reasons(model, request, input_tokens=2001, output_tokens=1000) == [
        "context_window"
    ]
    request = request.override(model_settings={"max_tokens": 2500})
    assert ineligible_reasons(model, request, input_tokens=100, output_tokens=1000) == [
        "output_budget"
    ]


def test_missing_profile_fields_remain_compatible() -> None:
    model = FakeMessagesListChatModel(responses=[])
    request = ModelRequest(model=model, messages=[], response_format=AutoStrategy(Answer))

    assert ineligible_reasons(model, request, input_tokens=100, output_tokens=1000) == []


def test_small_configured_output_limit_overrides_default_reserve() -> None:
    model = FakeMessagesListChatModel(
        responses=[], profile={"max_input_tokens": 1000, "max_output_tokens": 100}
    )
    request = ModelRequest(model=model, messages=[], model_settings={"max_tokens": 100})

    assert ineligible_reasons(model, request, input_tokens=900, output_tokens=1000) == []


@pytest.mark.parametrize("choice", ["auto", "none", {"type": "auto"}, {"type": "none"}])
def test_automatic_tool_choice_does_not_require_forced_tool_support(
    choice: str | dict[str, str],
) -> None:
    model = FakeMessagesListChatModel(
        responses=[], profile={"tool_calling": True, "tool_choice": False}
    )
    request = ModelRequest(model=model, messages=[], tool_choice=choice)

    assert ineligible_reasons(model, request, input_tokens=100, output_tokens=1000) == []
