"""Tests for Gemini tool-schema repair and error translation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deepagents_code._gemini_schema import (
    repair_tools_for_gemini,
    translate_gemini_tool_schema_error,
)
from deepagents_code.configurable_model import ConfigurableModelMiddleware


def _make_gemini_model() -> MagicMock:
    """Create a mock model that reports the google_genai provider."""
    model = MagicMock(spec=BaseChatModel)
    model.model_name = "gemini-2.5-flash"
    model._get_ls_params.return_value = {"ls_provider": "google_genai"}
    return model


def _make_request(model: BaseChatModel, tools: list[Any]) -> ModelRequest:
    """Create a ModelRequest carrying a Gemini-ish model and given tools."""
    runtime = SimpleNamespace(context=None)
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        tools=tools,
        runtime=cast("Any", runtime),
    )


def _walk_arrays_have_items(schema: object) -> bool:
    """Return True if every `type: array` under `schema` has a valid `items`."""
    if isinstance(schema, dict):
        if schema.get("type") == "array":
            items = schema.get("items")
            if not isinstance(items, (dict, list)) or (
                isinstance(items, dict) and not items
            ):
                return False
        for value in schema.values():
            if not _walk_arrays_have_items(value):
                return False
    elif isinstance(schema, list):
        for value in schema:
            if not _walk_arrays_have_items(value):
                return False
    return True


class TestRepairToolsForGemini:
    """Repair passes injected `items` on array properties missing it."""

    def test_basetool_with_list_params_gets_items_on_every_array(self) -> None:
        @tool
        def sample(
            sections: list[str],
            rows: list[dict],
            datasets: list[list[str]],
        ) -> str:
            """Sample tool with array-typed params."""
            _ = (sections, rows, datasets)
            return "ok"

        repaired = repair_tools_for_gemini([sample])
        assert len(repaired) == 1
        assert isinstance(repaired[0], dict)

        params = repaired[0]["function"]["parameters"]
        for name in ("sections", "rows", "datasets"):
            assert _walk_arrays_have_items(params["properties"][name]), (
                f"array property {name!r} still missing items after repair"
            )

    def test_dict_tool_with_bare_array_property_gets_default_items(self) -> None:
        tool_dict = {
            "type": "function",
            "function": {
                "name": "widget",
                "description": "a widget",
                "parameters": {
                    "type": "object",
                    "properties": {"names": {"type": "array"}},
                },
            },
        }
        repaired = repair_tools_for_gemini([tool_dict])
        prop = repaired[0]["function"]["parameters"]["properties"]["names"]
        assert prop["items"] == {"type": "string"}

    def test_repair_does_not_replace_valid_items(self) -> None:
        tool_dict = {
            "type": "function",
            "function": {
                "name": "widget",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
        }
        repaired = repair_tools_for_gemini([tool_dict])
        prop = repaired[0]["function"]["parameters"]["properties"]["names"]
        assert prop["items"] == {"type": "integer"}

    def test_repair_walks_nested_anyof_and_defs(self) -> None:
        tool_dict = {
            "type": "function",
            "function": {
                "name": "widget",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "anyOf": [
                                {"type": "array"},
                                {"type": "null"},
                            ],
                        },
                    },
                    "$defs": {
                        "Nested": {
                            "type": "object",
                            "properties": {
                                "items_field": {"type": "array"},
                            },
                        },
                    },
                },
            },
        }
        repaired = repair_tools_for_gemini([tool_dict])
        params = repaired[0]["function"]["parameters"]
        assert params["properties"]["field"]["anyOf"][0]["items"] == {"type": "string"}
        nested = params["$defs"]["Nested"]["properties"]["items_field"]
        assert nested["items"] == {"type": "string"}


class TestTranslateGeminiToolSchemaError:
    """Gemini `400` responses referencing tool schemas are translated."""

    def test_translates_message_with_property_names(self) -> None:
        body = (
            "400 INVALID_ARGUMENT. Invalid value at "
            "'GenerateContentRequest.tools[0].function_declarations[23]"
            ".parameters.properties[sections].items' (TYPE_INT64), "
            "and function_declarations[24].parameters.properties[rows]"
        )
        exc = Exception(body)
        message = translate_gemini_tool_schema_error(exc)
        assert message is not None
        assert "function_declarations[23].parameters.properties[sections]" in message
        assert "function_declarations[24].parameters.properties[rows]" in message

    def test_returns_none_for_unrelated_error(self) -> None:
        exc = Exception("500 Internal Server Error")
        assert translate_gemini_tool_schema_error(exc) is None

    def test_generic_fallback_when_indices_unparsed(self) -> None:
        exc = Exception(
            "GenerateContentRequest.tools something something function_declarations"
        )
        message = translate_gemini_tool_schema_error(exc)
        assert message is not None
        assert "rejected" in message.lower()


class TestConfigurableModelMiddlewareGemini:
    """Integration checks that the middleware repairs schemas and translates errors."""

    def test_middleware_repairs_tools_before_handler(self) -> None:
        @tool
        def sample(names: list[str]) -> str:
            """Docstring."""
            _ = names
            return "ok"

        model = _make_gemini_model()
        request = _make_request(model, [sample])
        middleware = ConfigurableModelMiddleware(persist_model_state=False)

        captured: list[ModelRequest] = []

        def handler(r: ModelRequest) -> ModelResponse[Any]:
            captured.append(r)
            return ModelResponse(result=[AIMessage(content="ok")])

        middleware.wrap_model_call(request, handler)

        assert len(captured) == 1
        outbound_tools = captured[0].tools
        assert len(outbound_tools) == 1
        outbound_tool = outbound_tools[0]
        assert isinstance(outbound_tool, dict)
        prop = outbound_tool["function"]["parameters"]["properties"]["names"]
        assert prop.get("items"), "middleware failed to repair Gemini tool schema"

    def test_middleware_translates_gemini_400_into_visible_error(self) -> None:
        model = _make_gemini_model()
        request = _make_request(model, [])
        middleware = ConfigurableModelMiddleware(persist_model_state=False)

        error_body = (
            "400 INVALID_ARGUMENT: GenerateContentRequest.tools[0]"
            ".function_declarations[3].parameters.properties[rows]"
        )

        def handler(_request: ModelRequest) -> ModelResponse[Any]:
            raise RuntimeError(error_body)

        with pytest.raises(RuntimeError, match="Gemini") as excinfo:
            middleware.wrap_model_call(request, handler)

        message = str(excinfo.value)
        assert message, "translated Gemini error must not be empty"
        assert "function_declarations[3]" in message

    def test_middleware_leaves_non_gemini_errors_untouched(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        model.model_name = "gpt-5.5"
        model._get_ls_params.return_value = {"ls_provider": "openai"}
        request = _make_request(model, [])
        middleware = ConfigurableModelMiddleware(persist_model_state=False)

        original = ValueError("some unrelated 500")

        def handler(_request: ModelRequest) -> ModelResponse[Any]:
            raise original

        with pytest.raises(ValueError, match="unrelated") as excinfo:
            middleware.wrap_model_call(request, handler)
        assert excinfo.value is original
