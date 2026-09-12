"""Provider compatibility for open MCP argument objects."""

from copy import deepcopy

import pytest
from fastmcp import Client, FastMCP
from jsonschema import Draft202012Validator
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp.types import Tool

from deepagents_code.mcp_tools import _build_mcp_tool, _normalize_mcp_schema


@pytest.mark.filterwarnings(
    r"default:`langchain\.mcp` is in beta:langchain_core._api.LangChainBetaWarning"
)
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "object", "properties": {}},
        {"type": "object"},
        {"type": ["object", "null"], "properties": {}},
    ],
)
async def test_open_payload_survives_provider_conversion(
    payload: dict[str, object],
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "payload": payload,
        },
        "required": ["operation"],
    }
    original = deepcopy(schema)
    mcp_tool = Tool(name="server_call_operation", input_schema=schema)
    tool = await _build_mcp_tool(
        mcp_tool=mcp_tool,
        server_name="server",
        client=Client(FastMCP("test")),
    )
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert parameters["properties"]["payload"]["additionalProperties"] is True
    assert parameters["required"] == ["operation"]
    assert parameters["properties"]["payload"]["type"] == payload["type"]
    assert schema == original
    assert mcp_tool.input_schema == original


@pytest.mark.parametrize("additional", [False, True, {"type": "string"}])
def test_explicit_constraints_are_preserved(additional: object) -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": additional}
    assert _normalize_mcp_schema(schema) == schema


def test_nested_schemas_are_normalized_without_changing_literal_data() -> None:
    empty = {"type": "object", "properties": {}}
    schema = {
        "$defs": {"payload": empty},
        "type": "array",
        "items": {"anyOf": [empty, {"type": "null"}]},
        "default": empty,
        "examples": [empty],
    }
    normalized = _normalize_mcp_schema(schema)
    assert normalized["$defs"]["payload"]["additionalProperties"] is True
    assert normalized["items"]["anyOf"][0]["additionalProperties"] is True
    assert normalized["default"] == empty
    assert normalized["examples"] == [empty]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {}, "unevaluatedProperties": False},
        {
            "allOf": [{"type": "object", "properties": {}}],
            "unevaluatedProperties": False,
        },
        {
            "$defs": {"payload": {"type": "object", "properties": {}}},
            "$ref": "#/$defs/payload",
            "unevaluatedProperties": {"type": "string"},
        },
    ],
)
def test_unevaluated_properties_constraints_are_preserved(
    schema: dict[str, object],
) -> None:
    normalized = _normalize_mcp_schema(schema)
    assert not Draft202012Validator(schema).is_valid({"page": 1})
    assert not Draft202012Validator(normalized).is_valid({"page": 1})
    assert normalized == schema


def test_literal_unevaluated_properties_do_not_disable_normalization() -> None:
    schema = {
        "type": "object",
        "properties": {},
        "default": {"unevaluatedProperties": False},
    }
    normalized = _normalize_mcp_schema(schema)
    assert normalized["additionalProperties"] is True
    assert normalized["default"] == schema["default"]
