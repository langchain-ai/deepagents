"""Provider compatibility for open MCP argument objects."""

from copy import deepcopy

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents_code.mcp_tools import _normalize_mcp_schema


def test_open_payload_survives_provider_conversion() -> None:
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string"},
            "payload": {"type": "object", "properties": {}},
        },
        "required": ["operation"],
    }
    original = deepcopy(schema)
    tool = StructuredTool(
        name="call_operation",
        description="Call an operation.",
        args_schema=_normalize_mcp_schema(schema),
    )
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert parameters["properties"]["payload"]["additionalProperties"] is True
    assert parameters["required"] == ["operation"]
    assert schema == original


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
