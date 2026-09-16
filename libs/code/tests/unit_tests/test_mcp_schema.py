"""Provider compatibility for open MCP argument objects."""

from copy import deepcopy

import pytest
from fastmcp import Client, FastMCP
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp.types import Tool

from deepagents_code.mcp_tools import _build_mcp_tool


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


@pytest.mark.parametrize(
    "constraint",
    [
        {"additionalProperties": False},
        {"additionalProperties": True},
        {"additionalProperties": {"type": "string"}},
        {"unevaluatedProperties": False},
    ],
)
async def test_explicit_constraints_are_preserved(
    constraint: dict[str, object],
) -> None:
    payload = {"type": "object", **constraint}
    schema = {"type": "object", "properties": {"payload": payload}}
    mcp_tool = Tool(name="server_call_operation", input_schema=schema)
    tool = await _build_mcp_tool(
        mcp_tool=mcp_tool,
        server_name="server",
        client=Client(FastMCP("test")),
    )
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]
    assert parameters["properties"]["payload"] == payload
    assert mcp_tool.input_schema == schema
