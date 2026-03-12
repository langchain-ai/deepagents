from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException, tool
from pydantic import BaseModel, Field, create_model

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryScorer, final_text_contains, run_agent, tool_call

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

DATA_PATH = Path(__file__).parent / "data" / "curated_external_evals.json"
_PACK = json.loads(DATA_PATH.read_text())
CASES: list[dict[str, Any]] = _PACK["cases"]

TOOL_SOURCES = {"toolbench", "bfcl", "gorillabench"}
TOOL_CASES = [case for case in CASES if case["source"] in TOOL_SOURCES]
HOTPOT_CASES = [case for case in CASES if case["source"] == "hotpotqa"]


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "Tool"


def _schema_type_to_python(schema: dict[str, Any]) -> object:
    schema_type = str(schema.get("type", "string")).lower()
    type_map = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "boolean": bool,
        "bool": bool,
        "array": list[object],
        "object": dict[str, object],
        "dict": dict[str, object],
    }
    return type_map.get(schema_type, object)


def _build_args_model(tool_name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields: dict[str, tuple[Any, Field]] = {}
    for key, schema in properties.items():
        annotation = _schema_type_to_python(schema)
        description = schema.get("description", "")
        if key in required:
            fields[key] = (annotation, Field(description=description))
        else:
            fields[key] = (annotation | None, Field(default=None, description=description))
    return create_model(f"{_sanitize_name(tool_name)}Args", **fields)


def _string_is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _coerce_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _string_is_number(value):
        return float(value)
    return None


def _values_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        return all(_values_equal(a, e) for a, e in zip(actual, expected, strict=False))
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    if isinstance(actual, bool) or isinstance(expected, bool):
        actual_value = actual.strip().lower() if isinstance(actual, str) else actual
        expected_value = expected.strip().lower() if isinstance(expected, str) else expected
        return actual_value == expected_value
    actual_number = _coerce_number(actual)
    expected_number = _coerce_number(expected)
    if actual_number is not None and expected_number is not None:
        return actual_number == expected_number
    return actual == expected


def _matches_allowed_args(actual: dict[str, Any], allowed_args: dict[str, list[Any]]) -> bool:
    for key, allowed_values in allowed_args.items():
        if key not in actual or actual[key] is None:
            if any(value == "" for value in allowed_values):
                continue
            return False
        if not any(_values_equal(actual[key], value) for value in allowed_values if value != ""):
            return False
    return True


def _build_tool(tool_spec: dict[str, Any], expected_calls: list[dict[str, Any]]) -> StructuredTool:
    args_model = _build_args_model(tool_spec["name"], tool_spec["parameters"])
    expected_for_tool = [call for call in expected_calls if call["name"] == tool_spec["name"]]

    def _invoke(**kwargs: Any) -> str:
        for call in expected_for_tool:
            if _matches_allowed_args(kwargs, call["allowed_args"]):
                return call["output"]
        if expected_for_tool:
            msg = f"Unexpected arguments for {tool_spec['name']}: {kwargs}"
            raise ToolException(msg)
        return f"{tool_spec['name']} is available but not relevant for this request."

    return StructuredTool.from_function(
        func=_invoke,
        name=tool_spec["name"],
        description=tool_spec["description"],
        args_schema=args_model,
    )


def create_tool_case_agent(case: dict[str, Any], model: BaseChatModel):
    tools = [_build_tool(tool_spec, case["expected_calls"]) for tool_spec in case["tools"]]
    return create_deep_agent(model=model, tools=tools)


def create_tool_case_scorer(case: dict[str, Any]) -> TrajectoryScorer:
    scorer = TrajectoryScorer().success(
        *(final_text_contains(snippet, case_insensitive=True) for snippet in case["answer_substrings"])
    )
    return scorer.expect(
        tool_call_requests=len(case["expected_calls"]),
        tool_calls=[tool_call(name=call["name"]) for call in case["expected_calls"]],
    )


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def create_hotpot_agent(case: dict[str, Any], model: BaseChatModel):
    documents = {doc["title"]: doc["text"] for doc in case["documents"]}

    @tool
    def search_reference_notes(query: str) -> list[dict[str, str]]:
        """Search the local reference notes and return the most relevant titles and snippets."""
        query_tokens = _normalize_tokens(query)
        ranked = []
        for title, text in documents.items():
            haystack_tokens = _normalize_tokens(f"{title} {text}")
            score = len(query_tokens & haystack_tokens)
            ranked.append((score, title, text))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [{"title": title, "snippet": text[:240]} for score, title, text in ranked[:3]]

    @tool
    def read_reference_note(title: str) -> str:
        """Read a local reference note by its exact title."""
        for doc_title, text in documents.items():
            if doc_title.lower() == title.lower():
                return text
        msg = f"Unknown reference title: {title}"
        raise ToolException(msg)

    return create_deep_agent(model=model, tools=[search_reference_notes, read_reference_note])


def create_hotpot_scorer(case: dict[str, Any]) -> TrajectoryScorer:
    return TrajectoryScorer().success(
        *(final_text_contains(snippet, case_insensitive=True) for snippet in case["answer_substrings"])
    ).expect(
        tool_calls=[
            tool_call(name="search_reference_notes"),
            tool_call(name="read_reference_note"),
        ]
    )


def run_tool_case(case: dict[str, Any], model: BaseChatModel) -> None:
    agent = create_tool_case_agent(case, model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        scorer=create_tool_case_scorer(case),
    )


def run_hotpot_case(case: dict[str, Any], model: BaseChatModel) -> None:
    agent = create_hotpot_agent(case, model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        scorer=create_hotpot_scorer(case),
    )
