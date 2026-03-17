from __future__ import annotations

import inspect
from types import GenericAlias
from typing import Any, NotRequired

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_quickjs._foreign_function_docs import (
    _collect_referenced_types,
    _format_annotation,
    _format_typed_dict_structure,
    _get_return_annotation,
    _render_typed_dict_definition,
    format_foreign_function_docs,
    render_external_functions_section,
    render_foreign_function_section,
)
from langchain_quickjs._foreign_functions import get_ptc_implementations


class UserLookup(TypedDict):
    id: int
    name: str


class OptionalUserLookup(TypedDict):
    id: int
    nickname: NotRequired[str]


class NameRecord(BaseModel):
    name: str


@tool
def find_users_by_name(name: str) -> list[UserLookup]:
    """Find users with the given name.

    Args:
        name: The user name to search for.
    """
    return [{"id": 1, "name": name}]


@tool
def get_user_location(user_id: int) -> int:
    """Get the location id for a user.

    Args:
        user_id: The user identifier.
    """
    return user_id


@tool
def get_city_for_location(location_id: int) -> str:
    """Get the city for a location.

    Args:
        location_id: The location identifier.
    """
    return f"City {location_id}"


@tool
def combine_user_details(name: str, city: str, active: bool) -> str:
    """Combine user details into a summary string.

    Args:
        name: The user name.
        city: The user's city.
        active: Whether the user is active.
    """
    return f"{name} in {city} active={active}"


@tool
def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""
    return name.strip().lower()


async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"Weather for {city}"


def summarize_lookup(user: OptionalUserLookup) -> dict[str, str]:
    return {"id": str(user["id"])}


def no_doc(value: int):
    return value


class BareTool(BaseTool):
    name: str = "bare_tool"
    description: str = "bare"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "ok"


@tool
async def async_tool(city: str) -> str:
    """Resolve a city name asynchronously."""
    return city


def test_format_foreign_function_docs_for_plain_function() -> None:
    assert (
        format_foreign_function_docs("normalize_name", normalize_name)
        == """/**
 * Normalize a user name for matching.
 */
function normalize_name(name: string): string"""
    )


def test_format_foreign_function_docs_for_tool_with_args_and_return_type() -> None:
    assert (
        format_foreign_function_docs("get_user_location", get_user_location)
        == """/**
 * Get the location id for a user.
 *
 * @param user_id The user identifier.
 */
function get_user_location(user_id: number): number"""
    )


def test_format_foreign_function_docs_for_async_function() -> None:
    assert (
        format_foreign_function_docs("fetch_weather", fetch_weather)
        == """/**
 * Fetch the current weather for a city.
 */
async function fetch_weather(city: string): Promise<string>"""
    )


def test_format_foreign_function_docs_for_three_arg_tool() -> None:
    assert (
        format_foreign_function_docs("combine_user_details", combine_user_details)
        == """/**
 * Combine user details into a summary string.
 *
 * @param name The user name.
 * @param city The user's city.
 * @param active Whether the user is active.
 */
function combine_user_details(name: string, city: string, active: boolean): string"""
    )


def test_format_foreign_function_docs_for_single_line_tool_docstring() -> None:
    assert (
        format_foreign_function_docs("greet_user", greet_user)
        == """/**
 * Greet a user by name.
 */
function greet_user(name: string): string"""
    )


def test_format_annotation_handles_additional_shapes() -> None:
    assert {
        "list": _format_annotation(list[str]),
        "ellipsis_tuple": _format_annotation(tuple[int, ...]),
        "fixed_tuple": _format_annotation(tuple[int, str]),
        "dict": _format_annotation(dict[str, int]),
        "type": _format_annotation(type[str]),
        "union": _format_annotation(str | None),
        "generic": _format_annotation(GenericAlias(NameRecord, (str,))),
        "dotted": _format_annotation(NameRecord),
    } == {
        "list": "string[]",
        "ellipsis_tuple": "number[]",
        "fixed_tuple": "[number, string]",
        "dict": "Record<string, number>",
        "type": "new (...args: any[]) => string",
        "union": "string | null",
        "generic": "NameRecord<string>",
        "dotted": "NameRecord",
    }


def test_get_ptc_implementations_and_external_section_without_docs() -> None:
    implementations = get_ptc_implementations([normalize_name, greet_user, object()])

    assert implementations == {
        "normalize_name": normalize_name,
        "greet_user": greet_user,
    }
    assert (
        render_external_functions_section(implementations, add_docs=False)
        == "\n\nAvailable foreign functions:\n- normalize_name\n- greet_user"
    )
    assert render_external_functions_section({}, add_docs=False) == ""


def test_render_external_functions_section_with_docs() -> None:
    assert render_external_functions_section(
        {"normalize_name": normalize_name}, add_docs=True
    ).startswith("\n\nAvailable foreign functions:\n")


def test_format_typed_dict_structure_variants() -> None:
    assert OptionalUserLookup.__optional_keys__ == frozenset()
    assert _format_typed_dict_structure(OptionalUserLookup) == (
        "Return structure `OptionalUserLookup`:\n"
        "- id: number (required)\n"
        "- nickname?: string (optional)"
    )
    assert _format_typed_dict_structure(list[OptionalUserLookup]) == (
        "Contained `OptionalUserLookup` structure:\n"
        "Return structure `OptionalUserLookup`:\n"
        "- id: number (required)\n"
        "- nickname?: string (optional)"
    )
    assert _format_typed_dict_structure(str) is None


def test_render_typed_dict_definition_and_referenced_type_filtering() -> None:
    referenced = _collect_referenced_types(
        {
            "find_users_by_name": find_users_by_name,
            "duplicate": find_users_by_name,
            "summarize_lookup": summarize_lookup,
            "normalize_name": normalize_name,
            "bare_tool": BareTool(),
        }
    )

    assert referenced == [UserLookup]
    assert _render_typed_dict_definition(OptionalUserLookup) == (
        "type OptionalUserLookup = {\n  id: number\n  nickname?: string\n}"
    )


def test_format_foreign_function_docs_fallbacks() -> None:
    assert format_foreign_function_docs("bare_tool", BareTool()) == (
        "function bare_tool(...args: any[]): any"
    )
    assert format_foreign_function_docs("no_doc", no_doc) == (
        "function no_doc(value: number): any"
    )
    assert format_foreign_function_docs("async_tool", async_tool) == (
        """/**
 * Resolve a city name asynchronously.
 */
async function async_tool(city: string): Promise<string>"""
    )


def test_get_return_annotation_without_hints() -> None:
    assert _get_return_annotation(no_doc) is inspect.Signature.empty


def test_render_foreign_function_section() -> None:
    actual = render_foreign_function_section(
        {
            "find_users_by_name": find_users_by_name,
            "get_user_location": get_user_location,
            "get_city_for_location": get_city_for_location,
            "combine_user_details": combine_user_details,
            "greet_user": greet_user,
            "normalize_name": normalize_name,
            "fetch_weather": fetch_weather,
        }
    )

    assert (
        actual
        == """Available foreign functions:

These are JavaScript-callable foreign functions exposed inside QuickJS. The TypeScript-style signatures below document argument and return shapes.

```ts
/**
 * Find users with the given name.
 *
 * @param name The user name to search for.
 */
function find_users_by_name(name: string): UserLookup[]

/**
 * Get the location id for a user.
 *
 * @param user_id The user identifier.
 */
function get_user_location(user_id: number): number

/**
 * Get the city for a location.
 *
 * @param location_id The location identifier.
 */
function get_city_for_location(location_id: number): string

/**
 * Combine user details into a summary string.
 *
 * @param name The user name.
 * @param city The user's city.
 * @param active Whether the user is active.
 */
function combine_user_details(name: string, city: string, active: boolean): string

/**
 * Greet a user by name.
 */
function greet_user(name: string): string

/**
 * Normalize a user name for matching.
 */
function normalize_name(name: string): string

/**
 * Fetch the current weather for a city.
 */
async function fetch_weather(city: string): Promise<string>
```

Referenced types:
```ts
type UserLookup = {
  id: number
  name: string
}
```"""  # noqa: E501
    )
