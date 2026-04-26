from __future__ import annotations

import inspect
from typing import Any, NotRequired
from unittest.mock import patch

from langchain_core.tools import StructuredTool, tool
from typing_extensions import TypedDict

from langchain_monty._foreign_function_docs import (
    _collect_referenced_types,
    _get_return_annotation,
    _render_function_stub,
    _render_typed_dict_definition,
    _render_typed_dict_fields,
    _unwrap_typed_dict_annotation,
    format_annotation,
    format_foreign_function_docs,
    format_typed_dict_structure,
    get_foreign_function_mode,
    get_tool_doc_target,
    render_foreign_function_section,
)


class UserLookup(TypedDict):
    id: int
    name: str


class OptionalLookup(TypedDict, total=False):
    id: int
    nickname: NotRequired[str]


class EmptyLookup(TypedDict):
    pass


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


def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""
    return name.strip().lower()


async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"Weather for {city}"


def unannotated(value):
    return value


def dotted_annotation(value: str) -> str:
    return value


def fully_annotated(value: int) -> str:
    return str(value)


def optional_lookup() -> OptionalLookup:
    return {"id": 1}


def empty_lookup() -> EmptyLookup:
    return {}


def test_render_foreign_function_section() -> None:
    actual = render_foreign_function_section(
        {
            "find_users_by_name": find_users_by_name,
            "get_user_location": get_user_location,
            "get_city_for_location": get_city_for_location,
            "normalize_name": normalize_name,
            "fetch_weather": fetch_weather,
        }
    )

    assert (
        actual
        == '''Available foreign functions:

```python
def find_users_by_name(name: str) -> list[UserLookup]:
    """Find users with the given name.

    Args:
        name: The user name to search for.
    """

def get_user_location(user_id: int) -> int:
    """Get the location id for a user.

    Args:
        user_id: The user identifier.
    """

def get_city_for_location(location_id: int) -> str:
    """Get the city for a location.

    Args:
        location_id: The location identifier.
    """

def normalize_name(name: str) -> str:
    """Normalize a user name for matching."""

async def fetch_weather(city: str) -> str:
    """Fetch the current weather for a city."""
```

Referenced types:
```python
class UserLookup(TypedDict):
    id: int
    name: str
```'''
    )


def test_format_annotation_handles_fallback_cases() -> None:
    assert format_annotation(Any) == "Any"
    assert format_annotation(inspect.Signature.empty) == "Any"
    assert format_annotation(list[int]) == "list[int]"
    assert format_annotation("pkg.mod.CustomType") == "CustomType"


def test_format_typed_dict_structure_handles_optional_and_container_types() -> None:
    assert format_typed_dict_structure(OptionalLookup) == (
        "Return structure `OptionalLookup`:\n"
        "- id: int (optional)\n"
        "- nickname: str (optional)"
    )
    assert format_typed_dict_structure(list[UserLookup]) == (
        "Contained `UserLookup` structure:\n"
        "Return structure `UserLookup`:\n"
        "- id: int (required)\n"
        "- name: str (required)"
    )
    assert format_typed_dict_structure(str) is None


def test_get_tool_doc_target_and_mode_cover_tool_edge_cases() -> None:
    nameless_tool = StructuredTool.from_function(
        func=lambda value: value,
        name="nameless_tool",
        description="tool without docs",
    )
    assert get_tool_doc_target(find_users_by_name) is not None
    assert get_tool_doc_target(nameless_tool) is not None
    assert get_foreign_function_mode(find_users_by_name) == "sync"
    assert get_foreign_function_mode(find_users_by_name, async_context=True) == "async"
    assert get_foreign_function_mode(fetch_weather) == "async"


def test_format_foreign_function_docs_covers_unannotated_and_no_doc_cases() -> None:
    assert (
        format_foreign_function_docs("unannotated", unannotated)
        == "def unannotated(value)"
    )
    assert format_foreign_function_docs("fully_annotated", fully_annotated) == (
        "def fully_annotated(value: int) -> str"
    )


def test_collect_referenced_types_and_typed_dict_rendering_cover_fallbacks() -> None:
    referenced = _collect_referenced_types(
        {
            "find_users_by_name": find_users_by_name,
            "optional_lookup": optional_lookup,
            "empty_lookup": empty_lookup,
            "normalize_name": normalize_name,
        }
    )
    assert referenced == [UserLookup, OptionalLookup, EmptyLookup]
    assert _render_typed_dict_definition(EmptyLookup) == (
        "class EmptyLookup(TypedDict):\n    pass"
    )


def test_rendering_helpers_cover_fallback_and_empty_paths() -> None:
    annotation, prefix = _unwrap_typed_dict_annotation(tuple[UserLookup])
    assert (annotation, prefix) == (
        UserLookup,
        "Contained `UserLookup` structure:\n",
    )
    assert _render_typed_dict_fields(EmptyLookup, prefix="") is None

    with patch(
        "langchain_monty._foreign_function_docs.get_type_hints",
        side_effect=TypeError("boom"),
    ):
        rendered_definition = _render_typed_dict_definition(UserLookup)
        assert rendered_definition == (
            "class UserLookup(TypedDict):\n"
            "    id: test_foreign_function_docs)\n"
            "    name: test_foreign_function_docs)"
        )
        assert _get_return_annotation(fully_annotated) is inspect.Signature.empty
        assert (
            _collect_referenced_types(
                {
                    "find_users_by_name": find_users_by_name,
                    "optional_lookup": optional_lookup,
                    "normalize_name": normalize_name,
                }
            )
            == []
        )


def test_render_function_stub_covers_tool_target_and_signature_fallbacks() -> None:
    tool_without_target = StructuredTool.from_function(
        func=lambda value: value,
        name="tool_without_target",
        description="tool without callable target",
    )
    tool_without_target.func = None
    tool_without_target.coroutine = None
    assert _render_function_stub("tool_without_target", tool_without_target) == (
        "def tool_without_target(...)"
    )

    with patch(
        "langchain_monty._foreign_function_docs.get_type_hints",
        side_effect=TypeError("boom"),
    ):
        assert _render_function_stub("fully_annotated", fully_annotated) == (
            "def fully_annotated(...)"
        )


def test_render_foreign_function_section_without_referenced_types() -> None:
    assert render_foreign_function_section({"normalize_name": normalize_name}) == (
        "Available foreign functions:\n\n"
        "```python\n"
        'def normalize_name(name: str) -> str:\n    """Normalize a user '
        'name for matching."""\n'
        "```"
    )

    assert render_foreign_function_section(
        {"get_user_location": get_user_location},
        async_context=True,
    ) == (
        "Available foreign functions:\n\n"
        "```python\n"
        "async def get_user_location(user_id: int) -> int:\n"
        '    """Get the location id for a user.\n\n'
        "    Args:\n"
        "        user_id: The user identifier.\n"
        '    """\n'
        "```"
    )
