"""Shared recursive types for JSON-compatible data."""

from pydantic import JsonValue as PydanticJsonValue, TypeAdapter

type JsonScalar = str | int | float | bool | None

# Plain re-export, not a `type` alias: `pydantic.JsonValue` is already a
# `TypeAliasType`, and wrapping it in a second one defeats pydantic's recursive
# alias collapsing, which more than doubles the length of validation-error
# titles on every JSON payload this module validates.
JsonValue = PydanticJsonValue

type JsonObject = dict[str, JsonValue]

JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)
JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
