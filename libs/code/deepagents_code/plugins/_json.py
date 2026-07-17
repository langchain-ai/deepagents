"""Internal JSON normalization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.plugins.models import JsonObject, JsonValue


def json_value(value: object) -> JsonValue | None:
    """Normalize a decoded value to the supported JSON type.

    Returns:
        The normalized value, or `None` for an unsupported value.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        normalized: list[JsonValue] = []
        for item in value:
            converted = json_value(item)
            if converted is not None or item is None:
                normalized.append(converted)
        return normalized
    if isinstance(value, dict):
        normalized_object: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            converted = json_value(item)
            if converted is not None or item is None:
                normalized_object[key] = converted
        return normalized_object
    return None


def json_object(value: object) -> JsonObject:
    """Normalize a decoded value to a JSON object.

    Returns:
        The normalized object, or an empty object for a non-object value.
    """
    converted = json_value(value)
    return converted if isinstance(converted, dict) else {}
