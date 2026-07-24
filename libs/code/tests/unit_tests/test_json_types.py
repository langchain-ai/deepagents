"""Tests for shared JSON type aliases and validators."""

import pytest
from pydantic import ValidationError

from deepagents_code.json_types import (
    JSON_OBJECT_ADAPTER,
    JSON_VALUE_ADAPTER,
    JsonObject,
    JsonValue,
)
from deepagents_code.plugins.models import (
    JsonObject as PluginJsonObject,
    JsonValue as PluginJsonValue,
)


def test_plugin_json_types_are_compatibility_reexports() -> None:
    """Plugin imports resolve to the canonical shared aliases."""
    assert PluginJsonValue is JsonValue
    assert PluginJsonObject is JsonObject


def test_json_adapters_validate_recursive_values() -> None:
    """Cached adapters validate recursive JSON values and objects."""
    value = {"nested": [1, True, None, {"name": "dcode"}]}

    assert JSON_VALUE_ADAPTER.validate_python(value) == value
    assert JSON_OBJECT_ADAPTER.validate_python(value) == value


def test_json_object_adapter_rejects_non_object_root() -> None:
    """Object validation rejects a JSON array at the root."""
    with pytest.raises(ValidationError):
        JSON_OBJECT_ADAPTER.validate_python(["not", "an", "object"])
