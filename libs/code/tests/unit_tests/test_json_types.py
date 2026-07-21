"""Tests for shared JSON type aliases."""

from deepagents_code.json_types import JsonObject, JsonValue
from deepagents_code.plugins.models import (
    JsonObject as PluginJsonObject,
    JsonValue as PluginJsonValue,
)


def test_plugin_json_types_are_compatibility_reexports() -> None:
    """Plugin imports resolve to the canonical shared aliases."""
    assert PluginJsonValue is JsonValue
    assert PluginJsonObject is JsonObject
