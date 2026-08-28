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
