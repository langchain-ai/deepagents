"""Public contract for dcode Python extensions."""

from deepagents_code.extensions.api import ExtensionAPI
from deepagents_code.extensions.models import (
    ExtensionError,
    ExtensionFile,
    LoadedExtension,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)
from deepagents_code.extensions.registry import ExtensionRegistry
from deepagents_code.extensions.settings import (
    ExtensionSettings,
    TrustPolicy,
    load_extension_settings,
)
from deepagents_code.extensions.trust import (
    is_project_extensions_trusted,
    trust_project_extensions,
)

__all__ = [
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionFile",
    "ExtensionRegistry",
    "ExtensionSettings",
    "LoadedExtension",
    "SourceInfo",
    "TrustPolicy",
    "UnitOrigin",
    "UnitScope",
    "is_project_extensions_trusted",
    "load_extension_settings",
    "trust_project_extensions",
]
