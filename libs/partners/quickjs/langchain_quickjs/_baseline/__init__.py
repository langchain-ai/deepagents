"""Built-in baseline interpreter extensions for Code Interpreter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_quickjs._baseline.edit_file import EditFileExtension
from langchain_quickjs._baseline.filesystem import FilesystemExtension
from langchain_quickjs._baseline.glob import GlobExtension
from langchain_quickjs._baseline.llm import LlmExtension
from langchain_quickjs._baseline.shared import BASELINE_RESERVED_GLOBALS
from langchain_quickjs._baseline.subagent import SubagentExtension

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from langchain_quickjs._extensions import InterpreterExtension


def build_baseline_extensions(
    *,
    subagent_max_in_flight: int,
    llm_max_in_flight: int,
    subagent_timeout_s: float | None,
    llm_timeout_s: float | None,
    llm_model: str | BaseChatModel | None,
) -> list[InterpreterExtension]:
    """Return default-on baseline extensions in registration order."""
    return [
        SubagentExtension(
            max_in_flight=subagent_max_in_flight,
            timeout_s=subagent_timeout_s,
        ),
        LlmExtension(
            max_in_flight=llm_max_in_flight,
            timeout_s=llm_timeout_s,
            default_model=llm_model,
        ),
        FilesystemExtension(),
        GlobExtension(),
        EditFileExtension(),
    ]


def extension_export_names(ext: InterpreterExtension) -> set[str]:
    """Return extension-owned global exports used for collision checks."""
    return set(ext.exported_globals)


__all__ = [
    "BASELINE_RESERVED_GLOBALS",
    "EditFileExtension",
    "FilesystemExtension",
    "GlobExtension",
    "LlmExtension",
    "SubagentExtension",
    "build_baseline_extensions",
    "extension_export_names",
]
