"""Tests for the rubric grader's tool module contract."""

from __future__ import annotations

from deepagents_code import _rubric_grader_tools


def test_module_keeps_concrete_annotations() -> None:
    """The grader tool module must not enable postponed annotations.

    The wrappers reuse the SDK filesystem tools' `args_schema`, so LangChain
    identifies the injected `ToolRuntime` from each wrapper's own annotations.
    Under postponed evaluation those annotations are strings, the runtime is not
    recognized as injected, and it is dropped during input validation. A module
    that imports `annotations` from `__future__` binds that feature as a module
    attribute, so its absence pins the requirement.
    """
    assert getattr(_rubric_grader_tools, "annotations", None) is None
