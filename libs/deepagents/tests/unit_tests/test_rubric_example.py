"""Tests for the runnable rubric middleware example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _DotenvModule(ModuleType):
    """Minimal `dotenv` replacement for importing the example."""

    @staticmethod
    def find_dotenv(*, usecwd: bool) -> str:
        assert usecwd
        return ""

    @staticmethod
    def load_dotenv(dotenv_path: str) -> bool:
        return bool(dotenv_path)


def _load_example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "dotenv", _DotenvModule("dotenv"))
    script_path = Path(__file__).parents[4] / "examples" / "rubric_middleware" / "rubric_agent.py"
    spec = importlib.util.spec_from_file_location("rubric_agent_example", script_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load rubric example from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
