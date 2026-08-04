"""Regression tests for the main-agent recursion limit default.

Guards that the runnable-config default the agent applies via `.with_config`
stays at the intended value and remains single-sourced from the manifest, so a
stray edit to either constant is caught immediately.
"""

from __future__ import annotations

from deepagents_code import config
from deepagents_code.config_manifest import RECURSION_LIMIT_DEFAULT


def test_config_recursion_limit_is_default() -> None:
    """The module-level runnable config pins the raised default."""
    assert config.config["recursion_limit"] == 2000


def test_config_recursion_limit_single_sourced() -> None:
    """The runnable-config default is sourced from the manifest constant."""
    assert config.config["recursion_limit"] == RECURSION_LIMIT_DEFAULT
