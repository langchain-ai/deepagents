from __future__ import annotations

import importlib


def test_fleet_deepagents_export_imports_against_talon_deepagents_pin() -> None:
    package = importlib.import_module("fleet_deepagents_export")
    skills = importlib.import_module("fleet_deepagents_export.skills")

    assert package.StaticSkillsLoader is skills.StaticSkillsLoader
