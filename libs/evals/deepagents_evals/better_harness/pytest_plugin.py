"""Pytest plugin that patches Deep Agents from a saved better-harness variant."""

from __future__ import annotations

from deepagents_evals.better_harness.variants import patch_deepagents_from_env

patch_deepagents_from_env()
