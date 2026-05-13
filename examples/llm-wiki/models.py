"""Shared data models for LLM wiki workflows."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from collections.abc import Callable, Sequence


Mode = Literal["init", "ingest", "query", "lint"]


@dataclass(frozen=True)
class RunnerConfig:
    """Parsed runner configuration."""

    mode: Mode
    topic: str
    repo: str
    owner: str | None
    topic_dir: Path
    sources: tuple[Path, ...]
    note: str | None
    question: str | None
    model: str | None
    description: str | None
    review: bool


@dataclass(frozen=True)
class CliDeps:
    """Injectable dependencies for tests."""

    run_langsmith_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
    run_agent_mode: Callable[[Path, str, str, str | None], str]
    run_agent_review_mode: Callable[[Path, str, str, str | None], str]
    ask_user: Callable[[str], str]
    tempdir_factory: Callable[[], tempfile.TemporaryDirectory[str]]


@dataclass(frozen=True)
class RunResult:
    """Output from a runner invocation."""

    answer: str | None
    hub_url: str | None
