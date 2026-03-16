from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from tests.evals.utils import (
    TrajectoryScorer,
    file_contains,
    file_excludes,
    final_text_contains,
    final_text_excludes,
    run_agent,
    tool_call,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime
    from langchain_core.language_models import BaseChatModel


@pytest.mark.langsmith
def test_memory_basic_recall(model: BaseChatModel) -> None:
    """Agent recalls project context from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/project/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/project/AGENTS.md": """# Project Memory

This is the TurboWidget project. The main goal is to process widgets efficiently.

## Key Facts
- Project name: TurboWidget
- Primary language: Python
- Test framework: pytest
""",
        },
        query="What is the name of this project? Answer with just the project name.",
        # 1st step: answer directly using memory context, no tools needed.
        # 0 tool calls: memory is already loaded in context.
        scorer=(TrajectoryScorer().expect(agent_steps=1, tool_call_requests=0).success(final_text_contains("TurboWidget"))),
    )


@pytest.mark.langsmith
def test_memory_guided_behavior_naming_convention(model: BaseChatModel) -> None:
    """Agent follows naming convention guidelines from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/project/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/project/AGENTS.md": """# Project Guidelines

## Naming Conventions
All configuration files must use the prefix "config_" followed by the purpose.
Example: config_database.txt, config_settings.txt

This rule is mandatory. If a user requests a configuration file path that does not
follow this convention (e.g., "/api.txt"), create the correctly named config file
instead (e.g., "/config_api.txt") without asking for confirmation.
""",
        },
        query="Create a configuration file for API settings at /api.txt with content 'API_KEY=secret'.",
        # 1st step: write file following the naming convention from memory.
        # 1 tool call request: write_file.
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="write_file", step=1, args_contains={"file_path": "/config_api.txt"})],
            )
            .success(file_contains("/config_api.txt", "API_KEY=secret"))
        ),
    )


@pytest.mark.langsmith
def test_memory_influences_file_content(model: BaseChatModel) -> None:
    """Agent applies code style guidelines from memory when creating files."""
    agent = create_deep_agent(
        model=model,
        memory=["/style/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/style/AGENTS.md": """# Code Style Guide

## Comment Requirements
Every function must start with a comment line that says "# Purpose: " followed by a brief description.
""",
        },
        query="Write a simple Python function that adds two numbers to /add.py. Keep it minimal.",
        # 1st step: write file following the style guide from memory.
        # 1 tool call request: write_file.
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=2, tool_call_requests=1)
            .success(file_contains("/add.py", "# Purpose:"), file_contains("/add.py", "def "))
        ),
    )


@pytest.mark.langsmith
def test_memory_multiple_sources_combined(model: BaseChatModel) -> None:
    """Agent accesses information from multiple memory sources."""
    agent = create_deep_agent(
        model=model,
        memory=["/user/AGENTS.md", "/project/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/user/AGENTS.md": """# User Preferences

My preferred programming language is Python.
""",
            "/project/AGENTS.md": """# Project Info

The project uses the FastAPI framework.
""",
        },
        query="What programming language do I prefer and what framework does the project use? Be concise.",
        # 1st step: answer using both memory sources, no tools needed.
        # 0 tool calls: both memory files loaded in context.
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("Python", case_insensitive=True),
                final_text_contains("FastAPI", case_insensitive=True),
            )
        ),
    )


@pytest.mark.langsmith
def test_memory_with_missing_file_graceful(model: BaseChatModel) -> None:
    """Agent handles missing memory files gracefully and still functions."""
    agent = create_deep_agent(
        model=model,
        memory=["/missing/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        query="What is 5 + 3? Answer with just the number.",
        # 1st step: answer directly even though memory file is missing.
        # 0 tool calls: simple math doesn't require tools.
        scorer=TrajectoryScorer().expect(agent_steps=1, tool_call_requests=0),
    )


@pytest.mark.langsmith
def test_memory_prevents_unnecessary_file_reads(model: BaseChatModel) -> None:
    """Agent uses memory context instead of reading documentation files."""
    agent = create_deep_agent(
        model=model,
        memory=["/docs/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/docs/AGENTS.md": """# API Documentation

## Endpoints
- GET /users - Returns list of all users
- POST /users - Creates a new user
- GET /users/{id} - Returns a specific user
""",
            "/docs/api.md": "This file contains the same API documentation.",
        },
        query="What are the API endpoints? List them briefly.",
        # 1st step: answer using memory, no need to read /docs/api.md.
        # 0 tool calls: documentation already in memory.
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("/users", case_insensitive=True),
                final_text_contains("GET", case_insensitive=True),
            )
        ),
    )


@pytest.mark.langsmith
def test_memory_does_not_persist_transient_info(model: BaseChatModel) -> None:
    """Agent should not write temporary status updates into durable memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/project/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/project/AGENTS.md": "# Project Memory\n\nStable preferences and project facts live here.\n",
        },
        query="I'm at a coffee shop right now. What's 2 + 2?",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("4"),
                file_excludes("/project/AGENTS.md", "coffee shop"),
            )
        ),
    )


@pytest.mark.langsmith
def test_memory_updates_user_formatting_preference(model: BaseChatModel) -> None:
    """Agent writes durable formatting preferences into memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/project/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/project/AGENTS.md": "# Project Memory\n\nCurrent preferences:\n- Use clear technical writing.\n",
        },
        query="For future responses, I prefer bullet points over paragraphs.",
        scorer=(
            TrajectoryScorer()
            .success(
                file_contains("/project/AGENTS.md", "bullet points"),
                final_text_contains("bullet", case_insensitive=True),
            )
            .expect(
                tool_call_requests=1,
                tool_calls=[tool_call(name="edit_file", args_contains={"file_path": "/project/AGENTS.md"})],
            )
        ),
    )


@pytest.mark.langsmith
def test_memory_missing_file_graceful_without_claiming_context(model: BaseChatModel) -> None:
    """Agent handles a missing memory file without inventing its contents."""
    agent = create_deep_agent(
        model=model,
        memory=["/missing/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        query="What coding preferences are saved in memory? If none are available, say so briefly.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_excludes("snake_case", case_insensitive=True),
                final_text_excludes("pytest", case_insensitive=True),
            )
        ),
    )


@pytest.mark.langsmith
def test_memory_middleware_composite_backend(model: BaseChatModel) -> None:
    """Test that agent can access memory from store backend via composite backend routing."""
    store = InMemoryStore()
    now = datetime.now(UTC).isoformat()
    store.put(
        ("filesystem",),
        "/AGENTS.md",
        {
            "content": ["Your name is Jackson"],
            "created_at": now,
            "modified_at": now,
        },
    )

    def sample_backend(rt: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": StoreBackend(rt),
            },
        )

    agent = create_deep_agent(
        model=model,
        backend=sample_backend,
        memory=["/memories/AGENTS.md"],
        store=store,
    )

    # Agent should be able to answer based on memory file
    run_agent(
        agent,
        model=model,
        query="What is your name?",
        scorer=(TrajectoryScorer().expect(agent_steps=1, tool_call_requests=0).success(final_text_contains("Jackson"))),
    )
