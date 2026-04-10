"""Eval tests for prompt caching with middleware ordering.

Tests whether the middleware ordering affects prompt cache efficiency.
Two test types:
1. Single-turn tests with varying memory content (like test_memory.py but
   every test has memory, no missing files, no multi-step tool calls).
2. Multi-turn conversation test: one agent, 10 sequential user messages,
   same memory throughout.

Written for the deepagents eval suite to measure prompt cache behavior.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("prompt_cache")]


# ---------------------------------------------------------------------------
# Part 1: Single-turn tests with varying memory content
#
# Each test creates a fresh agent with DIFFERENT memory content and asks
# a single question. No missing files, no tool calls. Every call exercises
# the middleware ordering difference equally.
# ---------------------------------------------------------------------------


@pytest.mark.langsmith
def test_cache_single_turn_project_info(model: BaseChatModel) -> None:
    """Single-turn: recall project name from memory."""
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
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("TurboWidget"))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_user_prefs(model: BaseChatModel) -> None:
    """Single-turn: recall user preferences from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/user/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/user/AGENTS.md": """# User Preferences

- Preferred language: Python
- Editor: VS Code
- Formatting: bullet points over paragraphs
""",
        },
        query="What is my preferred programming language? Answer briefly.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("Python", case_insensitive=True))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_api_docs(model: BaseChatModel) -> None:
    """Single-turn: recall API endpoints from memory."""
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
- DELETE /users/{id} - Deletes a user
""",
        },
        query="List the API endpoints briefly.",
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
def test_cache_single_turn_deploy_info(model: BaseChatModel) -> None:
    """Single-turn: recall deployment details from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/ops/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/ops/AGENTS.md": """# Deployment Guide

## Infrastructure
- Cloud provider: AWS
- Compute: ECS Fargate
- Database: RDS PostgreSQL
- Cache: ElastiCache Redis
""",
        },
        query="What cloud provider and database does this project use? Be concise.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("AWS", case_insensitive=True),
                final_text_contains("PostgreSQL", case_insensitive=True),
            )
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_team_info(model: BaseChatModel) -> None:
    """Single-turn: recall team structure from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/team/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/team/AGENTS.md": """# Team Structure

- Tech lead: Alice
- Backend: Bob, Charlie
- Frontend: Dana
- DevOps: Eve
""",
        },
        query="Who is the tech lead? Answer with just the name.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("Alice"))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_conventions(model: BaseChatModel) -> None:
    """Single-turn: recall coding conventions from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/style/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/style/AGENTS.md": """# Code Conventions

- Use snake_case for all Python identifiers
- Maximum line length: 88 characters
- Use type hints on all public functions
- Docstrings: Google style
""",
        },
        query="What naming convention should I use for Python? Answer briefly.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("snake_case", case_insensitive=True))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_testing_info(model: BaseChatModel) -> None:
    """Single-turn: recall testing strategy from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/testing/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/testing/AGENTS.md": """# Testing Strategy

- Framework: pytest
- Coverage target: 80%
- Integration tests use a real PostgreSQL database
- E2E tests run in CI via GitHub Actions
""",
        },
        query="What test framework and coverage target do we use? Be concise.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("pytest", case_insensitive=True),
                final_text_contains("80", case_insensitive=True),
            )
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_security(model: BaseChatModel) -> None:
    """Single-turn: recall security policy from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/security/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/security/AGENTS.md": """# Security Policy

- Authentication: OAuth 2.0 with JWT tokens
- Authorization: RBAC with three roles (admin, editor, viewer)
- All API calls require HTTPS
- Secrets stored in AWS Secrets Manager
""",
        },
        query="What authentication method does this project use? Answer briefly.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("OAuth", case_insensitive=True))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_roadmap(model: BaseChatModel) -> None:
    """Single-turn: recall project roadmap from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/roadmap/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/roadmap/AGENTS.md": """# Project Roadmap

## Q2 2026
- Migrate to PostgreSQL 17
- Add real-time notifications via WebSockets
- Launch mobile API v2
""",
        },
        query="What database migration is planned? Answer briefly.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(final_text_contains("PostgreSQL 17", case_insensitive=True))
        ),
    )


@pytest.mark.langsmith
def test_cache_single_turn_dependencies(model: BaseChatModel) -> None:
    """Single-turn: recall project dependencies from memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/deps/AGENTS.md"],
    )
    run_agent(
        agent,
        model=model,
        initial_files={
            "/deps/AGENTS.md": """# Key Dependencies

- Web framework: FastAPI 0.115
- ORM: SQLAlchemy 2.0
- Task queue: Celery with Redis broker
- HTTP client: httpx
""",
        },
        query="What web framework and ORM does this project use? Be concise.",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=1, tool_call_requests=0)
            .success(
                final_text_contains("FastAPI", case_insensitive=True),
                final_text_contains("SQLAlchemy", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Part 2: Multi-turn conversation test
#
# One agent, one conversation thread, 10 sequential messages.
# Same memory file throughout. Tests whether the cached prefix
# persists across turns within a single conversation.
# ---------------------------------------------------------------------------


@pytest.mark.langsmith
def test_cache_multi_turn_conversation(model: BaseChatModel) -> None:
    """Multi-turn: 10 messages in one conversation with consistent memory."""
    agent = create_deep_agent(
        model=model,
        memory=["/project/AGENTS.md"],
    )

    memory_content = """# Project Memory

This is the TurboWidget project. The main goal is to process widgets efficiently.

## Key Facts
- Project name: TurboWidget
- Primary language: Python
- Test framework: pytest
- Database: PostgreSQL
- Deployment: AWS ECS
- Cache: Redis
- Auth: OAuth 2.0 with JWT
- Team lead: Alice

## User Preferences
- Preferred language: Python
- Use bullet points over paragraphs
- Keep responses concise
"""

    turns = [
        ("What is the name of this project? Answer briefly.", "TurboWidget"),
        ("What programming language does it use?", "Python"),
        ("What test framework?", "pytest"),
        ("What database?", "PostgreSQL"),
        ("How is it deployed?", "AWS"),
        ("What caching layer?", "Redis"),
        ("What auth method?", "OAuth"),
        ("Who is the team lead?", "Alice"),
        ("What are my formatting preferences?", "bullet"),
        ("Give a one-line summary of this project.", "TurboWidget"),
    ]

    # Run all turns in the same thread
    thread_id = str(uuid.uuid4())
    for i, (query, expected) in enumerate(turns):
        run_agent(
            agent,
            model=model,
            initial_files={
                "/project/AGENTS.md": memory_content,
            },
            query=query,
            thread_id=thread_id,
            scorer=(
                TrajectoryScorer()
                .expect(agent_steps=1, tool_call_requests=0)
                .success(final_text_contains(expected, case_insensitive=True))
            ),
        )
