from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.utils import create_file_data
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _smoke_model() -> GenericFakeChatModel:
    """Return a fake model with enough canned responses for prompt snapshot tests."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="hello!") for _ in range(4)]))


def _system_message_as_text(message: SystemMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)


def _invoke_for_snapshot(agent: object, payload: dict[str, Any]) -> None:
    """Invoke the agent and tolerate fake-model exhaustion after the first call."""
    try:
        if not hasattr(agent, "invoke"):
            msg = f"Expected compiled agent with invoke(), got {type(agent)!r}"
            raise TypeError(msg)
        agent.invoke(payload)
    except RuntimeError as exc:
        if "StopIteration" not in str(exc):
            raise


def _assert_snapshot(snapshot_path: Path, actual: str, *, update_snapshots: bool) -> None:
    if update_snapshots or not snapshot_path.exists():
        snapshot_path.write_text(actual, encoding="utf-8")
        if update_snapshots:
            return
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)

    expected = snapshot_path.read_text(encoding="utf-8")
    assert actual == expected


def _tools_as_openai_snapshot(tools: list[Any]) -> str:
    formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
    return json.dumps(formatted_tools, indent=2, sort_keys=True) + "\n"


def _assert_tools_snapshot(
    snapshots_dir: Path,
    snapshot_name: str,
    tools: list[Any],
    *,
    update_snapshots: bool,
) -> None:
    snapshot_path = snapshots_dir / snapshot_name
    _assert_snapshot(
        snapshot_path,
        _tools_as_openai_snapshot(tools),
        update_snapshots=update_snapshots,
    )


def test_system_prompt_snapshot_with_execute(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = _smoke_model()
    backend = LocalShellBackend(root_dir=Path.cwd(), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

    _assert_tools_snapshot(
        snapshots_dir,
        "system_prompt_with_execute_tools.json",
        history[0]["tools"],
        update_snapshots=update_snapshots,
    )

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "system_prompt_with_execute.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )


def test_system_prompt_snapshot_without_execute(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

    _assert_tools_snapshot(
        snapshots_dir,
        "system_prompt_without_execute_tools.json",
        history[0]["tools"],
        update_snapshots=update_snapshots,
    )

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "system_prompt_without_execute.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )


def test_custom_system_message_snapshot(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt="You are Bobby a virtual assistant for company X",
    )

    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

    _assert_tools_snapshot(
        snapshots_dir,
        "custom_system_message_tools.json",
        history[0]["tools"],
        update_snapshots=update_snapshots,
    )

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "custom_system_message.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )


def test_system_prompt_snapshot_with_sync_and_async_subagents(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = _smoke_model()
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        subagents=[
            {
                "name": "code-reviewer",
                "description": "Reviews code for quality and security issues",
                "system_prompt": "You are a code reviewer. Analyze code for bugs, security vulnerabilities, and style issues.",
            },
            {
                "name": "remote-researcher",
                "description": "Researches topics on a remote LangGraph server",
                "graph_id": "research_graph",
                "url": "http://localhost:8123",
            },
            {
                "name": "remote-analyst",
                "description": "Analyzes data on a remote LangGraph server",
                "graph_id": "analysis_graph",
                "url": "http://localhost:8123",
            },
        ],
    )

    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

    _assert_tools_snapshot(
        snapshots_dir,
        "system_prompt_with_sync_and_async_subagents_tools.json",
        history[0]["tools"],
        update_snapshots=update_snapshots,
    )

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "system_prompt_with_sync_and_async_subagents.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )


def test_system_prompt_with_memory_and_skills(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = _smoke_model()

    agent = create_deep_agent(
        model=model,
        memory=["/memory/AGENTS.md", "/memory/user/AGENTS.md"],
        skills=["/skills/user/", "/skills/project/"],
    )

    user_skill_content = """\
---
name: web-research
description: Structured approach to conducting thorough web research on any topic
---

# Web Research Skill

## When to Use
- User asks you to research a topic
- You need to gather information from the web
"""

    project_skill_content = """\
---
name: code-review
description: Systematic code review process following best practices and style guides
---

# Code Review Skill

## When to Use
- User asks you to review code
- You need to provide feedback on a pull request
"""

    memory_content = """\
# Project Memory

- Always use Python type hints
- Prefer functional programming patterns
"""

    user_memory_content = """\
# User Memory

- Preferred language: Python
- Always add docstrings to public functions
"""

    files = {
        "/skills/user/web-research/SKILL.md": create_file_data(user_skill_content),
        "/skills/project/code-review/SKILL.md": create_file_data(project_skill_content),
        "/memory/AGENTS.md": create_file_data(memory_content),
        "/memory/user/AGENTS.md": create_file_data(user_memory_content),
    }

    _invoke_for_snapshot(agent, {"messages": [HumanMessage(content="hi")], "files": files})

    history = model.call_history
    assert len(history) >= 1

    _assert_tools_snapshot(
        snapshots_dir,
        "system_prompt_with_memory_and_skills_tools.json",
        history[0]["tools"],
        update_snapshots=update_snapshots,
    )

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "system_prompt_with_memory_and_skills.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )
