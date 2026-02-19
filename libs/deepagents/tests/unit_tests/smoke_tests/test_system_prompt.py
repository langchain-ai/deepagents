from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.utils import create_file_data
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _system_message_as_text(message: SystemMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)


def _assert_snapshot(snapshot_path: Path, actual: str, *, update_snapshots: bool) -> None:
    if update_snapshots or not snapshot_path.exists():
        snapshot_path.write_text(actual)
        if update_snapshots:
            return
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)

    expected = snapshot_path.read_text()
    assert actual == expected


def test_system_prompt_snapshot_with_execute(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))
    backend = LocalShellBackend(root_dir=Path.cwd(), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    agent.invoke({"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

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
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)

    agent.invoke({"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

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
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt="You are Bobby a virtual assistant for company X",
    )

    agent.invoke({"messages": [HumanMessage(content="hi")]})

    history = model.call_history
    assert len(history) >= 1

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "custom_system_message.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )


def test_system_prompt_with_memory_and_skills(snapshots_dir: Path, *, update_snapshots: bool) -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))

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

    agent.invoke({"messages": [HumanMessage(content="hi")], "files": files})

    history = model.call_history
    assert len(history) >= 1

    messages = history[0]["messages"]
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) >= 1

    snapshot_path = snapshots_dir / "system_prompt_with_memory_and_skills.md"
    _assert_snapshot(
        snapshot_path,
        _system_message_as_text(system_messages[0]),
        update_snapshots=update_snapshots,
    )
