from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def _system_message_as_text(message: SystemMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)


def test_system_prompt_snapshot_with_execute(snapshots_dir: Path) -> None:
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
    actual = _system_message_as_text(system_messages[0])
    if snapshot_path.exists():
        expected = snapshot_path.read_text()
        assert actual == expected
    else:
        snapshot_path.write_text(actual)
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)


def test_system_prompt_snapshot_without_execute(snapshots_dir: Path) -> None:
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
    actual = _system_message_as_text(system_messages[0])
    if snapshot_path.exists():
        expected = snapshot_path.read_text()
        assert actual == expected
    else:
        snapshot_path.write_text(actual)
        msg = f"Created snapshot at {snapshot_path}. Re-run tests."
        raise AssertionError(msg)
