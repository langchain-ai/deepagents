from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends import FilesystemBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_agent_reads_skill_file() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="hello!")]))
    backend = FilesystemBackend(root_dir="/", virtual_mode=True)

    backend.write(
        "/skills/user/example/SKILL.md",
        "---\nname: example\ndescription: example skill\n---\n\n# Example\n",
    )

    backend.ls_info("/")
    agent = create_deep_agent(
        model=model,
        backend=backend,
        skills=["/skills/user"],
    )
    agent.invoke({"messages": [HumanMessage(content="use the example skill")]})

    history = model.call_history
    assert len(history) >= 1

    system_prompt = history[0]["messages"][0]
    system_text = system_prompt.content
    if isinstance(system_text, list):
        system_text = "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in system_text
        )

    assert "## Skills System" in system_text
    assert "You have access to a skills library" in system_text
