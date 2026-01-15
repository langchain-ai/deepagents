from pathlib import Path
from textwrap import dedent

import requests
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend


def _write_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_summarize_continues_task(tmp_path: Path):
    # Pull sample file
    response = requests.get(
        "https://raw.githubusercontent.com/langchain-ai/langchain/3356d0555725c3e0bbb9408c2b3f554cad2a6ee2/libs/partners/openai/langchain_openai/chat_models/base.py"
    )

    # Populate backend
    root = tmp_path
    fp = root / "base.py"
    _write_file(fp, response.text)
    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    checkpointer = InMemorySaver()

    model = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
    model.profile["max_input_tokens"] = 30_000

    system_prompt = dedent(
        """
        ## File Reading Best Practices

        When exploring codebases or reading multiple files, use pagination to prevent context overflow.

        **Pattern for codebase exploration:**
        1. First scan: `read_file(path, limit=100)` - See file structure and key sections
        2. Targeted read: `read_file(path, offset=100, limit=200)` - Read specific sections if needed
        3. Full read: Only use `read_file(path)` without limit when necessary for editing

        **When to paginate:**
        - Reading any file >500 lines
        - Exploring unfamiliar codebases (always start with limit=100)
        - Reading multiple files in sequence

        **When full read is OK:**
        - Small files (<500 lines)
        - Files you need to edit immediately after reading
        """
    )

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[],
        backend=backend,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1"}}
    input_message = {
        "role": "user",
        "content": "Can you read the entirety of base.py and summarize it?",
    }
    result = agent.invoke({"messages": [input_message]}, config)

    # Check we summarized
    assert result["messages"][0].additional_kwargs["lc_source"] == "summarization"
    # Check we got to the end of the file
    for message in reversed(result["messages"]):
        if message.type == "tool":
            assert message.content.endswith("4609\t    )")
