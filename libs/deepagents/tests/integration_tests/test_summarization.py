import json
from pathlib import Path
from textwrap import dedent

import pytest
import requests
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

# URL for a large file that will trigger summarization
LARGE_FILE_URL = "https://raw.githubusercontent.com/langchain-ai/langchain/3356d0555725c3e0bbb9408c2b3f554cad2a6ee2/libs/partners/openai/langchain_openai/chat_models/base.py"

SYSTEM_PROMPT = dedent(
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


def _write_file(p: Path, content: str) -> None:
    """Helper to write a file, creating parent directories."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _setup_summarization_test(tmp_path: Path, model_name: str):
    """Common setup for summarization tests.

    Returns:
        Tuple of `(agent, backend, root_path, config)`
    """
    response = requests.get(LARGE_FILE_URL, timeout=30)
    response.raise_for_status()

    root = tmp_path
    fp = root / "base.py"
    _write_file(fp, response.text)

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    checkpointer = InMemorySaver()

    model = init_chat_model(model_name)
    if model.profile is None:
        model.profile = {}
    # Lower artificially to trigger summarization more easily
    model.profile["max_input_tokens"] = 30_000

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        backend=backend,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1"}}

    return agent, backend, root, config


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("anthropic:claude-sonnet-4-5-20250929", id="claude-sonnet"),
    ],
)
def test_summarize_continues_task(tmp_path: Path, model_name: str) -> None:
    """Test that summarization triggers and the agent can continue reading a large file."""
    agent, _, _, config = _setup_summarization_test(tmp_path, model_name)

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
            break


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("anthropic:claude-sonnet-4-5-20250929", id="claude-sonnet"),
    ],
)
def test_summarization_offloads_to_filesystem(tmp_path: Path, model_name: str) -> None:
    """Test that conversation history is offloaded to filesystem during summarization.

    This verifies the summarization middleware correctly writes conversation history
    JSON files to the backend at /conversation_history/{thread_id}/{timestamp}.json.
    """
    agent, _, root, config = _setup_summarization_test(tmp_path, model_name)

    input_message = {
        "role": "user",
        "content": "Can you read the entirety of base.py and summarize it?",
    }
    result = agent.invoke({"messages": [input_message]}, config)

    # Check we summarized
    assert result["messages"][0].additional_kwargs["lc_source"] == "summarization"

    # Verify conversation history was offloaded to filesystem
    conversation_history_root = root / "conversation_history"
    assert conversation_history_root.exists(), f"Conversation history root directory not found at {conversation_history_root}"

    # Find all JSON files in conversation_history (may be multiple from main agent + subagents)
    json_files = list(conversation_history_root.rglob("*.json"))
    assert len(json_files) >= 1, f"Expected at least one JSON file in {conversation_history_root}, found {len(json_files)}"

    # Verify structure of at least one offloaded conversation history file
    history_file = json_files[0]
    with history_file.open() as f:
        payload = json.load(f)

    # Check required fields in the payload
    assert "timestamp" in payload, "Missing 'timestamp' in offloaded payload"
    assert "thread_id" in payload, "Missing 'thread_id' in offloaded payload"
    assert "message_count" in payload, "Missing 'message_count' in offloaded payload"
    assert "messages" in payload, "Missing 'messages' in offloaded payload"

    # Verify messages were actually stored
    assert payload["message_count"] > 0, "No messages were offloaded"
    assert len(payload["messages"]) == payload["message_count"]

    # Verify the summary message references the conversation_history path
    summary_message = result["messages"][0]
    assert "conversation_history" in summary_message.content
