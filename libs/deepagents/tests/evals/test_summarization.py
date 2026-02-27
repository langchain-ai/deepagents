import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
import requests
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.load import load
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
    compute_summarization_defaults,
)
from tests.evals.utils import AgentTrajectory, run_agent

# URL for a large file that will trigger summarization
LARGE_FILE_URL = "https://raw.githubusercontent.com/langchain-ai/deepagents/5c90376c02754c67d448908e55d1e953f54b8acd/libs/deepagents/deepagents/middleware/summarization.py"

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

ls_client = Client()


def _write_file(p: Path, content: str) -> None:
    """Helper to write a file, creating parent directories."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _setup_summarization_test(
    tmp_path: Path,
    model_name: str,
    max_input_tokens: int,
    middleware: Sequence[AgentMiddleware] = (),
    *,
    include_compact_tool: bool = False,
) -> tuple[Any, FilesystemBackend, Path]:
    """Common setup for summarization tests.

    Returns:
        Tuple of `(agent, backend, root_path)`
    """
    response = requests.get(LARGE_FILE_URL, timeout=30)
    response.raise_for_status()

    root = tmp_path
    fp = root / "summarization.py"
    _write_file(fp, response.text)

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    checkpointer = InMemorySaver()

    model = init_chat_model(model_name)
    if model.profile is None:
        model.profile = {}
    # Lower artificially to trigger summarization more easily
    model.profile["max_input_tokens"] = max_input_tokens

    all_middleware: list[AgentMiddleware] = list(middleware)
    if include_compact_tool:
        summarization_defaults = compute_summarization_defaults(model)
        summarization_middleware = SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        )
        all_middleware.append(SummarizationToolMiddleware(summarization_middleware))

    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        backend=backend,
        checkpointer=checkpointer,
        middleware=all_middleware,
    )

    return agent, backend, root


@pytest.mark.langsmith
def test_summarize_continues_task(tmp_path: Path, model: str) -> None:
    """Test that summarization triggers and the agent can continue reading a large file."""
    agent, _, _ = _setup_summarization_test(tmp_path, model, 15_000)
    thread_id = uuid.uuid4().hex[:8]

    trajectory = run_agent(
        agent,
        model=model,
        query="Can you read the entirety of summarization.py, 500 lines at a time, and summarize it?",
        thread_id=thread_id,
    )

    # Check we summarized
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    assert state.values["_summarization_event"]

    # Verify the agent made substantial progress reading the file after summarization.
    # We check the highest line number seen across all tool observations to confirm
    # the agent continued working after context was summarized.
    max_line_seen = 0
    reached_eof = False

    for step in trajectory.steps:
        for obs in step.observations:
            # Check for EOF error (indicates agent tried to read past end)
            if "exceeds file length" in obs.content:
                reached_eof = True
            # Extract line numbers from formatted output (e.g., "4609\t    )")
            line_numbers = re.findall(r"^\s*(\d+)\t", obs.content, re.MULTILINE)
            if line_numbers:
                max_line_seen = max(max_line_seen, *[int(n) for n in line_numbers])

    assert max_line_seen >= 959 or reached_eof, (
        f"Expected agent to make substantial progress reading file. Max line seen: {max_line_seen}, reached EOF: {reached_eof}"
    )


@pytest.mark.langsmith
def test_summarization_offloads_to_filesystem(tmp_path: Path, model: str) -> None:
    """Test that conversation history is offloaded to filesystem during summarization.

    This verifies the summarization middleware correctly writes conversation history
    as markdown to the backend at /conversation_history/{thread_id}.md.
    """
    agent, _, root = _setup_summarization_test(tmp_path, model, 15_000)
    thread_id = uuid.uuid4().hex[:8]

    _ = run_agent(
        agent,
        model=model,
        query="Can you read the entirety of summarization.py, 500 lines at a time, and summarize it?",
        thread_id=thread_id,
    )

    # Check we summarized
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    assert state.values["_summarization_event"]

    # Verify conversation history was offloaded to filesystem
    conversation_history_root = root / "conversation_history"
    assert conversation_history_root.exists(), f"Conversation history root directory not found at {conversation_history_root}"

    # Verify the markdown file exists for thread_id
    history_file = conversation_history_root / f"{thread_id}.md"
    assert history_file.exists(), f"Expected markdown file at {history_file}"

    # Read and verify markdown content
    content = history_file.read_text()

    # Should have timestamp header(s) from summarization events
    assert "## Summarized at" in content, "Missing timestamp header in markdown file"

    # Should contain human-readable message content (from get_buffer_string)
    assert "Human:" in content or "AI:" in content, "Missing message content in markdown file"

    # Verify the summary message references the conversation_history path
    summary_message = state.values["_summarization_event"]["summary_message"]
    assert "conversation_history" in summary_message.content
    assert f"{thread_id}.md" in summary_message.content

    # --- Needle in the haystack follow-up ---
    # Ask about a specific detail from the beginning of the file that was read
    # before summarization. The agent should read the conversation history to find it.
    # The first standard library import in summarization.py (after `from __future__`) is `import base64`.
    followup_trajectory = run_agent(
        agent,
        model=model,
        query=(
            "What is the first standard library import in summarization.py? (After "
            "the `from __future__` import.) Check the conversation history if needed."
        ),
        thread_id=thread_id,
    )

    # The agent should retrieve the answer from the conversation history
    final_answer = followup_trajectory.answer

    # Check that the answer mentions "base64" (the first standard library import)
    assert "logging" in final_answer.lower(), f"Expected agent to find 'logging' as the first import. Got: {final_answer}"


def _called_compact(trajectory: AgentTrajectory) -> bool:
    """Check if `compact_conversation` was called in any step."""
    return any(tc.get("name") == "compact_conversation" for step in trajectory.steps for tc in step.action.tool_calls)


def _load_seed_messages() -> list[AnyMessage]:
    """Load seed messages from the shared LangSmith run."""
    run = ls_client.read_run("7c1618cc-0447-40b4-8c4e-c4dc5ad32c21")
    return load(run.outputs["messages"])


@pytest.mark.langsmith
def test_compact_tool_new_task(tmp_path: Path, model: str) -> None:

    agent, _, _ = _setup_summarization_test(tmp_path, model, 35_000, include_compact_tool=True)

    seed = _load_seed_messages()
    query = "Thanks. Let's move on to a completely new task. To prepare, first spec out how to upgrade a web app to Typescript 5.5"
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert _called_compact(trajectory)


@pytest.mark.langsmith
def test_compact_tool_not_overly_sensitive(tmp_path: Path, model: str) -> None:

    agent, _, _ = _setup_summarization_test(tmp_path, model, 35_000, include_compact_tool=True)

    seed = _load_seed_messages()
    query = "Moving on, what are the two primary OpenAI APIs supported?"
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert not _called_compact(trajectory)


@pytest.mark.langsmith
def test_compact_tool_large_reads(tmp_path: Path, model: str) -> None:
    another_large_file = "https://raw.githubusercontent.com/langchain-ai/deepagents/5c90376c02754c67d448908e55d1e953f54b8acd/libs/deepagents/deepagents/middleware/filesystem.py"

    response = requests.get(another_large_file, timeout=30)
    response.raise_for_status()

    agent, backend, _ = _setup_summarization_test(
        tmp_path,
        model,
        35_000,
        middleware=[ModelCallLimitMiddleware(run_limit=3)],
        include_compact_tool=True,
    )
    backend.upload_files([("/filesystem.py", response.content)])

    seed = _load_seed_messages()
    query = "OK, done with that. Now do the same for filesystem.py."
    trajectory = run_agent(
        agent,
        model=model,
        query=[*seed, HumanMessage(query)],
    )
    assert _called_compact(trajectory)
