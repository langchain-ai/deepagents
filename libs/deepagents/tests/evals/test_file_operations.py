from __future__ import annotations

import pytest

from deepagents import create_deep_agent
from tests.evals.utils import TrajectoryExpectations, run_agent


@pytest.mark.langsmith
def test_read_file_seeded_state_backend_file(model: str) -> None:
    """Reads a seeded file and answers a question."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/foo.md": "alpha beta gamma\none two three four\n"},
        query="Read /foo.md and tell me the 3rd word on the 2nd line.",
        # 1st step: request a tool call to read /foo.md.
        # 2nd step: answer the question using the file contents.
        # 1 tool call request: read_file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1).require_final_text_contains(
            "three",
            case_insensitive=True,
        ),
    )


@pytest.mark.langsmith
def test_write_file_simple(model: str) -> None:
    """Writes a file then answers a follow-up."""
    agent = create_deep_agent(model=model, system_prompt="Your name is Foo Bar.")
    trajectory = run_agent(
        agent,
        model=model,
        query="Write your name to a file called /foo.md and then tell me your name.",
        # 1st step: request a tool call to write /foo.md.
        # 2nd step: tell the user the name.
        # 1 tool call request: write_file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert "Foo Bar" in trajectory.files["/foo.md"]
    assert "Foo Bar" in trajectory.steps[-1].action.text


@pytest.mark.langsmith
def test_write_files_in_parallel(model: str) -> None:
    """Writes two files in parallel then confirms."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        query='Write "bar" to /a.md and "bar" to /b.md. Do the writes in parallel, then confirm you did it.',
        # 1st step: request 2 write_file tool calls in parallel.
        # 2nd step: confirm the writes.
        # 2 tool call requests: write_file to /a.md and write_file to /b.md.
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=2)
            .require_tool_call(step=1, name="write_file", args_contains={"file_path": "/a.md"})
            .require_tool_call(step=1, name="write_file", args_contains={"file_path": "/b.md"})
        ),
    )
    assert trajectory.files["/a.md"] == "bar"
    assert trajectory.files["/b.md"] == "bar"


@pytest.mark.langsmith
def test_ls_directory_contains_file_yes_no(model: str) -> None:
    """Uses ls then answers YES/NO about a directory entry."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/foo/a.md": "a",
            "/foo/b.md": "b",
            "/foo/c.md": "c",
        },
        query="Is there a file named c.md in /foo? Answer with [YES] or [NO] only.",
        # 1st step: request a tool call to list /foo.
        # 2nd step: answer YES/NO.
        # 1 tool call request: ls.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1).require_final_text_contains(
            "[YES]",
        ),
    )


@pytest.mark.langsmith
def test_ls_directory_missing_file_yes_no(model: str) -> None:
    """Uses ls then answers YES/NO about a missing directory entry."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/foo/a.md": "a",
            "/foo/b.md": "b",
        },
        query="Is there a file named c.md in /foo? Answer with [YES] or [NO] only.",
        # 1st step: request a tool call to list /foo.
        # 2nd step: answer YES/NO.
        # 1 tool call request: ls.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1).require_final_text_contains("[no]", case_insensitive=True),
    )


@pytest.mark.langsmith
def test_edit_file_replace_text(model: str) -> None:
    """Edits a file by replacing text, then validates the edit."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        initial_files={"/note.md": "cat cat cat\n"},
        model=model,
        query=(
            "Replace all instances of 'cat' with 'dog' in /note.md, then tell me "
            "how many replacements you made. Do not read the file before editing it."
        ),
        # 1st step: request a tool call to edit /note.md.
        # 2nd step: report completion.
        # 1 tool call request: edit_file.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    assert trajectory.files["/note.md"] == "dog dog dog\n"


@pytest.mark.langsmith
def test_read_then_write_derived_output(model: str) -> None:
    """Reads a file and writes a derived output file."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={"/data.txt": "alpha\nbeta\ngamma\n"},
        query="Read /data.txt and write the lines reversed (line order) to /out.txt.",
        # 1st step: request a tool call to read /data.txt.
        # 2nd step: request a tool call to write /out.txt.
        # 2 tool call requests: read_file, write_file.
        expect=TrajectoryExpectations(num_agent_steps=3, num_tool_call_requests=2),
    )
    assert trajectory.files["/out.txt"].splitlines() == ["gamma", "beta", "alpha"]


@pytest.mark.langsmith
def test_avoid_unnecessary_tool_calls(model: str) -> None:
    """Answers a trivial question without using tools."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        query="What is 2+2? Answer with just the number.",
        model=model,
        # 1 step: answer directly.
        # 0 tool calls: no files/tools needed.
        expect=TrajectoryExpectations(num_agent_steps=1, num_tool_call_requests=0),
    )
    assert trajectory.steps[-1].action.text.strip() == "4"


@pytest.mark.langsmith
def test_read_files_in_parallel(model: str) -> None:
    """Performs two independent read_file calls in a single agent step."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/a.md": "same",
            "/b.md": "same",
        },
        query="Read /a.md and /b.md in parallel and tell me if they are identical. Answer with [YES] or [NO] only.",
        # 1st step: request 2 read_file tool calls in parallel.
        # 2nd step: answer YES/NO.
        # 2 tool call requests: read_file /a.md and read_file /b.md.
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=2)
            .require_tool_call(step=1, name="read_file", args_contains={"file_path": "/a.md"})
            .require_tool_call(step=1, name="read_file", args_contains={"file_path": "/b.md"})
            .require_final_text_contains("[YES]")
        ),
    )


@pytest.mark.langsmith
def test_grep_finds_matching_paths(model: str) -> None:
    """Uses grep to find matching files and reports the matching paths."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={
            "/a.txt": "haystack\nneedle\n",
            "/b.txt": "haystack\n",
            "/c.md": "needle\n",
        },
        query="Using grep, find which files contain the word 'needle'. Answer with the matching file paths only.",
        # 1st step: request a tool call to grep for 'needle'.
        # 2nd step: answer with the matching paths.
        # 1 tool call request: grep.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    answer = trajectory.steps[-1].action.text
    assert "/a.txt" in answer
    assert "/c.md" in answer
    assert "/b.txt" not in answer


@pytest.mark.langsmith
def test_glob_lists_markdown_files(model: str) -> None:
    """Uses glob to list files matching a pattern."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={
            "/foo/a.md": "a",
            "/foo/b.txt": "b",
            "/foo/c.md": "c",
        },
        query="Using glob, list all markdown files under /foo. Answer with the file paths only.",
        # 1st step: request a tool call to glob for markdown files.
        # 2nd step: answer with the matching paths.
        # 1 tool call request: glob.
        expect=TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1),
    )
    answer = trajectory.steps[-1].action.text
    assert "/foo/a.md" in answer
    assert "/foo/c.md" in answer
    assert "/foo/b.txt" not in answer
