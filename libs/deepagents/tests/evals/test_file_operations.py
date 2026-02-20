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
def test_tool_error_recovery_read_file_then_ls(model: str) -> None:
    """User supplies a misspelled file path; agent recovers via ls instead of guessing."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/docs/readme.md": "hello\n",
            "/docs/release_notes.md": """Version: 1.2.3

Important: The MAGIC_TOKEN is SAPPHIRE-13.
""",
        },
        query=(
            "Read /docs/releasse_notes.md and tell me the MAGIC_TOKEN value. "
            "If the file path is wrong, do not guess the filename: list /docs to find the correct file."
        ),
        expect=(
            TrajectoryExpectations(num_agent_steps=4, num_tool_call_requests=3)
            .require_tool_call(step=1, name="read_file", args_contains={"file_path": "/docs/releasse_notes.md"})
            .require_tool_call(step=2, name="ls", args_contains={"path": "/docs"})
            .require_tool_call(step=3, name="read_file", args_contains={"file_path": "/docs/release_notes.md"})
            .require_final_text_contains("SAPPHIRE-13")
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
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1)
            .require_final_text_contains("/a.txt")
            .require_final_text_contains("/c.md")
        ),
    )
    assert "/b.txt" not in trajectory.answer


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
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1)
            .require_final_text_contains("/foo/a.md")
            .require_final_text_contains("/foo/c.md")
        ),
    )
    assert "/foo/b.txt" not in trajectory.answer


@pytest.mark.langsmith
def test_find_magic_phrase_deep_nesting(model: str) -> None:
    """Finds a magic phrase in a deeply nested directory efficiently."""
    agent = create_deep_agent(model=model)
    magic_phrase = "cobalt-otter-17"
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={
            "/a/b/c/d/e/notes.txt": "just some notes\n",
            "/a/b/c/d/e/readme.md": "project readme\n",
            "/a/b/c/d/e/answer.txt": f"MAGIC_PHRASE: {magic_phrase}\n",
            "/a/b/c/d/other.txt": "nothing here\n",
            "/a/b/x/y/z/nope.txt": "still nothing\n",
        },
        query=(
            "Find the file that contains the line starting with 'MAGIC_PHRASE:' and reply with the phrase value only. "
            "Be efficient: use grep to locate it before reading."
        ),
        # 1st step: grep for MAGIC_PHRASE to locate the file.
        # 2nd step: read the file (if needed) and answer with the phrase.
        # 1 tool call requests: grep
        expect=(
            TrajectoryExpectations(num_agent_steps=2, num_tool_call_requests=1)
            .require_tool_call(step=1, name="grep", args_contains={"pattern": "MAGIC_PHRASE:"})
            .require_final_text_contains(magic_phrase)
        ),
    )
    assert "MAGIC_PHRASE" not in trajectory.answer


@pytest.mark.langsmith
def test_identify_quote_author_from_directory_parallel_reads(model: str) -> None:
    """Identifies which quote matches a target author by reading a directory efficiently."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={
            "/quotes/q1.txt": """Quote: The analytical engine weaves algebraic patterns.
Clues: discusses an engine for computation and weaving patterns.
""",
            "/quotes/q2.txt": """Quote: I have always been more interested in the future than in the past.
Clues: talks about anticipating the future; broad and general.
""",
            "/quotes/q3.txt": """Quote: The most dangerous phrase in the language is, 'We've always done it this way.'
Clues: emphasizes changing established processes; often associated with early computing leadership.
""",
            "/quotes/q4.txt": """Quote: Sometimes it is the people no one can imagine anything of who do the things no one can imagine.
Clues: about imagination and doing the impossible; inspirational.
""",
            "/quotes/q5.txt": """Quote: Programs must be written for people to read, and only incidentally for machines to execute.
Clues: about programming readability; software craftsmanship.
""",
        },
        query=(
            "In the /quotes directory, there are several small quote files. "
            "Which file most likely contains a quote by Grace Hopper? Reply with the file path only. "
            "Be efficient: list the directory, then read the quote files in parallel to decide. "
            "Do not use grep."
        ),
        # 1st step: list the directory to discover files.
        # 2nd step: read all quote files in parallel.
        # 3rd step: answer with the selected path.
        # 6 tool call requests: 1 ls + 5 read_file.
        expect=(
            TrajectoryExpectations(num_agent_steps=3, num_tool_call_requests=6)
            .require_tool_call(step=1, name="ls", args_contains={"path": "/quotes"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q1.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q2.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q3.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q4.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q5.txt"})
            .require_final_text_contains("/quotes/q3.txt")
        ),
    )
    assert trajectory.answer.strip() == "/quotes/q3.txt"


@pytest.mark.langsmith
def test_identify_quote_author_from_directory_unprompted_efficiency(model: str) -> None:
    """Identifies which quote matches a target author without explicit efficiency instructions."""
    agent = create_deep_agent(model=model)
    trajectory = run_agent(
        agent,
        model=model,
        initial_files={
            "/quotes/q1.txt": """Quote: The analytical engine weaves algebraic patterns.
Clues: discusses an engine for computation and weaving patterns.
""",
            "/quotes/q2.txt": """Quote: I have always been more interested in the future than in the past.
Clues: talks about anticipating the future; broad and general.
""",
            "/quotes/q3.txt": """Quote: The most dangerous phrase in the language is, 'We've always done it this way.'
Clues: emphasizes changing established processes; often associated with early computing leadership.
""",
            "/quotes/q4.txt": """Quote: Sometimes it is the people no one can imagine anything of who do the things no one can imagine.
Clues: about imagination and doing the impossible; inspirational.
""",
            "/quotes/q5.txt": """Quote: Programs must be written for people to read, and only incidentally for machines to execute.
Clues: about programming readability; software craftsmanship.
""",
        },
        query=(
            "In the /quotes directory, there are a few small quote files. "
            "Which file most likely contains a quote by Grace Hopper? Reply with the file path only."
        ),
        # 1st step: list the directory to discover files.
        # 2nd step: read all quote files (ideally in parallel).
        # 3rd step: answer with the selected path.
        # 6 tool call requests: 1 ls + 5 read_file.
        expect=(
            TrajectoryExpectations(num_agent_steps=3, num_tool_call_requests=6)
            .require_tool_call(step=1, name="ls", args_contains={"path": "/quotes"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q1.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q2.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q3.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q4.txt"})
            .require_tool_call(step=2, name="read_file", args_contains={"file_path": "/quotes/q5.txt"})
            .require_final_text_contains("/quotes/q3.txt")
        ),
    )
    assert trajectory.answer.strip() == "/quotes/q3.txt"
