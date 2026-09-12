"""Eval tests for file operations and tool efficiency.

Tests whether the agent can correctly use the built-in file tool surface
(read, write, edit, ls, grep, glob) including parallel invocation,
pagination recovery for large files, and avoiding unnecessary tool calls.

Written internally for the deepagents eval suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    file_absent,
    file_contains,
    file_equals,
    file_excludes,
    final_text_contains,
    final_text_excludes,
    run_agent,
    tool_call,
)


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_read_file_seeded_state_backend_file(model: BaseChatModel) -> None:
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
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(final_text_contains("three", case_insensitive=True)),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_file_overwrites_existing(model: BaseChatModel) -> None:
    """Overwrites an existing file without reading it first."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/note.md": "old content\n"},
        query='Rewrite /note.md so it contains only "new content". Reply with DONE only.',
        # 1st step: write_file to /note.md (no read needed).
        # 2nd step: reply DONE.
        # 1 tool call request: write_file.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(name="write_file", step=1, args_contains={"file_path": "/note.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_equals("/note.md", "new content"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_file_overwrite_drops_old_content(model: BaseChatModel) -> None:
    """Overwriting a file replaces it entirely, no trace of old content should remain."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/log.txt": "stale line\n"},
        query='Write "fresh line" to /log.txt. Reply with DONE only.',
        # 1st step: write_file to /log.txt.
        # 2nd step: reply DONE.
        # 1 tool call request: write_file.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            final_text_contains("DONE"),
            file_contains("/log.txt", "fresh line"),
            file_excludes("/log.txt", "stale"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_file_prefers_edit_for_targeted_change(model: BaseChatModel) -> None:
    """Uses edit_file (not write_file) when only a targeted in-place change is needed."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/note.md": "cat dog bird\n"},
        query="In /note.md, replace 'cat' with 'lion'. Reply with DONE only.",
        # 1st step: edit_file on /note.md (targeted change, rest of file preserved).
        # 2nd step: reply DONE.
        # 1 tool call request: edit_file.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(name="edit_file", step=1, args_contains={"file_path": "/note.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_contains("/note.md", "lion"),
            file_excludes("/note.md", "cat"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_file_simple(model: BaseChatModel) -> None:
    """Writes a file then answers a follow-up."""
    agent = create_deep_agent(model=model, system_prompt="Your name is Foo Bar.")
    run_agent(
        agent,
        model=model,
        query="Write your name to a file called /foo.md and then tell me your name.",
        # 1st step: request a tool call to write /foo.md.
        # 2nd step: tell the user the name.
        # 1 tool call request: write_file.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            file_contains("/foo.md", "Foo Bar"),
            final_text_contains("Foo Bar"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_files_in_parallel(model: str) -> None:
    """Writes two files in parallel without post-write verification or extra tool calls."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=(
            'Write "bar" to /a.md and "bar" to /b.md. Do the writes in parallel. Do NOT read any files afterward. Reply with DONE only.'
        ),
        # 1st step: request 2 write_file tool calls in parallel.
        # 2nd step: respond with "done".
        # 2 tool call requests: write_file to /a.md and write_file to /b.md.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="write_file", step=1, args_contains={"file_path": "/a.md"}),
                tool_call(name="write_file", step=1, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_equals("/a.md", "bar"),
            file_equals("/b.md", "bar"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_files_in_parallel_confirm_with_verification(model: str) -> None:
    """Writes two files in parallel, reads them back in parallel, then replies DONE."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=(
            'Write "bar" to /a.md and "bar" to /b.md in parallel. Then read both files in parallel to verify. Reply with DONE only.'
        ),
        # 1st step: request 2 write_file tool calls in parallel.
        # 2nd step: request 2 read_file tool calls in parallel.
        # 3rd step: confirm.
        # 4 tool call requests: 2 write_file + 2 read_file.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=4,
            tool_calls=[
                tool_call(name="write_file", step=1, args_contains={"file_path": "/a.md"}),
                tool_call(name="write_file", step=1, args_contains={"file_path": "/b.md"}),
                tool_call(name="read_file", step=2, args_contains={"file_path": "/a.md"}),
                tool_call(name="read_file", step=2, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_equals("/a.md", "bar"),
            file_equals("/b.md", "bar"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_files_in_parallel_ambiguous_confirmation(model: BaseChatModel) -> None:
    """Intentionally ambiguous: the user asks for a reply but doesn't constrain verification.

    We keep this prompt ambiguous on purpose to measure default efficiency in the harness.
    The most efficient behavior is to do the parallel writes and then reply DONE without
    any post-write `read_file` calls (the harness already provides `trajectory.files`).
    Some models will choose to verify by reading the files back anyway.

    This test therefore only enforces that the writes happen in parallel, and does not
    enforce step/tool-call counts.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query='Write "bar" to /a.md and "bar" to /b.md. Do the writes in parallel, then reply DONE.',
        # Intentionally ambiguous: some models will confirm directly; others may read back to verify.
        # Only enforce the parallel writes; do not enforce step/tool-call counts.
        scorer=TrajectoryScorer()
        .expect(
            tool_calls=[
                tool_call(name="write_file", step=1, args_contains={"file_path": "/a.md"}),
                tool_call(name="write_file", step=1, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(
            file_equals("/a.md", "bar"),
            file_equals("/b.md", "bar"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_ls_directory_contains_file_yes_no(model: BaseChatModel) -> None:
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
        query="Is there a file named c.md in /foo? Answer with `[YES]` or `[NO]` only.",
        # 1st step: request a tool call to list /foo.
        # 2nd step: answer YES/NO.
        # 1 tool call request: ls.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(final_text_contains("[YES]")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_ls_directory_missing_file_yes_no(model: BaseChatModel) -> None:
    """Uses ls then answers YES/NO about a missing directory entry."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/foo/a.md": "a",
            "/foo/b.md": "b",
        },
        query="Is there a file named c.md in /foo? Answer with `[YES]` or `[NO]` only.",
        # 1st step: request a tool call to list /foo.
        # 2nd step: answer YES/NO.
        # 1 tool call request: ls.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(final_text_contains("[no]", case_insensitive=True)),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_edit_file_replace_text(model: BaseChatModel) -> None:
    """Edits a file by replacing text, then validates the edit."""
    agent = create_deep_agent(model=model)
    run_agent(
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
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(file_equals("/note.md", "dog dog dog\n")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_read_then_write_derived_output(model: BaseChatModel) -> None:
    """Reads a file and writes a derived output file."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/data.txt": "alpha\nbeta\ngamma\n"},
        query="Read /data.txt and write the lines reversed (line order) to /out.txt.",
        # 1st step: request a tool call to read /data.txt.
        # 2nd step: request a tool call to write /out.txt.
        # 2 tool call requests: read_file, write_file.
        scorer=TrajectoryScorer()
        .expect(agent_steps=3, tool_call_requests=2)
        .success(
            file_contains("/out.txt", "gamma\nbeta\nalpha"),
            file_contains("/out.txt", "gamma"),
            file_contains("/out.txt", "beta"),
            file_contains("/out.txt", "alpha"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_avoid_unnecessary_tool_calls(model: BaseChatModel) -> None:
    """Answers a trivial question without using tools."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        query="What is 2+2? Answer with just the number.",
        model=model,
        # 1 step: answer directly.
        # 0 tool calls: no files/tools needed.
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(final_text_contains("4")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_read_files_in_parallel(model: BaseChatModel) -> None:
    """Performs two independent read_file calls in a single agent step."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/a.md": "same",
            "/b.md": "same",
        },
        query="Read /a.md and /b.md in parallel and tell me if they are identical. Answer with `[YES]` or `[NO]` only.",
        # 1st step: request 2 read_file tool calls in parallel.
        # 2nd step: answer YES/NO.
        # 2 tool call requests: read_file /a.md and read_file /b.md.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="read_file", step=1, args_contains={"file_path": "/a.md"}),
                tool_call(name="read_file", step=1, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(final_text_contains("[YES]")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_finds_matching_paths(model: BaseChatModel) -> None:
    """Uses grep to find matching files and reports the matching paths."""
    agent = create_deep_agent(model=model)
    run_agent(
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
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            final_text_contains("/a.txt"),
            final_text_contains("/c.md"),
            final_text_excludes("/b.txt"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_alternation_regex_recovers(model: BaseChatModel) -> None:
    """Recovers when an `|` alternation grep misses, finding both literal terms.

    `grep` matches literal text, so `cat|dog` is searched verbatim and misses.
    A model that reaches for regex should be steered by the no-match hint into
    running a separate literal search per term instead of burning calls on
    regex variants. Either way it must end up reporting both matching files.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/pets/a.txt": "the cat sat\n",
            "/pets/b.txt": "the dog ran\n",
            "/pets/c.txt": "the bird flew\n",
        },
        query="Using grep, find which files mention either 'cat' or 'dog'. Answer with the matching file paths only.",
        # 1st step (ideal): two literal greps, one per term.
        # 2nd step: answer with the matching paths.
        # A single `cat|dog` regex attempt misses and surfaces the literal hint.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=2)
        .success(
            final_text_contains("/pets/a.txt"),
            final_text_contains("/pets/b.txt"),
            final_text_excludes("/pets/c.txt"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_wildcard_regex_recovers(model: BaseChatModel) -> None:
    """Recovers when a `.*` wildcard grep misses, locating the literal line.

    A model may search `TODO.*cache`; literal grep treats `.*` verbatim and
    misses. The no-match hint should steer it back to a plain literal search
    (e.g. `TODO`) so it still finds and reports the file.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/src/app.py": "x = 1\n# TODO: add a cache layer here\n",
            "/src/util.py": "y = 2\n",
        },
        query="Using grep, find which file has a TODO comment about a cache. Answer with the matching file path only.",
        # 1st step (ideal): literal grep for 'TODO'.
        # 2nd step: answer with the matching path.
        # A `TODO.*cache` regex attempt misses and surfaces the literal hint.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            final_text_contains("/src/app.py"),
            final_text_excludes("/src/util.py"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_grep_escaped_metachar_regex_recovers(model: BaseChatModel) -> None:
    """Recovers when an escaped-regex grep misses a literal filename.

    A model may search `config\\.yaml`; literal grep matches the backslash
    verbatim and misses `config.yaml`. The no-match hint should steer it back
    to a plain literal search so it still finds and reports the file.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/svc/readme.md": "Edit config.yaml to set up the service.\n",
            "/svc/notes.txt": "nothing relevant here\n",
        },
        query="Using grep, find which file mentions the config.yaml file. Answer with the matching file path only.",
        # 1st step (ideal): literal grep for 'config.yaml'.
        # 2nd step: answer with the matching path.
        # A `config\\.yaml` escaped-regex attempt misses and surfaces the literal hint.
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            final_text_contains("/svc/readme.md"),
            final_text_excludes("/svc/notes.txt"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_glob_lists_markdown_files(model: BaseChatModel) -> None:
    """Uses glob to list files matching a pattern."""
    agent = create_deep_agent(model=model)
    run_agent(
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
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(
            final_text_contains("/foo/a.md"),
            final_text_contains("/foo/c.md"),
            final_text_excludes("/foo/b.txt"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_find_magic_phrase_deep_nesting(model: BaseChatModel) -> None:
    """Finds a magic phrase in a deeply nested directory efficiently."""
    agent = create_deep_agent(model=model)
    magic_phrase = "cobalt-otter-17"
    run_agent(
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
            "Find the file that contains the line starting with 'MAGIC_PHRASE:' and reply with the phrase value only. Be efficient: use grep."
        ),
        # 1st step: grep for MAGIC_PHRASE to locate the file.
        # 2nd step: read the file (if needed) and answer with the phrase.
        # 1 tool call requests: grep
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="grep", step=1, args_contains={"pattern": "MAGIC_PHRASE:"})],
        )
        .success(
            final_text_contains(magic_phrase),
            final_text_excludes("MAGIC_PHRASE"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_identify_quote_author_from_directory_parallel_reads(
    model: BaseChatModel,
) -> None:
    """Identifies which quote matches a target author by reading a directory efficiently."""
    agent = create_deep_agent(model=model)
    run_agent(
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
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=6,
            tool_calls=[
                tool_call(name="ls", step=1, args_contains={"path": "/quotes"}),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q1.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q2.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q3.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q4.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q5.txt"},
                ),
            ],
        )
        .success(final_text_contains("/quotes/q3.txt")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
def test_identify_quote_author_from_directory_unprompted_efficiency(
    model: BaseChatModel,
) -> None:
    """Identifies which quote matches a target author without explicit efficiency instructions."""
    agent = create_deep_agent(model=model)
    run_agent(
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
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=6,
            tool_calls=[
                tool_call(name="ls", step=1, args_contains={"path": "/quotes"}),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q1.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q2.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q3.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q4.txt"},
                ),
                tool_call(
                    name="read_file",
                    step=2,
                    args_contains={"file_path": "/quotes/q5.txt"},
                ),
            ],
        )
        .success(final_text_contains("/quotes/q3.txt")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_read_file_remaining_lines_notice_enables_tail_jump(
    model: BaseChatModel,
) -> None:
    """Uses pagination metadata to jump directly from the first page to the file's tail."""
    agent = create_deep_agent(model=model)
    last_line = "opal-fox-91"
    initial = "x\n" * 300
    initial_files = {"/big.txt": initial + last_line + "\n"}
    run_agent(
        agent,
        model=model,
        initial_files=initial_files,
        query=(
            "Use read_file yourself to read /big.txt and tell me the exact contents of the "
            "last non-empty line. Do not delegate this task. Reply with that line only."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(step=1, name="read_file", args_contains={"file_path": "/big.txt"}),
                tool_call(step=2, name="read_file", args_contains={"file_path": "/big.txt"}),
            ],
        )
        .success(final_text_contains(last_line)),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_read_file_empty_file_reports_empty(model: BaseChatModel) -> None:
    """Empty files should be reported as empty rather than hallucinated."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/empty.txt": ""},
        query="Read /empty.txt. If it is empty, reply with exactly: EMPTY. Do not fabricate any content.",
        scorer=TrajectoryScorer()
        .expect(agent_steps=2, tool_call_requests=1)
        .success(final_text_contains("EMPTY")),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_delete_simple(model: BaseChatModel) -> None:
    """Deletes a seeded file and confirms it is gone."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/foo.md": "delete me\n"},
        query="Delete the file /foo.md, then reply with DONE only.",
        # 1st step: request a delete tool call for /foo.md.
        # 2nd step: reply DONE.
        # 1 tool call request: delete.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(name="delete", step=1, args_contains={"file_path": "/foo.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_absent("/foo.md"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_delete_one_of_several_files(model: BaseChatModel) -> None:
    """Deletes a single target file, leaving the others untouched."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={
            "/a.md": "a",
            "/b.md": "b",
            "/c.md": "c",
        },
        query="Delete /b.md only. Leave the other files untouched. Reply with DONE only.",
        # 1st step: request a delete tool call for /b.md.
        # 2nd step: reply DONE.
        # 1 tool call request: delete.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[
                tool_call(name="delete", step=1, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_absent("/b.md"),
            file_equals("/a.md", "a"),
            file_equals("/c.md", "c"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_deletes_in_parallel(model: BaseChatModel) -> None:
    """Deletes two files in parallel without extra tool calls."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/a.md": "a", "/b.md": "b"},
        query=(
            "Delete /a.md and /b.md. Do the deletes in parallel. Do NOT read any files afterward. Reply with DONE only."
        ),
        # 1st step: request 2 delete tool calls in parallel.
        # 2nd step: reply DONE.
        # 2 tool call requests: delete /a.md and delete /b.md.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=2,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="delete", step=1, args_contains={"file_path": "/a.md"}),
                tool_call(name="delete", step=1, args_contains={"file_path": "/b.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_absent("/a.md"),
            file_absent("/b.md"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_write_then_delete_same_file(model: BaseChatModel) -> None:
    """Writes a file and then deletes it, leaving no trace."""
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=('Write "scratch" to /tmp.md, then delete /tmp.md. Reply with DONE only.'),
        # 1st step: request a write_file tool call for /tmp.md.
        # 2nd step: request a delete tool call for /tmp.md.
        # 3rd step: reply DONE.
        # 2 tool call requests: write_file then delete.
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="write_file", step=1, args_contains={"file_path": "/tmp.md"}),
                tool_call(name="delete", step=2, args_contains={"file_path": "/tmp.md"}),
            ],
        )
        .success(
            final_text_contains("DONE"),
            file_absent("/tmp.md"),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.eval_category("file_operations")
@pytest.mark.langsmith
def test_delete_missing_file_reports_absence(model: BaseChatModel) -> None:
    """A delete of a nonexistent file is reported, not faked.

    The `delete` tool returns a "File ... not found" error for a path
    that does not exist. The agent should surface that the file is missing
    rather than claim a successful deletion, and must leave the real file
    untouched. Tool-call shape is intentionally not enforced: a model may
    list/read first, or attempt the delete and recover from the error.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        initial_files={"/exists.md": "still here\n"},
        query=(
            "Delete the file /missing.md. If it does not exist, reply with exactly: NOT_FOUND. "
            "Otherwise reply with exactly: DONE."
        ),
        scorer=TrajectoryScorer().success(
            final_text_contains("NOT_FOUND"),
            file_equals("/exists.md", "still here\n"),
        ),
    )
