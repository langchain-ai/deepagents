"""Tests for better-harness."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from better_harness import (
    Case,
    Experiment,
    IterResult,
    SplitResult,
    ToolCall,
    Trace,
    Turn,
    load_experiment,
    load_trace,
    render_trace_md,
    run_split,
    slug,
    TRACE_ENV,
)
import better_harness.optimize as _opt_module
from better_harness.optimize import run_optimization
from better_harness.workspace import build_workspace, read_proposal, read_edited_harness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    case_id: str = "task-a",
    split: str = "train",
    score: float = 0.0,
    task: str = "Do something",
    turns: list[Turn] | None = None,
    final_output: str = "I couldn't do it.",
    failure: str | None = "Expected X, got Y",
) -> Trace:
    return Trace(
        case_id=case_id,
        split=split,
        score=score,
        task=task,
        turns=turns or [],
        final_output=final_output,
        failure=failure,
    )


def _make_experiment(tmp_path: Path, harness_content: str = 'PROMPT = "base"') -> tuple[Experiment, Path]:
    harness = tmp_path / "agent.py"
    harness.write_text(harness_content)

    tasks_root = tmp_path / "tasks"
    for task in ("task-a", "task-b", "task-c", "task-d"):
        (tasks_root / task).mkdir(parents=True)
        (tasks_root / task / "task.toml").write_text(f'name = "{task}"\n')

    config = tmp_path / "cases.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "test"
        harness = "agent.py"
        max_iterations = 2

        [better_agent]
        model = "claude-sonnet-4-6"
        max_turns = 100

        [runner]
        command = ["echo"]
        tasks_root = "{tasks_root}"
        pass_threshold = 1.0

        [[cases]]
        id = "task-a"
        split = "train"

        [[cases]]
        id = "task-b"
        split = "train"

        [[cases]]
        id = "task-c"
        split = "holdout"

        [[cases]]
        id = "task-d"
        split = "holdout"
        """).strip() + "\n"
    )
    return load_experiment(config), harness


# ---------------------------------------------------------------------------
# slug
# ---------------------------------------------------------------------------


def test_slug_replaces_special_chars():
    assert slug("my/task-name!v2") == "my-task-name-v2"
    assert slug("---") == "case"   # all hyphens stripped → "case" fallback
    assert slug("") == "case"


# ---------------------------------------------------------------------------
# Trace serialization
# ---------------------------------------------------------------------------


def test_trace_round_trips_json(tmp_path: Path):
    trace = _make_trace(
        turns=[
            Turn(
                agent="I'll run the shell",
                calls=[
                    ToolCall(tool="run_shell", input={"command": "ls"}, output="file.txt"),
                    ToolCall(tool="run_shell", input={"command": "cat x"}, error="No such file"),
                ],
            ),
            Turn(agent="Done", calls=[]),
        ]
    )
    path = tmp_path / "trace.json"
    trace.save(path)
    loaded = Trace.load(path)

    assert loaded.case_id == trace.case_id
    assert loaded.score == trace.score
    assert len(loaded.turns) == 2
    assert loaded.turns[0].calls[0].tool == "run_shell"
    assert loaded.turns[0].calls[0].output == "file.txt"
    assert loaded.turns[0].calls[1].error == "No such file"
    assert loaded.final_output == trace.final_output
    assert loaded.failure == trace.failure


def test_trace_passed_threshold():
    assert _make_trace(score=1.0).passed() is True
    assert _make_trace(score=0.5).passed() is False
    assert _make_trace(score=0.0).passed() is False


# ---------------------------------------------------------------------------
# load_trace
# ---------------------------------------------------------------------------


def test_load_trace_from_json(tmp_path: Path):
    data = {
        "task": "Fix the bug",
        "turns": [
            {
                "agent": "Let me check the file",
                "calls": [
                    {"tool": "read_file", "input": {"path": "main.py"}, "output": "def foo(): pass", "error": None}
                ],
            }
        ],
        "final_output": "Fixed.",
        "failure": None,
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(data))

    trace = load_trace(trace_path, case_id="fix-bug", split="train", score=1.0)
    assert trace.case_id == "fix-bug"
    assert trace.score == 1.0
    assert trace.task == "Fix the bug"
    assert len(trace.turns) == 1
    assert trace.turns[0].calls[0].tool == "read_file"
    assert trace.turns[0].calls[0].output == "def foo(): pass"
    assert trace.passed() is True


def test_load_trace_missing_file(tmp_path: Path):
    trace = load_trace(
        tmp_path / "nonexistent.json",
        case_id="x",
        split="train",
        score=0.0,
    )
    assert trace.case_id == "x"
    assert trace.score == 0.0
    assert trace.failure is not None
    assert "trace.json not found" in trace.failure


def test_load_trace_patches_score_and_failure(tmp_path: Path):
    data = {"task": "T", "turns": [], "final_output": "done", "failure": "original"}
    (tmp_path / "trace.json").write_text(json.dumps(data))

    trace = load_trace(
        tmp_path / "trace.json",
        case_id="t",
        split="holdout",
        score=0.0,
        failure="harbor says: wrong answer",
    )
    # score and failure from caller win over what's in the file
    assert trace.score == 0.0
    assert trace.failure == "harbor says: wrong answer"


# ---------------------------------------------------------------------------
# render_trace_md
# ---------------------------------------------------------------------------


def test_render_trace_md_failed_case():
    trace = _make_trace(
        case_id="write-report",
        task="Write a weekly sales report",
        turns=[
            Turn(
                agent="I'll send the report",
                calls=[
                    ToolCall(
                        tool="send_email",
                        input={"to": "boss@co.com"},
                        error="Tool not found: send_email",
                    )
                ],
            )
        ],
        final_output="Failed to send report.",
        failure="Expected email sent, got error",
    )
    md = render_trace_md(trace)

    assert "write-report" in md
    assert "FAILED" in md
    assert "Write a weekly sales report" in md
    assert "send_email" in md
    assert "✗" in md
    assert "Tool not found" in md
    assert "Failed to send report" in md
    assert "Expected email sent" in md


def test_render_trace_md_passed_case():
    trace = _make_trace(score=1.0, failure=None)
    md = render_trace_md(trace)
    assert "PASSED" in md
    assert "Why it failed" not in md


def test_render_trace_md_no_turns():
    trace = _make_trace(turns=[])
    md = render_trace_md(trace)
    assert "no turns captured" in md


def test_render_trace_md_tool_output_shown():
    trace = _make_trace(
        turns=[
            Turn(
                agent="checking files",
                calls=[ToolCall(tool="ls", input={"path": "/"}, output="bin\nusr\nvar")],
            )
        ]
    )
    md = render_trace_md(trace)
    assert "`ls(" in md
    assert "bin" in md


def test_render_trace_md_truncates_long_output():
    long_out = "x" * 1000
    trace = _make_trace(
        turns=[Turn(agent="a", calls=[ToolCall(tool="t", input={}, output=long_out)])]
    )
    md = render_trace_md(trace)
    assert len(md) < 3000  # long output should be truncated


# ---------------------------------------------------------------------------
# SplitResult
# ---------------------------------------------------------------------------


def test_split_result_counts():
    traces = [
        _make_trace("a", score=1.0, failure=None),
        _make_trace("b", score=0.0),
        _make_trace("c", score=0.5),
    ]
    sr = SplitResult(split="train", traces=traces)
    assert sr.passed == 1
    assert sr.total == 3
    assert abs(sr.pass_rate - 1 / 3) < 0.001
    assert sr.summary() == "1/3"
    assert [t.case_id for t in sr.failing()] == ["b", "c"]


# ---------------------------------------------------------------------------
# load_experiment
# ---------------------------------------------------------------------------


def test_load_experiment_basic(tmp_path: Path):
    harness = tmp_path / "agent.py"
    harness.write_text("# harness")
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    config = tmp_path / "exp.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "demo"
        harness = "agent.py"
        max_iterations = 3

        [better_agent]
        model = "claude-sonnet-4-6"
        max_turns = 500

        [runner]
        command = ["echo"]
        tasks_root = "{tasks_root}"

        [[cases]]
        id = "t1"
        split = "train"

        [[cases]]
        id = "h1"
        split = "holdout"
        """).strip() + "\n"
    )
    exp = load_experiment(config)
    assert exp.name == "demo"
    assert exp.max_iterations == 3
    assert len(exp.train_cases()) == 1
    assert len(exp.holdout_cases()) == 1
    assert exp.harness_path == harness


def test_load_experiment_missing_harness(tmp_path: Path):
    config = tmp_path / "exp.toml"
    config.write_text(
        dedent("""
        [experiment]
        name = "x"
        harness = "nonexistent.py"

        [runner]
        command = ["echo"]
        tasks_root = "."

        [[cases]]
        id = "a"
        split = "train"

        [[cases]]
        id = "b"
        split = "holdout"
        """).strip() + "\n"
    )
    with pytest.raises(ValueError, match="harness not found"):
        load_experiment(config)


def test_load_experiment_no_holdout_raises(tmp_path: Path):
    harness = tmp_path / "agent.py"
    harness.write_text("x")
    config = tmp_path / "exp.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "x"
        harness = "agent.py"

        [runner]
        command = ["echo"]
        tasks_root = "{tmp_path}"

        [[cases]]
        id = "a"
        split = "train"
        """).strip() + "\n"
    )
    with pytest.raises(ValueError, match="holdout"):
        load_experiment(config)


# ---------------------------------------------------------------------------
# build_workspace
# ---------------------------------------------------------------------------


def test_build_workspace_creates_expected_files(tmp_path: Path):
    harness = tmp_path / "agent.py"
    harness.write_text("SYSTEM_PROMPT = 'base'\n")

    failing_trace = _make_trace(
        case_id="task-a",
        turns=[Turn(agent="tried something", calls=[
            ToolCall(tool="run_shell", input={"cmd": "ls"}, error="not found")
        ])],
    )
    passing_trace = _make_trace(case_id="task-b", score=1.0, failure=None)
    holdout_trace = _make_trace(case_id="task-c", split="holdout", score=0.0)

    train = SplitResult(split="train", traces=[failing_trace, passing_trace])
    holdout = SplitResult(split="holdout", traces=[holdout_trace])

    ws = build_workspace(
        harness_path=harness,
        train_result=train,
        holdout_result=holdout,
        history=[],
        workspace_dir=tmp_path / "workspace",
    )

    assert (ws / "agent.py").exists()
    assert (ws / "scores.json").exists()
    assert (ws / "task.md").exists()
    assert (ws / "history.md").exists()
    assert (ws / "proposal.md").exists()

    # Only failing train cases get a trace file (in traces/failing/).
    trace_files = list((ws / "traces" / "failing").iterdir())
    assert len(trace_files) == 1
    assert "task-a" in trace_files[0].name

    # Trace content should show the tool call.
    content = trace_files[0].read_text()
    assert "run_shell" in content
    assert "not found" in content

    # Passing traces directory exists.
    assert (ws / "traces" / "passing").exists()

    # Scratchpad and case_status exist.
    assert (ws / "scratchpad.md").exists()
    assert (ws / "case_status.md").exists()


def test_build_workspace_all_passing(tmp_path: Path):
    harness = tmp_path / "agent.py"
    harness.write_text("x")

    train = SplitResult(split="train", traces=[_make_trace(score=1.0)])
    holdout = SplitResult(split="holdout", traces=[_make_trace(score=1.0)])

    ws = build_workspace(
        harness_path=harness,
        train_result=train,
        holdout_result=holdout,
        history=[],
        workspace_dir=tmp_path / "ws",
    )
    assert (ws / "traces" / "failing" / "ALL_PASSING.md").exists()


def test_build_workspace_history_rendered(tmp_path: Path):
    harness = tmp_path / "agent.py"
    harness.write_text("x")

    train = SplitResult(split="train", traces=[_make_trace()])
    holdout = SplitResult(split="holdout", traces=[_make_trace()])
    history = [
        IterResult(
            iteration=1, accepted=True,
            train=SplitResult("train", [_make_trace(score=1.0)]),
            holdout=SplitResult("holdout", [_make_trace(score=1.0)]),
            proposal="Added send_email tool. Fixed 2 cases.",
        )
    ]
    ws = build_workspace(
        harness_path=harness,
        train_result=train,
        holdout_result=holdout,
        history=history,
        workspace_dir=tmp_path / "ws",
    )
    history_text = (ws / "history.md").read_text()
    assert "ACCEPTED" in history_text
    assert "Added send_email tool" in history_text
    assert "Iteration 1" in history_text


def test_read_proposal_strips_template(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    # Write the exact blank template that build_workspace produces.
    (ws / "proposal.md").write_text(
        "# Proposal\n\n"
        "**What I changed:**\n\n"
        "**Why (tie to scratchpad hypothesis):**"
    )
    assert read_proposal(ws) == ""


def test_read_proposal_returns_content(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "proposal.md").write_text("# Proposal\n\nAdded send_email tool.")
    assert "send_email" in read_proposal(ws)


# ---------------------------------------------------------------------------
# End-to-end loop with mock Harbor
# ---------------------------------------------------------------------------


def _write_mock_harbor(path: Path, pass_tasks: list[str]) -> None:
    """Write a mock harbor script that passes specific tasks and writes trace.json."""
    pass_json = json.dumps(pass_tasks)
    path.write_text(
        dedent(f"""
        import argparse, json, os, sys
        from pathlib import Path

        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        r = sub.add_parser("run")
        r.add_argument("-p", dest="tasks_root")
        r.add_argument("--task-name")
        r.add_argument("-o", dest="output_dir")
        r.add_argument("--job-name")
        r.add_argument("--agent-import-path")
        r.add_argument("-l", default="1")
        r.add_argument("-n", default="1")
        args = p.parse_args()

        pass_tasks = {pass_json}
        score = 1.0 if args.task_name in pass_tasks else 0.0

        jobs = Path(args.output_dir) / args.job_name / args.task_name
        jobs.mkdir(parents=True, exist_ok=True)
        (jobs / "result.json").write_text(json.dumps({{"score": score, "message": "ok" if score else "failed"}}))

        # Also write trace.json via the env var, simulating what a real harness would do.
        trace_path = os.environ.get("BETTER_HARNESS_TRACE_FILE")
        if trace_path:
            trace = {{
                "task": args.task_name,
                "turns": [
                    {{
                        "agent": "I processed the task",
                        "calls": [
                            {{"tool": "run_shell", "input": {{"command": "echo hi"}}, "output": "hi", "error": None}}
                        ]
                    }}
                ],
                "final_output": "done" if score else "failed",
                "failure": None if score else "task failed",
            }}
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            Path(trace_path).write_text(json.dumps(trace))
        """).strip() + "\n"
    )


def test_run_split_with_mock_harbor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    harness = tmp_path / "agent.py"
    harness.write_text("SYSTEM_PROMPT = 'test'\n")

    mock_harbor = tmp_path / "mock_harbor.py"
    _write_mock_harbor(mock_harbor, pass_tasks=["task-a"])

    tasks_root = tmp_path / "tasks"
    for t in ("task-a", "task-b"):
        (tasks_root / t).mkdir(parents=True)

    experiment = Experiment(
        name="test",
        harness_path=harness,
        cases=[
            Case("task-a", "train"),
            Case("task-b", "train"),
        ],
        runner_config={
            "command": ["python3", str(mock_harbor)],
            "tasks_root": str(tasks_root),
            "pass_threshold": 1.0,
        },
    )

    result = run_split(
        experiment=experiment,
        harness_path=harness,
        split="train",
        output_dir=tmp_path / "out",
    )

    assert result.total == 2
    assert result.passed == 1
    assert result.summary() == "1/2"

    # task-a passed, task-b failed
    passing = [t for t in result.traces if t.passed()]
    failing = [t for t in result.traces if not t.passed()]
    assert len(passing) == 1 and passing[0].case_id == "task-a"
    assert len(failing) == 1 and failing[0].case_id == "task-b"

    # Traces should have been captured.
    assert all(len(t.turns) > 0 for t in result.traces)
    assert result.traces[0].turns[0].calls[0].tool == "run_shell"


def test_full_optimization_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: baseline fails, outer agent improves harness, candidate accepted."""
    harness = tmp_path / "agent.py"
    harness.write_text("SYSTEM_PROMPT = 'base'\n# version: 0\n")

    mock_harbor = tmp_path / "mock_harbor.py"
    # Initially nothing passes; after the outer agent edits, task-a and task-c pass.
    _write_mock_harbor(mock_harbor, pass_tasks=[])

    tasks_root = tmp_path / "tasks"
    for t in ("task-a", "task-b", "task-c", "task-d"):
        (tasks_root / t).mkdir(parents=True)

    config = tmp_path / "cases.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "loop-test"
        harness = "agent.py"
        max_iterations = 2

        [better_agent]
        model = "claude-sonnet-4-6"
        max_turns = 100

        [runner]
        command = ["python3", "{mock_harbor}"]
        tasks_root = "{tasks_root}"
        pass_threshold = 1.0

        [[cases]]
        id = "task-a"
        split = "train"

        [[cases]]
        id = "task-b"
        split = "train"

        [[cases]]
        id = "task-c"
        split = "holdout"

        [[cases]]
        id = "task-d"
        split = "holdout"
        """).strip() + "\n"
    )

    experiment = load_experiment(config)

    # Fake outer agent: edits the harness and updates mock harbor to pass train cases.
    call_count: list[int] = [0]

    def fake_outer_agent(*, experiment, workspace_dir):  # noqa: ARG001
        call_count[0] += 1
        # Rewrite mock harbor to now pass task-a and task-c.
        _write_mock_harbor(mock_harbor, pass_tasks=["task-a", "task-c"])
        # Edit the harness (outer agent "improves" it).
        (workspace_dir / "agent.py").write_text(
            "SYSTEM_PROMPT = 'improved'\n# version: 1\n"
        )
        (workspace_dir / "proposal.md").write_text(
            "# Proposal\n\nAdded better prompt. Fixed task-a."
        )

    monkeypatch.setattr(_opt_module, "run_outer_agent", fake_outer_agent)

    output_dir = tmp_path / "runs"
    summary = run_optimization(experiment, output_dir=output_dir, max_iterations=2)

    # Outer agent ran at least once (may run a second time before detecting no-change).
    assert call_count[0] >= 1

    # Baseline: 0/2 train, 0/2 holdout.
    assert summary["baseline"]["train"] == "0/2"
    assert summary["baseline"]["holdout"] == "0/2"

    # Final: 1/2 train (task-a passes), 1/2 holdout (task-c passes).
    assert summary["final"]["train"] == "1/2"
    assert summary["final"]["holdout"] == "1/2"

    # One iteration accepted.
    assert summary["iterations_accepted"] == 1
    assert summary["history"][0]["accepted"] is True
    assert "Added better prompt" in summary["history"][0]["proposal"]

    # Report files written.
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()

    # Workspace files exist.
    ws = output_dir / "iter-001" / "workspace"
    assert (ws / "agent.py").exists()
    assert (ws / "scores.json").exists()
    assert (ws / "task.md").exists()
    assert (ws / "history.md").exists()
    assert (ws / "traces").exists()


def test_optimization_loop_rejects_no_improvement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If the outer agent makes no improvement, all iterations are rejected."""
    harness = tmp_path / "agent.py"
    harness.write_text("SYSTEM_PROMPT = 'base'\n")

    mock_harbor = tmp_path / "mock_harbor.py"
    _write_mock_harbor(mock_harbor, pass_tasks=[])  # nothing ever passes

    tasks_root = tmp_path / "tasks"
    for t in ("t-train", "t-holdout"):
        (tasks_root / t).mkdir(parents=True)

    config = tmp_path / "cases.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "reject-test"
        harness = "agent.py"
        max_iterations = 2

        [better_agent]
        model = "claude-sonnet-4-6"
        max_turns = 100

        [runner]
        command = ["python3", "{mock_harbor}"]
        tasks_root = "{tasks_root}"
        pass_threshold = 1.0

        [[cases]]
        id = "t-train"
        split = "train"

        [[cases]]
        id = "t-holdout"
        split = "holdout"
        """).strip() + "\n"
    )

    experiment = load_experiment(config)

    def fake_outer_agent(*, experiment, workspace_dir):  # noqa: ARG001
        # Make a change (different text) but harbor still fails everything.
        (workspace_dir / "agent.py").write_text("SYSTEM_PROMPT = 'changed'\n")
        (workspace_dir / "proposal.md").write_text("# Proposal\n\nTried something.")

    monkeypatch.setattr(_opt_module, "run_outer_agent", fake_outer_agent)

    summary = run_optimization(experiment, output_dir=tmp_path / "runs", max_iterations=2)

    assert summary["iterations_accepted"] == 0
    assert all(not h["accepted"] for h in summary["history"])
    # Final should equal baseline (0/1).
    assert summary["final"]["train"] == summary["baseline"]["train"]


def test_optimization_loop_stops_if_no_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Loop stops early if the outer agent makes no changes to agent.py."""
    harness = tmp_path / "agent.py"
    harness.write_text("SYSTEM_PROMPT = 'base'\n")

    mock_harbor = tmp_path / "mock_harbor.py"
    _write_mock_harbor(mock_harbor, pass_tasks=[])

    tasks_root = tmp_path / "tasks"
    for t in ("t-train", "t-holdout"):
        (tasks_root / t).mkdir(parents=True)

    config = tmp_path / "cases.toml"
    config.write_text(
        dedent(f"""
        [experiment]
        name = "no-change"
        harness = "agent.py"
        max_iterations = 3

        [better_agent]
        model = "claude-sonnet-4-6"
        max_turns = 100

        [runner]
        command = ["python3", "{mock_harbor}"]
        tasks_root = "{tasks_root}"
        pass_threshold = 1.0

        [[cases]]
        id = "t-train"
        split = "train"

        [[cases]]
        id = "t-holdout"
        split = "holdout"
        """).strip() + "\n"
    )

    experiment = load_experiment(config)
    calls: list[int] = [0]

    def fake_outer_agent(*, experiment, workspace_dir):  # noqa: ARG001
        calls[0] += 1
        # Don't change anything — agent.py stays the same as harness.

    monkeypatch.setattr(_opt_module, "run_outer_agent", fake_outer_agent)

    summary = run_optimization(experiment, output_dir=tmp_path / "runs", max_iterations=3)

    # Should stop after first iteration (no change detected).
    assert calls[0] == 1
    assert summary["iterations_run"] == 0  # no iteration was accepted or produced a result


def test_trace_env_set_in_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TRACE_ENV is set in the subprocess environment so the harness can write traces."""
    recorded_envs: list[dict] = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        recorded_envs.append(dict(kwargs.get("env") or {}))

        class R:
            returncode = 0
            stdout = stderr = ""

        return R()

    import subprocess as _subprocess  # noqa: PLC0415
    from better_harness.runner import run_case as _run_case  # noqa: PLC0415

    monkeypatch.setattr(_subprocess, "run", fake_run)

    harness = tmp_path / "agent.py"
    harness.write_text("x")
    tasks_root = tmp_path / "tasks"
    (tasks_root / "t").mkdir(parents=True)

    experiment = Experiment(
        name="e",
        harness_path=harness,
        cases=[Case("t", "train")],
        runner_config={"command": ["echo"], "tasks_root": str(tasks_root), "pass_threshold": 1.0},
    )

    try:
        _run_case(
            experiment=experiment,
            harness_path=harness,
            case=Case("t", "train"),
            case_dir=tmp_path / "out",
        )
    except Exception:  # noqa: BLE001
        pass  # We only care that subprocess.run was called with TRACE_ENV set.

    assert any(TRACE_ENV in env for env in recorded_envs), (
        f"{TRACE_ENV} not set in subprocess env"
    )
