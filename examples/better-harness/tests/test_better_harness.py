from __future__ import annotations

import importlib
import json
from pathlib import Path
from textwrap import dedent

import pytest

from better_harness import (
    CaseOutcome,
    EvalCase,
    Experiment,
    SplitResult,
    Surface,
    load_experiment,
    main,
    run_experiment,
)
from better_harness.agent import build_proposer_workspace
from better_harness.core import RunLayout
from better_harness.patching import (
    build_baseline_variant,
    build_variant,
    patch_module_attrs,
    workspace_override_context,
)
from better_harness.runners import parse_harbor_case, parse_pytest_outcomes


def _write_minimal_pytest_experiment(tmp_path: Path) -> Path:
    workspace = tmp_path / "demo_workspace"
    workspace.mkdir(parents=True)
    (workspace / "demo_agent.py").write_text(
        '"""Tiny demo harness under test."""\n\n'
        'BASE_PROMPT = "If the request is ambiguous, ask questions before acting."\n'
    )
    (workspace / "tools.py").write_text('"""Demo tool surface."""\n\nTOOLS = ["run_shell"]\n')
    (workspace / "skills.md").write_text("# Demo skills\n\nBe generally helpful.\n")
    (workspace / "middleware.py").write_text('"""Demo middleware surface."""\n\nMIDDLEWARE = []\n')

    project_root = workspace / "evals"
    tests_dir = project_root / "tests"
    tests_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "better-harness-demo-evals"
            version = "0.1.0"
            requires-python = ">=3.12"
            dependencies = []

            [dependency-groups]
            test = [
                "pytest>=8.4.2",
            ]

            [tool.pytest.ini_options]
            testpaths = ["tests"]
            """
        ).strip()
        + "\n"
    )
    (tests_dir / "conftest.py").write_text(
        dedent(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            import pytest

            COUNTS = {"passed": 0, "failed": 0, "skipped": 0}


            def pytest_addoption(parser: pytest.Parser) -> None:
                parser.addoption("--model", action="store", default="demo-model")
                parser.addoption("--evals-report-file", action="store", default="")


            @pytest.fixture
            def model(pytestconfig: pytest.Config) -> str:
                return str(pytestconfig.getoption("--model"))


            def pytest_configure(config: pytest.Config) -> None:
                del config
                for key in COUNTS:
                    COUNTS[key] = 0


            def pytest_runtest_logreport(report: pytest.TestReport) -> None:
                if report.when != "call":
                    return
                if report.passed:
                    COUNTS["passed"] += 1
                elif report.failed:
                    COUNTS["failed"] += 1
                elif report.skipped:
                    COUNTS["skipped"] += 1


            def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
                del exitstatus
                summary_file = str(session.config.getoption("--evals-report-file"))
                if not summary_file:
                    return
                total = COUNTS["passed"] + COUNTS["failed"] + COUNTS["skipped"]
                payload = {
                    "created_at": "demo",
                    "sdk_version": "demo",
                    "model": str(session.config.getoption("--model")),
                    "passed": COUNTS["passed"],
                    "failed": COUNTS["failed"],
                    "skipped": COUNTS["skipped"],
                    "total": total,
                    "correctness": 0.0 if total == 0 else COUNTS["passed"] / total,
                }
                path = Path(summary_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2) + "\\n")
            """
        ).strip()
        + "\n"
    )
    (tests_dir / "test_harness.py").write_text(
        dedent(
            """
            from __future__ import annotations

            from pathlib import Path

            import demo_agent

            WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


            def read_surface(path: str) -> str:
                return (WORKSPACE_ROOT / path).read_text()


            def test_prompt_train() -> None:
                assert "minimum number of followup questions" in demo_agent.BASE_PROMPT


            def test_tools_train() -> None:
                assert "send_report" in read_surface("tools.py")


            def test_skills_train() -> None:
                assert "domain-defining questions" in read_surface("skills.md")


            def test_middleware_train() -> None:
                assert "duplicate tool calls" in read_surface("middleware.py")


            def test_prompt_holdout() -> None:
                assert "Use reasonable defaults" in demo_agent.BASE_PROMPT


            def test_tools_holdout() -> None:
                tools = read_surface("tools.py")
                assert "run_shell" in tools
                assert "send_report" in tools


            def test_skills_holdout() -> None:
                assert "clarify the domain" in read_surface("skills.md")


            def test_middleware_holdout() -> None:
                assert "reuse prior successful results" in read_surface("middleware.py")


            def test_final_eval_combined() -> None:
                prompt = demo_agent.BASE_PROMPT
                tools = read_surface("tools.py")
                skills = read_surface("skills.md")
                middleware = read_surface("middleware.py")
                assert "minimum number of followup questions" in prompt
                assert "Use reasonable defaults" in prompt
                assert "send_report" in tools
                assert "domain-defining questions" in skills
                assert "duplicate tool calls" in middleware


            def test_final_eval_story() -> None:
                assert "Do not ask for details the user already supplied" in demo_agent.BASE_PROMPT
            """
        ).strip()
        + "\n"
    )

    config = tmp_path / "minimal_pytest.toml"
    config.write_text(
        dedent(
            f"""
            [experiment]
            name = "minimal-pytest"
            runner = "pytest"
            workspace_root = "{workspace}"
            model = "demo-model"
            max_iterations = 4

            [better_agent]
            model = "claude-sonnet-4-6"
            max_turns = 40

            [runner.pytest]
            project_root = "{project_root}"
            model_flag = "--model"
            summary_flag = "--evals-report-file"
            pytest_args = ["-q"]

            [surfaces.prompt]
            kind = "module_attr"
            target = "demo_agent:BASE_PROMPT"
            filename = "prompt.txt"
            base_value = \"\"\"
            If the request is ambiguous, ask questions before acting.
            \"\"\"

            [surfaces.tools]
            kind = "workspace_file"
            target = "tools.py"
            filename = "tools.py"
            base_value = \"\"\"
            TOOLS = ["run_shell"]
            \"\"\"

            [surfaces.skills]
            kind = "workspace_file"
            target = "skills.md"
            filename = "skills.md"
            base_value = \"\"\"
            # Demo skills

            Be generally helpful.
            \"\"\"

            [surfaces.middleware]
            kind = "workspace_file"
            target = "middleware.py"
            filename = "middleware.py"
            base_value = \"\"\"
            MIDDLEWARE = []
            \"\"\"

            [[cases]]
            case_id = "tests/test_harness.py::test_prompt_train"
            split = "train"
            stratum = "prompt"

            [[cases]]
            case_id = "tests/test_harness.py::test_tools_train"
            split = "train"
            stratum = "tools"

            [[cases]]
            case_id = "tests/test_harness.py::test_skills_train"
            split = "train"
            stratum = "skills"

            [[cases]]
            case_id = "tests/test_harness.py::test_middleware_train"
            split = "train"
            stratum = "middleware"

            [[cases]]
            case_id = "tests/test_harness.py::test_prompt_holdout"
            split = "holdout"
            stratum = "prompt"

            [[cases]]
            case_id = "tests/test_harness.py::test_tools_holdout"
            split = "holdout"
            stratum = "tools"

            [[cases]]
            case_id = "tests/test_harness.py::test_skills_holdout"
            split = "holdout"
            stratum = "skills"

            [[cases]]
            case_id = "tests/test_harness.py::test_middleware_holdout"
            split = "holdout"
            stratum = "middleware"

            [[cases]]
            case_id = "tests/test_harness.py::test_final_eval_combined"
            split = "scorecard"
            stratum = "combined"

            [[cases]]
            case_id = "tests/test_harness.py::test_final_eval_story"
            split = "scorecard"
            stratum = "combined"
            """
        ).strip()
        + "\n"
    )
    return config


def test_load_experiment_normalizes_scorecard_aliases(tmp_path: Path):
    prompt = tmp_path / "base.txt"
    prompt.write_text("base")
    config = tmp_path / "experiment.toml"
    config.write_text(
        f"""
[experiment]
name = "demo"
runner = "pytest"
workspace_root = "{tmp_path}"
model = "demo-model"

[better_agent]
model = "claude-sonnet-4-6"
max_turns = 9

[runner.pytest]
project_root = "{tmp_path}"
pytest_args = ["-q"]

[surfaces.prompt]
kind = "module_attr"
target = "demo_mod:PROMPT"
base_file = "{prompt.name}"

[[cases]]
case_id = "tests/test_demo.py::test_a[{{model}}]"
split = "train"
stratum = "tool_use"

[[cases]]
case_id = "tests/test_demo.py::test_b[{{model}}]"
split = "holdout"
stratum = "tool_use"

[[cases]]
case_id = "tests/test_demo.py::test_c[{{model}}]"
split = "final_eval"
stratum = "tool_use"
"""
    )
    experiment = load_experiment(config)
    assert experiment.name == "demo"
    assert experiment.better_agent_model == "claude-sonnet-4-6"
    assert experiment.better_agent_max_turns == 9
    assert experiment.rendered_case_ids("scorecard") == ["tests/test_demo.py::test_c[demo-model]"]


def test_build_variant_tracks_changed_surfaces(tmp_path: Path):
    experiment = Experiment(
        path=tmp_path / "demo.toml",
        name="demo",
        runner="pytest",
        workspace_root=tmp_path,
        model="demo-model",
        max_iterations=3,
        better_agent_model="claude-sonnet-4-6",
        better_agent_max_turns=20,
        better_agent_deepagents_root=None,
        better_agent_system_prompt=None,
        runner_config={"project_root": str(tmp_path)},
        surfaces={
            "prompt": Surface("prompt", "module_attr", "demo_mod:PROMPT", "base prompt", "prompt.txt"),
            "tools": Surface("tools", "workspace_file", "tools.py", "BASE = 1", "tools.py"),
        },
        cases=(
            EvalCase("tests/test_demo.py::test_a[{model}]", "train", "tool_use"),
            EvalCase("tests/test_demo.py::test_b[{model}]", "holdout", "tool_use"),
        ),
    )
    baseline = build_baseline_variant(experiment)
    variant = build_variant(
        experiment=experiment,
        label="iter-001",
        values={"prompt": "patched prompt", "tools": "BASE = 1"},
    )
    assert baseline.changed_surfaces == ()
    assert variant.changed_surfaces == ("prompt",)
    assert variant.file_overrides() == {"tools.py": "BASE = 1"}


def test_patch_module_attrs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "demo_mod.py"
    module_path.write_text("PROMPT = 'base'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    patch_module_attrs({"demo_mod:PROMPT": "patched"})
    demo_mod = importlib.import_module("demo_mod")
    assert demo_mod.PROMPT == "patched"


def test_workspace_override_context_restores_files(tmp_path: Path):
    target = tmp_path / "tools.py"
    target.write_text("BASE = 1\n")
    with workspace_override_context(tmp_path, {"tools.py": "BASE = 2\n"}):
        assert target.read_text() == "BASE = 2\n"
    assert target.read_text() == "BASE = 1\n"


def test_parse_pytest_outcomes_marks_failed_cases(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="1" failures="1">
    <testcase file="tests/test_demo.py" classname="tests.test_demo" name="test_case[demo-model]" time="0.1">
      <failure message="boom">boom</failure>
    </testcase>
  </testsuite>
</testsuites>
"""
    )
    outcomes = parse_pytest_outcomes(
        junit_path=junit,
        cases=[EvalCase("tests/test_demo.py::test_case[{model}]", "train", "tool_use")],
        model="demo-model",
        artifacts_dir=tmp_path,
    )
    assert outcomes[0].status == "failed"
    assert outcomes[0].failure_message == "boom"


def test_parse_harbor_case_reads_result_json(tmp_path: Path):
    result_dir = tmp_path / "jobs" / "job" / "task"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(json.dumps({"score": 1.0, "message": "ok"}))
    score, payload, failure = parse_harbor_case(jobs_dir=tmp_path / "jobs", pass_threshold=1.0)
    assert score == 1.0
    assert payload == {"score": 1.0, "message": "ok"}
    assert failure is None


def test_build_proposer_workspace_copies_train_context(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_root = workspace / "evals"
    test_file = project_root / "tests" / "test_demo.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_case():\n    assert True\n")

    experiment = Experiment(
        path=tmp_path / "demo.toml",
        name="demo",
        runner="pytest",
        workspace_root=workspace,
        model="demo-model",
        max_iterations=2,
        better_agent_model="claude-sonnet-4-6",
        better_agent_max_turns=20,
        better_agent_deepagents_root=None,
        better_agent_system_prompt=None,
        runner_config={"project_root": str(project_root)},
        surfaces={
            "prompt": Surface("prompt", "module_attr", "demo_mod:PROMPT", "base", "prompt.txt"),
        },
        cases=(
            EvalCase("tests/test_demo.py::test_case[{model}]", "train", "prompt"),
            EvalCase("tests/test_demo.py::test_holdout[{model}]", "holdout", "prompt"),
        ),
    )
    baseline = build_baseline_variant(experiment)
    train = SplitResult(
        split="train",
        variant="baseline",
        model="demo-model",
        passed=0,
        total=1,
        score=0.0,
        returncode=1,
        run_dir="run",
        outcomes=(
            CaseOutcome(
                case_id="tests/test_demo.py::test_case[demo-model]",
                split="train",
                stratum="prompt",
                status="failed",
                score=0.0,
                duration_s=0.0,
                failure_message="missing prompt policy",
            ),
        ),
    )
    layout = RunLayout(tmp_path / "run")
    layout.write_manifest(experiment)
    prior_iteration_dir = layout.visible_iterations_dir / "000"
    prior_iteration_dir.mkdir(parents=True, exist_ok=True)
    (prior_iteration_dir / "decision.json").write_text('{"iteration": 0, "decision": "accepted", "train_passed": 1, "train_total": 1}\n')
    prior_train_dir = layout.visible_root / "train" / "baseline"
    prior_train_dir.mkdir(parents=True, exist_ok=True)
    (prior_train_dir / "result.json").write_text('{"split":"train","variant":"baseline","model":"demo-model","passed":0,"total":1,"score":0.0,"correctness":0.0,"returncode":1,"run_dir":"run","outcomes":[]}\n')
    proposer_workspace = build_proposer_workspace(
        experiment=experiment,
        current=baseline,
        train_result=train,
        layout=layout,
        iteration=1,
    )
    assert (proposer_workspace.root / "task.md").exists()
    assert (proposer_workspace.root / "train_failures.json").exists()
    assert (proposer_workspace.root / "train_cases" / "tests" / "test_demo.py").exists()
    assert (proposer_workspace.root / "history" / "prior_visible" / "iterations" / "000" / "decision.json").exists()
    assert (proposer_workspace.root / "history" / "prior_visible" / "train" / "baseline" / "result.json").exists()
    assert proposer_workspace.surface_files["prompt"].read_text() == "base"


def test_run_end_to_end_pytest_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_proposer(*, experiment, workspace):
        del experiment
        (workspace.current_dir / "prompt.txt").write_text(
            "If the request is ambiguous, ask questions before acting.\n\n"
            "## Clarifying Requests\n\n"
            "- If a request is underspecified, ask only the minimum number of followup questions needed to take the next useful action.\n"
            "- Do not ask for details the user already supplied.\n"
            "- Use reasonable defaults when the request clearly implies them.\n"
        )
        (workspace.current_dir / "tools.py").write_text(
            '"""Demo tool surface."""\n\nTOOLS = ["run_shell", "send_report"]\n'
        )
        (workspace.current_dir / "skills.md").write_text(
            "# Demo skills\n\n"
            "Be generally helpful.\n\n"
            "- Ask domain-defining questions before implementation questions.\n"
            "- First clarify the domain before proposing execution details.\n"
        )
        (workspace.current_dir / "middleware.py").write_text(
            '"""Demo middleware surface."""\n\n'
            'MIDDLEWARE = ["duplicate tool calls", "reuse prior successful results"]\n'
        )
        workspace.proposal_file.write_text("# Proposal\n\nFixed all four demo surfaces.\n")
        return "Updated prompt, tools, skills, and middleware."

    monkeypatch.setattr(
        "better_harness.agent.invoke_deepagents_proposer",
        fake_proposer,
    )

    config = _write_minimal_pytest_experiment(tmp_path / "fixture")
    output_dir = tmp_path / "pytest-run"
    report = run_experiment(
        load_experiment(config),
        output_dir=output_dir,
        max_iterations=4,
    )
    assert report.final_train.passed == 4
    assert report.final_holdout.passed == 4
    assert report.final_scorecard is not None
    assert report.final_scorecard.passed == 2
    assert (output_dir / "history" / "visible" / "train").exists()
    assert (output_dir / "history" / "private" / "holdout").exists()
    assert (output_dir / "history" / "visible" / "iterations" / "001" / "decision.json").exists()


def test_run_end_to_end_harbor_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_proposer(*, experiment, workspace):
        del experiment
        (workspace.current_dir / "prompt.txt").write_text(
            "If the request is ambiguous, ask questions before acting.\n\n"
            "## Clarifying Requests\n\n"
            "- If a request is underspecified, ask only the minimum number of followup questions needed to take the next useful action.\n"
            "- Do not ask for details the user already supplied.\n"
            "- Use reasonable defaults when the request clearly implies them.\n"
        )
        (workspace.current_dir / "tools.py").write_text('TOOLS = ["run_shell", "send_report"]\n')
        workspace.proposal_file.write_text("# Proposal\n\nPatched prompt and tools.\n")
        return "Patched prompt and tools."

    monkeypatch.setattr(
        "better_harness.agent.invoke_deepagents_proposer",
        fake_proposer,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "demo_agent.py").write_text(
        'BASE_PROMPT = "If the request is ambiguous, ask questions before acting."\n'
    )
    (workspace / "tools.py").write_text('TOOLS = ["run_shell"]\n')

    prompt_base = tmp_path / "prompt.txt"
    prompt_base.write_text('If the request is ambiguous, ask questions before acting.')
    tools_base = tmp_path / "tools_base.py"
    tools_base.write_text('TOOLS = ["run_shell"]\n')

    tasks_root = tmp_path / "tasks"
    for task_name in ("prompt-train", "tool-train", "prompt-holdout", "tool-holdout", "scorecard-story"):
        task_dir = tasks_root / task_name
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(f'name = "{task_name}"\n')

    mock_harbor = tmp_path / "mock_harbor.py"
    mock_harbor.write_text(
        """from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
run = sub.add_parser("run")
run.add_argument("-p", "--project", dest="tasks_root", required=True)
run.add_argument("--task-name", required=True)
run.add_argument("-o", "--output-dir", required=True)
run.add_argument("--job-name", required=True)
run.add_argument("--agent-import-path")
run.add_argument("-l", default="1")
run.add_argument("-n", default="1")
args = parser.parse_args()
workspace_root = Path(os.environ["BETTER_HARNESS_WORKSPACE_ROOT"])
demo_agent = importlib.import_module("demo_agent")
prompt_text = str(demo_agent.BASE_PROMPT)
tools_text = (workspace_root / "tools.py").read_text()
score = 0.0
if args.task_name == "prompt-train":
    score = 1.0 if "minimum number of followup questions" in prompt_text else 0.0
elif args.task_name == "tool-train":
    score = 1.0 if "send_report" in tools_text else 0.0
elif args.task_name == "prompt-holdout":
    score = 1.0 if "Use reasonable defaults" in prompt_text else 0.0
elif args.task_name == "tool-holdout":
    score = 1.0 if "run_shell" in tools_text and "send_report" in tools_text else 0.0
elif args.task_name == "scorecard-story":
    score = 1.0 if "minimum number of followup questions" in prompt_text and "send_report" in tools_text else 0.0
jobs_dir = Path(args.output_dir) / args.job_name / args.task_name
jobs_dir.mkdir(parents=True, exist_ok=True)
(jobs_dir / "result.json").write_text(json.dumps({"score": score, "message": "ok" if score >= 1 else "missing"}, indent=2) + "\\n")
(jobs_dir / "reward.txt").write_text(f"{score}\\n")
"""
    )

    config = tmp_path / "harbor.toml"
    config.write_text(
        f"""
[experiment]
name = "minimal-harbor"
runner = "harbor"
workspace_root = "{workspace}"
model = "demo-model"
max_iterations = 2

[better_agent]
model = "claude-sonnet-4-6"
max_turns = 20

[runner.harbor]
tasks_root = "{tasks_root}"
command = ["python3", "{mock_harbor}"]
agent_import_path = "demo_agent:AutoAgent"
pass_threshold = 1.0

[surfaces.prompt]
kind = "module_attr"
target = "demo_agent:BASE_PROMPT"
base_file = "{prompt_base}"

[surfaces.tools]
kind = "workspace_file"
target = "tools.py"
base_file = "{tools_base}"

[[cases]]
case_id = "prompt-train"
split = "train"
stratum = "prompt"

[[cases]]
case_id = "tool-train"
split = "train"
stratum = "tools"

[[cases]]
case_id = "prompt-holdout"
split = "holdout"
stratum = "prompt"

[[cases]]
case_id = "tool-holdout"
split = "holdout"
stratum = "tools"

[[cases]]
case_id = "scorecard-story"
split = "scorecard"
stratum = "combined"
"""
    )

    output_dir = tmp_path / "harbor-run"
    report = run_experiment(load_experiment(config), output_dir=output_dir, max_iterations=2)
    assert report.final_train.passed == 2
    assert report.final_holdout.passed == 2
    assert report.final_scorecard is not None
    assert report.final_scorecard.passed == 1
    assert (output_dir / "history" / "private" / "scorecard").exists()


def test_cli_inventory_and_split_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = _write_minimal_pytest_experiment(tmp_path / "fixture")
    inventory_path = tmp_path / "inventory.json"
    assert main(["inventory", str(config), "--output", str(inventory_path)]) == 0
    payload = json.loads(inventory_path.read_text())
    assert "tests/test_harness.py::test_prompt_train" in payload["cases"]

    split_dir = tmp_path / "split"
    assert main(["split", str(config), "--output-dir", str(split_dir)]) == 0
    split_payload = json.loads((split_dir / "split.json").read_text())
    assert "scorecard" in split_payload

    captured = capsys.readouterr()
    assert str(inventory_path) in captured.out
