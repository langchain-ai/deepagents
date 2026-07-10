"""Static checks for Harbor LangSmith plugin integration."""

from __future__ import annotations

import tomllib
from pathlib import Path

from deepagents_evals.tau3_subset import DATASET, INCLUDE_TASKS, TASKS

ROOT = Path(__file__).parents[4]
EVALS = ROOT / "libs" / "evals"


def test_evals_uses_published_harbor_langsmith_dependency() -> None:
    pyproject = tomllib.loads((EVALS / "pyproject.toml").read_text())

    assert "harbor[langsmith]>=0.16.1,<0.17.0" in pyproject["project"]["dependencies"]
    assert "harbor" not in pyproject["tool"]["uv"]["sources"]


def test_langsmith_make_target_uses_harbor_plugin_and_langgraph_agent() -> None:
    makefile = (EVALS / "Makefile").read_text()
    _, target = makefile.split("run-terminal-bench-langsmith:", maxsplit=1)
    target = target.split("\n\n", maxsplit=1)[0]

    assert "HARBOR_AGENT_IMPL ?= dcode" in makefile
    assert (
        "HARBOR_AGENT_GRAPH = $(if $(filter bare,$(HARBOR_AGENT_IMPL)),bare_deepagent,deepagent)"
        in makefile
    )
    assert "HARBOR_AGENT_ARGS = --agent langgraph" in makefile
    assert "HARBOR_LANGGRAPH_PROJECT = deepagents_harbor/langgraph_project" in makefile
    assert "--agent-kwarg project_path=$(HARBOR_LANGGRAPH_PROJECT)" in makefile
    assert "--agent-kwarg config=langgraph.json" in makefile
    assert "--agent-kwarg graph=$(HARBOR_AGENT_GRAPH)" in makefile
    assert "stage-harbor-local-deps:" in makefile
    assert "../deepagents/ $(HARBOR_LOCAL_DEPS_DIR)/deepagents/" in makefile
    assert "../code/ $(HARBOR_LOCAL_DEPS_DIR)/deepagents-code/" in makefile
    assert "HARBOR_AGENT_ENV_ARGS ?=" in makefile
    assert "HARBOR_TERMINAL_BENCH_DATASET ?= terminal-bench/terminal-bench-2" in makefile
    assert "--agent-env 'ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY}'" in makefile
    assert "--agent-env 'LANGSMITH_API_KEY=$${LANGSMITH_API_KEY}'" in makefile
    assert "$(HARBOR_AGENT_ARGS)" in target
    assert "$(HARBOR_AGENT_ENV_ARGS)" in target
    assert "--jobs-dir $(HARBOR_TERMINAL_BENCH_JOBS_DIR)" in target
    assert "--plugin langsmith" in target
    assert "--dataset $(HARBOR_TERMINAL_BENCH_DATASET)" in target
    assert "--plugin-kwarg dataset_name=$(HARBOR_TERMINAL_BENCH_DATASET)" in target
    assert "--plugin-kwarg experiment_name=" in target
    assert "--agent-import-path deepagents_harbor:DeepAgentsWrapper" not in target


def test_makefile_no_longer_uses_custom_harbor_wrapper() -> None:
    makefile = (EVALS / "Makefile").read_text()

    assert "--agent-import-path deepagents_harbor:DeepAgentsWrapper" not in makefile
    assert "AGENT_MODE" not in makefile
    assert "HARBOR_HELLO_WORLD_JOBS_DIR ?= harbor-jobs/hello-world" in makefile
    assert "HARBOR_TERMINAL_BENCH_JOBS_DIR ?= harbor-jobs/terminal-bench" in makefile
    for target_name in [
        "run-hello-world",
        "run-terminal-bench-modal",
        "run-terminal-bench-daytona",
        "run-terminal-bench-docker",
        "run-terminal-bench-runloop",
    ]:
        _, target = makefile.split(f"{target_name}:", maxsplit=1)
        target = target.split("\n\n", maxsplit=1)[0]
        assert "$(HARBOR_AGENT_ENV_ARGS)" in target
        assert "stage-harbor-local-deps" in target


def test_harbor_workflow_uses_plugin_instead_of_manual_experiment_steps() -> None:
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()

    assert "create-experiment" not in workflow
    assert "add-feedback" not in workflow
    assert "agent_impl:" in workflow
    assert 'default: "dcode"' in workflow
    assert "          - dcode" in workflow
    assert "dataset:" in workflow
    assert "Terminal-bench dataset to run through Harbor." in workflow
    assert '- "terminal-bench/terminal-bench-2"' in workflow
    assert '- "terminal-bench/terminal-bench-2-1"' in workflow
    assert '- "sierra-research/tau3-bench"' in workflow
    assert "dataset_override:" in workflow
    assert "          - tau3" in workflow
    assert "include_tasks:" in workflow
    assert "Space-separated task-name globs" in workflow
    assert "rollouts_per_task:" in workflow
    assert 'default: "1"' in workflow
    assert "HARBOR_AGENT_IMPL: ${{ inputs.agent_impl }}" in workflow
    assert (
        "HARBOR_DATASET: ${{ inputs.dataset_override || inputs.dataset || "
        "'terminal-bench/terminal-bench-2' }}"
    ) in workflow
    assert "HARBOR_INCLUDE_TASKS: ${{ inputs.include_tasks }}" in workflow
    assert "HARBOR_ROLLOUTS_PER_TASK: ${{ inputs.rollouts_per_task }}" in workflow
    assert "from fnmatch import fnmatch" in workflow
    assert "No Harbor tasks matched include_tasks filters" in workflow
    assert "select_shard_tasks(selected, [], n_tasks, n, i)" in workflow
    assert 'echo "| \\`dataset\\` | \\`${DATASET}\\` |"' in workflow
    assert "INCLUDE_TASKS: ${{ inputs.include_tasks }}" in workflow
    assert 'echo "| \\`include_tasks\\` | \\`${INCLUDE_TASKS}\\` |"' in workflow
    assert 'echo "| \\`rollouts_per_task\\` | \\`${ROLLOUTS_PER_TASK}\\` |"' in workflow
    assert '[[ "$HARBOR_ROLLOUTS_PER_TASK" =~ ^[1-9][0-9]*$ ]]' in workflow
    assert '[[ "$HARBOR_DATASET" =~ ^[A-Za-z0-9._/-]+$ ]]' in workflow
    assert '[[ "$t" =~ ^[A-Za-z0-9._/?*-]+$ ]]' in workflow
    assert '"${include_args[@]}"' in workflow
    assert '--n-attempts "$HARBOR_ROLLOUTS_PER_TASK"' in workflow
    assert 'echo "- Included tasks: ${HARBOR_INCLUDE_TASKS}"' in workflow
    assert 'echo "- Included tasks: all"' in workflow
    assert 'echo "- Rollouts per task: ${HARBOR_ROLLOUTS_PER_TASK}"' in workflow
    assert (
        'HARBOR_LANGSMITH_DATASET="${HARBOR_LANGSMITH_DATASET_NAME:-$HARBOR_DATASET}"' in workflow
    )
    assert "HARBOR_AGENT_GRAPH=deepagent" in workflow
    assert "HARBOR_AGENT_GRAPH=bare_deepagent" in workflow
    assert "--agent langgraph" in workflow
    assert "--agent-kwarg project_path=deepagents_harbor/langgraph_project" in workflow
    assert "--agent-kwarg config=langgraph.json" in workflow
    assert '--agent-kwarg graph="$HARBOR_AGENT_GRAPH"' in workflow
    assert 'local_deps_dir="deepagents_harbor/langgraph_project/.local_deps"' in workflow
    assert '../deepagents/ "$local_deps_dir/deepagents/"' in workflow
    assert '../code/ "$local_deps_dir/deepagents-code/"' in workflow
    assert "agent_env_args=(" in workflow
    assert "--agent-env 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}'" in workflow
    assert (
        "fireworks)\n"
        "              agent_env_args+=(\n"
        "                --agent-env 'FIREWORKS_API_KEY=${FIREWORKS_API_KEY}'\n"
        "                --agent-env UV_PRERELEASE=allow\n"
        "              )" in workflow
    )
    assert "--agent-env 'LANGSMITH_API_KEY=${LANGSMITH_API_KEY}'" in workflow
    assert "--agent-env 'OLLAMA_HOST=${OLLAMA_HOST}'" in workflow
    assert '"${agent_env_args[@]}"' in workflow
    assert '--dataset "$HARBOR_DATASET"' in workflow
    assert "--plugin langsmith" in workflow
    assert "--jobs-dir harbor-jobs/terminal-bench" in workflow
    assert 'Path("harbor-jobs/terminal-bench")' in workflow
    assert "libs/evals/harbor-jobs/terminal-bench" in workflow
    assert '--plugin-kwarg dataset_name="$HARBOR_LANGSMITH_DATASET"' in workflow
    assert '--plugin-kwarg experiment_name="$HARBOR_LANGSMITH_EXPERIMENT"' in workflow


def test_harbor_workflow_scopes_secrets_to_runtime_steps() -> None:
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()

    _, harbor_job = workflow.split("  harbor:", maxsplit=1)
    job_env = harbor_job.split("    steps:", maxsplit=1)[0]
    install_step = harbor_job.split('      - name: "📦 Install Dependencies"', maxsplit=1)[1]
    install_step = install_step.split("      - name:", maxsplit=1)[0]
    run_step = harbor_job.split('      - name: "⚓ Run Harbor"', maxsplit=1)[1]
    run_step = run_step.split("      - name:", maxsplit=1)[0]

    for secret in [
        "ANTHROPIC_API_KEY",
        "BASETEN_API_KEY",
        "FIREWORKS_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_SANDBOX_API_KEY",
        "NVIDIA_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ]:
        assert secret not in job_env

    assert "secrets." not in install_step
    assert "LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}" in run_step
    assert "inputs.sandbox_env == 'langsmith'" in run_step
    assert "startsWith(matrix.model, 'fireworks:')" in run_step
    assert "startsWith(matrix.model, 'ollama:')" in run_step


def test_harbor_workflow_only_exposes_docker_and_langsmith_sandboxes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()

    _, sandbox_input = workflow.split("sandbox_env:", maxsplit=1)
    sandbox_input = sandbox_input.split("agent_impl:", maxsplit=1)[0]

    assert "- docker" in sandbox_input
    assert "- langsmith" in sandbox_input
    for sandbox in ["daytona", "modal", "runloop", "vercel"]:
        assert f"- {sandbox}" not in sandbox_input
        assert f"{sandbox})" not in workflow

    for secret in [
        "DAYTONA_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "RUNLOOP_API_KEY",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
        "VERCEL_TOKEN",
    ]:
        assert secret not in workflow


def test_contributing_docs_use_langsmith_sandbox_example() -> None:
    contributing = (EVALS / "CONTRIBUTING.md").read_text()

    assert "# Run via Daytona" not in contributing
    assert "# Run via LangSmith sandboxes" in contributing
    assert "--jobs-dir harbor-jobs/terminal-bench" in contributing
    assert "--env langsmith" in contributing
    assert "--plugin langsmith" in contributing


def test_eval_workflow_scopes_secrets_away_from_dependency_install() -> None:
    workflow = (ROOT / ".github" / "workflows" / "_eval.yml").read_text()

    _, eval_job = workflow.split("  eval:", maxsplit=1)
    job_env = eval_job.split("    steps:", maxsplit=1)[0]
    install_step = eval_job.split('      - name: "📦 Install Dependencies"', maxsplit=1)[1]
    install_step = install_step.split("      - name:", maxsplit=1)[0]
    run_step = eval_job.split('      - name: "📊 Run Evals"', maxsplit=1)[1]
    run_step = run_step.split("      - name:", maxsplit=1)[0]
    analysis_step = eval_job.split('      - name: "🧠 Analyze eval failures"', maxsplit=1)[1]
    analysis_step = analysis_step.split("      - name:", maxsplit=1)[0]

    for secret in [
        "ANTHROPIC_API_KEY",
        "BASETEN_API_KEY",
        "FIREWORKS_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "LANGSMITH_API_KEY",
        "NVIDIA_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ]:
        assert secret not in job_env

    assert "secrets." not in install_step
    assert "LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}" in run_step
    assert "inputs.provider == 'fireworks'" in run_step
    assert "inputs.provider == 'ollama'" in run_step
    assert "startsWith(inputs.analysis_model, 'anthropic:')" in analysis_step


def test_harbor_workflow_wires_tau3_subset() -> None:
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()

    # "tau3-subset" is a selectable dataset.
    assert '- "tau3-subset"' in workflow
    # Its resolution step pulls the committed task filter, runs against the real
    # registry dataset, and names the LangSmith dataset "tau3-subset".
    assert "if: env.HARBOR_DATASET == 'tau3-subset'" in workflow
    assert "from deepagents_evals.tau3_subset import INCLUDE_TASKS" in workflow
    assert "HARBOR_DATASET=sierra-research/tau3-bench" in workflow
    assert "HARBOR_LANGSMITH_DATASET_NAME=tau3-subset" in workflow


def test_tau3_subset_constant_is_well_formed() -> None:
    tiers = [t.tier for t in TASKS]
    assert len(TASKS) == 30
    assert (tiers.count("easy"), tiers.count("medium"), tiers.count("hard")) == (
        2,
        7,
        21,
    )
    entries = INCLUDE_TASKS.split()
    assert len(entries) == 30
    assert all(e.startswith(f"{DATASET}__tau3-") for e in entries)
    assert all(t.justification for t in TASKS)
