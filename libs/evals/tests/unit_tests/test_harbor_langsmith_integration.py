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
    assert "HARBOR_AGENT_GRAPH = $(HARBOR_AGENT_IMPL)" in makefile
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
    """Harbor drives LangSmith via its plugin (not the retired manual
    create-experiment / add-feedback steps), and the run wires the langgraph
    agent to its graph, dataset, attempts, jobs dir, and plugin arguments.

    The functional flag checks run against the EXTRACTED `⚓ Run Harbor` step (as
    the secret-scoping test below does), not the whole file, so a flag that only
    appears in a comment or unrelated step wouldn't satisfy them — while cosmetic
    edits elsewhere (descriptions, input defaults, dataset options, echo lines)
    don't break the test.
    """
    workflow = (ROOT / ".github" / "workflows" / "_harbor_run.yml").read_text()

    # The retired manual experiment wiring must stay gone.
    assert "create-experiment" not in workflow
    assert "add-feedback" not in workflow

    # Isolate the actual `harbor run` invocation.
    run_step = workflow.split('      - name: "⚓ Run Harbor"', maxsplit=1)[1]
    run_step = run_step.split("      - name:", maxsplit=1)[0]

    # Agent is the langgraph deep agent wired to the selected graph.
    assert "--agent langgraph" in run_step
    assert '--agent-kwarg graph="$HARBOR_AGENT_GRAPH"' in run_step
    # Dataset and per-task attempts come from the dispatch inputs.
    assert '--dataset "$HARBOR_DATASET"' in run_step
    assert '--n-attempts "$HARBOR_ROLLOUTS_PER_TASK"' in run_step
    # Results are written under a jobs dir the aggregate job later collects.
    assert "--jobs-dir harbor-jobs/" in run_step
    # LangSmith is driven by the plugin, with dataset + experiment names passed to it.
    assert "--plugin langsmith" in run_step
    assert '--plugin-kwarg dataset_name="$HARBOR_LANGSMITH_DATASET"' in run_step
    assert '--plugin-kwarg experiment_name="$HARBOR_LANGSMITH_EXPERIMENT"' in run_step


def test_harbor_run_step_validates_dispatch_inputs_before_use() -> None:
    """The `⚓ Run Harbor` step must allowlist-validate every dispatch input it
    interpolates into a shell command, so a malicious `workflow_dispatch` value
    can't inject shell. These regexes are security-relevant; the prior whole-file
    test asserted them but the functional-flag refactor dropped that coverage, so
    pin them here — scoped to the extracted step, which is stronger than a
    whole-file substring match (a regex in an unrelated step wouldn't satisfy it).
    """
    workflow = (ROOT / ".github" / "workflows" / "_harbor_run.yml").read_text()
    run_step = workflow.split('      - name: "⚓ Run Harbor"', maxsplit=1)[1]
    run_step = run_step.split("      - name:", maxsplit=1)[0]

    # rollouts must be a positive integer; dataset and resolved task names must
    # match a conservative allowlist before reaching the shell / harbor CLI.
    assert '[[ "$HARBOR_ROLLOUTS_PER_TASK" =~ ^[1-9][0-9]*$ ]]' in run_step
    assert '[[ "$HARBOR_DATASET" =~ ^[A-Za-z0-9._/-]+$ ]]' in run_step
    assert '[[ "$t" =~ ^[A-Za-z0-9._/?*-]+$ ]]' in run_step


def test_harbor_workflow_scopes_secrets_to_runtime_steps() -> None:
    workflow = (ROOT / ".github" / "workflows" / "_harbor_run.yml").read_text()

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
        "NVIDIA_API_KEY",
        "OLLAMA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
    ]:
        assert secret not in job_env

    assert "secrets." not in install_step
    assert "LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}" in run_step
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
    # "tau3-subset" is a selectable dataset on the dispatch workflow.
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()
    assert '- "tau3-subset"' in workflow

    # Its resolution + run wiring lives in the reusable leaf.
    leaf = (ROOT / ".github" / "workflows" / "_harbor_run.yml").read_text()
    # Its resolution step pulls the committed task filter, runs against the real
    # registry dataset, and names the LangSmith dataset "tau3-subset".
    assert "if: env.HARBOR_DATASET == 'tau3-subset'" in leaf
    assert "from deepagents_evals.tau3_subset import INCLUDE_TASKS" in leaf
    assert "HARBOR_DATASET=sierra-research/tau3-bench" in leaf
    assert "HARBOR_LANGSMITH_DATASET_NAME=tau3-subset" in leaf
    # Injecting the task filter is the whole point of the step: it must be
    # written and it must land in $GITHUB_ENV, or the run silently uses the full
    # dataset. Guard both the payload line and the redirect.
    assert "HARBOR_INCLUDE_TASKS=$include_tasks" in leaf
    assert '} >> "$GITHUB_ENV"' in leaf
    # The resolve step must fail loudly on a wrong-sized filter (empty => full
    # dataset), so the count tripwire must stay wired.
    assert 'if [ "$task_count" -ne 30 ]; then' in leaf
    # tau3's verifier/judge is always an OpenAI model, so the leaf provides the
    # OpenAI key unconditionally, and the preflight fails loudly if it is missing
    # for a tau3 run whose agent model is hosted by another provider.
    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in leaf
    assert '[[ "$HARBOR_DATASET" == *tau3* ]]' in leaf
    assert '[ "$model_provider" != "openai" ]' in leaf


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
    # A swap-duplicate task_id keeps len(entries)==30 and the tier tuple intact,
    # so pin uniqueness and the exact {DATASET}__{task_id} round-trip explicitly.
    assert len(set(entries)) == len(TASKS)
    assert set(entries) == {f"{DATASET}__{t.task_id}" for t in TASKS}
