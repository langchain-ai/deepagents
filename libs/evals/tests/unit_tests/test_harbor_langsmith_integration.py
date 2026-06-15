"""Static checks for Harbor LangSmith plugin integration."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[4]
EVALS = ROOT / "libs" / "evals"


def test_evals_uses_harbor_langsmith_fork_source() -> None:
    pyproject = tomllib.loads((EVALS / "pyproject.toml").read_text())

    assert pyproject["tool"]["uv"]["sources"]["harbor"] == {
        "git": "https://github.com/nick-hollon-lc/harbor.git",
        "branch": "nh/integration",
    }


def test_langsmith_make_target_uses_harbor_plugin_and_langgraph_agent() -> None:
    makefile = (EVALS / "Makefile").read_text()
    _, target = makefile.split("run-terminal-bench-langsmith:", maxsplit=1)
    target = target.split("\n\n", maxsplit=1)[0]

    assert "HARBOR_AGENT_ARGS = --agent langgraph" in makefile
    assert "--agent-kwarg project_path=deepagents_harbor/langgraph_project" in makefile
    assert "--agent-kwarg config=langgraph.json" in makefile
    assert "--agent-kwarg graph=deepagent" in makefile
    assert "HARBOR_AGENT_ENV_ARGS ?=" in makefile
    assert "--agent-env 'ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY}'" in makefile
    assert "--agent-env 'LANGSMITH_API_KEY=$${LANGSMITH_API_KEY}'" in makefile
    assert "$(HARBOR_AGENT_ARGS)" in target
    assert "$(HARBOR_AGENT_ENV_ARGS)" in target
    assert "--jobs-dir $(HARBOR_TERMINAL_BENCH_JOBS_DIR)" in target
    assert "--plugin langsmith" in target
    assert "--plugin-kwarg dataset_name=terminal-bench@2.0" in target
    assert "--plugin-kwarg experiment_name=" in target
    assert "--agent-import-path deepagents_harbor:DeepAgentsWrapper" not in target


def test_makefile_no_longer_uses_custom_harbor_wrapper() -> None:
    makefile = (EVALS / "Makefile").read_text()

    assert "--agent-import-path deepagents_harbor:DeepAgentsWrapper" not in makefile
    assert "AGENT_MODE" not in makefile
    assert "HARBOR_TERMINAL_BENCH_JOBS_DIR ?= ../harbor-jobs/terminal-bench" in makefile


def test_harbor_workflow_uses_plugin_instead_of_manual_experiment_steps() -> None:
    workflow = (ROOT / ".github" / "workflows" / "harbor.yml").read_text()

    assert "create-experiment" not in workflow
    assert "add-feedback" not in workflow
    assert "--agent langgraph" in workflow
    assert "--agent-kwarg project_path=deepagents_harbor/langgraph_project" in workflow
    assert "--agent-kwarg config=langgraph.json" in workflow
    assert "--agent-kwarg graph=deepagent" in workflow
    assert "agent_env_args=(" in workflow
    assert "--agent-env 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}'" in workflow
    assert "--agent-env 'LANGSMITH_API_KEY=${LANGSMITH_API_KEY}'" in workflow
    assert '"${agent_env_args[@]}"' in workflow
    assert "--plugin langsmith" in workflow
    assert "--jobs-dir ../harbor-jobs/terminal-bench" in workflow
    assert 'Path("../harbor-jobs/terminal-bench")' in workflow
    assert '--plugin-kwarg dataset_name="$HARBOR_LANGSMITH_DATASET"' in workflow
    assert '--plugin-kwarg experiment_name="$HARBOR_LANGSMITH_EXPERIMENT"' in workflow
