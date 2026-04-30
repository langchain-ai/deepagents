.PHONY: lint format type typecheck test help test_integration test_watch run-hello-world run-terminal-bench-modal run-terminal-bench-docker run-terminal-bench-daytona run-terminal-bench-runloop evals format_unsafe radar radar-from-summary model-groups eval-catalog

.DEFAULT_GOAL := help

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests
PYTEST_EXTRA ?=

test: ## Run unit tests
	uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket $(TEST_FILE)

evals: ## Run evals (set MODEL=<id>; required)
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL is required. Example: make evals MODEL=claude-opus-4-7" >&2; \
		exit 1; \
	fi
	LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals -v --tb=short --model $(MODEL) $(PYTEST_EXTRA)

test_watch: ## Run tests in watch mode
	uv run --group test ptw . -- $(TEST_FILE)

# Harbor jobs
# -n = concurrent trials (parallel sandbox slots), NOT task count
# -l = max tasks to run (omit for all)
# AGENT_MODE: cli (local default) or sdk (CI default)
AGENT_MODE ?= cli
AGENT_KWARG = --agent-kwarg use_cli_agent=$(if $(filter cli,$(AGENT_MODE)),true,false)

run-hello-world: ## Run hello-world job
	@mkdir -p jobs/hello-world
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset hello-world --task-name hello-world -n 1 --jobs-dir tmp/hello-world --env docker $(AGENT_KWARG)

run-terminal-bench-modal: ## Run terminal-bench on Modal (4 concurrent)
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 4 --jobs-dir jobs/terminal-bench --env modal $(AGENT_KWARG)

run-terminal-bench-daytona: ## Run terminal-bench on Daytona (40 concurrent)
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 40 --jobs-dir jobs/terminal-bench --env daytona $(AGENT_KWARG)

run-terminal-bench-docker: ## Run terminal-bench on Docker (sequential)
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker $(AGENT_KWARG)

run-terminal-bench-runloop: ## Run terminal-bench on Runloop (10 concurrent)
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env runloop $(AGENT_KWARG)

######################
# CHARTS
######################

RADAR_OUTPUT ?= charts/radar.png

radar: ## Generate eval radar chart (toy data by default)
	uv run --extra charts python scripts/generate_radar.py --toy -o $(RADAR_OUTPUT)

SUMMARY_JSON ?= evals_summary.json

radar-from-summary: ## Generate radar chart from evals_summary.json
	uv run --extra charts python scripts/generate_radar.py --summary $(SUMMARY_JSON) -o $(RADAR_OUTPUT)

######################
# MODEL GROUPS
######################

model-groups: ## Regenerate MODEL_GROUPS.md from the canonical registry
	uv run python scripts/generate_model_groups.py

######################
# EVAL CATALOG
######################

eval-catalog: ## Regenerate EVAL_CATALOG.md from eval test files
	uv run python scripts/generate_eval_catalog.py

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
lint format: PYTHON_FILES=deepagents_evals/ deepagents_harbor/ tests/
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=. --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint: ## Run linters and type checker
lint lint_diff:
	[ "$(PYTHON_FILES)" = "" ] ||	uv run --group test ruff format $(PYTHON_FILES) --diff
	@if [ "$(LINT)" != "minimal" ]; then \
		if [ "$(PYTHON_FILES)" != "" ]; then \
			uv run --group test ruff check $(PYTHON_FILES) --diff; \
		fi; \
	fi
	$(MAKE) type PYTHON_FILES="$(PYTHON_FILES)"
	uv run python scripts/generate_eval_catalog.py --check

type: ## Run type checker (eval + harbor source + unit tests)
type typecheck:
	uv run --group test ty check deepagents_evals/ deepagents_harbor/ tests/unit_tests/

format: ## Run code formatters
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff check --fix $(PYTHON_FILES)

format_unsafe: ## Run formatters with unsafe fixes
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff format --unsafe-fixes $(PYTHON_FILES)

######################
# HELP
######################

help: ## Show this help message
	@echo "Usage: make [target] [TEST_FILE=path/to/tests/]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
