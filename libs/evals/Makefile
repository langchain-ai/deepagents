.PHONY: lint format type typecheck test help test_integration test_watch run-hello-world run-harbor evals format_unsafe radar radar-from-summary

.DEFAULT_GOAL := help

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests
PYTEST_EXTRA ?=

test: ## Run unit tests
	uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket $(TEST_FILE)

evals: ## Run evals
	LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest tests/evals

test_watch: ## Run tests in watch mode
	uv run --group test ptw . -- $(TEST_FILE)

# Harbor jobs
AGENT_MODE ?= cli
HARBOR_MODEL ?=
HARBOR_SANDBOX_ENV ?= docker
HARBOR_N_TASKS ?= 0
HARBOR_CONCURRENCY ?= 1

run-hello-world: ## Run hello-world job
	@mkdir -p jobs/hello-world
	uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset hello-world --task-name hello-world -n 1 --jobs-dir tmp/hello-world --env docker --agent-kwarg use_cli_agent=$(if $(filter cli,$(AGENT_MODE)),true,false)

run-harbor: ## Run full Harbor workflow (e.g. make run-harbor HARBOR_MODEL=anthropic:claude-sonnet-4-6)
	@if [ -z "$(HARBOR_MODEL)" ]; then echo "Error: HARBOR_MODEL is required (e.g. make run-harbor HARBOR_MODEL=anthropic:claude-sonnet-4-6)" >&2; exit 1; fi
	uv run python scripts/run_harbor.py \
		--model $(HARBOR_MODEL) \
		--env $(HARBOR_SANDBOX_ENV) \
		--concurrency $(HARBOR_CONCURRENCY) \
		--n-tasks $(HARBOR_N_TASKS) \
		--agent-mode $(AGENT_MODE)

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
