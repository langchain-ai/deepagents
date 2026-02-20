.PHONY: lint format test help test_integration test_watch run-hello-world run-terminal-bench-modal run-terminal-bench-docker run-terminal-bench-daytona run-terminal-bench-runloop format_unsafe

.DEFAULT_GOAL := help

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests
INTEGRATION_FILES ?= tests/integration_tests

test: ## Run unit tests
	uv run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_integration: ## Run integration tests
	uv run pytest $(INTEGRATION_FILES)

test_watch: ## Run tests in watch mode
	uv run ptw . -- $(TEST_FILE)


# Run jobs
run-hello-world: ## Run hello-world job
	@mkdir -p jobs/hello-world
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset hello-world --task-name hello-world -n 1 --jobs-dir tmp/hello-world --env docker

run-terminal-bench-modal: ## Run terminal-bench on Modal
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 4 --jobs-dir jobs/terminal-bench --env modal

run-terminal-bench-daytona: ## Run terminal-bench on Daytona
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 40 --jobs-dir jobs/terminal-bench --env daytona

run-terminal-bench-docker: ## Run terminal-bench on Docker
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker

run-terminal-bench-runloop: ## Run terminal-bench on Runloop
	@mkdir -p jobs/terminal-bench
	harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env runloop

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
lint format: PYTHON_FILES=deepagents_harbor/ tests/
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=. --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint: ## Run linters and type checker
lint lint_diff:
	[ "$(PYTHON_FILES)" = "" ] ||	uv run --group test ruff format $(PYTHON_FILES) --diff
	@if [ "$(LINT)" != "minimal" ]; then \
		if [ "$(PYTHON_FILES)" != "" ]; then \
			uv run --group test ruff check $(PYTHON_FILES) --diff; \
		fi; \
	fi
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ty check $(PYTHON_FILES)

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
