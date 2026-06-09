.PHONY: lint format type typecheck test tests test_watch help

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

TEST_FILE ?= tests/
PYTEST_EXTRA ?=

test: ## Run unit tests
test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(PYTEST_EXTRA) $(TEST_FILE) \
		--timeout 10 --cov=deepagents_talon --cov-report=term-missing

test_watch: ## Run tests in watch mode
	uv run --group test ptw . -- $(TEST_FILE)

lint format: PYTHON_FILES=deepagents_talon/ tests/
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/talon --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')

lint: ## Run linters and type checker
lint lint_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff format $(PYTHON_FILES) --diff
	$(MAKE) type

type: ## Run type checker
type typecheck:
	uv run --group test ty check deepagents_talon

format: ## Run code formatters
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group test ruff check --fix $(PYTHON_FILES)

help: ## Show this help message
	@echo "Usage: make [target] [TEST_FILE=path/to/tests/]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
