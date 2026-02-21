.PHONY: format lint test tests evals integration_test integration_tests test_watch benchmark help run lint_package lint_tests check_imports coverage typecheck update-snapshots

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
SMOKE_TESTS ?= tests/unit_tests/smoke_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test: ## Run unit tests
test tests:
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket $(TEST_FILE)

update-snapshots: ## Update smoke test snapshots
	# update snapshots for smoke tests
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket --update-snapshots $(SMOKE_TESTS)

coverage: ## Run unit tests with coverage
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

integration_test: ## Run integration tests
integration_test integration_tests:
	uv run --group test pytest -n auto -vvv --timeout 30 $(TEST_FILE)

evals: ## Run evals
	LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest -n auto tests/evals

test_watch: ## Run tests in watch mode
	uv run --group test ptw --now . -- -vv $(TEST_FILE)

benchmark: ## Run benchmark tests
	uv run --group test pytest ./tests -m benchmark

run: ## Reinstall and run package
	uvx --no-cache --reinstall .


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/deepagents --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: ## Lint only the package
lint_package: PYTHON_FILES=deepagents
lint_tests: ## Lint only tests
lint_tests: PYTHON_FILES=tests

lint: ## Run linters and type checker
lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	uv run --all-groups ty check deepagents

typecheck: ## Run type checker
	uv run --all-groups ty check deepagents

format: ## Run code formatters
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: ## Check imports
check_imports: $(shell find deepagents -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help: ## Show this help message
	@echo "Usage: make [target] [TEST_FILE=path/to/tests/]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
