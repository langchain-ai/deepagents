.PHONY: format lint type test tests integration_test integration_tests test_watch benchmark help run lint_package lint_tests check_imports coverage

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test: ## Run unit tests
test tests:
	uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

coverage: ## Run unit tests with coverage
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

integration_test: ## Run integration tests
integration_test integration_tests:
	uv run --group test pytest -n auto -vvv --timeout 30 $(TEST_FILE)

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
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/cli --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: ## Lint only the package
lint_package: PYTHON_FILES=deepagents_cli
lint_tests: ## Lint only tests
lint_tests: PYTHON_FILES=tests

lint: ## Run linters and type checker
lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ty check $(PYTHON_FILES)

type: ## Run type checker
	uv run --all-groups ty check $(PYTHON_FILES)

format: ## Run code formatters
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: ## Check imports
check_imports: $(shell find deepagents_cli -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help: ## Show this help message
	@echo "Usage: make [target] [TEST_FILE=path/to/tests/]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
