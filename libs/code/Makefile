.PHONY: format lint type typecheck test tests integration_test integration_tests test_watch benchmark bench bench-memory help run lint_package lint_tests check_imports coverage commands-catalog commands-catalog-check check bootstrap

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

REPO_ROOT := $(shell git rev-parse --show-toplevel)

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

COV_ARGS ?= --cov=deepagents_code --cov-report=term-missing
PYTEST_EXTRA ?=

test: ## Run unit tests
test tests:
	uv run --group test pytest -n auto --benchmark-disable --disable-socket --allow-unix-socket $(PYTEST_EXTRA) $(TEST_FILE) \
		$(COV_ARGS)

coverage: ## Run unit tests with coverage
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

integration_test: ## Run integration tests
integration_test integration_tests:
	uv run --group test pytest -n auto --benchmark-disable -vvv --timeout 30 $(TEST_FILE)

test_watch: ## Run tests in watch mode
	uv run --group test ptw --now . -- -vv $(TEST_FILE)

benchmark: ## Run benchmark tests
	uv run --group test pytest ./tests -m benchmark

bench: ## Run benchmarks under CodSpeed instrumentation
	uv run --group test pytest ./tests -m benchmark --codspeed

bench-memory: ## Run memory benchmarks under CodSpeed instrumentation
	uv run --group test pytest ./tests -m memory_benchmark --codspeed

run: ## Reinstall and run package
	uvx --no-cache --reinstall .


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/code --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: ## Lint only the package
lint_package: PYTHON_FILES=deepagents_code
lint_tests: ## Lint only tests
lint_tests: PYTHON_FILES=tests

lint: ## Run linters and type checker
lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	$(MAKE) type PYTHON_FILES="$(PYTHON_FILES)"
	uv run --all-groups python scripts/generate_commands_catalog.py --check

type: ## Run type checker
type typecheck:
	uv run --all-groups ty check $(PYTHON_FILES)

format: ## Run code formatters
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: ## Check imports
check_imports: $(shell find deepagents_code -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

commands-catalog: ## Regenerate COMMANDS.md from the slash-command registry
	uv run --all-groups python scripts/generate_commands_catalog.py

commands-catalog-check: ## Verify COMMANDS.md matches the slash-command registry
	uv run --all-groups python scripts/generate_commands_catalog.py --check

######################
# LOCAL CI PARITY
######################

bootstrap: ## Install dev dependencies and git hooks for local development
	uv sync --group test
	cd "$(REPO_ROOT)" && uvx pre-commit install --install-hooks
	@echo "✅ Dev environment ready: deps synced and git hooks installed."

check: ## Run the full local CI gauntlet (lint, types, imports, tests, repo gates)
check: lint check_imports test
	@echo "🔍 Verifying extras stay in sync with required deps..."
	uv run --all-groups python "$(REPO_ROOT)/.github/scripts/check_extras_sync.py" pyproject.toml
	@echo "🔍 Verifying pyproject.toml and _version.py agree..."
	uv run --all-groups python "$(REPO_ROOT)/.github/scripts/check_version_equality.py"
	@echo "🔒 Verifying uv.lock is up-to-date..."
	UV_FROZEN=false uv lock --check
	@echo "🔗 Checking SDK pin against workspace SDK (advisory)..."
	@uv run --all-groups python "$(REPO_ROOT)/.github/scripts/check_sdk_pin.py" || \
		{ rc=$$?; [ $$rc -eq 1 ] && echo "⚠️  Stale SDK pin (advisory, non-fatal)." || exit $$rc; }
	@echo "✅ All local CI checks passed!"

######################
# HELP
######################

help: ## Show this help message
	@echo "Usage: make [target] [TEST_FILE=path/to/tests/]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
