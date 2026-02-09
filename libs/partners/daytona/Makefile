.PHONY: all format lint test tests integration_test integration_tests test_watch benchmark help

all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TESTING
######################

TEST_FILE ?= tests/unit_tests/

integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test pytest -vvv --timeout 30 $(TEST_FILE)

test_watch:
	uv run --group test ptw --now . -- -vv $(TEST_FILE)

benchmark:
	uv run --group test pytest ./tests -m benchmark

######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/daytona --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_daytona

lint lint_diff lint_package:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'lint_package                 - lint only the package'
	@echo 'test                         - run unit tests'
	@echo 'integration_test             - run integration tests'
	@echo 'test_watch                   - run tests in watch mode'
	@echo 'benchmark                    - run benchmark tests'
