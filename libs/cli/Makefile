.PHONY: all format lint type test tests integration_test integration_tests test_watch benchmark help run lint_package lint_tests check_imports coverage

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest -n auto --disable-socket --allow-unix-socket $(TEST_FILE)

coverage:
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

integration_test integration_tests:
	uv run --group test pytest -n auto -vvv --timeout 30 $(TEST_FILE)

test_watch:
	uv run --group test ptw --now . -- -vv $(TEST_FILE)

benchmark:
	uv run --group test pytest ./tests -m benchmark

run:
	uvx --no-cache --reinstall .


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/cli --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=deepagents_cli
lint_tests: PYTHON_FILES=tests

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ty check $(PYTHON_FILES)

type:
	uv run --all-groups ty check $(PYTHON_FILES)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find deepagents_cli -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo 'check_imports                - check imports'
	@echo 'lint_package                 - lint only the package'
	@echo 'lint_tests                   - lint only tests'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_test             - run integration tests'
	@echo 'integration_tests            - run integration tests'
	@echo 'test_watch                   - run tests in watch mode'
	@echo 'benchmark                    - run benchmark tests'
