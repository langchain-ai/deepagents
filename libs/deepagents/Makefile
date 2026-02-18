.PHONY: all format lint test tests evals integration_test integration_tests test_watch benchmark help run lint_package lint_tests check_imports coverage typecheck

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# TESTING AND COVERAGE
######################

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
SMOKE_TESTS ?= tests/unit_tests/smoke_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket $(TEST_FILE)

update-snapshots:
	# update snapshots for smoke tests
	uv run --group test pytest -vvv --disable-socket --allow-unix-socket --update-snapshots $(SMOKE_TESTS)

coverage:
	uv run --group test pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered \
		$(TEST_FILE)

integration_test integration_tests:
	uv run --group test pytest -n auto -vvv --timeout 30 $(TEST_FILE)

evals:
	LANGSMITH_TEST_SUITE=deepagents-evals uv run --group test pytest -n auto tests/evals 

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
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/deepagents --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=deepagents
lint_tests: PYTHON_FILES=tests

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	uv run --all-groups ty check deepagents

typecheck:
	uv run --all-groups ty check deepagents

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find deepagents -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters and type checker'
	@echo 'typecheck                    - run type checker (ty)'
	@echo 'check_imports                - check imports'
	@echo 'lint_package                 - lint only the package'
	@echo 'lint_tests                   - lint only tests'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_test             - run integration tests'
	@echo 'integration_tests            - run integration tests'
	@echo 'evals                        - run eval scenarios (LangSmith test suite: deepagents-evals)'
	@echo 'test_watch                   - run tests in watch mode'
	@echo 'benchmark                    - run benchmark tests'
