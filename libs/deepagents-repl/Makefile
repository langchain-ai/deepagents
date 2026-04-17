.PHONY: test tests integration_test integration_tests lint format help

.DEFAULT_GOAL := help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test: ## Run unit tests
test tests:
	uv run --group test pytest -vvv $(PYTEST_EXTRA) $(TEST_FILE)

integration_test: ## Run integration tests
integration_test integration_tests:
	uv run --group test pytest -vvv --timeout 30 $(TEST_FILE)

lint: ## Run ruff
	uv run --group test ruff check deepagents_repl tests

format: ## Apply ruff fixes
	uv run --group test ruff check --fix deepagents_repl tests

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
