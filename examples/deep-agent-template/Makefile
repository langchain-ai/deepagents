.PHONY: help install dev serve test integration-tests lint format

help:
	@echo 'Targets:'
	@echo '  install             Sync runtime dependencies with uv'
	@echo '  dev                 Sync project + dev dependencies with uv'
	@echo '  serve               Start the local LangGraph dev server'
	@echo '  test                Run unit tests'
	@echo '  integration-tests   Run integration tests'
	@echo '  lint                Run Ruff checks'
	@echo '  format              Format with Ruff'

install:
	uv sync --no-dev

dev:
	uv sync

serve:
	uv run langgraph dev

test:
	uv run python -m pytest tests/unit_tests -q

integration-tests:
	uv run python -m pytest tests/integration_tests -q

lint:
	uv run python -m ruff check src tests

format:
	uv run python -m ruff format src tests
