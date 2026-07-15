.PHONY: all format lint test test_watch

PYTHON_FILES = deepagents_browser tests
UV_FROZEN = true

all: lint test

format:
	uv run --group test ruff format $(PYTHON_FILES)
	uv run --group test ruff check --fix $(PYTHON_FILES)

lint:
	uv run --group test ruff format --check $(PYTHON_FILES)
	uv run --group test ruff check $(PYTHON_FILES)
	uv run --group test ty check deepagents_browser

test:
	uv run --group test pytest --disable-socket --allow-unix-socket

test_watch:
	uv run --group test ptw -- --disable-socket --allow-unix-socket
