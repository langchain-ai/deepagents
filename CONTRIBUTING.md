# Contributing to Deep Agents

Thank you for your interest in contributing to Deep Agents! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/deepagents.git
   cd deepagents
   ```
3. Install dependencies:
   ```bash
   make setup
   ```

## Development Workflow

### Package Management

We use `uv` for package management. Key commands:

```bash
# Add a dependency
uv add <package>

# Add a development dependency
uv add --group dev <package>

# Sync dependencies
uv sync
```

### Code Quality

We use `ruff` for linting and formatting, and `ty` for type checking.

```bash
# Run linters
make lint

# Format code
make format

# Type check
make typecheck
```

### Testing

```bash
# Run all tests
make test

# Run a specific test file
uv run --group test pytest tests/unit_tests/path/to/test.py
```

- Unit tests are in `tests/unit_tests/` — no network calls allowed
- Integration tests are in `tests/integration_tests/` — network calls permitted

### Working on Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b pr/<type>/<short-description>
   ```

2. Make your changes, following our [development guidelines](AGENTS.md).

3. Write tests for new features or bug fixes.

4. Ensure all checks pass before submitting.

5. Commit using Conventional Commits format:
   ```bash
   git commit -m "type(scope): description"
   ```

## Pull Request Process

1. **Title format**: Follow Conventional Commits with a scope. Example: `feat(sdk): add new feature`
2. **Description**: Explain the *why* behind your changes. Reference any related issues.
3. **Tests**: Ensure new functionality is covered by tests.
4. **Breaking changes**: Discuss with maintainers before submitting.

## Code Standards

- All Python code MUST include type hints
- Use Google-style docstrings for public functions
- Prefer descriptive, self-explanatory variable names
- Follow existing patterns in the codebase
- Keep functions focused — refactor large functions (>20 lines) when appropriate

## Security

- Never use `eval()`, `exec()`, or `pickle` on user-controlled input
- Always handle exceptions properly (no bare `except:`)
- Remove unreachable or commented code before committing

## License

By contributing to Deep Agents, you agree that your contributions will be licensed under the MIT License.
