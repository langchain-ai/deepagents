# Local Development Setup

## Prerequisites
- Python 3.11+ 
- [uv](https://docs.astral.sh/uv/getting-started/) (recommended) or pip
- Git

## Quick Start

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/libs/deepagents
uv venv && source .venv/bin/activate
uv sync
```

## Environment Variables

Create `.env` in project root:
```bash
OPENAI_API_KEY=your_key          # or ANTHROPIC_API_KEY
TAVILY_API_KEY=your_key          # Optional: for web search
```

## Verify Installation

```bash
uv run python -c "from deepagents import create_deep_agent; print('✓ Ready')"
```

## Common Tasks

**Run tests:**
```bash
uv run pytest tests/
uv run pytest tests/ --cov=deepagents
```

**Format & lint code:**
```bash
uv run ruff format .
uv run ruff check . --fix
```

**Development workflow:**
```bash
git checkout -b feature/name
# Make changes
uv run pytest tests/
uv run ruff format . && uv run ruff check . --fix
git commit -m "feat: description"
```

## Project Structure
```
libs/deepagents/          # Core library
libs/deepagents-cli/      # CLI tool
libs/acp/                 # Agent Client Protocol
libs/harbor/              # Harbor integration
examples/                 # Runnable examples
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: deepagents` | Run `uv sync` and activate venv |
| API key errors | Verify `.env` file exists with keys |
| Tests fail | Run `uv run pytest tests/ -v -s` for details |

## Resources
- [Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [Examples](examples/)