# 🚀🧠 Deep Agents CLI

The [deepagents](https://github.com/langchain-ai/deepagents) CLI is an open source coding assistant that runs in your terminal, similar to Claude Code.

For more information on usage and configuration, please see the [Deep Agents CLI docs](https://docs.langchain.com/oss/python/deepagents/cli).

## Development

### Running Tests

To run the test suite:

```bash
uv sync --all-groups

make test
```

### Running During Development

```bash
# From libs/deepagents-cli directory
uv run deepagents

# Or install in editable mode
uv pip install -e .
deepagents
```

### Modifying the CLI

- **UI changes** → Edit `ui.py` or `input.py`
- **Add new tools** → Edit `tools.py`
- **Change execution flow** → Edit `execution.py`
- **Add commands** → Edit `commands.py`
- **Agent configuration** → Edit `agent.py`
- **Skills system** → Edit `skills/` modules
- **Constants/colors** → Edit `config.py`
