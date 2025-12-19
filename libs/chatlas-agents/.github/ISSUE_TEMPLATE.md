# Issue Report Template

Please use this template when reporting issues with the ChATLAS agent.

## Description

A clear and concise description of the issue.

### Steps to Reproduce

1. First step...
2. Second step...
3. Expected behavior...
4. Actual behavior...

## Environment

- **Python Version**: (e.g., 3.11, 3.13)
- **OS**: (e.g., Linux, macOS, Windows)
- **Shell**: (e.g., bash, zsh)
- **Key Environment Variables**: 
  ```
  CHATLAS_LLM_MODEL=gpt-4-turbo
  CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
  ```

## Command Used

```bash
uv run python -m chatlas_agents.cli run --input "your query?" --verbose
```

## Error Message

```
[Paste full error output here, including traceback if available]
```

## Logs

Run with verbose flag and paste relevant logs:

```bash
uv run python -m chatlas_agents.cli run --input "test" --verbose
```

```
[Paste logs here]
```

## Checklist

- [ ] I've verified my CHATLAS_LLM_API_KEY is set correctly
- [ ] I've tested connectivity to MCP server: `curl https://chatlas-mcp.app.cern.ch/mcp`
- [ ] I've run the command with `--verbose` flag
- [ ] I've checked the troubleshooting section in AGENT_INSTRUCTIONS.md
- [ ] This issue doesn't duplicate an existing issue

## Additional Context

Add any other relevant context about the issue here.
