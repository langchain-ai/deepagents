# deepagents cli

This is the CLI for deepagents

## Skills

Skills are reusable agent capabilities that can be loaded into the CLI. Each agent has its own skills directory at `~/.deepagents/{AGENT_NAME}/skills/`.

For the default agent (named `agent`), skills are stored in `~/.deepagents/agent/skills/`.

### Example Skills

Example skills are provided in the `examples/skills/` directory:

- **web-research** - Structured web research workflow with planning, parallel delegation, and synthesis
- **langgraph-docs** - LangGraph documentation lookup and guidance

To use an example skill with the default agent, copy it to your agent's skills directory:

```bash
mkdir -p ~/.deepagents/agent/skills
cp -r examples/skills/web-research ~/.deepagents/agent/skills/
```

For a custom agent, replace `agent` with your agent name:

```bash
mkdir -p ~/.deepagents/my-agent/skills
cp -r examples/skills/web-research ~/.deepagents/my-agent/skills/
```

### Managing Skills

```bash
# List available skills
deepagents skills list

# Create a new skill from template
deepagents skills create my-skill

# View detailed information about a skill
deepagents skills info web-research
```

## Development

### Running Tests

To run the test suite:

```bash
uv sync --all-groups

make test
```
