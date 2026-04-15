# Deploy Subagents Support

**Date:** 2026-04-15
**Status:** Draft

## Overview

Add support for declaring sync and async subagents in the deepagents deploy filesystem convention. Sync subagents are defined as subdirectories under `subagents/` with their own config, prompt, skills, and MCP servers. Async subagents are references to remote deployed agents declared inline in the parent's `deepagents.toml`.

## Directory Structure

```
my-agent/
  deepagents.toml            # Parent config + async subagent refs
  AGENTS.md                  # Parent system prompt
  mcp.json                   # Parent MCP servers
  skills/                    # Parent skills
  user/                      # Per-user memory template
  subagents/
    researcher/
      deepagents.toml        # Required: [agent] with name, description, optional model
      AGENTS.md              # Required: subagent system prompt
      mcp.json               # Optional: subagent-specific MCP servers
      skills/                # Optional: subagent-specific skills
        deep-search/
          SKILL.md
    code-reviewer/
      deepagents.toml
      AGENTS.md
```

## Config Schema

### Parent `deepagents.toml`

`description` is added as an optional field to `[agent]`:

```toml
[agent]
name = "my-agent"
description = "A helpful coding assistant"  # Optional for parent
model = "anthropic:claude-sonnet-4-6"

[sandbox]
provider = "daytona"

# Async subagents — references to remote deployed agents
[[async_subagents]]
name = "content-writer"
description = "Writes blog posts, landing pages, and marketing copy"
graph_id = "content-writer-agent"
url = "https://my-langgraph-deployment.com"
```

### Async subagent fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name for the subagent |
| `description` | Yes | Description used for tool routing |
| `graph_id` | Yes | Remote graph/assistant ID |
| `url` | No | Remote server URL (defaults to platform URL) |
| `headers` | No | Additional auth headers (key-value pairs) |

### Subagent `deepagents.toml`

```toml
[agent]
name = "researcher"              # Required
description = "Research agent"   # Required
model = "anthropic:claude-sonnet-4-6"  # Optional, defaults to SDK default
```

Differences from parent config:
- `description` is **required** (SDK `SubAgent` needs it for tool descriptions)
- No `[sandbox]` section — shares parent's sandbox
- No `[[async_subagents]]` — no nesting
- No `subagents/` directory inside a subagent — one level only

## Inheritance Model

No inheritance. Subagents are fully explicit:
- No `model` field = SDK default (`claude-sonnet-4-6`)
- No `mcp.json` = no MCP tools
- No `skills/` = no skills
- No `user/` directory = no per-user memory (not supported for subagents)
- Sandbox always shared with parent

## Config Parsing

### New dataclasses in `config.py`

```python
@dataclass(frozen=True)
class AsyncSubAgentConfig:
    name: str
    description: str
    graph_id: str
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class SubAgentConfig:
    """Parsed from a subagent's deepagents.toml."""
    agent: AgentConfig  # name, description (required), model

@dataclass(frozen=True)
class SubAgentProject:
    """A discovered subagent directory with its parsed config."""
    config: SubAgentConfig
    root: Path
```

### Changes to existing dataclasses

`AgentConfig` gains an optional `description` field:

```python
@dataclass(frozen=True)
class AgentConfig:
    name: str
    description: str = ""
    model: str = "anthropic:claude-sonnet-4-6"
```

`DeployConfig` gains `async_subagents`:

```python
@dataclass(frozen=True)
class DeployConfig:
    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    async_subagents: list[AsyncSubAgentConfig] = field(default_factory=list)
```

### New function

`load_subagents(project_root: Path) -> dict[str, SubAgentProject]` — discovers and validates each subdirectory under `subagents/`.

### Validation rules

- Each subagent subdirectory must have `deepagents.toml` and `AGENTS.md`
- Subagent `description` is required (non-empty)
- Subagent `deepagents.toml` must not contain `[sandbox]` or `[[async_subagents]]`
- Subagent directory must not contain a `subagents/` subdirectory
- Subagent `mcp.json` follows same transport restrictions as parent (no stdio)
- Subagent names must be unique across both sync and async subagents
- `description` allowed as a key in `[agent]` section (update allowed-keys validation)

## Bundling

### Extended `_seed.json` schema

```json
{
  "memories": { "/AGENTS.md": "..." },
  "skills": { "/code-review/SKILL.md": "..." },
  "user_memories": { "/AGENTS.md": "..." },
  "subagents": {
    "researcher": {
      "config": {
        "name": "researcher",
        "description": "Research agent for deep web search",
        "model": "anthropic:claude-sonnet-4-6"
      },
      "memories": { "/AGENTS.md": "..." },
      "skills": { "/deep-search/SKILL.md": "..." },
      "mcp": { "servers": { "..." } }
    },
    "code-reviewer": {
      "config": {
        "name": "code-reviewer",
        "description": "Reviews code for quality",
        "model": "anthropic:claude-sonnet-4-6"
      },
      "memories": { "/AGENTS.md": "..." },
      "skills": {},
      "mcp": null
    }
  },
  "async_subagents": [
    {
      "name": "content-writer",
      "description": "Writes blog posts and marketing copy",
      "graph_id": "content-writer-agent",
      "url": "https://my-langgraph-deployment.com"
    }
  ]
}
```

- `subagents` key omitted entirely if no sync subagents exist
- `async_subagents` key omitted entirely if none declared
- Each subagent's `mcp` is `null` if no `mcp.json` present
- Subagent skills namespaced under their own entry, not mixed with parent

### New bundler function

`_build_subagent_seed(subagent_root: Path, config: SubAgentConfig) -> dict` — mirrors parent seed-building logic scoped to a subagent directory.

### Dependency inference

`pyproject.toml` generation considers subagent configs:
- Subagent model providers add their dependencies (e.g., different provider = additional package)
- Any subagent with MCP triggers `langchain-mcp-adapters` dependency

## Runtime Template

### Sync subagent construction

For each entry in `seed["subagents"]`, the generated `deploy_graph.py` builds a `SubAgent` dict:

```python
from deepagents.middleware.subagents import SubAgent

subagents = []
for name, data in seed.get("subagents", {}).items():
    sa: SubAgent = {
        "name": data["config"]["name"],
        "description": data["config"]["description"],
        "system_prompt": data["memories"]["/AGENTS.md"],
    }
    if data["config"].get("model"):
        sa["model"] = data["config"]["model"]
    if data.get("skills"):
        sa["skills"] = [...]  # skill paths seeded into store
    if data.get("mcp"):
        sa["tools"] = await _build_mcp_tools(data["mcp"])
    subagents.append(sa)
```

### Async subagent construction

```python
from deepagents.middleware.async_subagents import AsyncSubAgent

async_subagents = []
for entry in seed.get("async_subagents", []):
    asa: AsyncSubAgent = {
        "name": entry["name"],
        "description": entry["description"],
        "graph_id": entry["graph_id"],
    }
    if entry.get("url"):
        asa["url"] = entry["url"]
    async_subagents.append(asa)
```

### Passed to `create_deep_agent()`

```python
create_deep_agent(
    model="...",
    memory=[...],
    skills=[...],
    tools=[...],
    backend=backend_factory,
    permissions=[...],
    middleware=[...],
    subagents=subagents,
    async_subagents=async_subagents,
)
```

### Store namespacing

Subagent memories and skills seeded under `(assistant_id, "subagents", subagent_name)` namespace to avoid collisions with parent.

## Example: GTM Agent

```
examples/deploy-gtm-agent/
  deepagents.toml
  AGENTS.md
  mcp.json
  skills/
    competitor-analysis/
      SKILL.md
  subagents/
    market-researcher/
      deepagents.toml
      AGENTS.md
      skills/
        analyze-market/
          SKILL.md
```

**Parent `deepagents.toml`:**

```toml
[agent]
name = "gtm-agent"
description = "Go-to-market strategy agent that coordinates research and content creation"
model = "anthropic:claude-sonnet-4-6"

[sandbox]
provider = "none"

[[async_subagents]]
name = "content-writer"
description = "Writes blog posts, landing pages, and marketing copy"
graph_id = "content-writer-agent"
url = "https://my-langgraph-deployment.com"
```

**`subagents/market-researcher/deepagents.toml`:**

```toml
[agent]
name = "market-researcher"
description = "Researches market trends, competitors, and target audiences"
model = "anthropic:claude-sonnet-4-6"
```

This example exercises both sync and async subagents in one project.

## Testing Strategy

### Config parsing (`test_config.py`)

- Parse `deepagents.toml` with `[[async_subagents]]` entries
- Parse `description` field on parent agent config
- Validate subagent `deepagents.toml` — required `description`, rejected `[sandbox]`, rejected `[[async_subagents]]`
- Validate uniqueness of subagent names across sync + async
- Reject nested `subagents/` inside a subagent directory
- `load_subagents()` discovers and validates subdirectories

### Bundler (`test_bundler.py`)

- `_seed.json` includes `subagents` dict with correct structure
- `_seed.json` includes `async_subagents` list from parent config
- Subagent skills discovered and namespaced correctly
- Subagent MCP config included when present, `null` when absent
- `pyproject.toml` dependencies inferred from subagent model providers and MCP
- Keys omitted when no subagents exist (backwards compatible)

### Commands (`test_commands.py`)

- `deepagents deploy --dry-run` summary includes subagent info

## Scope Boundaries

**In scope:**
- Sync subagent filesystem convention and config parsing
- Async subagent declaration in parent TOML
- Bundling subagents into `_seed.json`
- Runtime template generation for both subagent types
- GTM agent example
- Unit tests for config, bundler, commands

**Out of scope (future work):**
- Recursive subagent nesting (subagents of subagents)
- Per-subagent sandbox configuration
- Per-user memory for subagents
- `deepagents init` scaffolding for subagents
- Async subagent type field for extensibility
