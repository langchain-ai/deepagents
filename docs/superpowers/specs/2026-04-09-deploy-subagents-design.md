# Subagents in `deepagents deploy`

## Summary

Add subagent support to `deepagents deploy` so that a deploy project can define nested subagents, each as a self-contained mini deploy unit with its own `AGENTS.md`, `deepagents.toml`, `skills/`, and `mcp.json`. The bundler auto-discovers them from `agents/` and wires them into the generated `deploy_graph.py` via `create_deep_agent(subagents=[...])`.

This also requires adding memory support to subagents in the SDK (`deepagents` library).

## Project Layout

```
my-agent/
    AGENTS.md               # main agent system prompt
    deepagents.toml         # main agent config
    .env                    # shared secrets
    mcp.json                # main agent MCP servers (optional)
    skills/                 # main agent skills (optional)
        review/
            SKILL.md
    agents/                 # NEW — subagent definitions
        researcher/
            AGENTS.md       # subagent system prompt (with name/description frontmatter)
            deepagents.toml # subagent config (same schema as main)
            skills/         # subagent-scoped skills (optional)
                summarize/
                    SKILL.md
            mcp.json        # subagent-scoped MCP servers (optional)
        code-reviewer/
            AGENTS.md
            deepagents.toml
```

Each subagent directory mirrors the top-level project structure exactly. A subagent is a self-contained deploy unit.

## Subagent `AGENTS.md` Format

Same as the existing CLI subagent format — YAML frontmatter with `name` and `description` (required), body becomes `system_prompt`:

```markdown
---
name: researcher
description: Research topics on the web before writing content
---

You are a research assistant with access to web search.

## Your Process
1. Search for relevant information
2. Summarize findings clearly
```

The `name` and `description` fields are **required** in subagent frontmatter. The `model` field is **not** in frontmatter — it comes from the subagent's `deepagents.toml`.

## Subagent `deepagents.toml` Format

Same schema as the main agent's toml. All fields optional except `[agent].name`:

```toml
[agent]
name = "researcher"
model = "anthropic:claude-haiku-4-5-20251001"

# Sandbox is optional — if omitted, inherits from parent
# [sandbox]
# provider = "none"
```

**Inheritance rules:**
- `model`: If omitted, inherits from the main agent's model.
- `sandbox`: If the entire `[sandbox]` section is omitted, inherits from the main agent's sandbox config. If present, fully overrides (no per-field merge).

**Validation:**
- `[agent].name` in `deepagents.toml` must match the `name` in `AGENTS.md` frontmatter. If they disagree, emit a validation error.
- The `name` must also match the directory name (`agents/{name}/`). If it doesn't, emit a warning (not a hard error — the `name` field is authoritative).

## Config Changes

### `config.py`

Add to the canonical filenames:

```python
AGENTS_DIRNAME = "agents"
```

Add a `SubagentConfig` dataclass:

```python
@dataclass(frozen=True)
class SubagentConfig:
    """A single subagent parsed from agents/{name}/."""

    agent: AgentConfig
    sandbox: SandboxConfig | None  # None means inherit from parent
    system_prompt: str
    description: str
    skills_dir: Path | None  # absolute path to subagent's skills/ if present
    mcp_path: Path | None    # absolute path to subagent's mcp.json if present
```

Extend `DeployConfig`:

```python
@dataclass(frozen=True)
class DeployConfig:
    agent: AgentConfig
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    subagents: list[SubagentConfig] = field(default_factory=list)
```

Add `_ALLOWED_SECTIONS` update — no change needed since subagent tomls use the same schema.

### Config loading

Add a `load_subagents(project_root: Path, parent_config: DeployConfig) -> list[SubagentConfig]` function:

1. Scan `project_root / "agents"` for subdirectories.
2. For each subdirectory, require `AGENTS.md` (error if missing).
3. Parse `AGENTS.md` frontmatter for `name` and `description`.
4. Load `deepagents.toml` if present; otherwise use defaults.
5. Resolve inheritance: if no `[sandbox]` section, set `sandbox = None` (signals inheritance).
6. Check for `skills/` and `mcp.json` presence.
7. Validate name consistency (toml name vs frontmatter name vs directory name).

### Validation

`DeployConfig.validate()` gains subagent validation:

- Each subagent's `AGENTS.md` must exist and have valid frontmatter.
- Each subagent's `mcp.json` (if present) must pass the same http/sse-only check.
- Model credentials are validated for each unique model across all subagents.
- Subagent names must be unique across all subagents.
- Subagent names must not be `"general-purpose"` (reserved).

## Bundler Changes

### `_build_seed()`

Extend the seed payload to include subagent data:

```json
{
    "memories": { "/AGENTS.md": "..." },
    "skills": { "/review/SKILL.md": "..." },
    "subagents": {
        "researcher": {
            "system_prompt": "...",
            "description": "...",
            "memories": { "/AGENTS.md": "..." },
            "skills": { "/summarize/SKILL.md": "..." }
        },
        "code-reviewer": {
            "system_prompt": "...",
            "description": "...",
            "memories": {},
            "skills": {}
        }
    }
}
```

Each subagent gets its own `memories` (seeded with its `AGENTS.md`) and `skills` sections. This keeps the seed self-contained.

### `bundle()`

New steps after the existing flow:

1. For each subagent, copy its `mcp.json` (if present) as `_mcp_{name}.json` into the build dir.
2. Include subagent data in `_seed.json`.
3. Pass subagent configs to the template renderer so `deploy_graph.py` can construct `SubAgent` specs.

### `_render_deploy_graph()`

The template needs to accept subagent configs and generate:
- Per-subagent MCP tool loaders (one `_load_mcp_tools_{name}()` per subagent with its own mcp.json).
- Per-subagent memory namespace seeding in `_seed_store_if_needed()`.
- A list of `SubAgent` dicts passed to `create_deep_agent(subagents=[...])`.

### `_render_pyproject()`

Infer additional model provider deps from subagent model strings. If the main agent uses Anthropic but a subagent uses OpenAI, `langchain-openai` must be added.

Similarly, if any subagent has MCP servers, include `langchain-mcp-adapters`.

## Template Changes (`deploy_graph.py`)

The generated `make_graph()` function changes from:

```python
return create_deep_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    memory=[f"{MEMORIES_PREFIX}AGENTS.md"],
    skills=[SKILLS_PREFIX],
    tools=tools,
    backend=backend_factory,
    middleware=[...],
)
```

To:

```python
# Build subagent specs
subagents = []
for sa_config in SUBAGENT_CONFIGS:
    sa_tools = []
    if sa_config.get("mcp_loader"):
        sa_tools.extend(await sa_config["mcp_loader"]())

    sa_memory_prefix = f"/memories/agents/{sa_config['name']}/"
    sa_skills_prefix = f"/skills/agents/{sa_config['name']}/"

    subagents.append({
        "name": sa_config["name"],
        "description": sa_config["description"],
        "system_prompt": sa_config["system_prompt"],
        "model": sa_config.get("model", MODEL),
        "tools": sa_tools,
        "memory": [f"{sa_memory_prefix}AGENTS.md"],
        "skills": [sa_skills_prefix],
    })

return create_deep_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    memory=[f"{MEMORIES_PREFIX}AGENTS.md"],
    skills=[SKILLS_PREFIX],
    tools=tools,
    backend=backend_factory,
    subagents=subagents,
    middleware=[...],
)
```

### Store namespace layout

Subagent memories and skills get their own namespaces under the assistant:

```
(assistant_id, "memories")                          # main agent
(assistant_id, "skills")                            # main agent
(assistant_id, "memories", "agents", "researcher")  # subagent
(assistant_id, "skills", "agents", "researcher")    # subagent
```

The `ReadOnlyStoreBackend` routes expand to cover subagent paths:

```python
routes={
    MEMORIES_PREFIX: ReadOnlyStoreBackend(...),
    SKILLS_PREFIX: ReadOnlyStoreBackend(...),
    # Subagent routes are nested under the same prefixes,
    # so they're already covered by the prefix match.
}
```

### Subagent sandbox handling

If a subagent has `sandbox: None` (inherited), it uses the parent's `_get_or_create_sandbox()`. If it has its own sandbox config, the template generates a separate `_get_or_create_sandbox_{name}()` function with that provider's block. The subagent's backend factory uses the appropriate one.

For v1, **inherited sandbox is the common case**. Subagents share the parent's sandbox and backend. Per-subagent sandbox overrides generate additional sandbox blocks in the template.

## SDK Changes: Memory for Subagents

### Current state

In `graph.py`, `MemoryMiddleware` is only added to the main agent's middleware stack (line ~393). Subagents get `TodoListMiddleware`, `FilesystemMiddleware`, `SummarizationMiddleware`, `PatchToolCallsMiddleware`, and optionally `SkillsMiddleware` — but no `MemoryMiddleware`.

### Required change

Add `memory` as an optional field on the `SubAgent` TypedDict:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: NotRequired[Sequence[...]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    skills: NotRequired[list[str]]
    memory: NotRequired[list[str]]  # NEW
```

In `create_deep_agent()`, when processing `SubAgent` specs (the `else` branch starting at line 326), add memory middleware if `spec.get("memory")` is present:

```python
subagent_memory = spec.get("memory")
if subagent_memory:
    subagent_middleware.append(MemoryMiddleware(backend=backend, sources=subagent_memory))
```

This should be appended **after** all other middleware but **before** `AnthropicPromptCachingMiddleware`, matching the main agent's ordering (memory after caching so memory updates don't invalidate the cache prefix).

## `deepagents init` Changes

Update the scaffolded project to include an example subagent:

```
my-agent/
    ...existing files...
    agents/
        researcher/
            AGENTS.md
            deepagents.toml
```

Add starter generators:

```python
def generate_starter_subagent_agents_md() -> str:
    return """\
---
name: researcher
description: Research topics on the web before writing content
---

You are a research assistant. Search for relevant information
and summarize your findings clearly and concisely.
"""

def generate_starter_subagent_config() -> str:
    return """\
[agent]
name = "researcher"
model = "anthropic:claude-haiku-4-5-20251001"
"""
```

## `print_bundle_summary()` Changes

Extend the summary to show subagents:

```
  Agent: my-agent
  Model: anthropic:claude-sonnet-4-6

  Subagents (2):
    researcher (anthropic:claude-haiku-4-5-20251001)
      skills: 1, mcp: yes
    code-reviewer (anthropic:claude-sonnet-4-6)
      skills: 0, mcp: no

  Memory seed (1 file(s)):
    /AGENTS.md
  ...
```

## Error Messages

- `agents/{name}/AGENTS.md not found` — required file missing.
- `Subagent '{name}' in agents/{dir}/ has mismatched name in deepagents.toml: '{toml_name}'` — name mismatch between toml and frontmatter.
- `Duplicate subagent name: '{name}'` — two subagent dirs define the same name.
- `Subagent name 'general-purpose' is reserved` — cannot override the default.
- `Subagent '{name}' AGENTS.md missing required frontmatter field: 'description'` — invalid frontmatter.

## Files to Change

### CLI (`libs/cli/deepagents_cli/deploy/`)

| File | Change |
|------|--------|
| `config.py` | Add `SubagentConfig`, `AGENTS_DIRNAME`, extend `DeployConfig`, add `load_subagents()`, extend validation |
| `bundler.py` | Extend `_build_seed()` for subagent data, copy subagent mcp.json files, pass subagent configs to template |
| `templates.py` | Extend `DEPLOY_GRAPH_TEMPLATE` to generate subagent specs, per-subagent MCP loaders, namespace seeding |
| `commands.py` | Update `_init()` to scaffold example subagent, update `_deploy()` to call `load_subagents()` |

### SDK (`libs/deepagents/deepagents/`)

| File | Change |
|------|--------|
| `middleware/subagents.py` | Add `memory: NotRequired[list[str]]` to `SubAgent` TypedDict |
| `graph.py` | Add `MemoryMiddleware` to subagent middleware stack when `memory` is present |

### Tests

| File | Change |
|------|--------|
| `tests/cli/deploy/test_config.py` | Test subagent config loading, inheritance, validation |
| `tests/cli/deploy/test_bundler.py` | Test seed generation with subagents, mcp copying |
| `tests/cli/deploy/test_templates.py` | Test generated graph includes subagent specs |
| `tests/unit/test_graph.py` | Test `create_deep_agent` with `memory` on subagents |

## Out of Scope

- **Async subagents in deploy**: Remote `AsyncSubAgent` (graph_id-based) is not addressed here. That's a separate feature for multi-deployment orchestration.
- **Per-subagent `.env`**: Subagents share the top-level `.env`. No per-subagent secrets.
- **Subagent-specific sandbox providers with separate credentials**: While the schema supports it, credential validation for subagent-specific sandbox providers is a follow-up.
- **Nested subagents**: A subagent cannot itself define subagents (no `agents/` within `agents/researcher/`).
