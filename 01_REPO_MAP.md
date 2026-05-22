# 01_REPO_MAP.md — Deep Agents Codebase Map

## Package Layout

```
deepagents/                          # Root / monorepo orchestrator
├── Makefile                         # Root-level: delegates to libs/*/Makefile
├── AGENTS.md                        # Developer guidelines (conventions, principles)
├── README.md
└── libs/
    ├── deepagents/                  # Core SDK (primary package)
    ├── cli/                         # deepagents CLI tool
    ├── acp/                         # Agent Context Protocol support
    ├── code/                        # DeepAgents Code (terminal coding agent, "dcode")
    ├── evals/                       # Evaluation suite + Harbor integration
    └── partners/                    # Third-party integrations
        └── daytona/                 # Daytona sandbox integration

```

## Core SDK — `libs/deepagents/deepagents/`

### Public API (`__init__.py`)
- `create_deep_agent()` — primary entry point
- `DeepAgent`, `HarnessProfiles`, model config helpers

### Internal Structure

| File/Dir | Purpose |
|---|---|
| `graph.py` | LangGraph state graph construction |
| `_models.py` | Pydantic models (state, config, responses) |
| `_messages_reducer.py` | Message list reduction logic |
| `_tools.py` | Built-in tool definitions |
| `_api/` | Public API module |
| `_api/__init__.py` | API exports |
| `_api/deprecation.py` | Deprecation utilities |
| `_subagent_transformer.py` | Sub-agent graph transformation |
| `_excluded_middleware.py` | Middleware exclusion list |
| `middleware/` | Agent middleware pipeline |
| `backends/` | Pluggable backend implementations |
| `profiles/` | Model/provider profiles |

### Middleware (`middleware/`)

| File | Purpose |
|---|---|
| `__init__.py` | Middleware exports |
| `_message_eviction.py` | Evict old messages to stay within context |
| `_overflow_clip.py` | Clip overflowing content (70% coverage — area for improvement) |
| `_tool_exclusion.py` | Exclude tools from sub-agent context |
| `_utils.py` | Shared middleware utilities |
| `async_subagents.py` | Async sub-agent spawning and management |
| `filesystem.py` | Filesystem tool middleware (read/write/edit/search) |
| `memory.py` | Persistent memory across sessions |
| `patch_tool_calls.py` | Patch/transform tool calls |
| `permissions.py` | Permission checking for HITL |
| `skills.py` | Skill loading and invocation |
| `subagents.py` | Sub-agent delegation logic |
| `summarization.py` | Context summarization for long threads |

### Backends (`backends/`)

| File | Purpose |
|---|---|
| `protocol.py` | Backend protocol/interface definitions |
| `composite.py` | Composite backend (multiple backends combined) |
| `context_hub.py` | Context hub for tool output offloading |
| `filesystem.py` | Local/sandboxed/remote filesystem backend |
| `langsmith.py` | LangSmith-backed storage |
| `local_shell.py` | Local shell command execution |
| `sandbox.py` | Sandboxed execution backend |
| `state.py` | Agent state backend |
| `store.py` | Key-value store backend |
| `utils.py` | Backend utilities |

### Profiles (`profiles/`)

| File | Purpose |
|---|---|
| `__init__.py` | Profile exports |
| `_builtin_profiles.py` | Built-in model profiles |
| `_keys.py` | Profile key definitions |
| `harness/` | Harness profiles (model-specific defaults) |
| `harness/__init__.py`, `harness_profiles.py` | Harness profile registry |
| `harness/_anthropic_haiku_4_5.py` | Anthropic Haiku profile |
| `harness/_anthropic_opus_4_7.py` | Anthropic Opus profile |
| `harness/_anthropic_sonnet_4_6.py` | Anthropic Sonnet profile |
| `harness/_openai_codex.py` | OpenAI Codex profile |
| `provider/` | Provider-specific configurations |
| `provider/__init__.py` | Provider profile exports |
| `provider/_openai.py` | OpenAI provider config |
| `provider/_openrouter.py` | OpenRouter provider config |
| `provider/provider_profiles.py` | Provider profile registry |

## Tests

```
libs/deepagents/tests/
├── unit_tests/         # No network, pytest --disable-socket
├── integration_tests/  # Network-permitted, --timeout 30
├── benchmarks/         # pytest-benchmark + pytest-codspeed
├── smoke_tests/        # Snapshot-based smoke tests
├── utils.py
└── README.md
```

## CI/CD Workflows (`.github/workflows/`)

| Workflow | Purpose |
|---|---|
| `_lint.yml` | Ruff lint + format check |
| `_test.yml` | Unit tests (all packages) |
| `integration_tests.yml` | Integration tests |
| `evals.yml` / `evals_trials.yml` | Evaluation runs |
| `harbor.yml` | Harbor sandbox evals |
| `release.yml` / `release-please.yml` | Release automation |
| `auto-label-by-package.yml` | Auto-label PRs by changed package |
| `pr_labeler.yml` / `pr_lint.yml` | PR labeling and title linting |
| `check_extras_sync.yml` | Dependency sync checks |
| `check_lockfiles.yml` | Lockfile validation |
| `check_versions.yml` | Version consistency checks |
| `ci.yml` | Main CI orchestration |

## Examples (`examples/`)

- `nvidia_deep_agent/` — NVIDIA-hosted model agent
- `deep_research/` — Deep research agent
- `repl_swarm/` — REPL swarm example
- `llm-wiki/` — LLM-powered wiki
- `async-subagent-server/` — Async sub-agent server
- `text-to-sql-agent/` — Text-to-SQL agent
- `rlm_agent/` — RLM agent
- `better-harness/` — Custom harness example
- `content-builder-agent/` — Content building agent

## Key Technologies

- **Runtime**: Python 3.11–3.14, uv package manager
- **Agent framework**: LangGraph, LangChain
- **Models**: OpenAI, Anthropic, Google GenAI, any tool-calling LLM
- **Storage**: LangSmith, pluggable backends (local, sandbox, remote)
- **Sandbox**: Daytona, local shell
- **Testing**: pytest, pytest-asyncio, pytest-benchmark, pytest-codspeed
- **Quality**: ruff (lint/format), ty (type-check)