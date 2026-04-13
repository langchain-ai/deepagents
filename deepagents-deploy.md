---
name: deepagents-deploy
description: Deploy a model-agnostic, open source agent harness to production with a single command.
---

# Deep Agents Deploy (Beta)

Deploy a model-agnostic, open source agent to production with a single command.

Deep Agents Deploy is built on [Deep Agents](https://github.com/langchain-ai/deepagents) — an open source, model-agnostic agent harness. It handles orchestration, sandboxing, and endpoint setup so you can go from a local agent to a deployed service without managing infrastructure. Built on open standards:

- **Open source harness** — MIT licensed, available for [Python](https://github.com/langchain-ai/deepagents) and [TypeScript](https://github.com/langchain-ai/deepagentsjs)
- **[AGENTS.md](https://agents.md/)** — open standard for agent instructions
- **[Agent Skills](https://agentskills.io/)** — open standard for agent knowledge and actions
- **Any model, any sandbox** — no provider lock-in
- **Open protocols** — [MCP](https://modelcontextprotocol.io/docs/getting-started/intro), [A2A](https://a2a-protocol.org/latest/), [Agent Protocol](https://github.com/langchain-ai/agent-protocol)
- **Self-hostable** — LangSmith Deployments can be self-hosted so memory stays in your infrastructure

> [!WARNING] Warning: Beta
> `deepagents deploy` is currently in beta. APIs, configuration format, and behavior may change between releases. See the [releases page](https://github.com/langchain-ai/deepagents/releases) for detailed changelogs.

## Compare to Claude Managed Agents

|  | Deep Agents Deploy | Claude Managed Agents |
| --- | --- | --- |
| Model support | OpenAI, Anthropic, Google, Bedrock, Azure, Fireworks, Baseten, OpenRouter, [many more](https://docs.langchain.com/oss/integrations/providers/overview) | Anthropic only |
| Harness | Open source (MIT) | Proprietary, closed source |
| Sandbox | LangSmith, Daytona, Modal, Runloop, or [custom](https://docs.langchain.com/oss/contributing/implement-langchain#sandboxes) | Built in |
| MCP support | ✅ | ✅ |
| Skill support | ✅ | ✅ |
| AGENTS.md support | ✅ | ❌ |
| Agent endpoints | MCP, A2A, Agent Protocol | Proprietary |
| Self hosting | ✅ | ❌ |

## What you're deploying

`deepagents deploy` packages your agent configuration and deploys it as a [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment). You configure your agent with a few parameters:

| Parameter | Description |
| --- | --- |
| **`model`** | The LLM to use. Any provider works — see [Supported Models](#supported-models). |
| **`AGENTS.md`** | The system prompt, loaded at the start of each session. |
| **`skills`** | [Agent Skills](https://agentskills.io/) for specialized knowledge and actions. Skills are synced into the sandbox so the agent can execute them at runtime. See [Skills docs](https://docs.langchain.com/oss/python/deepagents/skills). |
| **`user/`** | Per-user memory templates. Files are seeded once per user (never overwritten) and writable at runtime. Preloaded into the agent's context via the memory middleware. |
| **`mcp.json`** | MCP tools (HTTP/SSE). See [MCP docs](https://docs.langchain.com/oss/python/langchain/mcp). |
| **`sandbox`** | Optional execution environment. See [Sandbox providers](#sandbox-providers). |

## Install

Install the CLI or run directly with `uvx`:

```bash
# Install globally
uv tool install deepagents-cli

# Or run without installing
uvx deepagents-cli deploy
```

## Usage

```bash
deepagents init [name] [--force]                                             # scaffold a new project
deepagents dev  [--config deepagents.toml] [--port 2024] [--allow-blocking]  # bundle and run locally
deepagents deploy [--config deepagents.toml] [--dry-run]                     # bundle and deploy
```

By default, `deepagents deploy` looks for `deepagents.toml` in the current directory. Pass `--config` to use a different path:

```bash
deepagents deploy --config path/to/deepagents.toml
```

### `deepagents init`

Scaffold a new agent project:

```bash
deepagents init my-agent
```

This creates the following files:

| File | Purpose |
| --- | --- |
| `deepagents.toml` | Agent config — name, model, optional sandbox |
| `AGENTS.md` | System prompt loaded at session start |
| `.env` | API key template (`ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, etc.) |
| `mcp.json` | MCP server configuration (empty by default) |
| `skills/` | Directory for [Agent Skills](https://agentskills.io/), with an example `review` skill |

After init, edit `AGENTS.md` with your agent's instructions and run `deepagents deploy`. Optionally add a `user/` directory with per-user memory templates — see [User Memory](#user-memory).

## Project layout

The deploy command uses a convention-based project layout. Place the following files alongside your `deepagents.toml` and they are automatically discovered:

```txt
my-agent/
├── deepagents.toml
├── AGENTS.md
├── .env
├── mcp.json
├── skills/
│   ├── code-review/
│   │   └── SKILL.md
│   └── data-analysis/
│       └── SKILL.md
└── user/
    └── preferences.md
```

| File/directory | Purpose | Required |
| --- | --- | --- |
| `AGENTS.md` | [Memory](https://docs.langchain.com/oss/python/deepagents/memory) for the agent. Provides persistent context (project conventions, instructions, preferences) that is always loaded at startup. Read-only at runtime. | Yes |
| `skills/` | Directory of [skill](https://docs.langchain.com/oss/python/deepagents/skills) definitions. Each subdirectory should contain a `SKILL.md` file. Read-only at runtime. | No |
| `user/` | Per-user memory templates. Files are seeded once per user on first access and never overwritten, so user edits persist. Writable at runtime — the agent can update these files. Preloaded into the agent's context at the start of each session. | No |
| `mcp.json` | [MCP](https://modelcontextprotocol.io/) server configuration. Only `http` and `sse` transports are supported in deployed contexts. | No |
| `.env` | Environment variables (API keys, secrets). Placed alongside `deepagents.toml` at the project root. | No |

> [!WARNING]
> `mcp.json` must only contain servers using `http` or `sse` transports. Servers using `stdio` transport are not supported in deployed environments because there is no local process to spawn.
>
> Convert stdio servers to HTTP or SSE before deploying.

## Configuration file

`deepagents.toml` configures the agent's identity and sandbox environment. Only the `[agent]` section is required. The `[sandbox]` section is optional and defaults to no sandbox.

### `[agent]`

(Required)

Core agent identity. For more on model selection and provider configuration, see [Supported Models](#supported-models).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | *(required)* | Name for the deployed agent. Used as the assistant identifier in LangSmith. |
| `model` | string | `anthropic:claude-sonnet-4-6` | Model identifier in `provider:model` format. See [Supported Models](#supported-models). |

```toml
[agent]
name = "research-assistant"
model = "anthropic:claude-sonnet-4-6"
```

> [!NOTE]
> The `name` field is the only required value in the entire configuration file. Everything else has defaults.

Skills, user memories, MCP servers, and model dependencies are auto-detected from the project layout — you don't declare them in `deepagents.toml`:

- **Skills** — the bundler recursively scans `skills/`, skipping hidden dotfiles, and bundles the rest.
- **User memories** — if `user/` exists, files are bundled as per-user memory templates. At runtime, each user gets their own copy (seeded on first access, never overwritten). The agent can read and write these files.
- **MCP servers** — if `mcp.json` exists, it is included in the deployment and [`langchain-mcp-adapters`](https://pypi.org/project/langchain-mcp-adapters/) is added as a dependency. Only HTTP/SSE transports are supported (stdio is rejected at bundle time).
- **Model dependencies** — the `provider:` prefix in the `model` field determines the required `langchain-*` package (e.g., `anthropic` -> `langchain-anthropic`).
- **Sandbox dependencies** — the `[sandbox].provider` value maps to its partner package (e.g., `daytona` -> `langchain-daytona`).

### `[sandbox]`

(Optional)

Configure the isolated execution environment where the agent runs code. Sandboxes provide a container with a filesystem and shell access, so untrusted code cannot affect the host.

When omitted or set to `provider = "none"`, the sandbox is disabled. Sandboxes are for if you need code execution or skill script execution.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | string | `none` | Sandbox provider. Supported values: `"none"`, `"daytona"`, `"modal"`, `"runloop"`, `"langsmith"` (private beta). See [Sandbox providers](#sandbox-providers). |
| `template` | string | `deepagents-deploy` | Provider-specific template name for the sandbox environment. |
| `image` | string | `python:3` | Base Docker image for the sandbox container. |
| `scope` | string | `thread` | Sandbox lifecycle scope. `"thread"` creates one sandbox per conversation. `"assistant"` shares a single sandbox across all conversations for the same assistant. |

**Scope behavior:**

- `"thread"` (default): Each conversation gets its own sandbox. Different threads get different sandboxes, but the same thread reuses its sandbox across turns. Use this when each conversation should start with a clean environment.
- `"assistant"`: All conversations share one sandbox. Files, installed packages, and other state persist across conversations. Use this when the agent maintains a long-lived workspace like a cloned repo.

### `.env`

Place a `.env` file alongside `deepagents.toml` with your API keys:

```bash
# Required — model provider keys
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
# ...etc.

# Required for deploy and LangSmith sandbox
LANGSMITH_API_KEY=lsv2_...

# Optional — sandbox provider keys
DAYTONA_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
RUNLOOP_API_KEY=...
```

## Supported models

Any provider supported by LangChain's [`init_chat_model()`](https://docs.langchain.com/oss/python/integrations/providers/overview) works. Use the `provider:model-name` format in `deepagents.toml`:

```toml
model = "anthropic:claude-sonnet-4-6"
model = "openai:gpt-5.4"
model = "google_genai:gemini-3.1-pro-preview"
# ...and so on
```

## Sandbox providers

Each sandbox provider requires specific configuration in `deepagents.toml` and environment variables in `.env`.

### Daytona

Cloud development environments with full workspace isolation. [Learn more.](https://www.daytona.io/)

```toml
[sandbox]
provider = "daytona"
```

```bash
# .env
DAYTONA_API_KEY=...
```

### Modal

Serverless compute — sandboxes spin up on demand. [Learn more.](https://modal.com/docs/guide/sandboxes)

```toml
[sandbox]
provider = "modal"
```

```bash
# .env (optional — can also use default Modal auth)
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### Runloop

Isolated DevBox environments for agent execution. [Learn more.](https://runloop.ai/)

```toml
[sandbox]
provider = "runloop"
```

```bash
# .env
RUNLOOP_API_KEY=...
```

### LangSmith Sandbox

> [!NOTE]
> LangSmith Sandbox is currently in private beta. [Sign up for the waitlist.](https://docs.langchain.com/langsmith/sandboxes#sandboxes-overview)

No additional setup beyond your LangSmith API key.

```toml
[sandbox]
provider = "langsmith"
template = "deepagents-deploy"
image = "python:3"
```

```bash
# .env
LANGSMITH_API_KEY=lsv2_...
```

### Custom sandbox

Implement your own sandbox provider. See the [custom sandbox guide](https://docs.langchain.com/oss/contributing/implement-langchain#sandboxes).

## Deployment endpoints

The deployed server exposes:

- [**MCP**](https://modelcontextprotocol.io/docs/getting-started/intro) — call your agent as a tool from other agents
- [**A2A**](https://a2a-protocol.org/latest/) — multi-agent orchestration via A2A protocol
- [**Agent Protocol**](https://github.com/langchain-ai/agent-protocol) — standard API for building UIs
- [**Human-in-the-loop**](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) — approval gates for sensitive actions
- [**Memory**](https://docs.langchain.com/oss/python/deepagents/memory) — short-term and long-term memory access

## Examples

A content writing agent with per-user preferences that the agent can update:

```toml
[agent]
name = "deepagents-deploy-content-writer"
model = "openai:gpt-4.1"
```

```txt
my-content-writer/
├── deepagents.toml
├── AGENTS.md
├── skills/
│   ├── blog-post/SKILL.md
│   └── social-media/SKILL.md
└── user/
    └── preferences.md      # writable — agent learns user preferences
```

A coding agent with a LangSmith sandbox for running code:

```toml
[agent]
name = "deepagents-deploy-coding-agent"
model = "anthropic:claude-sonnet-4-5"

[sandbox]
provider = "langsmith"
template = "coding-agent"
image = "python:3.12"
```

## User Memory

User memory gives each user their own writable files that persist across conversations. Place template files in the `user/` directory at your project root:

```txt
user/
└── preferences.md
```

At runtime, these files are mounted at `/memories/user/` and scoped per user via the `user_id` in the invocation config. The first time a user interacts with the agent, their namespace is seeded with a copy of each template file. Subsequent interactions reuse the existing files — the agent's edits persist, and redeployments never overwrite user data.

### How it works

1. **Bundle time** — the bundler scans `user/` and includes the files in the seed payload.
2. **Runtime (first access)** — when a user_id is seen for the first time, the seed templates are written to the store under that user's namespace. Existing entries are never overwritten.
3. **Preloaded** — user memory files are passed to the memory middleware, so the agent sees their contents in context at the start of every conversation.
4. **Writable** — the agent can update user memory files via `edit_file`. AGENTS.md and skills are read-only.

### Permissions

| Path | Writable | Scope |
| --- | --- | --- |
| `/memories/AGENTS.md` | No | Shared (assistant-scoped) |
| `/memories/skills/**` | No | Shared (assistant-scoped) |
| `/memories/user/**` | Yes | Per-user (user_id-scoped) |

### Passing user_id

Include `user_id` in the invocation's `configurable` dict:

```python
result = await client.runs.create(
    thread_id=thread_id,
    assistant_id="my-agent",
    input={"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"user_id": "user-123"}},
)
```

If `user_id` is omitted, it defaults to `"default"` — all users share the same memory namespace.

## Gotchas

- **AGENTS.md and skills are read-only at runtime.** Edit source files and redeploy to update them. User memory files (`/memories/user/`) are the exception — they are writable by the agent.
- **Full rebuild on deploy:** `deepagents deploy` creates a new revision on every invocation. Use `deepagents dev` for local iteration.
- **Sandbox lifecycle:** Thread-scoped sandboxes are provisioned per thread and will be re-created if the server restarts. Use `scope = "assistant"` if you need sandbox state that persists across threads.
- **MCP: HTTP/SSE only.** Stdio transports are rejected at bundle time.
- **No custom Python tools.** Use MCP servers to expose custom tool logic.
