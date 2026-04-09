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
> `deepagents deploy` is currently in beta. APIs, configuration format, and behavior may change between releases. Keep an eye on the [releases page](https://github.com/langchain-ai/deepagents/releases) for detailed changelogs.

## Comparing to Claude Managed Agents

|  | Deep Agents Deploy | Claude Managed Agents |
| --- | --- | --- |
| Model Support | OpenAI, Anthropic, Google, Bedrock, Azure, Fireworks, Baseten, OpenRouter, [many more](https://docs.langchain.com/oss/python/integrations/providers/overview) | Anthropic only |
| Harness | Open source (MIT) | Proprietary, closed source |
| Sandbox | LangSmith, Daytona, Modal, Runloop, or [custom](https://docs.langchain.com/oss/python/contributing/implement-langchain#sandboxes) | Built in |
| MCP Support | ✅ | ✅ |
| Skill Support | ✅ | ✅ |
| AGENTS.md Support | ✅ | ❌ |
| Agent Endpoints | MCP, A2A, Agent Protocol | Proprietary |
| Self Hosting | ✅ | ❌ |

## What you're deploying

`deepagents deploy` takes your agent configuration and deploys it as a [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment) — a horizontally scalable server with 30+ endpoints including MCP, A2A, Agent Protocol, human-in-the-loop, and memory APIs.

You configure your agent with a few parameters:

| Parameter | Description |
| --- | --- |
| **`model`** | The LLM to use. Any provider works — see [Supported Models](#supported-models). |
| **`AGENTS.md`** | The system prompt, loaded at the start of each session. |
| **`skills`** | [Agent Skills](https://agentskills.io/) for specialized knowledge and actions. Skills are synced into the sandbox so the agent can execute them at runtime. See [Skills docs](https://docs.langchain.com/oss/python/deepagents/skills). |
| **`mcp.json`** | MCP tools (HTTPS/SSE). See [MCP docs](https://docs.langchain.com/oss/python/langchain/mcp). |
| **`sandbox`** | Optional execution environment. See [Sandboxes](#sandboxes). |

## Project layout

```txt
my-agent/
    .env             # API keys and secrets
    AGENTS.md        # required — system prompt
    skills/          # optional — agent skills
    mcp.json         # optional — HTTP/SSE MCP servers
    deepagents.toml  # agent configuration
```

### `deepagents.toml`

```toml
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

# optional, but strongly encouraged when using skills or code execution
[sandbox]
provider = "langsmith"   # langsmith | daytona | modal | runloop
scope    = "thread"      # thread | assistant
```

Skills, MCP servers, and model dependencies are auto-detected from the project layout — you don't declare them in `deepagents.toml`:

- **Skills** — the bundler recursively scans `skills/`, skipping hidden dotfiles, and bundles the rest.
- **MCP servers** — if `mcp.json` exists, it is included in the deployment and [`langchain-mcp-adapters`](https://pypi.org/project/langchain-mcp-adapters/) is added as a dependency. Only HTTP/SSE transports are supported (stdio is rejected at bundle time).
- **Model dependencies** — the `provider:` prefix in the `model` field determines the required `langchain-*` package (e.g., `anthropic` -> `langchain-anthropic`).
- **Sandbox dependencies** — the `[sandbox].provider` value maps to its partner package (e.g., `daytona` -> `langchain-daytona`).

### `.env`

Place a `.env` file alongside `deepagents.toml` with your API keys:

```bash
cp .env.example .env
```

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

## Usage

```bash
deepagents init [name] [--force]               # scaffold a new project (name prompted if omitted)
deepagents dev  [--config deepagents.toml] [--port 2024] [--allow-blocking]   # bundle and run locally
deepagents deploy [--config deepagents.toml] [--dry-run]                      # bundle and deploy
```

### `deepagents init`

Scaffolds a new agent project with the full layout:

```bash
deepagents init my-agent
```

This creates:

| File | Purpose |
| --- | --- |
| `deepagents.toml` | Agent config — name, model, optional sandbox |
| `AGENTS.md` | System prompt loaded at session start |
| `.env` | API key template (`ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`), etc. |
| `mcp.json` | MCP server configuration (empty by default) |
| `skills/` | Directory for [Agent Skills](https://agentskills.io/), with an example `review` skill |

After init, edit `AGENTS.md` with your agent's instructions and run `deepagents deploy`.

## Supported models

Any provider supported by LangChain's [`init_chat_model()`](https://docs.langchain.com/oss/python/integrations/providers/overview) works. Use the `provider:model-name` format in `deepagents.toml`:

```toml
model = "anthropic:claude-sonnet-4-6"
model = "openai:gpt-5.4"
model = "google_genai:gemini-3.1-pro-preview"
# ...and so on
```

## Sandboxes

Sandboxes provide isolated execution environments for your agent to run code and scripts. Using a sandbox is optional, but strongly recommended for any operations that use Agent Skills or for code execution.

### Providers

#### Daytona

Cloud development environments with full workspace isolation. [Learn more.](https://www.daytona.io/)

```toml
[sandbox]
provider = "daytona"
```

```bash
# .env
DAYTONA_API_KEY=...
```

#### Modal

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

#### Runloop

Isolated DevBox environments for agent execution. [Learn more.](https://runloop.ai/)

```toml
[sandbox]
provider = "runloop"
```

```bash
# .env
RUNLOOP_API_KEY=...
```

#### LangSmith Sandbox (Private preview)

[(Documentation)](https://docs.langchain.com/langsmith/sandboxes)

No additional setup beyond your LangSmith API key.

Add the following to your `deepagents.toml`:

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

#### Custom sandbox

You can implement your own sandbox provider. See the [custom sandbox guide](https://docs.langchain.com/oss/python/contributing/implement-langchain#sandboxes).

### Sandbox scope

By default, each thread gets its own sandbox (`scope = "thread"`). Set `scope = "assistant"` to share one sandbox across all threads for the same assistant.

```toml
[sandbox]
provider = "langsmith"
scope = "assistant"   # shared across threads
```

## Deployment endpoints

The deployed server exposes:

- [**MCP**](https://modelcontextprotocol.io/docs/getting-started/intro) — call your agent as a tool from other agents
- [**A2A**](https://a2a-protocol.org/latest/) — multi-agent orchestration via [A2A protocol](https://a2a-protocol.org/latest/)
- **[Agent Protocol](https://github.com/langchain-ai/agent-protocol)** — standard API for building UIs
- **[Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)** — approval gates for sensitive actions
- **[Memory](https://docs.langchain.com/oss/python/deepagents/memory)** — short-term and long-term memory access

## Gotchas

- **Read-only at runtime:** `/memories/` and `/skills/` are synced into the sandbox but cannot be edited at runtime. Edit source files and redeploy.
- **Full rebuild on deploy:** `deepagents deploy` creates a new revision on every invocation. Use `deepagents dev` for local iteration.
- **Sandbox lifecycle:** Thread-scoped sandboxes are provisioned per thread and will be re-created if the server restarts. Use `scope = "assistant"` if you need sandbox state that persists across threads.
- **MCP: HTTP/SSE only.** Stdio transports are rejected at bundle time.
- **No custom Python tools.** Use MCP servers to expose custom tool logic.
