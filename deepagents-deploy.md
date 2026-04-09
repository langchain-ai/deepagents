---
name: deepagents-deploy
description: Deploy a model-agnostic, open source agent harness to production with a single command.
---

# Deep Agents Deploy (Beta)

> **Note:** `deepagents deploy` is currently in beta. APIs, configuration format, and behavior may change between releases.

Deploy a model-agnostic, open source agent to production with a single command.

Deep Agents Deploy is built on [Deep Agents](https://github.com/langchain-ai/deepagents) — an open source, model-agnostic agent harness. It handles orchestration, sandboxing, and endpoint setup so you can go from a local agent to a deployed service without managing infrastructure.

## What you're deploying

`deepagents deploy` takes your agent configuration and deploys it as a [LangSmith Deployment](https://docs.langchain.com/langsmith/deployment) — a horizontally scalable server with 30+ endpoints including MCP, A2A, Agent Protocol, human-in-the-loop, and memory APIs.

You configure your agent with a few parameters:

| Parameter | Description |
| --- | --- |
| **`model`** | The LLM to use. Any provider works — see [Supported Models](#supported-models). |
| **`AGENTS.md`** | The system prompt, loaded at the start of each session. |
| **`skills`** | [Agent Skills](https://agentskills.io/) for specialized knowledge and actions. Skills are synced into the sandbox so the agent can execute them at runtime. See [Skills docs](https://docs.langchain.com/oss/python/deepagents/skills). |
| **`mcp.json`** | MCP tools (HTTPS/SSE). |
| **`sandbox`** | Optional execution environment. See [Sandboxes](#sandboxes). |
| **`agents/`** | Optional [subagents](#subagents) — specialized agents the main agent can delegate to. |

## Project layout

```
my-agent/
  .env             # API keys and secrets
  AGENTS.md        # required — system prompt
  skills/          # optional — agent skills
  mcp.json         # optional — HTTP/SSE MCP servers
  deepagents.toml  # agent configuration
  agents/          # optional — subagent definitions
    researcher/
      AGENTS.md
      deepagents.toml
      skills/
      mcp.json
```

### `deepagents.toml`

```toml
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

# [sandbox] is optional — omit if not needed for skills or code execution.
[sandbox]
provider = "langsmith"   # langsmith | daytona | modal | runloop
scope    = "thread"      # thread | assistant
```

Skills, MCP servers, and model dependencies are auto-detected.

### `.env`

Place a `.env` file alongside `deepagents.toml` with your API keys:

```bash
cp .env.example .env
```

```bash
# Required — your model provider key
ANTHROPIC_API_KEY=sk-...

# Required for deploy and LangSmith sandbox
LANGSMITH_API_KEY=lsv2_...

# Optional — sandbox provider keys (only needed if using that provider)
DAYTONA_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
RUNLOOP_API_KEY=...
```

## CLI

```bash
deepagents init my-agent                # scaffold my-agent/ with full project layout
deepagents dev    [--config deepagents.toml] [--port 2024]
deepagents deploy [--config deepagents.toml] [--dry-run]
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
| `.env` | API key template (`ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`) |
| `mcp.json` | MCP server configuration (empty by default) |
| `skills/` | Directory for [Agent Skills](https://agentskills.io/) |
| `agents/` | Directory with a starter [subagent](#subagents) |

After init, edit `AGENTS.md` with your agent's instructions and run `deepagents deploy`.

## Supported models

Deep Agents works with any model provider. Use the `provider:model-name` format in `deepagents.toml`.

### Anthropic

```toml
model = "anthropic:claude-opus-4-6"
model = "anthropic:claude-sonnet-4-6"
model = "anthropic:claude-haiku-4-5-20251001"
```

### OpenAI

```toml
model = "openai:gpt-5.4"
model = "openai:gpt-5.4-mini"
model = "openai:o3"
model = "openai:o4-mini"
model = "openai:gpt-4.1"
model = "openai:gpt-4o"
```

### Google

```toml
model = "google_genai:gemini-3.1-pro-preview"
model = "google_genai:gemini-3-flash-preview"
model = "google_genai:gemini-2.5-pro"
model = "google_genai:gemini-2.5-flash"
```

### Azure OpenAI

```toml
model = "azure_openai:my-gpt4-deployment"
```

### Amazon Bedrock

```toml
model = "google_vertexai:claude-sonnet-4-6"
```

### xAI

```toml
model = "xai:grok-4"
model = "xai:grok-3-mini-fast"
```

### Fireworks

```toml
model = "fireworks:fireworks/deepseek-v3p2"
model = "fireworks:fireworks/qwen3-vl-235b-a22b-thinking"
model = "fireworks:fireworks/minimax-m2p5"
model = "fireworks:fireworks/kimi-k2p5"
model = "fireworks:fireworks/glm-5"
```

### Baseten

```toml
model = "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct"
model = "baseten:MiniMaxAI/MiniMax-M2.5"
model = "baseten:moonshotai/Kimi-K2.5"
model = "baseten:nvidia/Nemotron-120B-A12B"
```

### Groq

```toml
model = "groq:qwen/qwen3-32b"
model = "groq:moonshotai/kimi-k2-instruct"
```

### NVIDIA

```toml
model = "nvidia:nvidia/nemotron-3-super-120b-a12b"
```

### OpenRouter

```toml
model = "openrouter:minimax/minimax-m2.7"
model = "openrouter:nvidia/nemotron-3-super-120b-a12b"
```

### Ollama (local models)

```toml
model = "ollama:deepseek-v3.2:cloud"
model = "ollama:qwen3-coder:480b-cloud"
model = "ollama:nemotron-3-super"
model = "ollama:glm-5"
```

### Additional providers

Deep Agents also supports **Cohere**, **DeepSeek**, **Hugging Face**, **IBM**, **LiteLLM**, **Mistral AI**, **Perplexity**, and **Together AI**. Any provider supported by LangChain's `init_chat_model()` works out of the box.

| Provider | Environment Variable |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| xAI | `XAI_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Baseten | `BASETEN_API_KEY` |
| Groq | `GROQ_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral AI | `MISTRAL_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |
| Perplexity | `PERPLEXITYAI_API_KEY` |

## Sandboxes

Sandboxes provide isolated execution environments for your agent to run code and scripts.

### LangSmith (default)

No additional setup beyond your LangSmith API key.

```toml
[sandbox]
provider = "langsmith"
template = "deepagents-cli"
image = "python:3.12"
```

```bash
# .env
LANGSMITH_API_KEY=lsv2_...
```

Install: `pip install 'deepagents-cli[langsmith]'`

### Daytona

Cloud development environments with full workspace isolation.

```toml
[sandbox]
provider = "daytona"
```

```bash
# .env
DAYTONA_API_KEY=...
```

Install: `pip install 'deepagents-cli[daytona]'`

### Modal

Serverless compute — sandboxes spin up on demand.

```toml
[sandbox]
provider = "modal"
```

```bash
# .env (optional — can also use default Modal auth)
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

Install: `pip install 'deepagents-cli[modal]'`

### Runloop

Isolated DevBox environments for agent execution.

```toml
[sandbox]
provider = "runloop"
```

```bash
# .env
RUNLOOP_API_KEY=...
```

Install: `pip install 'deepagents-cli[runloop]'`

### Sandbox scope

By default, each thread gets its own sandbox (`scope = "thread"`). Set `scope = "assistant"` to share one sandbox across all threads for the same assistant.

```toml
[sandbox]
provider = "langsmith"
scope = "assistant"   # shared across threads
```

### No sandbox

Omit the `[sandbox]` section entirely if not needed for skills or code execution.

## Deployment endpoints

The deployed server exposes:

- **MCP** — call your agent as a tool from other agents
- **A2A** — multi-agent orchestration via [A2A protocol](https://a2a-protocol.org/latest/)
- **[Agent Protocol](https://github.com/langchain-ai/agent-protocol)** — standard API for building UIs
- **[Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)** — approval gates for sensitive actions
- **[Memory](https://docs.langchain.com/oss/python/deepagents/memory)** — short-term and long-term memory access

## Open ecosystem

- **Open source harness** — MIT licensed, available for [Python](https://github.com/langchain-ai/deepagents) and [TypeScript](https://github.com/langchain-ai/deepagentsjs)
- **[AGENTS.md](https://agents.md/)** — open standard for agent instructions
- **[Agent Skills](https://agentskills.io/)** — open standard for agent knowledge and actions
- **Any model, any sandbox** — no provider lock-in
- **Open protocols** — MCP, A2A, Agent Protocol
- **Self-hostable** — LangSmith Deployments can be self-hosted so memory stays in your infrastructure

## Comparing to Claude Managed Agents

|  | Deep Agents Deploy | Claude Managed Agents |
| --- | --- | --- |
| Model Support | OpenAI, Anthropic, Google, Bedrock, Azure, Fireworks, Baseten, OpenRouter, many more | Anthropic only |
| Harness | Open source (MIT) | Proprietary, closed source |
| Sandbox | LangSmith, Daytona, Modal, Runloop, or custom | Built in |
| MCP Support | Yes | Yes |
| Skill Support | Yes | Yes |
| AGENTS.md Support | Yes | No |
| Agent Endpoints | MCP, A2A, Agent Protocol | Proprietary |
| Self Hosting | Yes | No |

## Subagents

Subagents are specialized agents that your main agent can delegate to via the `task()` tool. Each subagent is a self-contained unit under `agents/` with its own system prompt, model, skills, and MCP servers.

### Defining subagents

Create a directory under `agents/` for each subagent. Each directory mirrors the top-level project structure:

```
agents/
  researcher/
    AGENTS.md           # required — system prompt with frontmatter
    deepagents.toml     # optional — model and sandbox config
    skills/             # optional — subagent-scoped skills
    mcp.json            # optional — subagent-scoped MCP servers
```

### Subagent `AGENTS.md`

Subagent `AGENTS.md` files require YAML frontmatter with `name` and `description`:

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

The main agent sees the `name` and `description` and uses them to decide when to delegate.

### Subagent `deepagents.toml`

Same schema as the main agent's config. If omitted, the subagent inherits the main agent's model. If the `[sandbox]` section is omitted, the subagent inherits the main agent's sandbox.

```toml
[agent]
name = "researcher"
model = "anthropic:claude-haiku-4-5-20251001"   # use a cheaper model for research

[sandbox]
provider = "langsmith"
scope = "thread"
```

### Example: deep research agent

A research orchestrator with two subagents — a fast researcher and a thorough synthesizer:

```
my-research-agent/
  AGENTS.md                         # orchestrator — decomposes questions, coordinates
  deepagents.toml                   # model = "anthropic:claude-sonnet-4-6"
  agents/
    researcher/
      AGENTS.md                     # web search, source evaluation, writes notes
      deepagents.toml               # model = "anthropic:claude-haiku-4-5-20251001"
      mcp.json                      # Tavily web search
      skills/
        search-strategy/SKILL.md
    synthesizer/
      AGENTS.md                     # reads research, writes structured report
      deepagents.toml               # model = "anthropic:claude-sonnet-4-6"
      skills/
        report-format/SKILL.md
```

The orchestrator delegates research tasks to the `researcher` (Haiku — fast and cheap for high-volume searches), then hands all findings to the `synthesizer` (Sonnet — stronger model for quality writing). Both subagents share the same sandbox so files flow naturally between them.

See [`examples/deploy-deep-research/`](examples/deploy-deep-research/) for the full working example.

### How it works

- The bundler auto-discovers `agents/` and includes subagent data in the deployment
- Each subagent's `AGENTS.md` is seeded as memory so the subagent loads its own instructions
- Subagent skills are scoped — each subagent only sees its own `skills/` directory
- Subagent MCP servers are scoped — each subagent loads only its own `mcp.json`
- Model provider dependencies are inferred from all subagent models (e.g., if a subagent uses OpenAI, `langchain-openai` is added automatically)

### Constraints

- Subagent names must be unique and cannot be `general-purpose` (reserved)
- The `name` in `AGENTS.md` frontmatter must match the `name` in `deepagents.toml`
- Subagents cannot define their own subagents (no nesting)
- MCP servers must use HTTP/SSE transport (no stdio)

## Gotchas

- **Read-only at runtime:** `/memories/` and `/skills/` are synced into the sandbox but cannot be edited at runtime. Edit source files and redeploy.
- **Full rebuild on deploy:** `deepagents deploy` creates a new revision on every invocation. Use `deepagents dev` for local iteration.
- **Sandbox lifecycle:** Thread-scoped sandboxes are provisioned per thread and will be re-created if the server restarts. Use `scope = "assistant"` if you need sandbox state that persists across threads.
- **MCP: HTTP/SSE only.** Stdio transports are rejected at bundle time.
- **No custom Python tools.** Use MCP servers to expose custom tool logic.
