# 🧠🤖 Deep Agents

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents?label=%20)](https://pypi.org/project/deepagents/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents)](https://pypistats.org/packages/deepagents)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [Deep Agents.js](https://github.com/langchain-ai/deepagentsjs).

To help you ship LangChain apps to production faster, check out [LangSmith](https://smith.langchain.com).
LangSmith is a unified developer platform for building, testing, and monitoring LLM applications.

## Quick Install

```bash
uv add deepagents
```

## 🤔 What is this?

Deep Agents is an open source agent harness — an opinionated agent that runs out of the box. Extend, override, or replace any piece.

**Principles:**

- **Opinionated** — defaults tuned for long-horizon, multi-step work
- **Extensible** — override or replace any piece without forking
- **Model-agnostic** — works with any LLM that supports tool calling: frontier, open-weight, or local
- **Production-ready** — built on LangGraph (streaming, persistence, checkpointing) with first-class tracing, evaluation, and deployment via LangSmith

**Features include:**

- **Sub-agents** — delegate tasks to agents with isolated context windows
- **Workflow mode** — opt-in `workflow` tool that runs a declarative multi-agent plan (parallel fan-out, then fan-in) in a single call
- **Filesystem** — read, write, edit, or search over pluggable local, sandboxed, or remote backends
- **Context management** — summarize long threads and offload tool outputs to disk
- **Shell access** — run commands in your sandbox of choice
- **Persistent memory** — pluggable state and store backends for cross-session recall
- **Human-in-the-loop** — approve, edit, or reject tool calls before they run
- **Skills** — reusable behaviors the agent can load on demand
- **Tools** — bring your own functions or any MCP server

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
result = agent.invoke({"messages": "Research LangGraph and write a summary"})
```

The agent can plan, read/write files, and manage its own context. Add your own tools, swap models, customize prompts, configure sub-agents, and more. For a full overview and quickstart of Deep Agents, the best resource is our [docs](https://docs.langchain.com/oss/python/deepagents/overview).

### Workflow mode

Set `workflow_mode=True` to give the agent a `workflow` tool. Instead of delegating to sub-agents one `task` call at a time, the agent can author a whole multi-stage plan in a single call: a list of **phases** that run sequentially, where the **steps** inside each phase run concurrently (fan-out). A later step consumes an earlier step's output by embedding `{{step_id}}` in its prompt (fan-in / pipeline). The engine runs the plan autonomously and returns only the final result, keeping the orchestrator's context small.

```python
agent = create_deep_agent(model="openai:gpt-5.5", workflow_mode=True)
```

The agent might then emit a workflow like:

```json
{
  "phases": [
    {"title": "Research", "steps": [
      {"id": "a", "subagent_type": "general-purpose", "description": "Research topic A", "prompt": "Research topic A."},
      {"id": "b", "subagent_type": "general-purpose", "description": "Research topic B", "prompt": "Research topic B."}
    ]},
    {"title": "Synthesize", "steps": [
      {"id": "s", "subagent_type": "general-purpose", "description": "Synthesize A and B", "depends_on": ["a", "b"],
       "prompt": "Compare and synthesize:\n\nA: {{a}}\n\nB: {{b}}"}
    ]}
  ]
}
```

Each step takes an optional `description` (shown in a plan preview before the workflow runs). Workflow steps delegate to the same sub-agents the `task` tool uses, so any sub-agent you configure is also available as a workflow step.

Pass `workflow_model=...` to run the workflow's worker sub-agents on a different (e.g. cheaper or faster) model than the orchestrator; it defaults to the deep agent's model:

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",          # orchestrator
    workflow_mode=True,
    workflow_model="openai:gpt-5.4-mini",  # workflow step runners
)
```

Why it helps: without workflow mode a single agent runs every step in one conversation, so each tool call's output piles into a growing context that the model re-sends on every turn — more tokens, and more room for it to drift or hallucinate as stale prior-step detail accumulates. Workflow mode runs each step in a fresh, isolated sub-agent that only sees its own task (plus any results passed in via `{{step_id}}`), so no single context carries everything.

See `examples/workflow_mode_demo.py` for a runnable demo that builds a small Python library (writing files and running tests in a real shell) two ways — a plain DeepAgent and a DeepAgent in workflow mode — and compares them on tokens, context behavior, and the final answer.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## ❓ FAQ

### How is this different from LangGraph or LangChain?

LangGraph is the graph runtime. LangChain's `create_agent` is a minimal agent harness on top of it. Deep Agents is a more opinionated harness on top of `create_agent` — same building blocks, but with filesystem, sub-agents, context management, and skills bundled in. For how the three relate, see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products).

### Does this work with open-weight or local models?

Yes. Any model that supports tool calling works — frontier APIs (OpenAI, Anthropic, Google), open-weight models hosted on providers like Baseten or Fireworks, and self-hosted models via Ollama, vLLM, or llama.cpp. Use any [LangChain chat model](https://docs.langchain.com/oss/python/langchain/models).

### Can I use this in production?

Yes! Deep Agents is built on LangGraph, designed for production agent deployments. Pair it with [LangSmith](https://docs.langchain.com/langsmith/home) for tracing, evaluation, and monitoring. See [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) for the full guide.

### When should I use Deep Agents vs. LangChain or LangGraph directly?

All three are layers in the same stack — see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products) for how they relate. Use **Deep Agents** when you want the full harness — planning, context management, delegation — out of the box. Use [**LangChain's `create_agent`**](https://docs.langchain.com/oss/python/langchain/agents) when you want a lighter harness without the bundled middleware. Drop to [**LangGraph**](https://docs.langchain.com/oss/python/langgraph/overview) when the agent loop itself isn't the right shape and you need a custom graph.

The layers compose: any LangGraph `CompiledStateGraph` can be passed in as a sub-agent to a Deep Agent, so custom orchestration plugs in alongside the harness's defaults.

## 📖 Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents)** — Full documentation
- **[LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products)** — how Deep Agents, LangChain, LangGraph, and LangSmith fit together
- **[API Reference](https://reference.langchain.com/python/deepagents/)** — Full SDK reference documentation
- **[Examples](https://github.com/langchain-ai/deepagents/tree/main/examples)** — Working agents and patterns
- **[Discussions](https://forum.langchain.com/c/oss-product-help-lc-and-lg/deep-agents/18)** — Community forum for technical questions, ideas, and feedback
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 🔒 Security

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police. See the [security policy](https://github.com/langchain-ai/deepagents?tab=security-ov-file) for more information.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
