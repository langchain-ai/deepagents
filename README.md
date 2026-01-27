<p align="center">
  <img src=".github/images/deepagents-banner.png" alt="Deep Agents" width="600"/>
</p>

<p align="center">
  <img src=".github/images/deepagents_logo.svg" alt="Deep Agents" height="40"/>
</p>

<p align="center">
  The batteries-included agent harness.
</p>

## What is Deep Agents?

Deep Agents is an agent harness.  An opinionated, ready-to-run agent out of the box. Instead of wiring up prompts, tools, and context management yourself, you get a working agent immediately and customize what you need.

**What's included:**

- **Planning** — `write_todos` / `read_todos` for task breakdown and progress tracking
- **Filesystem** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` for reading and writing context
- **Shell access** — `execute` for running commands (with sandboxing)
- **Sub-agents** — `task` for delegating work with isolated context windows
- **Smart defaults** — Prompts that teach the model how to use these tools effectively
- **Context management** — Auto-summarization when conversations get long, large outputs saved to files

## Quickstart

```bash
pip install deepagents
# or
uv add deepagents
```

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Research LangGraph and write a summary"}]})
```

The agent can plan, read/write files, and manage its own context. Add tools, customize prompts, or swap models as needed.

## Customization

Add your own tools, swap models, customize prompts, configure sub-agents, and more. See the [documentation](https://docs.langchain.com/oss/python/deepagents/overview) for full details.

```python
from langchain.chat_models import init_chat_model

agent = create_deep_agent(
    model=init_chat_model("openai:gpt-4o"),
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
)
```

MCP is supported via [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters).

## Deep Agents CLI

Try Deep Agents instantly from the terminal:

```bash
uv tool install deepagents-cli
deepagents
```

The CLI adds conversation resume, web search, remote sandboxes (Modal, Runloop, Daytona), persistent memory, custom skills, and human-in-the-loop approval. See the [CLI documentation](https://docs.langchain.com/oss/python/deepagents/cli) for more.  Using the Deep Agents requires setting an API Key before running (ex: ANTHROPIC_API_KEY).

## LangGraph Native

`create_deep_agent` returns a compiled [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) graph. Use it with streaming, Studio, checkpointers, or any LangGraph feature.

## FAQ

### Why should I use this?

- **100% open source** — MIT licensed, fully extensible
- **Provider agnostic** — Works with Claude, OpenAI, Google, or any LangChain-compatible model
- **Built on LangGraph** — Production-ready runtime with streaming, persistence, and checkpointing
- **Batteries included** — Planning, file access, sub-agents, and context management work out of the box
- **Get started in seconds** — `pip install deepagents` or `uv add deepagents` and you have a working agent
- **Customize in minutes** — Add tools, swap models, tune prompts when you need to

## Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents/overview)** — Full API reference and guides
- **[Examples](examples/)** — Working agents and patterns
- **[CLI](https://docs.langchain.com/oss/python/deepagents/cli)** — Interactive terminal interface

## Security

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police.
