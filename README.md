<div align="center">
  <a href="https://docs.langchain.com/oss/python/deepagents/overview#deep-agents-overview">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-light.svg">
      <img alt="Deep Agents Logo" src=".github/images/logo-dark.svg" width="80%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>The batteries-included agent harness.</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/deepagents" alt="PyPI - License"></a>
  <a href="https://pypistats.org/packages/deepagents" target="_blank"><img src="https://img.shields.io/pepy/dt/deepagents" alt="PyPI - Downloads"></a>
  <a href="https://pypi.org/project/deepagents/#history" target="_blank"><img src="https://img.shields.io/pypi/v/deepagents?label=%20" alt="Version"></a>
  <a href="https://x.com/langchain" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

Deep Agents is an agent harness. An opinionated, ready-to-run agent out of the box. Instead of wiring up prompts, tools, and context management yourself, you get a working agent immediately and customize what you need.

**What's included:**

- **Planning** — `write_todos` for task breakdown and progress tracking
- **Filesystem** — `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` for reading and writing context
- **Shell access** — `execute` for running commands (with sandboxing)
- **Sub-agents** — `task` for delegating work with isolated context windows
- **Smart defaults** — Prompts that teach the model how to use these tools effectively
- **Context management** — Auto-summarization when conversations get long, large outputs saved to files

> [!NOTE]
> Looking for the JS/TS library? Check out [deepagents.js](https://github.com/langchain-ai/deepagentsjs).

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

The CLI adds conversation resume, web search, remote sandboxes (Modal, Runloop, Daytona, & more), persistent memory, custom skills, headless mode, and human-in-the-loop approval. See the [CLI documentation](https://docs.langchain.com/oss/python/deepagents/cli) for more.

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

---

## Documentation

- [docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/overview) – Comprehensive documentation, including conceptual overviews and guides
- [reference.langchain.com/python](https://reference.langchain.com/python/deepagents/) – API reference docs for Deep Agents packages
- [Chat LangChain](https://chat.langchain.com/) – Chat with the LangChain documentation and get answers to your questions

**Discussions**: Visit the [LangChain Forum](https://forum.langchain.com) to connect with the community and share all of your technical questions, ideas, and feedback.

## Additional resources

- **[Examples](examples/)** — Working agents and patterns
- [API Reference](https://reference.langchain.com/python/deepagents/) – Detailed reference on navigating base packages and integrations for LangChain.
- [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) – Learn how to contribute to LangChain projects and find good first issues.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) – Our community guidelines and standards for participation.

## Packages

This is a monorepo containing all Deep Agents packages:

| Package | PyPI | Description |
| ------- | ---- | ----------- |
| [`deepagents`](libs/deepagents/) | [![Version](https://img.shields.io/pypi/v/deepagents?label=%20)](https://pypi.org/project/deepagents/) | Core SDK — `create_deep_agent`, middleware, backends |
| [`deepagents-cli`](libs/cli/) | [![Version](https://img.shields.io/pypi/v/deepagents-cli?label=%20)](https://pypi.org/project/deepagents-cli/) | Interactive terminal interface with TUI, web search, and sandboxes |
| [`deepagents-acp`](libs/acp/) | [![Version](https://img.shields.io/pypi/v/deepagents-acp?label=%20)](https://pypi.org/project/deepagents-acp/) | [Agent Client Protocol](https://agentclientprotocol.com) integration for editors like Zed |
| [`deepagents-harbor`](libs/harbor/) | - | [Harbor](https://harborframework.com) evaluation and benchmark framework |
| [`langchain-daytona`](libs/partners/daytona/) | [![Version](https://img.shields.io/pypi/v/langchain-daytona?label=%20)](https://pypi.org/project/langchain-daytona/) | Daytona sandbox integration |
| [`langchain-modal`](libs/partners/modal/) | [![Version](https://img.shields.io/pypi/v/langchain-modal?label=%20)](https://pypi.org/project/langchain-modal/) | Modal sandbox integration |
| [`langchain-runloop`](libs/partners/runloop/) | [![Version](https://img.shields.io/pypi/v/langchain-runloop?label=%20)](https://pypi.org/project/langchain-runloop/) | Runloop sandbox integration |

---

## Acknowledgements

This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.

## Security

Deep Agents follows a "trust the LLM" model. The agent can do anything its tools allow. Enforce boundaries at the tool/sandbox level, not by expecting the model to self-police.
