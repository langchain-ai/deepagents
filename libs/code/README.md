# 🧠🤖 Deep Agents Code

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents-code?label=%20)](https://pypi.org/project/deepagents-code/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents-code)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents-code)](https://pypistats.org/packages/deepagents-code)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

<p align="center">
  <img src="https://raw.githubusercontent.com/langchain-ai/deepagents/main/libs/code/images/tui.png" alt="Deep Agents Code" width="600"/>
</p>

## Quick Install

```bash
curl -LsSf https://langch.in/dcode | bash
```

```bash
# With model provider extras
# OpenAI, Anthropic, and Gemini are included by default
DEEPAGENTS_CODE_EXTRAS="nvidia,ollama" curl -LsSf https://langch.in/dcode | bash
```

Run:

```bash
dcode
```

## 🤔 What is this?

The fastest way to start using Deep Agents. `deepagents-code` is a pre-built coding agent in your terminal — similar to Claude Code or Cursor — powered by any LLM that supports tool calling. One install command and you're up and running, no code required.

**What `deepagents-code` adds on top of the SDK:**

- **Interactive TUI** — rich terminal interface with streaming responses
- **Conversation resume** — pick up where you left off across sessions
- **Web search** — ground responses in live information
- **Remote sandboxes** — run code in isolated environments (LangSmith, AgentCore, Daytona, Modal, Runloop, & more)
- **Persistent memory** — agent remembers context across conversations
- **Custom skills** — extend the agent with your own slash commands
- **Experimental plugins** — load marketplace skills, commands, agents, and MCP servers
- **Headless mode** — run non-interactively for scripting and CI
- **Human-in-the-loop** — approve or reject tool calls before execution

## 🔒 Security model

By default, `dcode` trusts the directory you run it in. Human-in-the-loop approval gates model-requested tool calls, but project artifacts are read before any approval prompt.

Do not run `dcode` in a directory you do not trust without a sandbox backend. For untrusted repositories, use a [remote sandbox](https://docs.langchain.com/oss/python/deepagents/code/remote-sandboxes) so execution is isolated from your machine. Running `dcode` in a directory lets that directory's files shape execution. See [`THREAT_MODEL.md`](https://github.com/langchain-ai/deepagents/blob/main/libs/code/THREAT_MODEL.md) for details.

Plugins can add executable MCP servers. Their exact executable surface must be
trusted before activation; see [`PLUGINS.md`](./PLUGINS.md). Plugin hook
declarations are detected but not executed.

## 📖 Resources

- **[Documentation](https://docs.langchain.com/deepagents-code)**
- **[Changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/code/CHANGELOG.md)**
- **[Source code](https://github.com/langchain-ai/deepagents/tree/main/libs/code)**
- **[Deep Agents SDK](https://github.com/langchain-ai/deepagents)** — underlying agent harness
- **[Plugin guide](./PLUGINS.md)** — marketplace, lifecycle, trust, and compatibility
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## 🤝 Acknowledgements

This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.
