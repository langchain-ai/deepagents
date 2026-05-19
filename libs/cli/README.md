# Deep Agents CLI — Deployment Tooling

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents-cli?label=%20)](https://pypi.org/project/deepagents-cli/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents-cli)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents-cli)](https://pypistats.org/packages/deepagents-cli)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

## Install

```bash
uv tool install deepagents-cli
```

Or with optional sandbox providers:

```bash
uv tool install 'deepagents-cli[all-sandboxes]'
```

## Usage

```bash
# Scaffold a new project folder
deepagents init my-agent

# Run a local langgraph dev server against the project
cd my-agent && deepagents dev

# Bundle and ship to LangSmith Deployment
deepagents deploy
```

## 📖 Resources

- **[CLI Documentation](https://docs.langchain.com/oss/python/deepagents/cli/overview)**
- **[Changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md)**
- **[Source code](https://github.com/langchain-ai/deepagents/tree/main/libs/cli)**
- **[Deep Agents SDK](https://github.com/langchain-ai/deepagents)** — underlying agent harness
- **[Deep Agents Code](https://pypi.org/project/deepagents-code/)** — coding agent

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
