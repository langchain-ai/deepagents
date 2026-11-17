# langchain-mxc

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-mxc?label=%20)](https://pypi.org/project/langchain-mxc/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-mxc)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-mxc)](https://pypistats.org/packages/langchain-mxc)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-mxc
npm install -g @microsoft/mxc-sdk  # installs the MXC executor
```

```python
from langchain_mxc import MXCSandbox

backend = MXCSandbox(root_dir="./my-project")
result = backend.execute("echo hello")
print(result.output)
```

## 🤔 What is this?

Microsoft Execution Containers (MXC) sandbox integration for Deep Agents. Shell commands from the agent's `execute` tool run in an MXC sandbox on the local machine, and the operating system limits which folders they can read and write and whether they can use the network. File tools run in the agent's process and are limited to `root_dir` by Deep Agents' path checks.

Requires Windows 11 24H2 (build 26100.9278 or newer), macOS 15 or newer on Apple silicon, or Linux with bubblewrap 0.5 or newer. The MXC executor ships in the `@microsoft/mxc-sdk` npm package, which needs Node.js 24 or newer to install. `MXCSandbox` finds it automatically, or set `executor_path` or the `MXC_EXECUTOR_PATH` environment variable.

On Windows, most developer tools (`git`, `node`, PowerShell, `pytest`) need `allow_ui=True` to start. See the [integration guide](https://docs.langchain.com/oss/python/integrations/sandboxes/mxc) for the default policy, configuration, and platform notes.

## 📕 Releases & Versioning

See the [changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/partners/mxc/CHANGELOG.md) for release history.

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
