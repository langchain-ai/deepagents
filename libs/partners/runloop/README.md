# langchain-runloop

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-runloop?label=%20)](https://pypi.org/project/langchain-runloop/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-runloop)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-runloop)](https://pypistats.org/packages/langchain-runloop)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-runloop
```

```python
import os

from langchain_runloop import RunloopProvider

api_key = os.environ["RUNLOOP_API_KEY"]
provider = RunloopProvider(api_key=api_key)

sandbox = provider.get_or_create()
try:
    result = sandbox.execute("echo hello")
    print(result.output)
finally:
    provider.delete(sandbox_id=sandbox.id)
```

Boot from a named blueprint (create-if-missing, same idea as LangSmith snapshots):

```python
sandbox = provider.get_or_create(snapshot="my-blueprint")
```

Or pin via env: `RUNLOOP_SANDBOX_BLUEPRINT_NAME`, `RUNLOOP_SANDBOX_BLUEPRINT_ID`
(ID wins; skips auto-build).

## 🤔 What is this?

Runloop sandbox integration for Deep Agents.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
