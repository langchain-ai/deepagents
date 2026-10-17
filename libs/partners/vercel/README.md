# langchain-vercel-sandbox

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-vercel-sandbox?label=%20)](https://pypi.org/project/langchain-vercel-sandbox/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-vercel-sandbox)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-vercel-sandbox)](https://pypistats.org/packages/langchain-vercel-sandbox)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-vercel-sandbox
```

```python
from vercel.sandbox import Sandbox

from langchain_vercel_sandbox import VercelSandbox

sandbox = Sandbox.create()

try:
    backend = VercelSandbox(sandbox=sandbox)
    result = backend.execute("echo hello")
    print(result.output)
finally:
    sandbox.stop()
```

## 🤔 What is this?

Vercel Sandbox integration for Deep Agents.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
