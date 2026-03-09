# langchain-e2b

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-e2b?label=%20)](https://pypi.org/project/langchain-e2b/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-e2b)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-e2b)](https://pypistats.org/packages/langchain-e2b)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain_e2b
```

```python
from e2b import Sandbox

from langchain_e2b import E2BSandbox

sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=sandbox)
result = backend.execute("echo hello")
print(result.output)
```

## What is this?

E2B sandbox integration for Deep Agents.

This package uses the low-level `e2b` sandbox SDK, not `e2b-code-interpreter`,
so it can support shell execution, file transfer, and sandbox lifecycle control.

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
