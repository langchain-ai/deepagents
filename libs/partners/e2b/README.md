# langchain-e2b

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-e2b?label=%20)](https://pypi.org/project/langchain-e2b/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-e2b)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-e2b)](https://pypistats.org/packages/langchain-e2b)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-e2b
```

```python
from e2b import Sandbox

from langchain_e2b import E2BSandbox

sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=sandbox)

try:
    result = backend.execute("echo hello")
    print(result.output)
finally:
    sandbox.kill()
```

## What is this?

`langchain-e2b` adapts an existing E2B sandbox to the Deep Agents sandbox
protocol. It uses the low-level `e2b` SDK so Deep Agents can run shell commands
and move files through the same backend interface used by the Daytona, Modal,
and Runloop integrations.

This package intentionally does not hide E2B sandbox lifecycle management. Use
the E2B SDK to create, connect to, configure, and kill sandboxes, then pass the
connected sandbox object to `E2BSandbox`.

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Development

```bash
uv sync --group test
make test
make lint
```

Integration tests require `E2B_API_KEY`:

```bash
E2B_API_KEY=... make integration_tests
```
