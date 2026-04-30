# langchain-cloudflare

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-cloudflare?label=%20)](https://pypi.org/project/langchain-cloudflare/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-cloudflare)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-cloudflare)](https://pypistats.org/packages/langchain-cloudflare)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-cloudflare
```

```python
from langchain_cloudflare import CloudflareSandbox

sandbox = CloudflareSandbox(
    base_url="https://your-worker.your-subdomain.workers.dev",
    sandbox_id="my-session",
    api_key="your-sandbox-api-key",
)
result = sandbox.execute("echo hello")
print(result.output)
```

## 🤔 What is this?

Cloudflare sandbox integration for Deep Agents.

This package communicates with the [Cloudflare Sandbox Bridge](https://developers.cloudflare.com/sandbox/bridge/)
HTTP API, which exposes Cloudflare's container-based sandboxes via a
Cloudflare Worker. You deploy the bridge Worker with sandbox bindings, then
point this client at its URL.

### Architecture

```
Python (this package)  ──HTTP──▶  Cloudflare Worker (bridge)  ──▶  Sandbox container
```

The bridge handles sandbox lifecycle, command execution (via SSE streaming),
and file I/O. This package translates those HTTP endpoints into the
`BaseSandbox` interface expected by Deep Agents.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
