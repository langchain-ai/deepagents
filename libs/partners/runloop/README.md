# langchain-runloop

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-runloop?label=%20)](https://pypi.org/project/langchain-runloop/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-runloop)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-runloop)](https://pypistats.org/packages/langchain-runloop)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-runloop
```

```python
import os

from runloop_api_client import RunloopSDK

from langchain_runloop import RunloopSandbox

api_key = os.environ["RUNLOOP_API_KEY"]
client = RunloopSDK(bearer_token=api_key)

devbox = client.devbox.create()
sandbox = RunloopSandbox(devbox=devbox)

try:
    result = sandbox.execute("echo hello")
    print(result.output)
finally:
    devbox.shutdown()
```

## 🤔 What is this?

Runloop sandbox integration for Deep Agents.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
