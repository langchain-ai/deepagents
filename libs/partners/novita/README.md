# langchain-novita

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-novita?label=%20)](https://pypi.org/project/langchain-novita/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-novita)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-novita)](https://pypistats.org/packages/langchain-novita)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain_novita
```

```python
from novita_sandbox.code_interpreter import Sandbox

from langchain_novita import NovitaSandbox

sdk_sandbox = Sandbox.create()
backend = NovitaSandbox(sandbox=sdk_sandbox)
result = backend.execute("echo hello")
print(result.output)
sdk_sandbox.kill()
```

## 🤔 What is this?

Novita AI sandbox integration for Deep Agents.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
