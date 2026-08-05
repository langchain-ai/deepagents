# langchain-cloud-run

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-cloud-run?label=%20)](https://pypi.org/project/langchain-cloud-run/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-cloud-run)](https://opensource.org/licenses/MIT)

Cloud Run sandbox integration for Deep Agents.

## Quick Install

```bash
uv add langchain-cloud-run
```

```python
from langchain_cloud_run import CloudRunSandbox
from deepagents import create_deep_agent

# Create an agent using the Cloud Run Sandbox backend
backend = CloudRunSandbox(
    allow_egress=False,
    env={"PYTHONPATH": "/workspace"},
    workdir="/workspace",
)

agent = create_deep_agent(
    model="google_genai:gemini-1.5-flash",
    backend=backend,
)

# Execute shell command string or list of command arguments
result = backend.execute(
    ["python3", "script.py", "arg1"],
    env={"LOG_LEVEL": "DEBUG"},
    workdir="/workspace",
)
print(result.output)
```

## 🤔 What is this?

Provides a `BaseSandbox` backend for Deep Agents that executes commands and code safely inside isolated Cloud Run containers using `/usr/local/gcp/bin/sandbox`.

### Features

- **Guest Kernel Isolation**: Enforces process sandboxing inside Cloud Run containers via `/usr/local/gcp/bin/sandbox do`.
- **Egress Network Control**: Restricts or permits outbound network calls via `allow_egress`.
- **Polyglot Execution**: Supports executing shell command strings (`str`) or direct binary invocation (`list[str]`).
- **Environment & Working Directory Support**: Custom environment variables (`env`) and working directory (`workdir`) per sandbox instance or per `execute()` call.
- **Pass-through Escape Hatch**: Support for `extra_sandbox_args` to pass any advanced `sandbox do` flags (`--write`, `--sync-tar`, `--mount`, `--debug`).
- **Fast Local Disk Transfers**: Sub-millisecond `upload_files` and `download_files` using shared container filesystem mounts.

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

See the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
