# langchain-microsandbox

Microsandbox integration for Deep Agents. It adapts an existing
[`microsandbox.Sandbox`](https://docs.microsandbox.dev/) to the Deep Agents
`BaseSandbox` interface.

## Install

```bash
uv add langchain-microsandbox
```

## Usage

Microsandbox's Python SDK is asynchronous, so construct the backend with
`await MicrosandboxSandbox.create(...)`. The adapter does not own the sandbox
lifecycle: the application that creates a sandbox must also stop it and choose
the appropriate persistence or removal policy. In the ephemeral example below,
stopping the sandbox removes it automatically.

The image must contain `python3` because inherited `BaseSandbox` file and search
operations execute small Python helpers inside the guest.

```python
from deepagents import create_deep_agent
from microsandbox import Sandbox

from langchain_microsandbox import MicrosandboxSandbox


sandbox = await Sandbox.create(
    "deepagents-session",
    image="python:3.13-slim",
    ephemeral=True,
)
try:
    backend = await MicrosandboxSandbox.create(sandbox)
    agent = create_deep_agent(backend=backend)
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a small Python package and run pytest.",
                }
            ]
        }
    )
finally:
    await sandbox.stop()
```

Both sync and async Deep Agents backend methods are available. Async calls use
the Microsandbox SDK directly; sync calls use a compatibility bridge and can be
called safely even when the calling thread already has a running asyncio event
loop.

## Releases and versioning

See the LangChain [release](https://docs.langchain.com/oss/python/release-policy)
and [versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

See the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
