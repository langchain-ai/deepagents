# langchain-mainbrella

Mainbrella sandbox backend for Deep Agents, using the official `mainbrella`
Python SDK. Install with `uv pip install langchain-mainbrella` and set
`MAINBRELLA_API_KEY` to an API key from your Mainbrella account.

```python
import os

from deepagents import create_deep_agent
from langchain_mainbrella import MainbrellaSandbox
from mainbrella import Mainbrella

client = Mainbrella(os.environ["MAINBRELLA_API_KEY"])
with client.create(catalog_id="python") as sandbox:
    backend = MainbrellaSandbox(sandbox=sandbox)
    agent = create_deep_agent(backend=backend)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write and run a Python script in /workspace.",
                }
            ]
        }
    )
```

The context manager stops its exact container generation. The backend wraps an
existing sandbox and does not provision or stop it automatically. Inherited
Deep Agents file tools require `python3`; choose the `python` catalog image or a
custom image with Python, `/bin/sh`, and GNU coreutils.

For lifecycle management, use `MainbrellaProvider`:

```python
from langchain_mainbrella import MainbrellaProvider

provider = MainbrellaProvider(api_key=os.environ["MAINBRELLA_API_KEY"])
backend = provider.get_or_create(size="lite")
try:
    print(backend.execute("python3 --version"))
    # Persist the entire ID when reconnecting, including its creation timestamp.
    attached = provider.get_or_create(sandbox_id=backend.id)
finally:
    provider.delete(sandbox_id=backend.id)
```

`backend.id` is `slot@createdAt`. Bare slot IDs are rejected, and stale
generations cannot attach to or stop replacement containers. Creation defaults
to the Python catalog image; `image_id` selects a custom image instead.
`base_url` selects an alternate API origin; credentials use HTTPS except for
loopback development. SDK errors retain sanitized error codes and creation
idempotency keys for reconciliation. Commands and file writes are not retried.

Commands must fit Mainbrella's 16 KiB command limit and default to a 60-second
timeout. Explicit timeouts from 1 to 900 seconds
are supported; zero and values above 900 are rejected. Commands above 60 seconds
use managed execution, retaining one of the container's 32 job records for one
hour. Provider timeouts preserve partial output and return exit code 124. Output
limits preserve the provider's truncation flag and unknown exit status.
Binary upload/download is limited to 1 MiB per file and requires absolute paths;
upload parent directories must already exist. Ordinary stop discards unsaved files.

The coding agent supports `dcode --sandbox mainbrella` after installing
`deepagents-code[mainbrella]`. It reads `MAINBRELLA_API_KEY` and optional
`MAINBRELLA_API_URL`, including `DEEPAGENTS_CODE_` overrides. Reconnect with
`--sandbox-id 'slot@createdAt'`; creation parameters such as `size` or `image_id`
can be supplied through the sandbox configuration.

From this directory, run `uv sync --all-groups`, `make test`, and `make lint`.
`make integration_test` creates and cleans up a live Python container and requires
`MAINBRELLA_API_KEY`; optional `MAINBRELLA_API_URL` selects the deployment.
