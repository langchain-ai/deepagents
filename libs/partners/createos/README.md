# langchain-createos

## Quick Install

```bash
uv add langchain-createos
```

```python
from langchain_createos import CreateOSSandbox

backend = CreateOSSandbox(
    sandbox_id="sb_01ABC...",
    api_key="your-api-key",
)
try:
    result = backend.execute("echo hello")
    print(result.output)
finally:
    backend.close()
```
