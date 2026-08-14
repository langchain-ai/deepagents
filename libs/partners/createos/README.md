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
result = backend.execute("echo hello")
print(result.output)
```
