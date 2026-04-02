# langchain-tensorlake

Tensorlake sandbox integration for Deep Agents.

## Quick install

```bash
pip install langchain-tensorlake
```

```python
from tensorlake.sandbox import SandboxClient
from langchain_tensorlake import TensorlakeSandbox

client = SandboxClient.for_cloud(api_key='...')
response = client.create(image='python:3.11')
backend = TensorlakeSandbox(
    sandbox=tensorlake.sandbox.Sandbox(
        sandbox_id=response.sandbox_id,
        api_key='...',
    )
)
print(backend.execute('echo hello').output)
```
