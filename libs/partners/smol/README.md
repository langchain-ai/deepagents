# langchain-smol

Use a local or Cloud Smol Machines microVM as a Deep Agents sandbox. Commands and file operations stay inside the VM; the host is not mounted into it by this integration.

## Install

```bash
uv add langchain-smol
```

## Example

```python
import smol
from deepagents import create_deep_agent
from langchain_smol import SmolSandbox

# Local works without a Cloud token. For Cloud, use target="cloud" and
# SMOL_CLOUD_TOKEN; the same sandbox adapter works for both targets.
connection = smol.ConnectOptions(target="local")
machine = smol.Machine.create(
    smol.MachineConfig(image="python:3.12-alpine", network=False), connection
)
try:
    sandbox = SmolSandbox(machine=machine)
    agent = create_deep_agent(backend=sandbox)
    result = agent.invoke({"messages": [{"role": "user", "content": "Create a hello.py file."}]})
    print(result["messages"][-1].content)
finally:
    machine.delete()
```

The VM image needs `sh` and `python3` for Deep Agents' command and file tools. Set `network=True` when a workload requires outbound access, or when Cloud needs to pull a cold image. Save the machine ID for later attachment with `smol.Machine.connect` if an agent run must survive process restarts. The caller owns the VM and should delete it after use. `SmolSandbox` does not transfer host credentials into the VM.
