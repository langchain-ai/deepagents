# langchain_daytona

Daytona sandbox integration for deepagents.

## Install

```bash
pip install langchain_daytona
```

## Usage

```python
from langchain_daytona import DaytonaProvider

provider = DaytonaProvider()
sandbox = provider.get_or_create()
result = sandbox.execute("echo hello")
print(result.output)
```

