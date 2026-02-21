# Deep Agents ACP integration

This repo contains an [Agent Client Protocol (ACP)](https://agentclientprotocol.com/overview/introduction) connector that allows you to run a Python [DeepAgent](https://docs.langchain.com/oss/python/deepagents/overview) within a text editor that supports ACP such as [Zed](https://zed.dev/).

![Deep Agents ACP Demo](./static/img/deepagentsacp.gif)

It includes an example coding agent that uses Anthropic's Claude models to write code with its built-in filesystem tools and shell, but you can also connect any Deep Agent with additional tools or different agent architectures!

## Getting started

First, make sure you have [Zed](https://zed.dev/) and [`uv`](https://docs.astral.sh/uv/) installed.

Next, clone this repo:

```sh
git clone git@github.com:langchain-ai/deepagents.git
```

Then, navigate into the newly created folder and run `uv sync`:

```sh
cd deepagents/libs/acp
uv sync
```

Rename the `.env.example` file to `.env` and add your [Anthropic](https://claude.com/platform/api) API key. You may also optionally set up tracing for your DeepAgent using [LangSmith](https://smith.langchain.com/) by populating the other env vars in the example file:

```ini
ANTHROPIC_API_KEY=""

# Set up LangSmith tracing for your DeepAgent (optional)

# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=""
# LANGSMITH_PROJECT="deepagents-acp"
```

Finally, add this to your Zed `settings.json`:

```json
{
  "agent_servers": {
    "DeepAgents": {
      "type": "custom",
      "command": "/your/absolute/path/to/deepagents-acp/run_demo_agent.sh"
    }
  }
}
```

You must also make sure that the `run_demo_agent.sh` entrypoint file is executable - this should be the case by default, but if you see permissions issues, run:

```sh
chmod +x run_demo_agent.sh
```

Now, open Zed's Agents Panel (e.g. with `CMD + Shift + ?`). You should see an option to create a new DeepAgent thread:

![](./static/img/newdeepagent.png)

And that's it! You can now use the DeepAgent in Zed to interact with your project.

If you need to upgrade your version of DeepAgents, run:

```sh
uv upgrade deepagents-acp
```

## Launch a custom DeepAgent with ACP

```sh
uv add deepagents-acp
```

```python
import asyncio

from acp import run_agent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP


async def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


async def main() -> None:
    agent = create_deep_agent(
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
        checkpointer=MemorySaver(),
    )
    server = AgentServerACP(agent)
    await run_agent(server)


if __name__ == "__main__":
    asyncio.run(main())
```

### Launch with Toad

```sh
uv tool install -U batrachian-toad --python 3.14

toad acp "python path/to/your_server.py" .
# or
toad acp "uv run python path/to/your_server.py" .
```
