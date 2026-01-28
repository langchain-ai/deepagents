# DeepAgents ACP integration

This repo contains an [Agent Client Protocol (ACP)](https://agentclientprotocol.com/overview/introduction) connector that allows you to run a Python [DeepAgent](https://docs.langchain.com/oss/python/deepagents/overview) within a text editor that supports ACP such as [Zed](https://zed.dev/).

The DeepAgent lives as code in `deepagents_acp/agent.py`, and can interact with the files of a project you have open in your ACP-compatible editor.

![DeepAgents ACP Demo](./static/img/deepagentsacp.gif)

Out of the box, your agent uses Anthropic's Claude models to do things like write code with its built-in filesystem tools, but you can also extend it with additional tools or agent architectures!

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
      "command": "/your/absolute/path/to/deepagents-acp/run.sh"
    }
  }
}
```

You must also make sure that the `run.sh` entrypoint file is executable - this should be the case by default, but if you see permissions issues, run:

```sh
chmod +x run.sh
```

Now, open Zed's Agents Panel (e.g. with `CMD + Shift + ?`). You should see an option to create a new DeepAgent thread:

![](./static/img/newdeepagent.png)

And that's it! You can now use the DeepAgent in Zed to interact with your project.
