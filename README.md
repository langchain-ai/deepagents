# 🧠🤖Deep Agents

Using an LLM to call tools in a loop is the simplest form of an agent. 
This architecture, however, can yield agents that are "shallow" and fail to plan and act over longer, more complex tasks. 
Applications like "Deep Research", "Manus", and "Claude Code" have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a TypeScript package that implements these in a general purpose way so that you can easily create a Deep Agent for your application.

![TIP] Looking for the Python version of this package? See here: hwchase17/deepagents

## Installation

```bash
yarn add deepagents
```

## Learn more

For more information, check out our docs: https://docs.langchain.com/labs/deep-agents/overview
