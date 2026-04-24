# LangChain Deep Agents + Browserbase

This example shows how to expose Browserbase Search, Fetch, and Stagehand browser sessions as Python tools for a LangChain Deep Agent.

The pattern is:

- give the main Deep Agent low-cost `search` and `fetch` tools for discovery and static pages
- add a browser specialist subagent for rendered extraction and interactive browser work
- require human review before stateful browser actions with Deep Agents `interrupt_on`

The example does not call the Browserbase CLI from inside the agent. Deep Agents already works with Python tools, subagents, and human-in-the-loop middleware, so Browserbase is exposed directly as LangChain tools.

## Architecture

| Component | Purpose |
| --- | --- |
| `browserbase_search` | Search result discovery through Browserbase Search |
| `browserbase_fetch` | Fast page retrieval through Browserbase Fetch |
| `browserbase_rendered_extract` | Stagehand-backed extraction from rendered pages |
| `browserbase_interactive_task` | Stagehand agent execution for clicking, typing, login, or form submission |
| `browser-specialist` | Subagent that isolates browser-heavy work from the main planner |

## Requirements

- Python 3.11+
- `BROWSERBASE_API_KEY` for Browserbase Search, Fetch, and hosted browser sessions
- `OPENAI_API_KEY` for direct OpenAI access, or an OpenAI-compatible `DEEPAGENT_BASE_URL`

The sample defaults to:

- Deep Agent model: `gpt-5.5`
- Stagehand rendered extraction model: `google/gemini-3-flash-preview`
- Stagehand interactive agent model: `anthropic/claude-sonnet-4-6`

You can override these with environment variables.

## Install

```bash
cd examples/langchain-browserbase
uv sync
cp .env.example .env
```

Then edit `.env` with your Browserbase key and model credentials.

## Run

Use the default research prompt:

```bash
uv run python main.py
```

Or pass your own:

```bash
uv run python main.py "Research the Browserbase Fetch API and explain when the agent should escalate to a full browser session."
```

## Approval Flow

The sample configures `interrupt_on` for `browserbase_interactive_task`.

When the agent wants to click, type, log in, or submit a form, the script pauses and asks you to approve, edit, or reject the proposed tool call. Approval happens at the Deep Agents tool boundary instead of being hidden inside ad hoc shell commands.

## Suggested Prompts

- `Research Browserbase Search, Fetch, and browser sessions. Give me a decision tree with citations.`
- `Open docs.browserbase.com and extract the limits of the Fetch API from the rendered docs page.`
- `Go to example.com and tell me whether any interactive action would be required to complete the task.`
