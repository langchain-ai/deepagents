# LangChain Deep Agents + Browserbase (Python)

This example shows the implementation pattern that fits LangChain Deep Agents best in Python:

- Give the main Deep Agent cheap Browserbase-backed tools for `search` and `fetch`
- Add a specialized browser subagent for heavier rendered or interactive browser work
- Gate stateful browser actions behind Deep Agents `interrupt_on`

It intentionally does **not** route the agent through the Browserbase CLI. Deep Agents already wants Python tools, subagents, and interrupt handling, so the clean integration is to expose Browserbase as Python tools directly.

## Architecture

- `browserbase_search`: fast discovery with Browserbase Search
- `browserbase_fetch`: cheap page retrieval with Browserbase Fetch
- `browserbase_rendered_extract`: Stagehand-backed rendered extraction for JS-heavy pages
- `browserbase_interactive_task`: a Stagehand `agent().execute(...)` workflow for clicks, typing, login, or form submission
- `browser-specialist` subagent: isolates browser-heavy work from the main planner

## Requirements

- Python 3.11+
- `BROWSERBASE_API_KEY` for Browserbase Search, Fetch, and browser sessions
- An OpenAI-compatible base URL for the Deep Agent model if you are not using direct OpenAI

The sample uses `BROWSERBASE_API_KEY` as the fallback API key for both:

- Browserbase primitives and Stagehand
- the LangChain chat model client

That means you do not need a second model-provider secret in this sample if you point the Deep Agent model at a compatible gateway endpoint.

The sample defaults to:

- Deep Agent model: `gpt-5.4`
- Stagehand rendered-extract model: `google/gemini-3-flash-preview`
- Stagehand interactive-agent model: `anthropic/claude-sonnet-4-6`

You can override either with environment variables.

## Install

```bash
cd examples/browserbase_deep_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

```bash
export BROWSERBASE_API_KEY="bb_..."

# Optional overrides
export DEEPAGENT_MODEL="gpt-5.4"
export DEEPAGENT_BASE_URL="https://<your-openai-compatible-gateway>"
export STAGEHAND_MODEL="google/gemini-3-flash-preview"
export STAGEHAND_AGENT_MODEL="anthropic/claude-sonnet-4-6"
```

## Run

Use the default research prompt:

```bash
python main.py
```

Or pass your own:

```bash
python main.py "Research the Browserbase Fetch API and explain when the agent should escalate to a full browser session."
```

## Approval flow

The sample configures `interrupt_on` for `browserbase_interactive_task`.

When the agent wants to click, type, log in, or submit a form, the script pauses and asks you to:

- `approve`
- `edit`
- `reject`

This is the right place to put human approval in a Deep Agents + Browserbase design, because the approval happens at the tool boundary instead of being hidden inside ad hoc shell calls.

## Notes

- The interactive tool now uses `stagehand.agent().execute(...)` instead of a single `sessions.act(...)` call. That makes it better suited to genuine multi-step browser tasks.
- Browserbase’s Stagehand quickstart documents that Model Gateway works with just `BROWSERBASE_API_KEY` for Stagehand browser workflows.
- I did not hardcode a Browserbase model-gateway URL for the LangChain model client because I did not find an official doc page in the Browserbase docs that specifies a general-purpose OpenAI-compatible endpoint for LangChain. The sample therefore accepts `DEEPAGENT_BASE_URL` or `OPENAI_BASE_URL` explicitly.

## Suggested prompts

- `Research Browserbase Search, Fetch, and browser sessions. Give me a decision tree with citations.`
- `Open docs.browserbase.com and extract the limits of the Fetch API from the rendered docs page.`
- `Go to example.com and tell me whether any interactive action would be required to complete the task.`
