# Parallel Subagent Coordination

Demonstrates how to use Deep Agents' subagent system to run multiple specialist agents in parallel and synthesize their results.

## What this example shows

- Defining multiple `SubAgent` specs with different roles and system prompts
- Launching subagents in parallel via the `task` tool
- Synthesizing results from parallel subagents into a unified report
- Using the `think` tool for strategic reflection within subagents

## Architecture

```
User Query
    |
    v
Orchestrator Agent
    |
    +---> market-researcher (parallel)
    +---> technical-analyst  (parallel)
    +---> user-researcher    (parallel)
    |
    v
Synthesized Report
```

The orchestrator receives a research query and delegates to three specialist subagents simultaneously. Each subagent operates in its own isolated context and returns a focused report. The orchestrator then combines all findings into a unified executive summary.

## Quickstart

**Prerequisites**: Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Navigate to this directory and install dependencies:

```bash
cd examples/parallel_subagents
uv sync
```

Set your API keys:

```bash
export ANTHROPIC_API_KEY=your_key_here
export LANGSMITH_API_KEY=your_key_here  # optional, for tracing
```

### Option 1: Run directly

```bash
uv run python agent.py
```

### Option 2: LangGraph Server

```bash
langgraph dev
```

Then open the Studio interface in your browser to interact with the agent.

## Customization

### Adding more subagents

Define additional `SubAgent` specs and pass them to `create_deep_agent`:

```python
from deepagents import SubAgent, create_deep_agent

financial_analyst: SubAgent = {
    "name": "financial-analyst",
    "description": "Analyze financial metrics and business models.",
    "system_prompt": "You are a financial analyst...",
    "tools": [think],
}

agent = create_deep_agent(
    model=model,
    subagents=[market_researcher, technical_analyst, user_researcher, financial_analyst],
)
```

### Using a different model

```python
from langchain.chat_models import init_chat_model

# Use OpenAI
model = init_chat_model("openai:gpt-4o", temperature=0.0)

# Use Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
```

### Per-subagent models

Override the model for individual subagents:

```python
technical_analyst: SubAgent = {
    "name": "technical-analyst",
    "description": "...",
    "system_prompt": "...",
    "model": "openai:gpt-4o",  # Use a different model for this subagent
}
```
