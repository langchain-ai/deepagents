### Deep Research Agent (OpenAI)

This example mirrors the Anthropic-based research agent but uses OpenAI via LangChain's `init_chat_model`. It researches a topic using Tavily search, writes a structured report to a virtual filesystem, and can spawn sub-agents for deep research and critique.

### Requirements

- **Python**: 3.11+
- **API keys**: `OPENAI_API_KEY`, `TAVILY_API_KEY`

### Setup (macOS and Linux)

1. Create and activate a virtual environment

```bash
cd examples/research_openai
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in this folder

```dotenv
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

### Running

- Launch Studio (recommended):

```bash
uv run langgraph dev
```

### Files

- `research_agent_openai.py`: Defines tools, sub-agents, and constructs the agent. Uses OpenAI via:

```python
from langchain.chat_models import init_chat_model
model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.2)
```

- `langgraph.json`: Wires the graph name `research_openai` to the `agent` object.
- `requirements.txt`: Example-specific dependencies.

### Customize the model

Edit the `init_chat_model(...)` call in `research_agent_openai.py`. Examples:

```python
model = init_chat_model(model="openai:gpt-4o", temperature=0.2)
# or a smaller model
model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)
```

Per-subagent overrides are also supported by the library if you add a `model_settings` dict to a sub-agent.

### How it works

- The agent uses built-in planning and a virtual filesystem (files are stored in state, not your disk). It commonly writes:
  - `question.txt`: the original user query
  - `final_report.md`: the final report in markdown
- Sub-agents:
  - `research-agent`: performs deep-dive research using `internet_search`
  - `critique-agent`: reviews the report and suggests improvements (does not edit files itself)

### Troubleshooting

- Ensure `.env` contains valid `OPENAI_API_KEY` and `TAVILY_API_KEY`.
- If you see OpenAI auth or model errors, verify your account has access to the requested model and the key is correct.
- If `langgraph` CLI isn’t found, ensure the virtual environment is activated and dependencies installed.
