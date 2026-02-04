# MCP Tools Integration Example

This example demonstrates how to integrate MCP (Model Context Protocol) tools with DeepAgents.

## Overview

This example provides 5 custom MCP tools that can be registered with DeepAgents:

| Tool | Description | Use Case |
|------|-------------|----------|
| `google_search_and_summarize` | Google search + web page fetching | Current events, web research |
| `rag_search` | RAG-based document search | Papers, technical docs |
| `weather_forecast` | 5-day weather forecast | Weather queries, planning |
| `sentinel_search` | Sentinel satellite imagery search | Satellite scene queries |
| `arxiv_search` | arXiv paper search | Academic papers, research |

## Installation

```bash
# From the examples directory
cd examples/mcp_tools_integration

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   ```bash
   # Required API keys
   OPENAI_API_KEY=your_key_here
   OPENWEATHER_API_KEY=your_key_here

   # MCP server paths (adjust to your setup)
   MCP_GOOGLE_SEARCH_SERVER=/path/to/google_search_server.py
   MCP_WEB_FETCH_SERVER=/path/to/web_fetch_server.py
   MCP_RAG_SERVER=/path/to/spaceops_rag.py
   MCP_WEATHER_SERVER=/path/to/weather.py
   MCP_SENTINEL_SERVER=/path/to/sentinel_server.py
   ```

## Usage

### Basic Usage - Single Tool

```python
from mcp_tools import google_search_and_summarize, weather_forecast

# Google search
result = google_search_and_summarize("트럼프 임기 기간")
for source in result["sources"]:
    print(f"Title: {source['title']}")
    print(f"URL: {source['url']}")
    print(f"Content: {source['text'][:300]}...")

# Weather
result = weather_forecast("Seoul", units="c", lang="kr")
print(result["forecast"])
```

### With DeepAgents

```python
from deepagents import create_deep_agent
from mcp_tools import ALL_MCP_TOOLS

# Create agent with all MCP tools
agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=ALL_MCP_TOOLS,
)

# Use the agent
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "서울 내일 날씨 어때?"}]},
    config={"configurable": {"thread_id": "my-thread"}},
)
print(result["messages"][-1].content)
```

### Router Agent (Advanced)

The router agent automatically selects appropriate tools based on query analysis:

```python
from router_agent import RouterAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
agent = RouterAgent(llm)

# Automatically routes to weather_forecast
answer = await agent.answer("서울 내일 날씨")

# Automatically routes to rag_search
answer = await agent.answer("SpaceOps 논문 검색")

# Routes to multiple tools and synthesizes
answer = await agent.answer("내일 대전 날씨와 위성 촬영 가능성")
```

## Running Examples

### Example Agent

```bash
# Run example queries
python example_agent.py

# Interactive mode
python example_agent.py --interactive

# Single query
python example_agent.py --query "서울 날씨"
```

### Router Agent

```bash
# Run example queries
python router_agent.py

# Interactive mode
python router_agent.py --interactive

# Single query
python router_agent.py --query "대전 위성 영상 검색"

# Custom model/API
python router_agent.py --interactive \
    --model openai/gpt-oss-20b \
    --api-base http://vllm_server:8000/v1 \
    --api-key EMPTY
```

## Tool Details

### google_search_and_summarize

```python
def google_search_and_summarize(
    query: str,
    num_results: int = 5,
    fetch_top_n: int = 3,
    max_chars_per_page: int = 2500,
) -> dict[str, Any]:
    """
    Search Google and fetch web page contents.

    Returns:
        {
            "success": True,
            "query": "...",
            "num_sources": 3,
            "sources": [
                {"title": "...", "url": "...", "snippet": "...", "text": "..."},
                ...
            ]
        }
    """
```

### rag_search

```python
def rag_search(
    query: str,
    k: int = 4,
    vectorstore_dir: str = "vectorstore.db",
) -> dict[str, Any]:
    """
    Search documents using RAG.

    Returns:
        {
            "success": True,
            "query": "...",
            "num_documents": 4,
            "documents": [
                {"content": "...", "metadata": {...}, "rank": 1},
                ...
            ]
        }
    """
```

### weather_forecast

```python
def weather_forecast(
    city: str,
    units: str = "c",  # "c", "f", "k"
    lang: str = "kr",  # "kr", "en"
) -> dict[str, Any]:
    """
    Get 5-day weather forecast.

    Returns:
        {
            "success": True,
            "city": "Seoul",
            "units": "c",
            "forecast": {...}
        }
    """
```

### sentinel_search

```python
def sentinel_search(
    query: str,
    sensor: str | None = None,     # "S1", "S2", "S3"
    aoi: str | None = None,        # "Daejeon" or "126.5,36.0,127.5,37.0"
    date_range: str | None = None, # "2024-01-01/2024-01-31"
    cloud_cover_max: int | None = None,  # 0-100
) -> dict[str, Any]:
    """
    Search Sentinel satellite imagery.

    Returns:
        {
            "success": True,
            "query": "...",
            "results": {...}
        }
    """
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DeepAgents Agent                        │
│                   (create_deep_agent)                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  MCP Tools      │ │  Built-in Tools │ │  Other Tools    │
│  (mcp_tools.py) │ │  (filesystem,   │ │  (custom)       │
│                 │ │   todos, etc.)  │ │                 │
└────────┬────────┘ └─────────────────┘ └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Servers (stdio)                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│  Google Search  │  RAG Server     │  Weather Server         │
│  Web Fetch      │  (vectorstore)  │  (OpenWeatherMap)       │
│                 │                 │  Sentinel Server        │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Router Agent Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Route Query    │ ─── LLM analyzes query
│  (LLM-based)    │     and selects tools
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌───────┐ ┌──────┐ ┌───────┐ ┌────────┐
│Weather│ │ RAG  │ │Sentinel│ │GSearch │
└───┬───┘ └──┬───┘ └───┬───┘ └───┬────┘
    │        │         │         │
    └────────┴────┬────┴─────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Synthesize    │ ─── LLM combines
         │  Results       │     all tool outputs
         └────────┬───────┘
                  │
                  ▼
            Final Answer
```

## Troubleshooting

### MCP Server Connection Failed

```
Error: No MCP tools available
```

1. Check MCP server paths in `.env`
2. Ensure MCP servers are executable: `chmod +x /path/to/server.py`
3. Test server manually: `python /path/to/server.py`

### Weather API Error

```
Error: 401 Unauthorized
```

1. Check `OPENWEATHER_API_KEY` in `.env`
2. Verify API key at https://openweathermap.org/

### Async Runtime Error

```
RuntimeError: This event loop is already running
```

The tools use `nest_asyncio` to handle this. Ensure it's installed:
```bash
pip install nest-asyncio
```
