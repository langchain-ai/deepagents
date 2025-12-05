# Using the CLI Agent with RAG (Semantic Code Search)

The deepagents CLI agent automatically includes RAG (Retrieval-Augmented Generation) for semantic code search when `OPENAI_API_KEY` is set.

## Quick Start

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-key-here'
   ```

2. **Run the CLI agent:**
   ```bash
   deepagents
   # or
   python -m deepagents_cli
   ```

3. **The agent now has access to semantic search!**

## Using Semantic Search in the CLI

Once the CLI agent is running, you can ask it to search code semantically:

```
> Find code that handles user authentication
> Search for error handling patterns in the codebase
> Where is database connection management implemented?
> Show me code related to API routing
```

The agent will automatically use the `semantic_search` tool to find relevant code by meaning, not just by keyword matching.

## How It Works

1. **Automatic Integration**: The semantic search tool is automatically added to the agent when `OPENAI_API_KEY` is set (see `main.py` line 296-305).

2. **First-Time Indexing**: When you first use semantic search on a folder, it will automatically index the codebase.

3. **Caching**: Indexes are cached in `~/.deepagents/rag_cache/` per workspace, so subsequent searches are fast.

4. **Works with Any Folder**: You can search any folder by specifying the `workspace_root` parameter when the agent uses the tool.

## Example CLI Session

```
$ deepagents
DeepAgents CLI - AI Coding Assistant
✓ Semantic search (RAG) enabled

> Find all authentication-related code in the current directory

[Agent uses semantic_search tool]
Found 5 relevant code chunks:
- libs/deepagents-cli/deepagents_cli/agent.py (score: 0.23)
  Contains: authentication logic, API key validation
  
- libs/deepagents-cli/deepagents_cli/config.py (score: 0.31)
  Contains: API key configuration, settings management
...

> Search for error handling in ../other-project

[Agent uses semantic_search with workspace_root="../other-project"]
...
```

## Requirements

- `OPENAI_API_KEY` environment variable set
- Dependencies installed: `langchain-community`, `langchain-openai`, `faiss-cpu`, `langchain-text-splitters`

These are automatically installed when you install deepagents-cli with:
```bash
uv sync
# or
pip install -e .
```

## Manual Tool Usage

The agent will automatically use semantic search when appropriate, but you can also explicitly request it:

```
> Use semantic search to find code that processes user input
> Search the codebase semantically for database query functions
```

The agent understands these requests and will use the `semantic_search` tool with appropriate parameters.
