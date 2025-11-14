# Deepagent Quickstarts 

## Overview

[Deepagents](https://github.com/langchain-ai/deepagents) is a simple, opinionated agent harness build on top of [LangGraph](https://github.com/langchain-ai/langgraph). It features a few general, built-in tools and middleware that are useful for building many type of agents. 

These general tools support various approaches for context engineering. They allow the agents to offload context (e.g., to files) when neccesary and isolate context (e.g., to subagents). Deepagents also middleware (hooks) to process tool results (e.g., summarization to reduce the size of context passed back to the LLM). 

This repo has a collection of quickstarts that demonstrate different agents that can he easily configured using the `deepagents` package.

### Tools

Every deepagent comes with a set of general tools that we've seen commonly used across popular agents such as [Manus](https://rlancemartin.github.io/2025/10/15/manus/) and [Claude Code](https://www.claude.com/product/claude-code): give agents  a small number of general atomic tools that allow them to use a computer (both filesystem and shell) as well as plan and delegate work. The following core tools by default:

| Tool Name | Description |
|-----------|-------------|
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows |
| `ls` | List all files in a directory (requires absolute path) |
| `read_file` | Read content from a file with optional pagination (offset/limit parameters) |
| `write_file` | Create a new file or completely overwrite an existing file |
| `edit_file` | Perform exact string replacements in files |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) |
| `grep` | Search for text patterns within files |
| `execute` | Run shell commands in a sandboxed environment (only if backend supports SandboxBackendProtocol) |
| `task` | Delegate tasks to specialized sub-agents with isolated context windows |

### Middleware

Middleware components extend agent capabilities by providing tools and implementing hooks that process model and tool interactions. Each middleware can:

1. **Provide tools** - Add new tools to the agent's toolkit (e.g., `FilesystemMiddleware` adds `ls`, `read_file`, `write_file`, etc.)
2. **Wrap model calls** - Inject system prompts and modify model requests before they're sent
3. **Wrap tool calls** - Process tool call results after tools execute (e.g., `SummarizationMiddleware` summarizes large conversation history)

Every deepagent includes the following middleware by default (applied in order). Some middleware are provided by the `deepagents` package (`FilesystemMiddleware`, `SubAgentMiddleware`, `PatchToolCallsMiddleware`), while others come from `langchain` (`TodoListMiddleware`, `SummarizationMiddleware`, `HumanInTheLoopMiddleware`) and `langchain-anthropic` (`AnthropicPromptCachingMiddleware`):

| Middleware | Tools Added | Where It Acts | What It Does |
|------------|-------------|---------------|--------------|
| **TodoListMiddleware** | `write_todos`, `read_todos` | `wrap_model_call`, `before_agent` | Provides task planning and progress tracking tools. Enables agents to create structured todo lists, break down complex tasks into steps, and track completion status. Injects todo usage instructions into system prompt and makes current todo state available to the agent. |
| **FilesystemMiddleware** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`* | `wrap_model_call`, `wrap_tool_call` | Provides file system operations and context offloading. Adds filesystem tools for reading, writing, and searching files. In `wrap_model_call`: Injects filesystem instructions into system prompt and filters out `execute` tool if backend doesn't support SandboxBackendProtocol. In `wrap_tool_call`: Intercepts large tool results (>20,000 tokens), automatically saves them to files, and returns summaries instead of full content to prevent context overflow. |
| **SubAgentMiddleware** | `task` | `wrap_model_call` | Enables task delegation to specialized subagents with isolated contexts. Provides the `task` tool for spawning ephemeral subagents that can handle complex, multi-step tasks independently. Subagents run with their own context windows and return concise summaries, preventing context pollution in the main agent. Injects detailed instructions about when and how to use subagents effectively (parallel execution, context isolation, etc.). |
| **SummarizationMiddleware** | N/A | `before_agent` | Prevents context window overflow via automatic summarization. Monitors conversation history token count before each agent turn. When tokens exceed 170,000, summarizes older messages while keeping the last 6 messages intact. Replaces summarized portion with a concise summary, maintaining conversation continuity while freeing up context space. |
| **AnthropicPromptCachingMiddleware** | N/A | `wrap_model_call` | Reduces API costs through prompt caching (Anthropic models only). Adds cache control headers to system prompts, marking static content for caching on Anthropic's servers. Configured with `unsupported_model_behavior="ignore"` to allow non-Anthropic models to work without errors. Significantly reduces costs for repeated system prompts by reusing cached prefixes. |
| **PatchToolCallsMiddleware** | N/A | `before_agent` | Fixes "dangling" tool calls from interrupted operations. Scans message history before each agent turn to find AIMessages with tool_calls that lack corresponding ToolMessages. Adds placeholder ToolMessages (e.g., "Tool call X was cancelled - another message came in before it could be completed") to prevent LangGraph validation errors. Essential for handling user interruptions and double-texting scenarios. |
| **HumanInTheLoopMiddleware** | N/A | `wrap_tool_call` | Enables human approval for sensitive operations. Intercepts tool calls for tools specified in the `interrupt_on` configuration. Creates LangGraph interrupts/breakpoints that pause execution and wait for human approval or rejection. Requires a checkpointer to maintain state across interruptions. Only included in deepagents when `interrupt_on` parameter is provided to `create_deep_agent()`. |

\* The `execute` tool is only available if the backend implements `SandboxBackendProtocol`

For each agent turn, hooks execute in this sequence:

  1. before_agent (all middleware)
     ├─ PatchToolCallsMiddleware: Fix dangling tool calls
     └─ SummarizationMiddleware: Summarize if needed

  2. wrap_model_call (all middleware)  
     ├─ FilesystemMiddleware: Inject filesystem instructions
     ├─ SubAgentMiddleware: Inject subagent instructions
     └─ AnthropicPromptCachingMiddleware: Add cache headers

  3. [Model generates response with tool calls]

  4. wrap_tool_call (all middleware, for each tool call)
     ├─ FilesystemMiddleware: Evict large results to files
     └─ HumanInTheLoopMiddleware: Pause for approval if configured

## Quickstarts

For each quickstart, we can amend the core deepagent harness with any custom tools, middleware, and / or instructions.

| Quickstart Name | Location | Description |
|----------------|----------|-------------|
| Deep Research | `examples/deep_research/` | A research agent that conducts multi-step web research using parallel sub-agents, strategic reflection, and context offloading to virtual files |

### Deep Research  

#### Instructions

The deep research agent uses several specialized instruction sets defined in `deep_research/research_agent/prompts.py`:

| Instruction Set | Purpose |
|----------------|---------|
| `RESEARCHER_INSTRUCTIONS` | Guides research sub-agents to conduct focused web searches with hard limits (2-3 searches for simple queries, max 5 for complex). Emphasizes strategic thinking, stopping when adequate information is gathered. |
| `TODO_USAGE_INSTRUCTIONS` | Instructs agents to create research plans as TODO lists, batch research tasks into single TODOs, and iteratively mark tasks complete while reflecting on progress. |
| `FILE_USAGE_INSTRUCTIONS` | Defines workflow for virtual filesystem: orient with `ls()`, save user requests, conduct research (tools auto-write files), then read files to synthesize answers. |
| `SUBAGENT_USAGE_INSTRUCTIONS` | Explains how to delegate research tasks to parallel sub-agents. Includes scaling rules: single agent for simple queries, multiple agents for comparisons/multi-faceted research. Limits iterations and emphasizes stopping when adequate information is collected. |

#### Tools

The deep research agent adds the following custom tools beyond the core deepagent tools:

| Tool Name | Description |
|-----------|-------------|
| `tavily_search` | Web search with context offloading: performs searches using Tavily API, fetches full webpage content via HTTP, converts HTML to markdown, saves full content to files, and returns only minimal summaries to avoid context overflow. Uses LangGraph `Command` to update both files and messages. |
| `think_tool` | Strategic reflection mechanism that helps the agent pause and assess progress between searches, analyze findings, identify gaps, and plan next steps. |
