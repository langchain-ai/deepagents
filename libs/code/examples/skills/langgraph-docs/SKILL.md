---
name: langgraph-docs
description: Fetches and references LangGraph Python documentation to build stateful agents, create multi-agent workflows, and implement human-in-the-loop patterns. Use when the user asks about LangGraph, graph agents, state machines, agent orchestration, LangGraph API, or needs LangGraph implementation guidance.
---

# langgraph-docs

## Workflow

### 1. Fetch the Documentation Index

Use `fetch_url` to read: https://docs.langchain.com/llms.txt

This returns a structured list of all available documentation with descriptions.

### 2. Select Relevant Documentation

Identify 2-4 most relevant URLs from the index. Prioritize:
- **Implementation questions** — specific how-to guides
- **Conceptual questions** — core concept pages
- **End-to-end examples** — tutorials
- **API details** — reference docs

### 3. Fetch and Apply

Use `fetch_url` on the selected URLs, then complete the user's request using the documentation content.

If `fetch_url` fails or returns empty content, retry once. If it fails again, inform the user and suggest checking https://langchain-ai.github.io/langgraph/ directly.
