# Deep Research Agent (Deploy Example)

A research orchestrator that uses specialized subagents to produce comprehensive, well-sourced research reports.

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Orchestrator │  (Sonnet) — decomposes question, coordinates subagents
│  AGENTS.md   │
└──────┬───────┘
       │
       ├──── task(researcher) ──── task(researcher) ──── ...
       │          │                      │
       │          ▼                      ▼
       │    /research/q1.md        /research/q2.md
       │                                           ← shared sandbox
       ├──── task(synthesizer)
       │          │
       │          ▼
       │    /output/report.md
       │
       ▼
  Final Report
```

### Subagents

| Agent | Model | Role | Tools |
|-------|-------|------|-------|
| **researcher** | Haiku | Fast web research, source evaluation | Web search (Tavily MCP), sandbox filesystem |
| **synthesizer** | Sonnet | Cross-reference findings, write report | Sandbox filesystem |

Both subagents share the same sandbox (scoped per thread), so the researcher writes files that the synthesizer can read.

## Project Layout

```
deploy-deep-research/
    AGENTS.md                   # orchestrator system prompt
    deepagents.toml             # main config (Sonnet, LangSmith sandbox)
    .env.example                # required API keys
    skills/
        research-plan/SKILL.md  # skill for decomposing questions
    agents/
        researcher/
            AGENTS.md           # researcher system prompt
            deepagents.toml     # Haiku model, LangSmith sandbox
            mcp.json            # Tavily web search
            skills/
                search-strategy/SKILL.md
        synthesizer/
            AGENTS.md           # synthesizer system prompt
            deepagents.toml     # Sonnet model, LangSmith sandbox
            skills/
                report-format/SKILL.md
```

## Setup

```bash
cd examples/deploy-deep-research

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY, LANGSMITH_API_KEY, TAVILY_API_KEY

# Deploy
deepagents deploy
```

## Usage

Ask a research question:

> "What are the tradeoffs between RAG and long-context models for enterprise knowledge management in 2025?"

The orchestrator will:
1. Break this into sub-questions (RAG architectures, long-context capabilities, cost comparison, enterprise requirements)
2. Send each to the researcher subagent for web research
3. Hand all findings to the synthesizer for a structured report
4. Review and deliver the final report
