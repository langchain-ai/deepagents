# MuAPI Creative Agent

A generative-media Deep Agent powered by [muapi.ai](https://muapi.ai) — 390+ models for images, video, audio, 3D, and more, exposed as LangChain tools and wired into a Deep Agent with a planner/specialist split.

**This example demonstrates how to build a media-generation agent through three filesystem primitives:**
- **Memory** (`AGENTS.md`) — persistent context: available models, decision tree, quality rules
- **Skills** (`skills/*/SKILL.md`) — loaded-on-demand workflows for generation and named recipes
- **Subagents** (`subagents.yaml`) — a `creative-specialist` for heavy multi-step work

## Quick Start

```bash
# Set API keys
export MUAPI_API_KEY="..."          # Get one at muapi.ai
export ANTHROPIC_API_KEY="..."

# Run (uv installs dependencies automatically)
cd examples/creative-agent
uv run python creative_agent.py "Generate a cinematic product photo of sneakers"

# With a budget cap
uv run python creative_agent.py --budget 200 "Animate my product image into a 5s video"

# Multi-step recipe
uv run python creative_agent.py "Make a 3-shot Instagram carousel for SunFizz mango water"
```

## How It Works

```
User brief
  │
  ├─ AGENTS.md          Loaded at startup — tells the agent what muapi can do
  ├─ skills/            Loaded on demand — step-by-step tool usage guides
  │
  ▼
Planner (Claude Sonnet)
  ├─ muapi_select       Discover best model/skill for the brief (free)
  ├─ muapi_generate     Single-shot generation: image / video / audio / 3D
  │
  └─ task ──────────────► creative-specialist subagent
                              ├─ muapi_run_skill     Named multi-step recipe
                              └─ muapi_creative_agent  Open-ended brief (HITL-gated)
```

`MuapiCostCallback` tracks credit spend in real time and aborts the run if the budget cap is hit.

## Tool Capability Gradient

| Tool | Tier | What it does | Credits |
|------|------|-------------|---------|
| `muapi_select` | Planner | Rank models + skills for an intent | Free |
| `muapi_generate` | Planner | Generate one asset (any modality) | Per call |
| `muapi_run_skill` | Specialist | Run a named multi-step recipe | Per recipe |
| `muapi_creative_agent` | Specialist | Hand a full brief to muapi's planner (HITL-gated) | Variable |

## File Structure

```
creative-agent/
├── AGENTS.md                     # Persistent agent context
├── creative_agent.py             # Main entry point
├── subagents.yaml                # creative-specialist definition
├── pyproject.toml
└── skills/
    ├── generate-asset/SKILL.md   # Single-asset generation workflow
    └── run-skill/SKILL.md        # Named recipe delegation workflow
```

## Environment Variables

| Variable | Required | Notes |
|----------|----------|-------|
| `MUAPI_API_KEY` | Yes | Get at [muapi.ai](https://muapi.ai) or run `muapi auth configure` |
| `ANTHROPIC_API_KEY` | Yes | Powers the planner (Claude Sonnet) |

## Related

- [muapi-langchain on PyPI](https://pypi.org/project/muapi-langchain/)
- [muapi.ai](https://muapi.ai) — API docs and model catalog
- [muapi CLI](https://github.com/SamurAIGPT/muapi-cli) — same auth, same client
