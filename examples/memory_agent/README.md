# Memory Agent

A deep agent that **improves over time** through learned memory. Demonstrates the runtime advantage of combining Deep Agents + LangGraph deployment + observability — the agent gets better the more you use it.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Memory Agent                          │
│                                                           │
│  System Prompt = base instructions                        │
│                + memory editing guidelines                 │
│                                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │            CompositeBackend                         │   │
│  │                                                     │   │
│  │  /memories/global.md ──► StoreBackend (persistent)  │   │
│  │  /memories/user.md   ──► StoreBackend (persistent)  │   │
│  │  everything else     ──► StateBackend (ephemeral)   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                           │
│  Agent reads/writes /memories/ live during conversations  │
└──────────────────────────┬───────────────────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  Cron Job (3am)   │
                  │  per-user         │
                  │                   │
                  │  1. Collect threads│
                  │  2. Analyze       │
                  │  3. Update memory │
                  └───────────────────┘
```

### Components

1. **Agent** (`agent.py`) — Deep agent with a `CompositeBackend` that routes:
   - `/memories/` → `StoreBackend` (persistent across threads via LangGraph Store)
   - Everything else → `StateBackend` (ephemeral per-thread)

   The agent can read and write memory files at any time during a conversation:
   - `/memories/global.md` — patterns learned across ALL users
   - `/memories/user.md` — preferences and context specific to the current user

2. **Cron Job** (`cron.py`) — Sleep-time compute, designed as **one invocation per user**:
   - `consolidate_user(store, user_id)` — the primary unit of work, runs independently per user
   - Collects that user's recent threads, analyzes with an LLM, updates both memory files
   - Scales horizontally — in production (LangGraph Cloud), each user gets their own cron trigger
   - `run_all_users()` — convenience wrapper for local dev that discovers all users and parallelizes via thread pool

3. **Eval** (`eval.py`) — Day-N evaluation framework that answers: *"Does the agent get better over time?"*
   - Simulates N days of usage with realistic conversations
   - Runs memory consolidation between days
   - Measures response quality with an LLM judge
   - Reports improvement trajectory (Day 1 score → Day N score)

## Setup

```bash
# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

### Deploy with LangGraph

```bash
langgraph up
# or
langgraph dev
```

The `langgraph.json` configures both the agent graph and the nightly cron job. No checkpointer needed — LangGraph adds one automatically.

### Run the agent locally

```python
from agent import create_memory_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
agent = create_memory_agent(store=store, user_id="alice")

# First conversation — no memory yet
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Help me write a status update"}]},
    config={"configurable": {"thread_id": "thread-1"}},
)
```

### Run memory consolidation manually

```bash
# Single user (production pattern — one invocation per user)
python cron.py --user alice

# All users (local dev convenience)
python cron.py --all
```

### Run the day-N evaluation

```bash
# Default: 3 days, all tasks
python eval.py

# Custom configuration
python eval.py --days 5 --tasks-per-day 3 --model anthropic:claude-sonnet-4-6
```

## How Memory Works

### Live Memory Editing

The agent has a persistent `/memories/` folder backed by LangGraph Store via `CompositeBackend`. During any conversation, the agent can:

- **Read** `/memories/global.md` and `/memories/user.md` to recall learned context
- **Write/edit** these files to save new learnings immediately

The system prompt includes guidelines for when to update each file:

| File | When to update |
|------|---------------|
| `/memories/global.md` | Patterns useful for ALL users: common task types, output formats, domain knowledge, tool usage patterns, mistakes to avoid |
| `/memories/user.md` | User-specific info: stated preferences, role/team/projects, communication style, corrections to agent behavior |

The agent updates memory proactively — it reads the existing file first, merges new information, and writes back. No duplicates, no announcements to the user.

### How the CompositeBackend Routes Files

```python
CompositeBackend(
    default=StateBackend(runtime),          # ephemeral per-thread
    routes={
        "/memories/": StoreBackend(runtime), # persistent across threads
    },
)
```

- Files under `/memories/` are stored in LangGraph Store and persist across all conversations
- All other files (todos, scratch work, etc.) use the default `StateBackend` and are ephemeral

### Sleep-Time Consolidation

The cron job uses an LLM to analyze conversation threads and extract memories. Each user's consolidation is an independent invocation:
1. Reads that user's recent threads
2. Loads current `/memories/global.md` and `/memories/user.md`
3. Asks an LLM to produce updated, consolidated memories
4. Writes the updated memories back to the store

This is designed to scale: in production, each user gets their own cron trigger (via LangGraph Cloud), so consolidation runs as many independent jobs in parallel as you have users. No single process bottleneck.

The key insight: **the agent doesn't need to be running to learn**. Background compute during "sleep time" processes raw conversations into structured, useful memory.

### Two Paths to Memory

Memory gets updated through two complementary paths:

1. **Live** — Agent writes to `/memories/` during conversations when it notices something worth remembering
2. **Cron** — Background consolidation processes conversation history to extract patterns the agent may have missed

Both write to the same store, so they compose naturally.

## Disabling Live Memory Editing

If you don't want the agent to edit its own memory during conversations (e.g., you only want cron-driven updates), use the `enable_live_memory=False` flag:

```python
from agent import create_memory_agent

agent = create_memory_agent(enable_live_memory=False)
```

This removes the `/memories/` route from the backend and strips the memory editing instructions from the system prompt. The agent will still benefit from memories written by the cron job (injected into the system prompt at conversation start), but it won't modify them during the conversation.

To modify the default agent for LangGraph deployment, edit `agent.py` and remove the `backend=create_backend` parameter:

```python
# Before (live memory enabled)
agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=AGENT_INSTRUCTIONS,
    backend=create_backend,
)

# After (cron-only memory)
agent = create_deep_agent(
    model=model,
    tools=[],
    system_prompt=AGENT_INSTRUCTIONS,  # swap to AGENT_INSTRUCTIONS_NO_LIVE
)
```

## The Day-N Eval

Traditional agent evals measure static performance. The day-N eval measures something more interesting: **does the agent improve with use?**

```
Day 1: avg score = 2.40  (no memory — generic responses)
Day 2: avg score = 3.60  (some memory — starting to personalize)
Day 3: avg score = 4.20  (rich memory — tailored responses)

Improvement: 2.40 → 4.20 (Δ = +1.80)
✅ Agent improved over time!
```

The eval simulates realistic multi-day usage:
- **Day 1**: Agent has no memory, gives generic responses
- **Between days**: Simulated conversations are stored, cron consolidates them
- **Day N**: Agent has accumulated memory, responses should be more personalized

Tasks span multiple categories:
- **Personalization** — Does it know the user's context?
- **Style adaptation** — Does it match preferred communication style?
- **Domain knowledge** — Does it remember technical details?
- **Code review** — Does it apply learned coding preferences?
