# Memory Agent

A deep agent that **improves over time** through learned memory. Demonstrates the runtime advantage of combining Deep Agents + LangGraph deployment + observability — the agent gets better the more you use it.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Memory Agent                       │
│                                                      │
│  System Prompt = base instructions                   │
│                + global memory (all users)            │
│                + user memory (per user)               │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ Global Memory │  │ User Memory  │                  │
│  │ (Store)       │  │ (Store)      │                  │
│  └──────┬───────┘  └──────┬───────┘                  │
│         │                  │                          │
│         └────────┬─────────┘                          │
│                  │                                    │
│         ┌────────▼────────┐                           │
│         │  LangGraph Store │                          │
│         │  (persistent)    │                          │
│         └────────┬────────┘                           │
│                  │                                    │
└──────────────────┼──────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   Cron Job (3am)    │
         │                     │
         │  1. Collect threads  │
         │  2. Analyze patterns │
         │  3. Update memories  │
         └─────────────────────┘
```

### Components

1. **Agent** (`agent.py`) — Deep agent with dynamic system prompt built from:
   - Base instructions
   - **Global memory** — patterns learned across all users (e.g., "users prefer bullet points", "always include error handling in code reviews")
   - **Per-user memory** — personalized context (e.g., "this user leads the platform team", "prefers concise responses")

2. **Cron Job** (`cron.py`) — Sleep-time compute that runs on a schedule to:
   - Collect recent conversation threads
   - Analyze them with an LLM to extract useful patterns
   - Update global and per-user memory in the store
   - Merge new learnings with existing memory (no duplicates)

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

### Deploy with LangGraph

```bash
langgraph up
```

The `langgraph.json` configures both the agent graph and the nightly cron job.

### Run memory consolidation manually

```bash
python cron.py
```

### Run the day-N evaluation

```bash
# Default: 3 days, all tasks
python eval.py

# Custom configuration
python eval.py --days 5 --tasks-per-day 3 --model anthropic:claude-sonnet-4-6
```

## How Memory Works

### Global Memory
Stored at namespace `("memory", "global")`. Contains patterns like:

```markdown
## Response Style
- Users generally prefer bullet-point format over paragraphs
- Include concrete examples when explaining technical concepts

## Common Patterns
- When reviewing code, always check error handling first
- For status updates, use the blockers → progress → next steps format
```

### Per-User Memory
Stored at namespace `("memory", "users", "<user_id>")`. Contains:

```markdown
## User Profile
- Senior engineer, platform team lead
- Reports to Sarah Chen (VP Engineering)

## Preferences
- Concise, no fluff
- Python with type hints
- Exponential backoff with jitter for retries (shared util in libs/retry.py)
```

### Sleep-Time Consolidation
The cron job uses an LLM to analyze conversation threads and extract memories. It:
1. Reads all recent threads for each user
2. Loads current global + user memory
3. Asks an LLM to produce updated, consolidated memories
4. Writes the updated memories back to the store

This is the key insight: **the agent doesn't need to be running to learn**. Background compute during "sleep time" processes raw conversations into structured, useful memory.

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
