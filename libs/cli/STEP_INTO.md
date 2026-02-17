# Step-Into Subagents: Interactive Context Branching

## Manifesto: Reclaiming Control in Agentic Systems

### The False Equivalence

The current paradigm of AI agent delegation rests on a flawed assumption: that **context isolation requires surrendering control**.

When we delegate to a subagent today, we fire a task into the void and wait for results. We call this "autonomy." But autonomy without oversight is abandonment. We've confused giving agents space to work with cutting the communication channel entirely.

**Context isolation** is a technical constraint. Tokens are finite. Verbose exploration pollutes focused conversation. Memory has boundaries.

**Control** is a human need. We steer. We course-correct. We say "not that direction" and "dig deeper here." We are not task-specifiers; we are collaborators.

These are orthogonal concerns. The industry has collapsed them into one.

### The Batch Processing Trap

Current subagent architectures are batch jobs wearing agent costumes:

```
INPUT → [BLACK BOX] → OUTPUT
```

You specify everything upfront. The agent runs. You receive results. If the results are wrong, you start over with better instructions.

This is 1960s computing with better marketing.

True collaboration looks different:

```
HUMAN ←→ AGENT ←→ HUMAN ←→ AGENT
         ↓
    [isolated context]
         ↓
      summary
```

The context can be isolated. The channel should not be.

### Principles

**1. Isolation is Memory, Not Communication**

Context isolation means: "Your working memory is separate from mine."
It does not mean: "We cannot speak."

A subagent should have its own context window, its own scratchpad, its own exploration space. But messages should flow. Questions should be askable. Guidance should be givable.

**2. Delegation is Not Abdication**

When you delegate to a human colleague, you don't seal them in a room and wait for a report. You check in. You redirect. You say "actually, focus on X instead."

Agents deserve the same interactive relationship. Delegation should be a tether, not a severance.

**3. Summarization is Export, Not the Only Exit**

The current model forces all subagent work through a summarization bottleneck. Everything must be compressed to return to the parent context.

But sometimes you want to *enter* the subagent's context. Explore what it found. Ask follow-up questions in its space. Then return with what *you* decide is relevant.

The summary should be one option, not the only option.

**4. Control Has Granularity**

Not all tasks need the same level of oversight:

- **Fire-and-forget**: "Search for X and summarize" - current model works fine
- **Checkpoint-based**: "Explore, but check with me before making decisions"
- **Interactive**: "Let's explore this together in a separate context"
- **Supervised**: "Do this, but stream your reasoning so I can interrupt"

One interaction model cannot serve all needs.

---

*Context isolation is about memory.*
*Control is about agency.*
*They were never the same thing.*

---

## How It Works

When the agent invokes the `task` tool to delegate work to a subagent, you now see a new option:

```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Task: Subagent Invocation                                │
│                                                             │
│ Subagent Type: general-purpose                              │
│                                                             │
│ Task Instructions:                                          │
│ ────────────────────────────────────────                    │
│ Research how JWT refresh tokens work in this codebase...    │
│ ────────────────────────────────────────                    │
│                                                             │
│ [Approve]  [Reject]  [Step into]  [Auto-accept all]         │
└─────────────────────────────────────────────────────────────┘
```

| Option | Behavior |
|--------|----------|
| **Approve** | Fire-and-forget (existing behavior) |
| **Reject** | Don't run the subagent |
| **Step into** | Enter the subagent conversation interactively |
| **Auto-accept all** | Approve all future tool calls |

## Usage

### Stepping Into a Subagent

1. When the agent wants to delegate, select **"Step into"** (press `S` or arrow down)
2. You enter an interactive session with the subagent
3. The prompt changes to show your depth: `[general-purpose:1] >`
4. Work with the subagent as you would normally
5. A summary file is created at `~/.deepagents/{agent}/branches/{id}/summary.md`

### New Commands

| Command | Description |
|---------|-------------|
| `/return` | Exit subagent, send summary to parent, return to previous context |
| `/summary` | View or edit the summary file (opens in `$EDITOR` if set) |
| `/context` | Show your current position in the context stack |

### Example Session

```
You: Research the authentication system in this codebase

Agent: I'll delegate this to a subagent for focused exploration.

[HITL prompt appears - select "Step into"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stepped into: general-purpose subagent
Summary file: ~/.deepagents/agent/branches/abc123/summary.md
Type /return to exit, /summary to edit summary, /context to see stack
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[general-purpose:1] > The agent starts working...

Agent: Found 3 auth approaches - JWT, session-based, and OAuth.
       Which should I focus on?

[general-purpose:1] > JWT. Ignore the others.

Agent: JWT implementation spans 4 files. The token refresh logic
       looks suspicious. Want me to dig into that?

[general-purpose:1] > Yes, and check if there are tests for it.

Agent: [explores, finds issues, reports back]

[general-purpose:1] > Add your findings to the summary.

Agent: [edits summary.md]

[general-purpose:1] > /return

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Returned to root
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Agent: Based on the subagent's findings, the JWT implementation has...
       [continues with the summary you helped craft]
```

### Viewing Context Stack

```
> /context
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context Stack:
  [0] root (main conversation)
  [1] general-purpose ← current
      task: "Research authentication system"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary: ~/.deepagents/agent/branches/abc123/summary.md
```

### Nested Step-Into

You can step into subagents from within subagents:

```
[general-purpose:1] Agent: I'll delegate the OAuth part to another subagent.

[Select "Step into" again]

[general-purpose:2] > Now you're two levels deep

[general-purpose:2] > /return  ← returns to depth 1

[general-purpose:1] > /return  ← returns to root
```

## Summary File

When you step into a subagent, a summary file is created:

**Location:** `~/.deepagents/{agent}/branches/{branch_id}/summary.md`

**Template:**
```markdown
# Branch Summary: general-purpose

## Task
[Original task description from parent]

## Findings

<!-- Add your findings here -->

## Conclusion

<!-- Summary for parent context -->
```

You can:
- Ask the agent to update it: "Add this to the summary"
- Edit it directly: `/summary` opens it in your `$EDITOR`
- View it: `/summary` displays contents if no editor is set

When you `/return`, this file's contents are sent to the parent agent.

## Comparison

| Aspect | Fire-and-Forget | Step Into |
|--------|-----------------|-----------|
| **Control** | None - wait for result | Full - guide the exploration |
| **Visibility** | Black box | See every step |
| **Course correction** | Start over with new prompt | Redirect mid-task |
| **Summary quality** | Agent decides what's important | You decide together |
| **Context isolation** | Yes | Yes |
| **Token efficiency** | Yes | Yes |

## Design Decisions

| Aspect | Decision |
|--------|----------|
| **Persistence** | Ephemeral - branches exist only in current session |
| **Inheritance** | Full - subagent has same tools/middleware as parent |
| **`/clear` behavior** | Resets all contexts back to root |

## Technical Details

- Uses LangGraph's thread-based checkpointing for context isolation
- Context stack tracks nested conversations
- Summary injection uses `agent.aupdate_state()` on return
- Same agent instance, different thread IDs per context
