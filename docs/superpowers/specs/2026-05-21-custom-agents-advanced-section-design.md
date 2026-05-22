# Design: "Advanced" section for customization.mdx

**Date:** 2026-05-21  
**Target file:** `docs/src/oss/deepagents/customization.mdx` in the docs repo  
**Branch:** `sr/custom-agents-advanced`

---

## Goal

Users want to build truly custom deep agents — not just tweak the defaults. The existing `customization.mdx` is a good parameter reference but doesn't teach the *mental model*. We need a section that shows how `create_deep_agent` provides a minimal base harness that you layer capabilities onto.

---

## Narrative arc

Start with the most minimal possible agent and build it up step by step, each step adding one capability layer. Reader can stop wherever their needs are met; the final snippet is the fully composed agent.

```
create_deep_agent()             # model + tools in a loop — the base harness
  + system_prompt + tools       # Step 1: give it a job and domain tools
  + backend/sandbox             # Step 2: execution environment
  + summarization/memory/skills # Step 3: context management
  + subagents                   # Step 4: planning & delegation
  + interrupt_on                # Step 5: steering (human-in-the-loop)
= fully composed custom agent
```

---

## Section structure

### `## Advanced`

**Opening paragraph** (2–3 sentences):  
`create_deep_agent` is a base harness — at its core, just a model calling tools in a loop. Every other capability is opt-in. This section shows how to compose those capabilities layer by layer.

---

### `### Step 1: A prompt and tools`

What it covers: the simplest meaningful agent — custom `system_prompt` + domain `tools`. No middleware beyond the defaults.

Key point: this is already enough for many use cases. Everything after this is additive.

Example: a web research agent with a `tavily_search` tool and a focused system prompt.

---

### `### Step 2: Execution environment`

What it covers: giving the agent a place to work — filesystem, sandbox, or REPL.

Maps to: `backend=` param (StateBackend, FilesystemBackend, SandboxBackend), `permissions=`.

Key point: once the agent has a filesystem, it can offload context, persist artifacts, and run code — it can do work that outlasts a single context window.

Example: add `backend=SandboxBackend()` to the running example so the research agent can write and run code.

---

### `### Step 3: Context management`

What it covers: keeping the agent effective over long sessions — summarization, context offloading, prompt caching, memory (AGENTS.md), and skills.

Maps to: `SummarizationMiddleware` (default), `memory=`, `skills=`, `AnthropicPromptCachingMiddleware` (auto).

Key point: these are mostly automatic — FilesystemMiddleware offloads large tool results, SummarizationMiddleware compresses history — but `memory` and `skills` are opt-in and high-value for domain-specific agents.

Example: add `memory=["./AGENTS.md"]` and `skills=["./skills/"]` to the running example.

---

### `### Step 4: Planning & delegation`

What it covers: breaking work into parallel or sequential subtasks via subagents.

Maps to: `subagents=` param (declarative SubAgent specs), `SubAgentMiddleware` (automatic), `AsyncSubAgentMiddleware` for background work.

Key point: subagents isolate context — each gets a fresh window — so complex multi-step work doesn't exhaust the main agent's context.

Example: add a `researcher` subagent to the running example that handles deep-dive searches.

---

### `### Step 5: Steering`

What it covers: keeping a human in the loop at critical decision points.

Maps to: `interrupt_on=` param, `HumanInTheLoopMiddleware`.

Key point: interrupts are per-tool and configurable — you can require approval only for high-impact actions (write_file, execute) while letting reads run freely.

Example: add `interrupt_on={"write_file": True, "execute": True}` + checkpointer to the running example.

---

### Final assembled example

Complete `create_deep_agent(...)` call with all 5 layers, preceded by a short "putting it all together" sentence. No new explanation — just the code, with comments marking each layer.

---

## Running example domain

**Research + code agent**: starts as a web researcher (Step 1), gains a sandbox to run code (Step 2), gets memory/skills for domain expertise (Step 3), delegates deep searches to a subagent (Step 4), and requires approval before writing files (Step 5).

Chosen because:
- It's realistic (maps to the deep_research and deploy-coding-agent examples in the repo)
- Each layer obviously adds value — the motivation for each step is self-evident
- Python-only for this draft (JS parity can follow)

---

## Format conventions

- Follows existing `customization.mdx` style: `:::python` blocks, Mintlify MDX
- No new snippet files — inline code only (the examples are short enough)
- Each step opens with 1 sentence explaining what it adds, ends with the updated code block
- Cross-links to existing detail pages (sandboxes, memory, skills, subagents, HITL) — don't duplicate their content

---

## What this section is NOT

- Not a replacement for the per-feature detail pages
- Not a comprehensive middleware reference (that's the existing "Middleware" section above)
- Not JS docs (out of scope for this PR; noted as follow-up)

---

## Out of scope / follow-up

- JavaScript parity
- Async subagents example
- Provider profiles in the advanced section
