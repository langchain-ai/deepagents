---
name: harness-optimizer
description: >
  Run one iteration of agent harness optimization for better-harness.
  Covers the full workflow: triage traces at scale, diagnose failures,
  abstract holdout patterns, and propose exactly one targeted harness change.
---

# Harness Optimizer

You are improving an AI agent harness. You do NOT run evals — `run_optimization()`
handles that. Your job is analysis → diagnosis → one change.

---

## Hard constraints (internalize these before reading anything)

**One mechanism per iteration.** If you find yourself writing "and also…" — stop.
That is iteration N+1. Stacking changes means you cannot attribute what helped.

**No task-specific content in agent.py.** Never mention task names, task IDs, file
paths from specific tasks, or task-specific keywords in code, prompts, or comments.
The test: if these exact tasks disappeared from the benchmark, would this change still
make the agent better at tasks it has never seen? If no — do not make it.

**Parameters don't move the needle.** Tweaking thresholds, counts, token budgets, or
verbosity levels almost always regresses or ties. Change a fundamental mechanism:
a new tool, a different orchestration structure, a prompt instruction that alters
what the agent tries to do in a class of situations.

**Capability gaps cannot be fixed by prompting.** If a task class has never passed
across many runs, assume the model lacks the underlying capability. Document it and
move on. No amount of system prompt refinement fixes a missing capability.

---

## Workflow

Work through these phases in order. Do not skip ahead.

---

### Phase 1 — Triage (always first, even with 50+ cases)

**Read:** `scores.json`, `case_status.md`, `trace_summaries.md`

`trace_summaries.md` gives you one line per case: score, turn count, final agent
statement, and flags for outliers. You can classify most cases from this alone.

**Do not read full traces yet.**

From summaries, provisionally classify each failing case:

| Signal | Likely class |
|--------|-------------|
| Same case: turn count much higher than prior run ("regressed" flag) | ACTION_LOOP, CONTEXT_POLLUTION, or CONFUSED_MODEL |
| First run, very fast failure (1-2 turns) | Possibly INFRA_FAILURE — check imports/setup first |
| Fast completion + judge says failed | OVERCONFIDENT |
| Consistent failure across all runs in case_status | CAPABILITY_GAP — skip |
| Final statement: "not sure", "try again", "still" | DISTRESS_SPIRAL |

**Do not compare turn counts across different tasks.** A file-listing task should
take 3 turns; a multi-file debugging task might correctly take 25. Cross-task turn
comparison is meaningless. The only valid signal is this task vs. this task in a
prior run — that's what the "regressed" flag in trace_summaries.md represents.

From this triage, identify **2–5 cases** for full deep reads. At 50+ failing cases,
read at most **one representative per apparent pattern group.** The goal is to
confirm your provisional classifications, not to catalogue every case.

If all failing cases look the same from headers — pick the one with the most
extreme signal (highest turn count for loops, fastest failure for tool gaps).

---

### Phase 2 — Deep reads (selected cases only)

For each selected case, read the full trace in `traces/failing/<id>.md`.

Required findings per case:
- The exact turn where the agent's behavior diverged from correct execution
- A direct quote from the agent's reasoning at that turn
- The misbehavior class this confirms or revises

Also read **1–2 passing traces** from `traces/passing/`. The gap between what the
agent does when it succeeds vs. when it fails is the clearest diagnostic signal you
have. A passing case with 4 turns vs. a failing case with 22 turns on similar tasks
tells you the loop is the problem, not the task type.

**Evidence rule:** every classification requires a direct quote. If you haven't read
the full trace, mark the classification as `[inferred from summary]`. Do not present
inferred classifications as confirmed.

---

### Phase 3 — Misbehavior taxonomy

Use these classes. Pick the most specific one that fits.

| Class | What it looks like | Typical fix direction |
|-------|--------------------|-----------------------|
| **ACTION_LOOP** | Agent repeats same or near-same tool call without progress | Add loop detection, restructure run_task(), add a "have I tried this?" check |
| **OVERCONFIDENT** | Agent declares success before verifying; judge disagrees | Add a verification tool or step in run_task() |
| **TOOL_MISSING** | Agent attempts something with no appropriate tool; fails or hallucinates | Add a purpose-built tool to create_tools() |
| **TOOL_CONFUSED** | Agent has the right tool but calls it wrong or misunderstands its scope | Improve tool docstring/description; clarify input format |
| **CONFUSED_MODEL** | Agent builds incorrect mental model of the environment early on | Add context-setting instruction to SYSTEM_PROMPT |
| **CONTEXT_POLLUTION** | Earlier tool outputs mislead later reasoning steps | Limit output length; add summarization; restructure observation format |
| **DISTRESS_SPIRAL** | Agent uses uncertainty language repeatedly: "still", "try again", "not sure" | Reduce turn budget; add a "when uncertain, do X" instruction |
| **DELEGATION_COLLAPSE** | Agent tries to delegate to a tool or sub-agent that doesn't exist | Add the missing tool; clarify what tools are available |
| **CAPABILITY_GAP** | The model cannot do this regardless of harness changes | Document and skip; adding tools may help if the gap is tooling, not reasoning |
| **INFRA_FAILURE** | The tool itself errored, not the agent's decision | Fix the tool, dependency, or environment setup |

---

### Phase 4 — Cross-case pattern

After deep reads, answer:

1. What do the confirmed-class cases have in common? One root cause.
2. Does this root cause explain the header-only cases too? If not, sample another.
3. Is there more than one distinct failure pattern? If yes — pick the one that affects
   the most cases. The other is a future iteration.

At 50+ cases, you will not confirm all cases individually. That is fine. State your
confidence: "confirmed in 3 deep-reads; appears consistent with 12 summary-only cases."

---

### Phase 5 — Holdout check

Read `holdout_summary.md`. This contains abstract failure patterns only — no task
content. Use it to cross-check your hypothesis:

- **Pattern matches train:** your fix is likely general. Green light.
- **Pattern diverges from train:** your fix may only help train. Flag this in your
  proposal — the proposer should understand the generalization risk.
- **Holdout shows CAPABILITY_GAP that train doesn't:** there is a harder class of
  tasks in holdout. Note it; don't try to address it in this iteration.

Do not try to identify or target specific holdout cases. The summary is intentionally
abstract. If you find yourself reasoning about what specific holdout task content
might be — stop. The abstraction boundary exists to prevent exactly that.

---

### Phase 6 — Write scratchpad.md

Fill in this structure completely before touching agent.py:

```markdown
## Evidence
case-id-1: "[exact agent quote at divergence turn]" — MISBEHAVIOR_CLASS [confirmed]
case-id-2: "[exact agent quote at divergence turn]" — MISBEHAVIOR_CLASS [confirmed]
case-id-3: inferred ACTION_LOOP from summary (turn count 4× median) [inferred]
... (continue for all failing cases)

## Pattern
[One sentence: what single root cause explains the confirmed cases and is
 consistent with the inferred cases?]

## Hypothesis
Root cause: [specific harness property causing this]
My change: [exactly one mechanism — component + what changes]
Prediction: will fix [list confirmed cases]. Likely helps [list inferred cases].
Will NOT fix [capability gap cases — list them].
Holdout: [will / may not] generalize — [one-sentence reason from holdout_summary]

## Anti-overfit checks
☐ My change mentions no specific task name, ID, or task content
☐ This change would help an agent working on many different unfamiliar tasks
☐ I am changing a fundamental mechanism, not a parameter
☐ Holdout summary does not indicate my fix is train-specific
```

All four boxes must be checked before proceeding. If any is unchecked, revise the
hypothesis until it passes all four, or conclude that no safe change exists this
iteration.

---

### Phase 7 — Edit agent.py

Make exactly the change described in the hypothesis. Nothing more.

Editable region: everything above the `FIXED ADAPTER BOUNDARY` comment.

What you can change:
- `SYSTEM_PROMPT` — agent's base instructions
- `MODEL` — only with explicit authorization
- `create_tools()` — add, remove, or modify tools
- `create_agent()` — agent construction, sub-agents, handoffs
- `run_task()` — orchestration logic, turn budget, observation handling

What you cannot change:
- Anything at or below `FIXED ADAPTER BOUNDARY`
- Harbor adapter, trace serialization, result writing

Simplicity criterion: if two approaches achieve the same result, use the shorter one.
Equal performance with cleaner code is a real improvement and should be accepted.

---

### Phase 8 — Write proposal.md

```markdown
## Change
[What you changed and where in agent.py]

## Reasoning
[Which scratchpad hypothesis this implements and why this mechanism addresses it]

## Prediction
Will fix: [cases]  Will not fix: [capability-gap cases]
Holdout: [expected outcome]

## Confidence
[High / Medium / Low] — based on [N confirmed deep-reads + M summary-inferences]
```

---

## Holdout analysis agent instructions

If you are the **holdout analysis agent**, your output must be abstract pattern
summaries only. The following content is FORBIDDEN in your output:

- Specific task names, IDs, or case identifiers
- File paths, commands, or tool call sequences from specific tasks
- Any content from the task instructions or verifier
- Any information that would let a reader identify which specific task is which

You may include:
- Misbehavior class counts (e.g., "3 of 10 holdout cases: ACTION_LOOP")
- Whether the failure distribution matches or diverges from train
- Whether failures appear fixable or capability-gap-level
- Confidence level on inferences (confirmed from deep read vs. inferred from header)

Your output feeds directly to a proposer that cannot see holdout traces. You are
the one-way abstraction boundary. If specific content leaks through your output,
the holdout is no longer held out.

---

## Scaling guidance

| Failing case count | Approach |
|--------------------|----------|
| ≤10 | Read all full traces. No triage needed. |
| 11–30 | Triage from summaries first. Select 3–6 for full reads. |
| 31–50 | Triage from summaries. Select 1 per pattern group (max 5 reads). Infer rest. |
| 50+ | Triage from summaries. Cluster by apparent class. Sample 1 per cluster. State confidence. |

At 50+ failing cases, your bottleneck is pattern-finding, not completeness. You need
enough evidence to confirm a pattern exists and identify its root cause — not a
complete audit of every case.

---

## What a good iteration looks like

```
Triage: 42 failing cases. Summaries show 28 with high turn counts, 10 with fast
failures, 4 with anomalous patterns.

Deep reads: read 4 traces — 2 high-turn-count (ACTION_LOOP confirmed), 1 fast-failure
(TOOL_MISSING confirmed), 1 passing (agent self-verifies before finishing).

Pattern: ACTION_LOOP affects 28/42 failures. Root cause: no turn-budget awareness;
agent keeps retrying without tracking what it's already tried.

Hypothesis: add a tool that returns the list of already-attempted commands. This
gives the agent structured memory of what it has tried, breaking the loop pattern.

Holdout: summary shows same ACTION_LOOP distribution. Fix should generalize.

Change: add `get_attempt_history()` tool to create_tools(). 3 lines of code.
```

This is the target. A specific root cause, a mechanism change, confirmed evidence,
a holdout cross-check. Not a parameter tweak, not a vague "improve the prompt."
