# Fireworks GLM-5p1 — harness profile findings (v1)

Working notes from the v1 prototype of `_fireworks_glm_5p1.py`. Logged so the v2
iteration starts from data, not memory.

## Sources

- **Initial baseline (no profile, pre-v1)**: GHA run [25403953867](https://github.com/langchain-ai/deepagents/actions/runs/25403953867) — 122 tests, solve_rate 0.478, correctness 0.81, 23 fails.
- **Re-baseline (no profile, immediately before v1)**: GHA run [25454256294](https://github.com/langchain-ai/deepagents/actions/runs/25454256294) — 119 tests, solve_rate 0.3746, correctness 0.81, 23 fails.
- **With profile v1**: GHA run [25455340145](https://github.com/langchain-ai/deepagents/actions/runs/25455340145) — 119 tests, solve_rate 0.4089, correctness 0.78, 26 fails.
- LangSmith experiment session for the profile run: `fireworks:accounts/fireworks/models/glm-5p1:ff3ce85a` (id `cbd56876-87e5-4970-883a-7a8abb18cfc0`).

The two no-profile runs differ on test count (122 vs. 119) — the dataset shifted between them, so head-to-head deltas use the 25454256294 baseline.

## v1 profile content

Four sections appended via `system_prompt_suffix`:

1. **Tool Execution Discipline** — restate-before-mutate, no defaulting to `""`/`"latest"`, no re-reading already-visible results.
2. **Parallel Tool Use** — batch independent calls.
3. **Stop Conditions** — close every TODO, do not double-execute mutations, write a brief end-of-turn confirmation.
4. **Output Channel** — `content` MUST never be empty; reasoning channel is internal-only; wrap single-token answers in a sentence.

## Initial-baseline failure clustering (run 25403953867, 23 fails)

Bucketing of model-fault failures (19 of 23) before v1:

| Cluster | Count | Symptom |
|---|---:|---|
| A. Plan / stop discipline | 7 | loops on read-only calls, drops a final mutation, or repeats one |
| B. Argument fidelity on mutating tools | 7 | right tool, wrong target ID / inverted enum / `team=''` |
| C. Function-composition syntax (Nexus) | 3 | pipeline `->` instead of nested calls |
| D. Communication-subscore only | 1 | tau2 task 14 |
| E. Output-channel quirk (empty `content`) | 1 | `test_avoid_unnecessary_tool_calls` |

Eval-design tension (4 of 23): `test_followup_question_quality` rubric-style cases where any model would struggle to satisfy a strict "exactly one question" / "ask about domain" item.

## Head-to-head (baseline 25454256294 → profile v1 25455340145)

| Metric | Baseline | With v1 | Δ |
|---:|---:|---:|---:|
| `solve_rate` | 0.3746 | **0.4089** | **+3.4 pts** |
| Passed (raw) | **96** | 93 | **−3** |
| `correctness` | **0.81** | 0.78 | −3.0 pts |
| `median_duration_s` | 12.19 | **10.67** | −1.52 s |
| `step_ratio` | 1.02 | **1.01** | −0.01 |
| `tool_call_ratio` | 1.11 | 1.14 | +0.03 |

`solve_rate` and pass count disagree. Headline metrics are mixed.

### Category scores

| Category | Baseline | With v1 | Δ |
|---|---:|---:|---:|
| **summarization** | 0.60 | **0.80** | **+0.20** ✓ |
| retrieval | 1.00 | 1.00 | — |
| tool_use | 0.87 | 0.87 | — |
| unit_test | 1.00 | 1.00 | — |
| file_operations | 0.92 | 0.85 | −0.07 ✗ |
| **conversation** | 0.43 | **0.29** | **−0.14** ✗✗ |

Conversation took a 14-point hit. That's the dominant negative.

### Test-level diff

**Fixed (5)**
- `test_compact_tool_new_task` (summarization)
- `test_chain_create_issue_then_notify`
- `test_metric_ranking_active_incident_highest_latency`
- `test_tau2_airline[task_32]`, `task_37`

**New failures (8)**
- `test_followup_question_quality[vague_data_analysis]` — *new*
- `test_followup_question_quality[vague_monitor_system]` — *new*
- `test_followup_question_quality[vague_summarize_emails]` — *new*
- `test_read_file_truncation_recovery_with_pagination` — agent emitted `'x'`
- `test_indirect_email_report` — agent gave up rather than searched
- `test_dependency_reasoning_active_incident_depending_on_identity_api` — wrong incident ID
- `test_single_tool_get_food_calories` — empty `content`, answer in `reasoning_content`
- `test_tau2_airline[task_9, task_35]` — borderline DB-state mismatches (likely noise)

**Still failing (18)** — incl. `test_avoid_unnecessary_tool_calls`, the original cluster-E case the Output Channel section was meant to fix.

## Root-cause findings from traces

1. **Suffix is wired correctly.** Trace `36d211f9` for `test_avoid_unnecessary_tool_calls` shows the system prompt sent to ChatFireworks containing all four sections, including the literal `"MUST NEVER be empty"` rule. The model is ignoring the rule, not missing it.

2. **Output Channel rule has a ceiling in the eval harness even though direct API probes were 8/8.** Both `test_avoid_unnecessary_tool_calls` and `test_single_tool_get_food_calories` routed the final token into `reasoning_content`. Likely cause: deepagents binds tools and emits a ~5 KB system prompt; in that mode GLM-5p1's tool-output discipline beats a late suffix rule. Direct probes with a small prompt and no tools bound don't reproduce. Pure prompting cannot fully close cluster E in the deepagents harness.

3. **`Stop Conditions` is the source of the conversation regression.** The "write a brief confirmation of what changed" + "ask one targeted question" combo makes the agent emit long numbered-question lists in vague-prompt scenarios, which trips rubric items demanding exactly-one-question or domain-first questions. All three new followup failures show 4–5 numbered questions instead of one.

4. **Tool Execution Discipline pulled its weight on the wins.** The summarization category (+0.20) and the tau2 fixes (32, 37) align with "do not re-issue read-only calls / do not double-execute mutations" landing.

5. **Parallel Tool Use was net-neutral.** Median duration dropped 1.5 s (consistent with batching) but no tests swung on it.

## v2 plan

1. **Drop the closure-confirmation bullet** from `Stop Conditions`. Keep only the "do not double-execute" / TODO-resolution lines. This should reverse most of the conversation regression.
2. **Promote Output Channel to the top of the suffix** and rewrite as an imperative ("Always begin your reply with the user-visible answer in plain text. Never leave content empty when you are not making a tool call."). Late soft rules lose to early imperatives in long prompts.
3. **Keep Tool Execution Discipline as-is** — it's carrying the wins.
4. **Keep Parallel Tool Use as-is** — net-neutral, mild latency win.
5. **Defer**: a `_FireworksReasoningContentMiddleware` in `extra_middleware` that surfaces `reasoning_content` into `content` when content is empty and there are no tool calls. Only path to a deterministic close on cluster E. Hold for v3 if v2 still leaves `avoid_unnecessary_tool_calls` failing.

## Next eval

After v2, re-run the same matrix and compare against both this run and the baseline. Watch for:
- Conversation score climbing back toward 0.43.
- Summarization + tau2 (32, 37, ...) fixes holding.
- `test_avoid_unnecessary_tool_calls` and `test_single_tool_get_food_calories` — if both still fail, ship the middleware.
