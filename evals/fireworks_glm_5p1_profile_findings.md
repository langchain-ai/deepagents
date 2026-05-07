# Fireworks GLM-5p1 — harness profile findings

Working notes from iterating on `_fireworks_glm_5p1.py`. Logged so future
iterations start from data, not memory.

**TL;DR — current state: v1 (4-section, Output Channel last).** v2 was
prototyped on a single-run signal that turned out to be a low-tail outlier
of v1's distribution; v2 regressed conversation by −0.16 outside the v1
N-trial noise band and was reverted. Section order is fixed by a unit test
as a regression guard.

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

---

## v2 results — negative; v1 reinstated

Two runs were kicked off after the v2 plan above:

- **v1 N-trials (variance baseline)**: GHA run [25460918393](https://github.com/langchain-ai/deepagents/actions/runs/25460918393), commit `9aee50ab2` (v1 + Output Channel section). 4 trials × 110 tests/trial. First 4 trials analyzed (run still going).
- **v2 (single run)**: GHA run [25461031650](https://github.com/langchain-ai/deepagents/actions/runs/25461031650), commit `2cf856fb7`. 119 tests.

### v1 noise band (4 trials)

| Metric | t0 | t1 | t2 | t3 | mean | stdev | range |
|---|---:|---:|---:|---:|---:|---:|---|
| `solve_rate` | 0.330 | 0.390 | 0.382 | 0.442 | 0.386 | 0.046 | 0.330–0.442 |
| `correctness` | 0.820 | 0.820 | 0.820 | 0.810 | 0.818 | 0.005 | 0.81–0.82 |
| `passed` | 90 | 90 | 90 | 89 | 89.75 | 0.5 | 89–90 |
| `median_duration_s` | 14.20 | 13.27 | 12.19 | 10.31 | 12.49 | 1.67 | — |
| **`conversation`** | **0.480** | **0.480** | **0.520** | **0.480** | **0.490** | **0.020** | **0.48–0.52** |
| `file_operations` | 0.92 | 0.92 | 0.92 | 1.00 | 0.940 | 0.040 | 0.92–1.00 |
| `tool_use` | 0.89 | 0.89 | 0.87 | 0.85 | 0.875 | 0.019 | 0.85–0.89 |
| `summarization` | 0.80 | 0.80 | 0.80 | 0.80 | 0.800 | 0.000 | — |
| `retrieval` | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 | — |

Per-test stability across the 4 trials:
- **14 stable fails** (4/4 trials): the model-fault clusters identified pre-v1.
- **14 flaky tests** (1–3/4 trials): the genuine noise floor.

### v2 vs v1 noise band

| Metric | v1 mean (±sd) | v1 range | v2 | Verdict |
|---|---:|---|---:|---|
| `solve_rate` | 0.386 ± 0.046 | 0.330–0.442 | 0.381 | **IN BAND** |
| `correctness` | 0.818 ± 0.005 | 0.81–0.82 | 0.81 | **IN BAND** |
| `conversation` | **0.490 ± 0.020** | **0.48–0.52** | **0.330** | **−0.16 BELOW BAND** 🔴 |
| `file_operations` | 0.940 ± 0.040 | 0.92–1.00 | 1.00 | IN BAND (top of) |
| `tool_use` | 0.875 ± 0.019 | 0.85–0.89 | 0.87 | IN BAND |
| `summarization` | 0.80 | — | 0.80 | TIED |
| `retrieval` | 1.00 | — | 1.00 | TIED |

- v2 fixed **0** of v1's 14 stable-fails.
- v2 introduced **2** net-new failures (`vague_data_analysis`, `tau2_airline[task_38]`) that never failed in any v1 trial.

### Why v2 lost

The v2 design was based on a **single-run** v1 signal (run 25455340145, conversation = 0.29) that I treated as the v1 mean. That run was actually the lower outlier of v1's distribution (mean 0.49, range 0.48–0.52). The "v1 conversation regression" v2 was meant to fix didn't exist — it was noise.

By removing the `Stop Conditions` section to "fix" a non-existent regression, v2 stripped framing the model was using to bound its responses on followup-quality and tau2-conversation tests. Promoting Output Channel to the top of the suffix didn't help cluster E either: `test_avoid_unnecessary_tool_calls` and `test_single_tool_get_food_calories` still emitted empty `content` with the answer in `reasoning_content`. The reorder gave us nothing and the deletion cost us the conversation score.

### Lesson — recorded for future iterations

**Profile changes must be evaluated against an N-trial baseline, not a single run.** The relevant noise band on this model is on the order of ±0.05 for solve_rate and ±0.02 for category scores. Single-run deltas inside that band are not signal. Single-run deltas of 0.10+ are *probably* signal but should still be confirmed with at least one repeat.

### v3 plan — when to attempt

The current v1 profile is the best-tested iteration. v3 only makes sense when there is:

1. A specific cluster of stable v1 fails to target (we have data for `test_avoid_unnecessary_tool_calls`, `test_single_tool_get_food_calories`, the BFCL tasks, and the persistent tau2 fails 7/14/23/29/39/44).
2. A change scoped tightly enough to not perturb conversation/tool_use scoring (e.g., a `_FireworksReasoningContentMiddleware` that only fires when content is empty and there are no tool calls — fully orthogonal to the suffix and likely safe).
3. An N-trials run on the candidate before declaring it better.

For now: do not edit the suffix. Route any further wins through middleware.
