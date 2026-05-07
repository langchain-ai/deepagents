# Fireworks GLM-5p1 — harness profile findings

Working notes from iterating on `_glm_5p1.py`. Logged so future
iterations start from data, not memory.

**TL;DR — current state: 3-section profile** (Tool Execution Discipline →
Parallel Tool Use → Stop Conditions). The earlier `Output Channel` section
was removed after a local ablation showed it was the *sole* cause of stable
regressions on `test_single_tool_get_food_calories` and
`test_single_tool_get_user_email`.

The full OpenRouter 2×2 (siliconflow/fp8 + atlas-cloud/fp8, no-profile vs.
with-profile, N=3 trials per cell) shows the profile produces **no
statistically meaningful effect on `solve_rate` at either provider**, with
small and inconsistent category-level lifts. Cluster E
(`reasoning_content` routing) was **not observed** in any of 244 failed
OpenRouter traces — that bug is Fireworks-specific. The deferred
middleware is no longer warranted for the OpenRouter deployment path.

The decision now is whether to keep the 3-section profile as a documented
null result or drop it. See the OpenRouter 2×2 and Cluster E sections at
the bottom for the data behind these conclusions.

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

---

## No-profile baseline (partial — trial 0 only)

GHA run [25473519302](https://github.com/langchain-ai/deepagents/actions/runs/25473519302) on `main` (no profile registered). 5 trials dispatched, cancelled after trial 0 finished (~5h). Trial 0 alone gives N=1 — useful but not a noise band.

**Test-set caveat**: profile-branch v1 N-trials excluded the `memory` category (110 tests). Main ran with default categories including memory (168 tests). So `correctness` (0.67 vs 0.82), `passed`, `failed`, and totals are not directly comparable. Per-category *rates* are.

### Category rates — v1 (N=4) vs no-profile (N=1)

| Category | v1 mean ± sd | v1 range | No-profile | Verdict |
|---|---:|---|---:|---|
| **conversation** | **0.490 ± 0.020** | 0.48–0.52 | **0.380** | **−0.11 BELOW band — profile helps** |
| **summarization** | **0.800 ± 0.000** | — | **0.600** | **−0.20 BELOW band — profile helps clearly** |
| file_operations | 0.940 ± 0.040 | 0.92–1.00 | 0.920 | IN band — no effect |
| retrieval | 1.000 | — | 1.000 | tied |
| tool_use | 0.875 ± 0.019 | 0.85–0.89 | 0.890 | IN band — no effect |
| `solve_rate` | 0.386 ± 0.046 | 0.330–0.442 | 0.403 | IN band (test sets differ — not load-bearing) |

### Per-test profile cost — 2 stable regressions

Tests that **pass without the profile** and **stably fail with it** (4/4 v1 trials):

- `test_single_tool_get_food_calories` — empty `content`, answer routed into `reasoning_content`
- `test_single_tool_get_user_email` — same pattern

Counterintuitively, these regress *despite* the suffix's `Output Channel` rule explicitly telling the model never to leave content empty. The larger profile prompt appears to push the model into `reasoning_content` routing on simple single-tool tasks more strongly than the rule counters. This is direct evidence that the empty-content section isn't paying for itself in the harness, even though direct API probes showed it working in isolation (8/8 vs. 1/8).

#### Local A/B verification (N=5 per cell, live Fireworks API)

Toggled the profile on/off via a pytest plugin that pops the GLM-5p1 entry from `_HARNESS_PROFILES` after bootstrap. Confirms the regression is real and large — not CI-run flakiness.

| Condition | Test | Pass | Fail |
|---|---|---:|---:|
| with-profile | `test_single_tool_get_food_calories` | 1 | 4 |
| with-profile | `test_single_tool_get_user_email` | 0 | 5 |
| no-profile | `test_single_tool_get_food_calories` | 5 | 0 |
| no-profile | `test_single_tool_get_user_email` | 5 | 0 |

Without the profile: **10/10 pass**. With it: **1/10 pass**. Same failure mode in every case (`got: ''`). The single passing `food_calories` run with profile shows the regression is probabilistic, not deterministic — but the magnitude (≈90% failure rate vs ≈0%) is unambiguous.

### Net assessment

The profile is a **modest net positive**: conversation +0.11 and summarization +0.20 (both outside v1's noise band) outweigh the 2-test regression on single-tool prompts. Headline `solve_rate` is statistically indistinguishable from no-profile, but the test sets differ enough that the headline isn't a fair comparison.

### Refined action items

1. **Keep v1 in place** — it pays for itself on the categories that change.
2. **Investigate the single-tool regressions before any further suffix edits.** If the suffix is causing the empty-content routing on these tests, that's the same cluster-E mechanism the rule was meant to fix — confirming it can't be solved at the prompt layer in this harness.
3. **Promote `_FireworksReasoningContentMiddleware` from "deferred" to "next."** It's the deterministic close on cluster E and would also fix the profile-induced regressions on the 2 single-tool tests, turning the profile from a net positive into a clean win.
4. Run a full 5-trial no-profile baseline at some point to firm up the conversation lift estimate (currently N=1, so ±0.11 is a point estimate, not a band). Match the test-set composition to the v1 N-trials run — exclude `memory` — so the comparison is apples-to-apples.

---

## Section ablation (local, live API) — Output Channel removed

Triggered by the verified single-tool regression: ran a 7-variant ablation against the live Fireworks API to identify which section was responsible. Each variant drops one section from the v1 4-section suffix, plus two controls (`ted-only` keeping only Tool Execution Discipline, `off` removing the profile entirely). N=5 per cell × 7 variants × 2 tests = 70 runs.

### Results

| Variant | `food_calories` | `user_email` | Total pass rate |
|---|---:|---:|---:|
| `v1-full` (TED + Parallel + Stop + Output) | 0/5 | 0/5 | 0/10 (0%) |
| **`no-output`** (drop Output Channel only) | **5/5** | **5/5** | **10/10 (100%)** |
| `no-stop` | 0/5 | 0/5 | 0/10 (0%) |
| `no-parallel` | 0/5 | 0/5 | 0/10 (0%) |
| `no-ted` | 0/5 | 0/5 | 0/10 (0%) |
| `ted-only` | 5/5 | 5/5 | 10/10 (100%) |
| `off` (no profile) | 5/5 | 5/5 | 10/10 (100%) |

### Interpretation

**`Output Channel` is the sole cause.** Every variant that retains the section fails 100%; every variant that drops it passes 100%. The other three sections do not contribute to the regression — `no-stop`, `no-parallel`, and `no-ted` all still fail at 0/10 because Output Channel is still present.

The "MUST NEVER be empty" rule appears to *cause* the very behavior it tried to prevent. Plausible mechanism: by explicitly naming the reasoning vs. content channel split and giving rules about which goes where, the suffix primes the model to treat the channels as distinct and route short final answers into the wrong one. The 8/8 success in direct API probes (small prompt, no tools bound) does not transfer to the deepagents harness's longer prompt with tools bound.

### Action taken

`_glm_5p1.py` is now a 3-section profile (TED + Parallel + Stop). Output Channel is removed. Module docstring records the ablation. Unit test `test_glm_5p1_has_harness_profile` asserts `## Output Channel` is *not* in the suffix, with a comment that reintroducing it requires fresh ablation data.

### Post-ablation verification (live API)

Confirmed the new 3-section profile against the same 2 tests:

| Test | Pass rate |
|---|---:|
| `test_single_tool_get_food_calories` | 5/5 |
| `test_single_tool_get_user_email` | 5/5 |

Cluster E (the original problem Output Channel was meant to solve — `test_avoid_unnecessary_tool_calls`, etc.) remains open and is the target of a separate `_FireworksReasoningContentMiddleware` change.

---

## OpenRouter 2×2 — full N=3 matrix (no-profile vs with-profile, two backends)

After cancelling the Fireworks with-profile arm for latency reasons, focus shifted to OpenRouter. Built out a full 2×2 design (provider pin × profile arm), N=3 sequential trials per cell, identical inputs otherwise.

### Run table

| OpenRouter pin | Arm | Run | Branch | Commit |
|---|---|---|---|---|
| `siliconflow/fp8` | no-profile | [25507995458](https://github.com/langchain-ai/deepagents/actions/runs/25507995458) | `main` | [`ee4f7104c`](https://github.com/langchain-ai/deepagents/tree/ee4f7104c58c2477c03c31a958edf4ca78a150c0) |
| `siliconflow/fp8` | with-profile | [25517643342](https://github.com/langchain-ai/deepagents/actions/runs/25517643342) | `mdrxy/evals/fg51-prof` | [`c13d71bdc`](https://github.com/langchain-ai/deepagents/tree/c13d71bdc89a14da216826403732f6a417178e38) |
| `atlas-cloud/fp8` | no-profile | [25518860368](https://github.com/langchain-ai/deepagents/actions/runs/25518860368) | `main` | [`fb0f0ba08`](https://github.com/langchain-ai/deepagents/tree/fb0f0ba08a58c3d2ae5db2435c07e3b826f395fa) |
| `atlas-cloud/fp8` | with-profile | [25517773053](https://github.com/langchain-ai/deepagents/actions/runs/25517773053) | `mdrxy/evals/fg51-prof` | [`c13d71bdc`](https://github.com/langchain-ai/deepagents/tree/c13d71bdc89a14da216826403732f6a417178e38) |

Inputs identical across cells: `model=openrouter:z-ai/glm-5.1`, `trials=3`, `parallel=false`, `eval_categories_exclude=memory,unit_test`, `analyze_failures=true`, `openrouter_allow_fallbacks=false`. Only differences are the two factors under study (provider pin, profile arm). All cells totals=110/trial.

Sanity-checked the comparison: the only deltas between baseline SHAs and branch tip outside the GLM-5.1 eval path are one quickjs commit (`libs/partners/quickjs/`) and one CLI commit (`libs/cli/`). Eval/SDK/workflow paths untouched.

### Headline metrics (mean ± stdev, N=3 trials)

| metric | sili NP | sili WP | atlas NP | atlas WP |
|---|---:|---:|---:|---:|
| `solve_rate` | 0.453 ± 0.014 | 0.452 ± 0.007 | 0.363 ± 0.040 | 0.355 ± 0.028 |
| `correctness` | 0.820 ± 0.026 | 0.817 ± 0.012 | 0.803 ± 0.031 | 0.827 ± 0.023 |
| `passed` (of 110) | 90.0 ± 2.6 | 89.7 ± 1.2 | 88.3 ± 3.1 | 90.7 ± 2.3 |
| `median_duration_s` | 10.08 ± 0.45 | 10.02 ± 0.54 | 11.83 ± 1.06 | 12.07 ± 1.64 |

### Category scores

| category | sili NP | sili WP | atlas NP | atlas WP |
|---|---:|---:|---:|---:|
| conversation | 0.427 ± 0.081 | 0.430 ± 0.050 | 0.397 ± 0.058 | 0.443 ± 0.100 |
| file_operations | 0.920 ± 0.000 | 0.973 ± 0.046 | 0.973 ± 0.046 | 1.000 ± 0.000 |
| retrieval | 1.000 | 1.000 | 1.000 | 1.000 |
| summarization | 0.800 | 0.733 ± 0.115 | 0.600 | 0.667 ± 0.115 |
| tool_use | 0.903 ± 0.015 | 0.893 ± 0.025 | 0.890 ± 0.040 | 0.903 ± 0.015 |

### Profile lift (with − no), per provider

| metric / category | siliconflow Δ | atlas-cloud Δ |
|---|---:|---:|
| `solve_rate` | **−0.001** | **−0.008** |
| `correctness` | −0.003 | +0.023 |
| conversation | +0.003 | +0.047 |
| file_operations | +0.053 | +0.027 |
| retrieval | 0.000 | 0.000 |
| summarization | −0.067 | +0.067 |
| tool_use | −0.010 | +0.013 |

### Verdict on OpenRouter

The profile produces **no statistically meaningful effect on `solve_rate`** at either provider — both deltas are well inside the within-provider noise band. Some category-level lifts (`file_operations` +0.05/+0.03, `conversation` +0.05 atlas-only) are large enough to be plausibly real but barely outside noise. The summarization signal is **opposite-sign across providers** (−0.07 sili, +0.07 atlas), which is consistent with noise rather than a real effect.

Provider effect (~9 pts on `solve_rate`) dominates everything: atlas-cloud is consistently lower-scoring and slower than siliconflow regardless of profile. The profile doesn't close that gap.

### Hard ceilings (model-level limitations)

10 tests fail 3/3 in **every** cell — these are the model's hard ceiling, not profile-fixable at the prompt layer:

- BFCL state mismatches (3): `multi_turn_composite_199`, `multi_turn_composite_97`, `multi_turn_miss_func_55`
- Tau2 conversation (4): tasks 7, 14, 23, 44
- Summarization (1): `test_compact_tool_large_reads`
- Tool-use composition (1): `nexus_placesapi_15`
- Followup-quality rubric (1): `detailed_calendar_brief`

These are the same 14 stable v1 fails identified earlier. Consistent picture: prompt-suffix tuning has hit its ceiling on this model.

### Profile-induced regressions on OpenRouter

The two single-tool tests that drove the local ablation (`test_single_tool_get_food_calories`, `test_single_tool_get_user_email`) **do not appear in the OpenRouter failure list at all** — they pass cleanly under both providers, both arms. Confirms those regressions were Output-Channel-section-specific *and* the post-ablation fix sticks.

---

## Cluster E — definitive answer (OpenRouter trace audit)

Goal: decide whether the deferred `_ReasoningContentMiddleware` is warranted for OpenRouter. Pulled all 245 failed-trace AIMessages across the 12 trial sessions in the 2×2 (≈60–65 per cell) and inspected each for the cluster-E pattern: `content == ""` AND no tool calls AND non-empty `reasoning_content`.

| Cell | Failed traces | Cluster-E hits |
|---|---:|---:|
| sili-NP | 60 | 0 |
| sili-WP | 61 | 0 |
| atlas-NP | 65 | 0 |
| atlas-WP | 58 | 0 |

**Zero cluster-E hits across all 244 OpenRouter trace failures.** Not a single one.

### What the OpenRouter failures actually look like

Representative sample from `sili-WP`:

| Test | Failure mode |
|---|---|
| `test_indirect_email_report` | Agent emitted full-text content (`"No status report file found..."`) — eval failed because the text didn't contain `"Week 10"`. `reasoning_content` is also populated, but `content` is fine. |
| `test_compact_tool_large_reads` | Agent hit run_limit (3/3) — `content` shows the limit-exceeded marker. Not a channel issue at all. |
| `test_followup_question_quality` | Agent emitted long `content` — eval failed on rubric ("did not ask one followup question"). `content` is full and non-empty. |

OpenRouter **does** surface `reasoning_content` (≈2/3 of failed traces have it populated), but as a parallel reasoning channel — `content` is also populated correctly. Cluster E (empty `content` with the answer routed into `reasoning_content`) appears to be **Fireworks-specific**, not a general GLM-5p1 trait. The OpenRouter inference path normalizes the output back to a populated `content` field.

### Verdict on `_ReasoningContentMiddleware`

**Not warranted for the OpenRouter deployment path.**

- 0/244 failed traces would be fixed by it.
- The two tests that exposed the local Fireworks regression (`test_single_tool_get_food_calories`, `test_single_tool_get_user_email`) don't appear in the OpenRouter failure list — they pass cleanly under both providers, both arms.
- All OpenRouter failures are content-quality issues (wrong answer, missing info, formatting / rubric mismatches), not channel-routing issues. None of these are addressable by a `reasoning_content` → `content` transform.

The middleware would still help Fireworks users (1/10 vs 10/10 in the local ablation), but the Fireworks arm was cancelled for latency reasons, and the OpenRouter route — where deployment is happening — does not have the bug. Building the middleware would be carrying weight this deployment doesn't pay for.

### Refined action items

1. **Profile** — given the null effect on `solve_rate` and the inconsistent / small category lifts, decide whether to keep the 3-section suffix as a documented null result, or drop it. The hard ceilings (10 stable fails) won't move regardless.
2. **Middleware** — close the ticket. Cluster E does not exist on OpenRouter; the only place it does (Fireworks) is not in scope.
3. **Future work** — if Fireworks comes back into scope, revisit the middleware. Until then, the prompt-layer wins are exhausted on this model.
