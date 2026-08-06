# Switchyard benchmark — results and findings

Deep Agents pytest eval suite, 145 non-memory tests, n=1, all arms routed
through a local Switchyard server so every arm is priced by one instrument.

Raw data: `runs/*.json` (Switchyard `/v1/stats` snapshots), `runs/*.log`
(pytest output incl. per-category correctness). Re-price with
`python collect_stats.py report runs/*.json`.

## Completed arms

| Arm | Correctness (95% CI) | Cache-aware cost | $/completed task |
|---|---|---|---|
| GLM 5.2 only | 84.1% [77.3, 89.2] | $1.21 | $0.0099 |
| Opus 4.8 only | 84.1% [77.3, 89.2] | $11.53 | $0.0945 |
| Escalation Opus↑GLM | 83.4% [76.6, 88.6] | $2.78 | $0.0230 |
| Nemotron 3.5 Nano only | 80.7% [73.5, 86.3] | $1.10 | $0.0094 |

Selected publication methodology (all input tokens at the base rate):

| Arm | Uncached cost | $/completed task |
|---|---|---|
| GLM 5.2 only | $5.19 | $0.0426 |
| Opus 4.8 only | $28.16 | $0.2308 |
| Escalation Opus↑GLM | $8.40 | $0.0694 |
| Nemotron 3.5 Nano only | $1.10 | $0.0094 |

Minimum detectable effect at n=145 is **11.5pp**, so the arms are statistically
indistinguishable. Do not claim a winner on accuracy.

## Findings

### 1. The suite is saturated for capable models

GLM 5.2 and Opus 4.8 tie at exactly 122/145 despite a 9.5x price gap, and Nano
3.5 (30B-A3B) tracked them exactly at the 102-test mark (79/102 for all three).
Five of seven categories sit at 88–100% for every model; residual failures
concentrate in `conversation` and `summarization`, where **all** tiers fail.

That is headroom no router can capture — escalating does not fix a task the
strong tier also fails. A routing benchmark needs a suite where the strong model
measurably beats the weak one.

### 2. Routing only pays above a cost threshold

Routing beats the strong model alone only when the weak-tier offload fraction
clears:

    f_min = judge_cost / (strong_cost - weak_cost)

Measured cache-aware on this suite: judge = $0.49/run, GLM = $1.21/run. Every weak-tier
candidate available lands ~$0.28–0.48 under GLM, i.e. under the judge's own
cost, so f_min exceeds 100% and no amount of tuning helps:

| Pairing | strong − weak | f_min | Verdict |
|---|---|---|---|
| GLM ↑ Nemotron Ultra | $0.47 | 104% | impossible |
| GLM ↑ GLM-4.7 | $0.48 | 102% | impossible |
| GLM ↑ Nano 3.5 | ~$0.28 (proj.) | ~175% | impossible |
| Opus ↑ GLM | $10.32 | 4.7% | achievable, and measured: 76% cheaper than Opus-only — but GLM alone is cheaper still, so it loses |

**The judge is the binding constraint, not the model choice.** It is a fixed
~$0.49 tax paid on every un-latched turn regardless of whether anything
escalates. Slimming it (`recent_turn_window` 28→10, `window_message_chars`
500→300, ~2,900→750 tokens/call) would take f_min for GLM↑Nano to ~43% —
achievable, since the Opus↑GLM arm ran 95% weak. **This is the untested
experiment most likely to change the answer.**

Under the selected uncached methodology, GLM costs $5.19 and Nano costs $1.10,
so GLM↑Nano's threshold is only ~12%. The pricing choice therefore changes not
just the headline but which routed pairing can plausibly win; Harbor should run
GLM↑Nano rather than extrapolating the cache-aware threshold.

`judge_min_turn` (skip judging the first N turns) exists in Switchyard's Python
`EscalationRouterConfig` but is **not exposed in the Rust TOML** — worth asking
NVIDIA to surface it, as it is the highest-leverage knob available.

### 3. Cheapest per token ≠ cheapest per task

Nemotron 3.5 Nano lists at $0.05/$0.20 per 1M — ~24x cheaper than GLM. It is
4.53x cheaper per completed task under uncached pricing but only 1.06x under
cache-aware pricing. Three compounding causes explain the difference:

- **No prompt caching on NIM.** Verified: `prompt_tokens_details: None` on every
  response, `prompt_tokens` unchanged across three identical 4k-token prompts,
  no latency warm-up. GLM on Baseten runs at a 90.5% cache-read hit rate, worth
  **77% off its bill** ($1.21 vs $5.19 priced without the cache detail).
- **Thinking tokens.** ~423 output tokens/call vs GLM's 107.
- **Turn inefficiency.** ~3x more model calls per test than GLM.

Effective rate after caching x tokens per call x calls per task is the unit that
matters. Per-token price lists are the wrong one.

Actionable for NVIDIA: **prompt caching on NIM would improve Nemotron's cost
story more than any price cut.** A 77% effective discount is not reachable by
trimming a rate card.

### 4. In the routed arm, 5% of calls were 41% of the spend

Opus↑GLM: 34 Opus calls vs 643 GLM calls, yet both cost ~$1.14. At a 9.5x price
ratio the latch rate is a razor-sharp dial — `confirmations` is the control that
matters most.

### 5. Escalation hurt exactly where model-swapping is most likely

Per-category, escalation beat both pure arms on `file_operations` (21/21) and
matched Opus on `retrieval` (13/13), but fell to 9/22 on `conversation` vs GLM's
11/22. Conversation has the longest multi-turn trajectories and the most
latching opportunity; after a latch the history holds GLM turns in OpenAI-chat
shape and Opus turns in Anthropic shape. At n=1 that is 2 tests and may be
noise, but it is the one place the mechanism predicts harm and where harm
appeared.

## Methodology caveats — must appear in any writeup

1. **n=1.** Sean ran n=2 with tight ranges, but this needs n>=2 before publishing.
2. **Harness profiles are disabled in all arms.** The harness resolves profiles
   by model spec (`graph.py:605` `_harness_profile_for_model`); routed runs
   present as `openai:switchyard`, which matches nothing, so
   model-specific accommodations are off. Uniform across arms, so the
   comparison is valid — but routed numbers are **not** comparable to a direct
   `--model <provider>:<model>` run.
3. **Cache-aware vs cache-ignored pricing.** The publication decision is to lead
   with uncached pricing for parity with NVIDIA's OpenRouter runs. Ignoring cache inflates GLM 4.3x,
   Opus 2.4x, escalation 3.0x, and Nano not at all — a systematic bias toward
   the non-caching model. Report both and state that the uncached headline is a
   methodology-normalized comparison, not the provider invoice.
4. **Provider asymmetry.** GLM is Baseten-direct; Opus and the judge go through
   the LangSmith gateway (an extra hop). Cost is unaffected; latency comparisons
   between tiers are not apples-to-apples.
5. **Each model at its shipped default.** Nano runs `enable_thinking = true`
   (NVIDIA's demonstrated default); GLM runs at Baseten's default. The pytest
   path applies no reasoning override — unlike the Harbor path, which forces
   GLM to `reasoning_effort: high`.

## Open questions for NVIDIA

- Does NIM support prefix caching for Nemotron 3.5 Nano, and is it billed?
- Was Sean's Lightning baseline (64.1%) run with thinking off? We measured Nano
  tracking GLM and Opus exactly at the 102-test mark, far above 64%.
- Sean's Opus↑GLM run escalated 1.3% of calls; ours escalated 5%. Did his client
  send a session id? `confirmations >= 2` retains its streak per session and
  escalation mode exposes no message-hash fallback, so without one the streak
  resets every turn and the router structurally under-escalates.
- Can `judge_min_turn` be exposed in the Rust TOML?
- Nemotron 3 Super 120B is not serverless on Fireworks and absent from Baseten;
  a dedicated deployment bills GPU-hours, which has no per-token price to compare.

## Where routing would plausibly pay

Not on this suite. The conditions it needs — an expensive strong tier, no
caching advantage for it, and tasks hard enough that the strong model actually
wins — point at Harbor / terminal-bench rather than a saturated behavioral
suite. Opus↔GLM parity here does not imply parity on hard agentic coding.
