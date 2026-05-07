# Kimi K2.6 harness-profile investigation

Working notes from the multi-day prototype + eval cycle. The branch under
investigation is `mdrxy/evals/bk26-prof`; the profile module lives at
`libs/deepagents/deepagents/profiles/harness/_kimi_k2_6.py` (renamed
from `_baseten_kimi_k2_6.py` once OpenRouter was added as a second
registration key).

## TL;DR

- We hand-wrote a harness profile (system-prompt suffix + one tool-description
  override) for Moonshot Kimi K2.6, motivated entirely by failure clusters
  observed in a baseline eval run.
- After 3 prompt iterations and 3-trial averaged evals on three providers, the
  profile **lifts OpenRouter Kimi (clean win on Fireworks pin, modest win on
  Cloudflare pin)** and **regresses Baseten Kimi**. Same model, opposite
  outcomes by provider.
- The compact-tool description override did **not** move the needle in
  multi-trial — the single-trial pass we attributed to it earlier was sampling
  variance.
- Recommended action: drop Baseten from the profile's registration tuple, keep
  OpenRouter. Investigate Baseten's serving setup before re-applying.

## Context: original failure-mode analysis

Source run: [GHA 25403883357](https://github.com/langchain-ai/deepagents/actions/runs/25403883357)
(single-trial, no profile, on `baseten:moonshotai/Kimi-K2.6`).

26 of 122 tests failed (solve_rate = 0.79). Worst categories: `conversation`
(0.52), `summarization` (0.60). After classifying every failure into
{model fault, eval/judge fragility, SDK/test version skew}:

- **~19 of 26 model faults** clustered by behavioral pattern:
  - Multi-turn planning & state tracking (9): tau2_airline tasks, bfcl
    multi_turn_*. Symptoms: redundant info-gathering loops
    (`get_flight_status` x10), wrong cancel-then-rebook ordering, hallucinated
    extra booking, missing terminal mutations, `user_stop` from the simulator
    after silent tool-call streaks.
  - Syntax / API composition (3): nexus benchmarks. Hallucinated `|>` pipe
    syntax, dropped outer `sort_results` wrapper, replaced literal `0.0` with
    `sin(radians=0.0)`.
  - Wrong tool selection / single-turn logic (4–5): incident-graph wrong
    incident ID; never-called `compact_conversation`; gave-up-on-search.
  - Followup over/under-asking (2): `vague_send_report` re-asked schedule
    after "every week"; `detailed_calendar_brief` asked 4 questions when 1
    was required.
- **4 eval/judge fragility**: substring-on-final-text checks failed because
  the agent did the side-effect correctly but didn't echo the substring;
  judge-graded followup-quality tests where the judge admitted "spirit met"
  but graded as fail.
- **3 SDK/test version skew (HITL trio)**: not a model fault. Test asserted
  `allowed_decisions == ["approve","edit","reject"]` but the SDK now emits
  `["approve","edit","reject","respond"]`. The test docstring even says
  *"These are SDK integration tests, not model capability evals."*

## Profile design

Mirrored the existing built-in harness profiles for Anthropic Opus/Sonnet/Haiku
and OpenAI Codex:

- **System-prompt suffix** (`HarnessProfile.system_prompt_suffix`): tagged
  XML-style sections, one per failure cluster, with `Bad:`/`Good:`
  contrastive examples (Moonshot-style few-shot prompting).
- **Tool description override** (`HarnessProfile.tool_description_overrides`):
  rewrote `compact_conversation`'s description from a soft "use proactively"
  hint to imperative trigger conditions.
- **No middleware changes, no tool exclusions, no model-construction
  overrides.** Suffix + one description was the entire surface area.

Inspired by [LangChain's *Tuning Deep Agents for Different Models*](https://www.langchain.com/blog/tuning-deep-agents-different-models)
post — the same "harness profile" mechanism the SDK already ships for
production model specs.

## Version history

5 commits on `mdrxy/evals/bk26-prof`; 3 distinct prompt-content iterations:

| Tag | Description | Suffix sections | Notes |
|---|---|---|---|
| **v1** (`init`) | Original prototype | 5: `<plan_before_mutate>`, `<no_redundant_reads>`, `<communicate_each_turn>`, `<respect_user_specifications>`, `<ground_in_provided_docs>` | Plain prose rules, no examples. Compact-conversation override added. |
| **v2** | Added few-shot examples | Same 5 sections | Added `Bad:`/`Good:` contrastive pairs to every section. |
| **v3** | Stripped two sections | 3: `<plan_before_mutate>`, `<no_redundant_reads>`, `<ground_in_provided_docs>` | Dropped `<communicate_each_turn>` (suspected to encourage tau2 user-simulator off-task chatter) and `<respect_user_specifications>` (flippy on followup-quality tests). |
| widen-providers | Multi-provider key | (no prompt change) | Added `openrouter:moonshotai/kimi-k2.6` alongside `baseten:moonshotai/Kimi-K2.6` in `_KIMI_K2_6_MODEL_SPECS` tuple. |
| rename file | `_baseten_kimi_k2_6.py` → `_kimi_k2_6.py` | (no behavior change) | Filename matched scope after multi-provider widen. |

The "version" referenced in eval results below is **v3** suffix content (3
sections + compact override) registered under both Baseten and OpenRouter
keys.

## Run inventory

### Baselines (no profile, on `main`)

| Run | Provider | Trials | Notes |
|---|---|---|---|
| [25403883357](https://github.com/langchain-ai/deepagents/actions/runs/25403883357) | `baseten:moonshotai/Kimi-K2.6` | 1 | Original failure-mode source |
| [25475600906](https://github.com/langchain-ai/deepagents/actions/runs/25475600906) | `baseten:moonshotai/Kimi-K2.6` | 1 | Single-trial baseline |
| [25506841185](https://github.com/langchain-ai/deepagents/actions/runs/25506841185) | `baseten:moonshotai/Kimi-K2.6` | 3 | **Multi-trial baseline (Baseten)** |
| [25508166079](https://github.com/langchain-ai/deepagents/actions/runs/25508166079) | `openrouter:moonshotai/kimi-k2.6` (cloudflare) | 3 | **Multi-trial baseline (OpenRouter / cloudflare)** |

### Profile runs

| Run | Profile | Provider | Trials | Notes |
|---|---|---|---|---|
| [25473000720](https://github.com/langchain-ai/deepagents/actions/runs/25473000720) | v1 | `baseten` | 1 | First profile run; +1 pass vs baseline |
| [25473977387](https://github.com/langchain-ai/deepagents/actions/runs/25473977387) | v2 | `baseten` | 1 | +3 passes vs baseline (compact tool flipped to passing — later shown to be variance) |
| [25518232947](https://github.com/langchain-ai/deepagents/actions/runs/25518232947) | v3 | `baseten:moonshotai/Kimi-K2.6` | 3 | **Multi-trial profile (Baseten)** — net regression |
| [25518242054](https://github.com/langchain-ai/deepagents/actions/runs/25518242054) | v3 | `openrouter` (cloudflare pin) | 3 | **Multi-trial profile (OpenRouter / cloudflare)** — modest win |
| [25518249967](https://github.com/langchain-ai/deepagents/actions/runs/25518249967) | v3 | `openrouter` (fireworks pin) | 3 | **Multi-trial profile (OpenRouter / fireworks)** — clean win, no Fireworks-pinned baseline counterpart |

All multi-trial runs: `trials=3`, `parallel=false`, `eval_categories_exclude=memory,unit_test`,
`analyze_failures=true`, `analysis_model=anthropic:claude-haiku-4-5-20251001`.

## Headline results (3-trial means)

| Metric | base-bt | prof-bt | Δ | base-cf | prof-cf | Δ | prof-fw | Δ vs base-cf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `correctness` | 0.78 | **0.76** | −0.02 | 0.78 | **0.80** | +0.01 | **0.81** | +0.03 |
| `passed` (mean) | 86.0 | **83.3** | −2.7 | 86.3 | **87.7** | +1.3 | **89.0** | +2.7 |
| `conversation` | 0.48 | **0.27** | −0.21 ⚠ | 0.46 | 0.41 | −0.05 | 0.46 | 0.00 |
| `file_operations` | 0.97 | 1.00 | +0.03 | 0.95 | 1.00 | +0.05 | 1.00 | +0.05 |
| `retrieval` | 0.93 | 1.00 | +0.07 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| `summarization` | 0.60 | 0.60 | 0.00 | 0.60 | 0.60 | 0.00 | 0.60 | 0.00 |
| `tool_use` | 0.84 | 0.85 | +0.01 | 0.84 | 0.87 | +0.03 | 0.88 | +0.04 |

**Headline:** same profile + same model name = opposite outcomes depending on
who's hosting Kimi. The profile is **not** provider-agnostic.

## What worked

- **`<ground_in_provided_docs>`** — flipped `nexus_multiversemath_18` (a
  documented case of inventing pipe syntax) from fail → pass. Cheap, narrow,
  no observed regressions on either provider.
- **`<plan_before_mutate>` + `<no_redundant_reads>`** on OpenRouter —
  improved tau2_airline tasks 14/23/35/37/39/44/7 (cloudflare) and tasks
  35/39 (fireworks). Reduced `bfcl multi_turn_miss_func_55` failures across
  the board.
- **Multi-provider registration** — registering one profile under both
  provider keys via a `_KIMI_K2_6_MODEL_SPECS` tuple worked cleanly (Codex
  pattern). The two keys resolve to literally the same `HarnessProfile`
  instance.

## What didn't work

- **Compact-conversation tool description override** — produced **zero**
  measurable lift in multi-trial. Both `test_compact_tool_large_reads` and
  `test_compact_tool_new_task` fail in 3/3 trials of every arm (baseline and
  profile, every provider). The `summarization` category mean is exactly
  0.60 with stdev=0 across all five arms. The single-trial v2 pass we'd
  attributed to the override was sampling variance.
- **Profile on Baseten** — regressed `conversation` from 0.48 to 0.27.
  Specifically:
  - 5 `tau2_airline` tasks got more failures (3 of them flipped from
    1/3-fail to 3/3-fail).
  - `vague_send_report` flipped from 1/3-fail to 3/3-fail (the
    `<respect_user_specifications>` section that previously fixed this was
    dropped in v3).
- **Stripping `<respect_user_specifications>` (v2 → v3)** — was load-bearing
  on **both** providers for followup-quality tests. After v3,
  `vague_send_report` and `vague_summarize_emails` regressed on both Baseten
  and OpenRouter. The original removal rationale (v1→v2 single-trial flake)
  doesn't hold up under multi-trial. Restoring the section is a candidate
  for v6.
- **`<communicate_each_turn>` on tau2_airline** — original v2 multi-trial
  data showed conversation regression on Baseten that we attributed to
  mid-trajectory chatter being scored off-task by the user simulator.
  v3 removed it. Conversation regression on Baseten *got worse* in v3
  multi-trial, so that section may not have been the actual culprit either.

## Findings & open questions

### The provider-divergence finding

Same profile, same model name, opposite outcomes:
- **OpenRouter** absorbs the suffix cleanly, especially on `<plan_before_mutate>`
  and `<ground_in_provided_docs>` — both pins (cloudflare and fireworks) lift.
- **Baseten** seems to react badly to `<plan_before_mutate>` specifically
  on tau2_airline tasks. Hypotheses:
  - Different model variant (base vs instruct vs an internal Moonshot fine-tune
    served by Baseten)
  - Different chat-template wrapping (Baseten may apply a system-prompt
    template that conflicts with the structured form of `<plan_before_mutate>`)
  - Different sampling defaults (temperature, top-p) leading to different
    suffix adherence
  - Tokenizer differences leading to different effective prompt budget

None of these are confirmed. The cleanest next data point would be a
`fireworks:` direct provider run (skipping the OpenRouter middle layer)
to triangulate whether the lift is OpenRouter routing or upstream serving.

### The cloudflare-vs-fireworks gap

Within OpenRouter, the same profile lifts cloudflare by +1.3 passes and
fireworks by +2.7. The two pins differ on which inference partner serves
the request. This is consistent with upstream-provider serving differences
(cloudflare may apply a different chat template or sampling default than
fireworks).

### Eval-suite friction worth flagging

- The HITL trio (`test_hitl_agent`, `test_subagent_with_hitl`,
  `test_subagent_with_custom_interrupt_on`) fails on **every** Kimi run
  (baseline and profile) due to the SDK/test version skew described above —
  not a model issue. These tests need their `allowed_decisions` assertions
  updated.
- The compact-tool tests likewise fail consistently, so they don't help
  discriminate between profile variants.
- LLM-judged followup-quality tests are the noisiest signal — single-trial
  runs swing pass/fail on the same trajectory due to judge variance.
  Multi-trial averaging is necessary; single-trial verdicts (including
  those in the company-wide model-comparison table) are misleading.

## Recommended next steps

1. **Drop Baseten from `_KIMI_K2_6_MODEL_SPECS`** — net-negative across 3
   trials, no metric where it wins materially. v5 candidate.
2. **Restore `<respect_user_specifications>`** (with multi-trial as the
   gate this time, not single-trial) — was load-bearing for followup-quality
   tests on both providers. v6 candidate.
3. **Run a `fireworks:moonshotai/...` direct baseline** (if Fireworks hosts
   Kimi K2.6 directly) to disambiguate "OpenRouter routing" from
   "upstream serving."
4. **Investigate Baseten's chat-template / sampling defaults** before
   re-attempting a Baseten profile.
5. **Update the company-wide model-comparison table** to use 3-trial means
   (current entries cherry-pick best-of-N from single-trial runs and
   misrepresent typical performance, especially for followup-quality
   sensitive categories).

## File pointers

- Profile module: `libs/deepagents/deepagents/profiles/harness/_kimi_k2_6.py`
- Bootstrap registration: `libs/deepagents/deepagents/profiles/_builtin_profiles.py`
  (note: this file was reverted out-of-session at one point during the
  investigation; the local file may not currently wire the import — verify
  before re-running.)
- Tests: `libs/deepagents/tests/unit_tests/test_models.py`
  (`TestBuiltInHarnessProfiles` class, `test_kimi_*` methods).
