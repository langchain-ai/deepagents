# Switchyard × Deep Agents — context and state of play

Orientation doc. Read this first; it links to everything else.

- **`RESULTS.md`** — measured numbers, findings, methodology caveats
- **`BRANCH_PLAN.md`** — the plan for running Harbor / unified evals in CI
- **`README.md`** — how to run the pytest arms

---

## What we are trying to do

NVIDIA launches **Switchyard**, an open-source LLM router, on **11 Aug 2026**,
alongside **Nemotron 3.5 Nano** (also called "Lightning"). They want a credible
external benchmark to launch with. LangChain is the partner evaluation — their
own notes say *"benchmarks are internal; no external customer results available
yet; partner evaluations underway."* Ours would be the first outside number.

**Deliverable**, after Harrison's call in Slack (*"zero change gateway; lets do
a deepagnets middleware"*): a **Deep Agents middleware** plus a devrel blog.
Explicitly *not* the LangChain LLM Gateway integration, which was the other
option on the table and needs PM scoping.

People: Sean Lopp (NVIDIA, running a parallel benchmark), Dhruv Nandakumar
(NVIDIA, owns Switchyard), Karan Singh (driving the blog), Vivek Trivedy
(LangChain lead).

**Metrics asked for:** cost per completed task, task success rate, latency.

## What Switchyard is

A router between an agent and its models that sends each LLM call to a cheap
("weak") or expensive ("strong") tier. Two forms, same engine:

- **Server** — OpenAI-compatible proxy. Point `base_url` at it.
- **Library** — importable Python with the routing engine as a PyO3 extension.
  This is what a middleware would use.

Routing algorithms that matter here:

- **stage_router** — reads tool-result history for signals. **Ruled out.** Needs
  dense coding-agent tool traffic; Sean tested it on our evals and it no-ops.
- **llm_classifier + `mode = "escalation"`** — start on weak, a small judge
  reads each completed turn, and after N consecutive "this is going badly"
  verdicts the session latches to strong. **This is what we use.**

Traps worth knowing up front (also in the `switchyard-config-surfaces` memory):
there is **no `cascade` route type** despite widespread references to one;
escalation is `type = "llm_classifier"` + `mode = "escalation"`, not
`type = "escalation"`; `/v1/stats` reports **no cost field**; and the Rust server
ships no prebuilt binary, with escalation requiring 0.2.0 which is not on PyPI.

## What has been built

A benchmark harness in this directory. Every arm routes **through** Switchyard —
including single-model baselines, as `passthrough` routes — so all arms are
priced by one instrument. Running baselines direct-to-provider would lose the
cache read/write split and make cost incomparable.

- `models.sh` — single edit point for the model pool
- `routes-*.toml.tmpl` + `render.sh` — six configs, all dry-run validated
- `collect_stats.py` — `/v1/stats` → per-model cost, judge broken out
- `run_arms.sh` (pytest) / `run_harbor_arms.sh` (Harbor)

Two code changes outside this directory, both inert unless routing:
- `tests/evals/conftest.py` — `--base-url`, forces `use_responses_api=False`,
  sends a per-test `x-switchyard-session-id`
- `deepagents_harbor/langgraph_project/langgraph_agent.py` —
  `_apply_router_session_header()`, same idea for Harbor

**The session header is load-bearing.** Escalation with `confirmations >= 2`
retains its streak per session and exposes no message-hash fallback; without an
id the streak resets every turn and the router silently never escalates — a null
result that looks real.

## What we found

Four arms, pytest suite, 145 non-memory tests, n=1. Full detail in `RESULTS.md`.

| Arm | Correctness | Uncached cost | $/completed task |
|---|---|---|---|
| GLM 5.2 only | 84.1% | $5.19 | $0.0426 |
| Opus 4.8 only | 84.1% | $28.16 | $0.2308 |
| Escalation Opus↑GLM | 83.4% | $8.40 | $0.0694 |
| Nemotron 3.5 Nano only | 80.7% | $1.10 | $0.0094 |

All inside the 11.5pp minimum detectable effect — statistically indistinguishable.

Three findings that matter more than the table:

1. **The suite is saturated.** Four models spanning 30B-to-frontier and ~24x in
   per-token price all land within 3.4pp. Five of seven categories sit at
   88–100% for everyone; the residual failures are tasks *no* model solves. A
   routing benchmark needs a suite where the strong model actually wins.
2. **Routing has a cost threshold:** `f_min = judge_cost / (strong − weak)`.
   Under cache-aware pricing, the judge is a fixed ~$0.49/run tax and GLM's
   cache advantage makes GLM↑Nano structurally uneconomic. Under the selected
   uncached methodology, GLM costs $5.19 vs Nano's $1.10, so the threshold falls
   to ~12%. That makes GLM↑Nano plausible and the highest-value Harbor arm.
3. **Caching dominates the real invoice.** Nano lists 24x cheaper than GLM and
   is 4.53x cheaper per completed task under uncached pricing, but only 1.06x
   under cache-aware pricing. Three causes compound: NIM does no prompt
   caching (GLM's 90.5% hit rate is worth **77% off its bill**), thinking mode
   produces ~4x the output per call, and those outputs become context that
   inflates input on every later turn.

Finding 3 is the most useful thing here and is actionable for NVIDIA:
**prompt caching on NIM would improve Nemotron's cost story more than any price
cut could.**

## Pricing decision

NVIDIA prices without cache tokens because OpenRouter does not publish them.
Ignoring cache inflates GLM 4.3x, Opus 2.4x, escalation 3.0x, and Nano **1.0x**
— the bias runs entirely toward NVIDIA's own model.

Per completed task, Nano vs GLM is **1.06x cheaper** cache-aware and **4.53x
cheaper** uncached. Same runs. The methodology alone moves the headline 4x.

Decision: lead with **uncached pricing** for comparability with NVIDIA's
OpenRouter runs. Preserve the cache-aware figures as a sensitivity analysis and
state explicitly that the uncached headline does not match the Baseten/Anthropic
invoice when prompt-cache discounts apply.

## What is next

The Harbor integration lives on `srimanth/evals/switchyard-harbor`.

**Then, in priority order:**

1. Publish a public, digest-pinned Switchyard image from the agreed upstream SHA.
2. Run the one-task `nano` smoke through the LangSmith compose sidecar.
3. Dispatch the autonomous lite baselines and GLM↑Nano/Opus↑Nano arms at n=1,
   then raise rollouts after measuring spend and variance. See `BRANCH_PLAN.md`.
4. Write up the pytest results and the cache sensitivity honestly.
5. The middleware itself. Escalation internals are importable
   (`ESCALATION_JUDGE_SYSTEM_PROMPT`, `EscalationVerdict`,
   `EscalationJudgeConfig`), so it wraps rather than reimplements. ~1–2 days.

## Questions outstanding with NVIDIA

- Does NIM support prefix caching for Nemotron 3.5 Nano, and is it billed?
- Was Sean's Lightning baseline (64.1%) run with thinking **off**? We measured
  80.7% with it on — that gap likely explains his Ultra↑Lightning conclusion.
- His Opus↑GLM run escalated 1.3% of calls; ours 5%. Did his client send a
  session id? Without one the router structurally under-escalates.
- Can `judge_min_turn` be exposed in the Rust TOML? It exists in the Python
  config and is the highest-leverage cost knob available.
- When does 0.2.0 reach PyPI? The middleware cookbook is awkward until it does.
