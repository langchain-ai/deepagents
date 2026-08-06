# Switchyard routing benchmark

Six arms are available, each measured through
[Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) so every arm is priced
by the same instrument:

| Arm | Route | What it answers |
|---|---|---|
| `glm` | `passthrough` → GLM 5.2 | GLM baseline |
| `opus` | `passthrough` → Opus 4.8 | Opus baseline |
| `nano` | `passthrough` → Nemotron 3.5 Nano | Nano baseline |
| `escalation` | Opus↑GLM | original routed arm |
| `glm-nano` | GLM↑Nano | cost-focused routed arm |
| `opus-nano` | Opus↑Nano | widest capability gap |

The baselines deliberately run **through** Switchyard rather than direct to the
provider. Direct runs give no per-model cache read/write split, and since cache
reads price at roughly a tenth of base input while writes price at roughly
1.25x, the raw split is required for the cache-aware sensitivity analysis. The
selected publication methodology prices all input tokens at the uncached base
rate for comparability with NVIDIA's OpenRouter runs.

## Unified evals through Harbor

The unified workflow runs Switchyard as a Docker Compose sidecar inside each
LangSmith sandbox. The agent reaches it at `http://switchyard:4000/v1`.

Prerequisite: publish a **public, digest-pinned** Switchyard image. NVIDIA does
not currently publish one. Build their Rust server Dockerfile at the Switchyard
commit being evaluated, push it to the agreed registry, and use the immutable
`registry/owner/image@sha256:<digest>` reference. Credentials are injected only
at runtime and must never be build arguments or image layers.

Dispatch `.github/workflows/unified_evals.yml` from this branch with:

```text
models: openai:switchyard
categories: autonomous
profile: lite
rollouts: 1
sandbox_env: langsmith
switchyard_config: glm-nano
switchyard_image: registry.example/owner/switchyard@sha256:<64 hex chars>
```

Run each arm as a separate dispatch so its task rewards and router statistics
remain attributable. The workflow stages the selected TOML, safely forwards the
route's provider variables, attaches the trial id as the router session id, and
captures `/v1/stats` before teardown. Each shard artifact contains the per-trial
snapshots plus `switchyard-stats-summary.json`.

For a one-task proof before dispatching CI:

```bash
cd libs/evals/switchyard
./run_harbor_arms.sh \
  --image 'registry.example/owner/switchyard@sha256:<64 hex chars>' \
  --n-tasks 1 --rollouts 1 glm-nano
```

The smoke is successful only when Harbor records a completed trial and reward,
and the job directory contains `switchyard-stats.json`.

Price a merged shard snapshot with the selected uncached methodology:

```bash
python collect_stats.py report \
  --pricing uncached \
  --rate nemotron-3.5-nano=0.05,0.20 \
  path/to/switchyard-stats-summary.json
```

## Setup

Switchyard's Rust server is not published as a binary, and the escalation router
is a 0.2.0 feature not yet on PyPI, so it has to be built from source. Building
in Docker keeps the Rust toolchain off the host — NVIDIA ships a Dockerfile for
exactly this:

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard
docker build -f benchmark/switchyard-rust-server.Dockerfile -t switchyard-server:local .
```

Then render the configs:

```bash
cd libs/evals/switchyard
./render.sh
```

Validate each before spending a run on it. The TOML loader uses
`deny_unknown_fields`, so a stray key is a hard load error rather than a
silently-ignored setting:

```bash
docker run --rm -v "$PWD/routes-escalation.toml:/c.toml:ro" \
  switchyard-server:local --config /c.toml --dry-run
```

Required in the environment: `BASETEN_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY` (judge), `LANGSMITH_API_KEY` (tracing).

## Running an arm

Same shape for all three — only the config and the LangSmith suite name change,
because every route publishes the same client-facing id (`switchyard`).

```bash
# terminal 1 — from libs/evals/switchyard.
# --host 0.0.0.0 is required: the server binds inside the container, and
# 127.0.0.1 there would not be reachable through the published port.
docker run --rm -p 4000:4000 \
  -v "$PWD/routes-glm.toml:/c.toml:ro" \
  -e BASETEN_API_KEY -e ANTHROPIC_API_KEY -e GOOGLE_API_KEY \
  switchyard-server:local --config /c.toml --host 0.0.0.0 --port 4000

# terminal 2, from libs/evals
python switchyard/collect_stats.py reset

LANGSMITH_TEST_SUITE=switchyard-glm-only \
uv run --group test pytest tests/evals -v --tb=short \
  --model openai:switchyard --base-url http://localhost:4000/v1 \
  --eval-category-exclude memory

python switchyard/collect_stats.py snapshot switchyard/runs/glm.json
```

`--eval-category-exclude memory` is what selects the 145 (of 199) tests.

Repeat with `routes-opus.toml` / `switchyard-opus-only`, then
`routes-escalation.toml` / `switchyard-escalation-opus-glm`.

## Reading the result

```bash
python switchyard/collect_stats.py report \
  switchyard/runs/glm.json switchyard/runs/opus.json switchyard/runs/esc.json
```

Prints calls, input, cache read, cache write, output, cache hit rate, p50
latency, and cost per model — judge traffic included as its own row, since its
tokens are a real cost of the routed arm.

Accuracy comes from pytest itself (per-category correctness in the run summary,
and in LangSmith under the `LANGSMITH_TEST_SUITE` name).

## Smoke test first

One file rather than 145, to prove the chain before spending a full run:

```bash
python switchyard/collect_stats.py reset
uv run --group test pytest tests/evals/test_tool_selection.py -v \
  --model openai:switchyard --base-url http://localhost:4000/v1
python switchyard/collect_stats.py snapshot /tmp/smoke.json
python switchyard/collect_stats.py report /tmp/smoke.json
```

Three things to check in that output:

1. **`total_requests` > 0** — traffic actually routed through the proxy.
2. **`cache R` > 0 on the Opus arm.** Switchyard marks the final content block
   as an ephemeral cache breakpoint when translating to Anthropic
   (`libsy-llm-client/src/client.rs:707`). If cache reads are zero, Opus is
   paying full input price and the cost comparison is off by 3–4x.
3. **A cost figure, not `unpriced`.** An unrecognised model id needs
   `report --provider <substring>=<provider>`.

On the escalation arm, also confirm the strong tier has a non-zero call count.
Zero means the route never latched — most likely a missing session id, which
`conftest.py` supplies per test via `x-switchyard-session-id`. `confirmations`
above 1 retains its streak per session, and escalation mode exposes no
message-hash fallback, so without that header the streak resets every turn and
the arm silently degrades to plain GLM.

## Notes

- `/v1/stats` reports no cost — there is no cost field anywhere in
  `crates/switchyard-server`. Cost here is computed from the token breakdown via
  `deepagents_code.cost_tracking` (genai-prices plus the repo's bundled
  overrides). The escalation doc's claim that the snapshot reports cost is a
  documentation bug.
- pytest runs serially, and each arm is a separate full run, so a
  `reset` → run → `snapshot` cycle attributes cleanly with no per-request
  tagging. That would not hold under `pytest -n`.
- `confirmations` is the main cost dial: `1` latches on the first escalate
  verdict and spends more on the strong tier; `2` (the benchmarked default)
  latches roughly a third as often.
