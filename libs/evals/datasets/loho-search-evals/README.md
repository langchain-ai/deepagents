# loho-search-evals

A Harbor dataset of long-horizon open-web search tasks for Deep Agents. Each task gives the
agent a paragraph of interlocking constraints and asks it to name the one entity that satisfies
all of them, using only `web_search` and `fetch_url`. There is no local corpus — the agent has
to find everything on the live web.

## Source

Tasks are derived from [**LoHoSearch**](https://huggingface.co/datasets/meituan-longcat/LoHoSearch)
(`benchmark` config, `test` split — 544 human-verified questions across 11 domains, built from a
Wikipedia knowledge graph of 7M+ entities). The paper is
[arXiv:2606.12837](https://arxiv.org/abs/2606.12837); the best published score is **34.74%**.

The knowledge graph was the authors' question-*authoring* tool and is not part of the released
dataset. Agents receive prose only.

## Nothing derived from the benchmark is committed

LoHoSearch publishes its questions and answers XOR-obfuscated, with a canary that forbids
republishing them as plain text — otherwise the benchmark gets scraped into training data and
dies. **This repo is public**, so:

- Only `dataset.toml`, `manifest.json`, and this README are committed.
- Task directories (`loho-*/`) and the downloaded CSV (`.cache/`) are git-ignored and
  regenerated on demand.
- CI excludes the agent instruction and verifier logs from uploaded artifacts.

**Before running**, generate the tasks:

```
uv run python -m harbor_adapters.lohosearch.main --populate datasets/loho-search-evals
uv run harbor run --path datasets/loho-search-evals ...
```

CI runs the populate step automatically before building task images.

## Task identity: content hash, not row number

LoHoSearch publishes no ID column, so row position is the only handle upstream offers — and row
position is exactly what breaks when a dataset is revised. All rows share one canary and the
cipher is a fixed XOR, so each row's ciphertext is byte-stable and unique;
`sha256(question_ciphertext)` is therefore a durable key.

`manifest.json` binds our task names to those hashes, and the upstream revision is deliberately
**not** pinned, so published corrections flow through automatically:

| Upstream change | Result |
|---|---|
| Rows re-ordered | Nothing — every task still resolves |
| Rows added | Invisible until they are selected |
| A selected question edited | `--populate` fails, naming the task |
| A selected row deleted | Same |

A hash miss is a hard error rather than a silent substitution: an edited question is a different
question, and any calibration recorded against it is void.

## Scoring

Graded exactly as the paper does — two independent LLM judges, and the reward is the mean of
their binary verdicts, so a trial scores **0.0, 0.5, or 1.0**:

| Judge | Prompt | Default model |
|---|---|---|
| A | BrowseComp grader | `gpt-4.1` (OpenAI) |
| B | SimpleQA grader | `qwen/qwen-2.5-72b-instruct` (OpenRouter) |

Both prompts are verbatim from [openai/simple-evals](https://github.com/openai/simple-evals).
Models and credentials come from `LOHO_JUDGE_A_*` / `LOHO_JUDGE_B_*` in the verifier
environment.

> **Scores here are LoHoSearch-*derived*, not the published metric.** The paper's judge B is
> Qwen2.5-**32B**, which no configured provider serves — Fireworks 404s on it, OpenRouter
> carries 72B/7B/coder-32B/VL-72B, and Groq only has Qwen3. Judge B is therefore the 72B of the
> same family and generation. Don't compare these numbers directly to the paper's 34.74%.

**A judge that answers "incorrect" scores 0 for its half — that is a real grade. A judge that
cannot be reached fails the trial instead**, so it lands in `errored` rather than contributing
a zero. This matters: when judge B was misconfigured, every trial silently had a ceiling of 0.5
and reported it as a legitimate `0.0`. `judge_status.json` (downloaded, plaintext-free by
construction) records each judge's model, whether its call succeeded, its verdict, and the HTTP
status on failure.

The headline metric is **`mean_reward@K`**, which is what the paper's 34.74% is measured in.
`pass@k` is reported too, but on this axis it means "both judges agreed at least once in K
rollouts" — useful for spotting broken or impossible tasks during calibration, not for ranking
models.

## Network policy

Tasks declare `network_mode = "public"` with no phase overrides.

The agent needs it by definition — this is an open-web benchmark and the agent runs inside the
sandbox. The narrower design would have been an allowlisted baseline with a `[verifier]` block
restricted to the judge endpoints, since the verifier is the only place the answer key exists.
**That is not expressible on the LangSmith sandbox.** Harbor validates the requested mode
against the backend's declared capabilities (`environments/base.py`,
`validate_network_policy_support`), and only the docker backend sets `network_allowlist`;
requesting an allowlist elsewhere aborts every trial at construction.

Remaining protections: `fetch_url`'s SSRF guards still block private and link-local addresses,
`TAVILY_API_KEY` is on the shell-tool env denylist so the agent cannot read its own search key,
and the answer key is never copied into the agent image — Harbor mounts `tests/` only at verify
time, after the agent is killed.

> Note: the same capability check means the existing `context` dataset, which declares
> `network_mode = "allowlist"`, cannot run on the LangSmith sandbox under the pinned Harbor
> either. That is a pre-existing issue independent of this dataset.

## Tracing

Runs trace to LangSmith automatically — the leaf workflow forwards
`LANGSMITH_API_KEY`/`LANGSMITH_TRACING` into the sandbox and sets
`LANGSMITH_PROJECT` to the experiment name.

| | |
|---|---|
| Dataset | `lohosearch-bench` |
| Experiment | `deepagents-harbor-<branch>-search-<model>-search-<run_id>-<attempt>` |

The trace is the place to inspect a run: every `web_search` query, every
`fetch_url`, the reasoning between them, and token counts. CI artifacts are
deliberately stripped to score-level records only, so LangSmith is the debugging
surface.

**Traces contain the decrypted question and answer.** Keep the workspace private
and do not share trace links publicly, or the benchmark leaks by a route the
artifact redaction does not cover.

## Task set

The current manifest holds 3 tasks and is **not calibrated** — it exists to exercise the
harness. A calibrated, stratified set replaces it before this dataset feeds the unified
scorecard.
