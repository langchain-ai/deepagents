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
| A | BrowseComp grader | `gpt-4.1` |
| B | SimpleQA grader | `qwen2p5-32b-instruct` (Fireworks) |

Both prompts are verbatim from [openai/simple-evals](https://github.com/openai/simple-evals).
Models and credentials come from `LOHO_JUDGE_A_*` / `LOHO_JUDGE_B_*` in the verifier
environment. Grading fails closed: an errored or unparseable judge scores 0 for its half.

The headline metric is **`mean_reward@K`**, which is what the paper's 34.74% is measured in.
`pass@k` is reported too, but on this axis it means "both judges agreed at least once in K
rollouts" — useful for spotting broken or impossible tasks during calibration, not for ranking
models.

## Network policy

The agent phase runs with `network_mode = "public"`: the benchmark is open-web, and the agent
runs inside the sandbox, so it needs real egress. `fetch_url`'s SSRF guards still block private
and link-local addresses, and `TAVILY_API_KEY` is on the shell-tool env denylist so the agent
cannot read its own search key.

## Task set

The current manifest holds 3 tasks and is **not calibrated** — it exists to exercise the
harness. A calibrated, stratified set replaces it before this dataset feeds the unified
scorecard.
