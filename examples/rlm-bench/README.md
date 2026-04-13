# RLM Benchmarks with Deep Agents CLI

Benchmark the [Deep Agents CLI](https://github.com/langchain-ai/deepagents) against long-context tasks from the **Recursive Language Models (RLMs)** paper ([Zhang, Kraska, Khattab 2025](https://arxiv.org/abs/2512.24601)).

## Background

The RLM paper shows that language models can handle arbitrarily long contexts by recursively decomposing them — spawning sub-LM calls over slices of the input rather than trying to process everything at once. This example tests whether the **deepagents CLI subagent mechanism** (the `task` tool) can achieve similar recursive decomposition, using `AGENTS.md` to prompt the agent into chunking context and delegating to subagents.

### Benchmarks

| Benchmark | Paper Task | Complexity | Description |
|-----------|-----------|------------|-------------|
| **S-NIAH** | Single Needle-in-a-Haystack | O(1) | Find a hidden phrase in a large body of text. From [RULER](https://arxiv.org/abs/2404.06654). |
| **OOLONG** | Long-context aggregation | O(N) | Classify and aggregate thousands of text entries. From [OOLONG](https://arxiv.org/abs/2511.02817). |
| **BrowseComp** | Multi-hop document QA | O(1) retrieval | Answer questions requiring reasoning across multiple documents. From [BrowseComp-Plus](https://arxiv.org/abs/2508.06600). |

The RLM paper also evaluates on **OOLONG-Pairs** (quadratic complexity) and **LongBench-v2 CodeQA**, which are not included here.

## How It Works

```
┌──────────────────────────────────────────────┐
│  run_benchmark.py                            │
│  ┌────────────────────────────────────────┐  │
│  │ For each benchmark task:               │  │
│  │  1. Generate/load task data            │  │
│  │  2. Write context to a temp file       │  │
│  │  3. Invoke deepagents CLI agent with   │  │
│  │     AGENTS.md prompt                   │  │
│  │  4. Agent reads context, spawns        │  │
│  │     subagents for chunks               │  │
│  │  5. Agent writes answer to output file │  │
│  │  6. Score predicted vs expected answer │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  AGENTS.md instructs the agent to:           │
│  - Treat context as external file            │
│  - Peek at structure (read with offset/limit)│
│  - Chunk and delegate to subagents           │
│  - Combine results into final answer         │
└──────────────────────────────────────────────┘
```

This mirrors the RLM paradigm: the context is stored externally (in a file, analogous to a REPL variable), and the agent programmatically examines, decomposes, and delegates sub-queries over it.

## Setup

```bash
cd examples/rlm-bench
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...
# Or for OpenAI:
# export OPENAI_API_KEY=sk-...
```

## Usage

### Run individual benchmarks

```bash
# S-NIAH: 10 tasks with 500-line haystacks (fast, good for testing)
uv run python run_benchmark.py s_niah --num-tasks 10

# OOLONG: 5 tasks with synthetic classification data
uv run python run_benchmark.py oolong --num-tasks 5

# BrowseComp: 5 tasks with 50 documents each
uv run python run_benchmark.py browsecomp --num-tasks 5 --num-documents 50
```

### Run all benchmarks

```bash
uv run python run_benchmark.py all --num-tasks 5
```

### Specify model

```bash
uv run python run_benchmark.py s_niah --model anthropic:claude-sonnet-4-20250514
uv run python run_benchmark.py s_niah --model openai:gpt-4o
```

### Scale up context length (S-NIAH)

```bash
# 2000-line haystacks (~8K tokens)
uv run python run_benchmark.py s_niah --num-lines 2000

# 10000-line haystacks (~40K tokens)
uv run python run_benchmark.py s_niah --num-lines 10000
```

## Results

Results are saved to `results/<benchmark>_results.json` with per-task scores and predictions.

### Evaluation Metrics

| Benchmark | Metric | From RLM Paper |
|-----------|--------|---------------|
| S-NIAH | String match (number in output) | Same |
| OOLONG | Exact match for labels; 0.75^|error| for numbers | Same |
| BrowseComp | Word overlap with expected answer | LLM-as-judge in paper |

## Data Sources

- **S-NIAH**: Fully synthetic, generated on the fly. No download needed.
- **OOLONG**: Uses synthetic data mimicking the `trec_coarse` split format. To use the real dataset, install `datasets` and it will attempt to load from `oolongbench/oolong-synth` on HuggingFace.
- **BrowseComp**: Uses synthetic multi-hop QA data. The real dataset is available at `Tevatron/browsecomp-plus` on HuggingFace (encrypted).

## Comparison to RLM Paper Results

From Table 1 of the paper (GPT-5 with RLM, sub-calls to GPT-5-mini):

| Task | Base GPT-5 | RLM(GPT-5) | Deep Agents (this example) |
|------|-----------|-------------|---------------------------|
| S-NIAH | ~100% (short) | ~100% | Run to find out! |
| OOLONG | 44.0 | 56.5 | Run to find out! |
| BrowseComp+ (1K) | 0.0* | 91.3 | Run to find out! |

*GPT-5 cannot fit 1000 documents in context.

## Architecture

```
rlm-bench/
├── AGENTS.md              # Agent instructions for recursive decomposition
├── run_benchmark.py       # Main benchmark runner
├── benchmarks/
│   ├── s_niah.py          # S-NIAH generator & evaluator
│   ├── oolong.py          # OOLONG loader & evaluator
│   └── browsecomp.py      # BrowseComp-Plus loader & evaluator
├── results/               # Output directory (gitignored)
├── pyproject.toml
└── README.md
```

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (2025)
- [RULER](https://arxiv.org/abs/2404.06654) — Hsieh et al. (2024)
- [OOLONG](https://arxiv.org/abs/2511.02817) — Bertsch et al. (2025)
- [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) — Chen et al. (2025)
- [RLM GitHub](https://github.com/alexzhang13/rlm)
