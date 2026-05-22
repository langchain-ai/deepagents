# 00_STATE.md — Deep Agents Repository Analysis

## Repository Identity

- **Upstream**: `langchain-ai/deepagents` (MIT licensed, active)
- **Fork**: `okwn/deepagents` (cloned to `/root/oss-pr-campaign/repos/deepagents`)
- **Current branch**: `main` (synced with upstream/main)
- **Version**: 0.6.3 (SDK), 0.6.4 released upstream
- **Languages**: Python (≥3.11, <4.0), TypeScript (deepagents.js)
- **Ecosystem**: LangGraph/LangChain agent harness

## Upstream Repository Stats

| Metric | Value |
|---|---|
| Open Issues | ~40 (checked via API) |
| Open PRs | ~40 (checked via API) |
| Stars | ~8k+ (implied by LangChain org) |
| License | MIT |
| Archived | No |

## Cloned Repository Structure

```
/root/oss-pr-campaign/repos/deepagents/
├── libs/
│   ├── deepagents/          # SDK package (MIT, v0.6.3)
│   ├── cli/                 # CLI tool
│   ├── acp/                 # Agent Context Protocol
│   ├── code/                # DeepAgents Code (terminal coding agent)
│   ├── evals/               # Evaluation suite + Harbor integration
│   └── partners/            # Integration packages (daytona, etc.)
├── .github/workflows/        # CI/CD (lint, test, release, evals)
├── examples/                # 10+ example agents (nvidia, repl_swarm, etc.)
├── Makefile                 # Root-level orchestration
└── README.md
```

## Baseline Test Results

```
make test (libs/deepagents) — pytest, 1613 passed, 93 skipped, 4 xfailed, 10 warnings
Duration: ~75s
Coverage: 91% overall, graph.py at 100%, some middleware at 70% (overflow_clip)
```

## Key Observations

- Monorepo with 5 independently-versioned packages under `libs/`
- Very active upstream: dozens of open PRs, daily commits
- Strong CI: lint, test, benchmark, release-please automation
- Follows strict Conventional Commits with scope prefixes
- Google-style docstrings, ruff linting, ty type-checking
- Many middleware files have <95% coverage (filesystem at 90%, overflow_clip at 70%)
- `open-swe` and `help wanted` labeled issues indicate external contribution areas

## Repository Health

- **Test suite**: Excellent (1613 tests, comprehensive coverage)
- **CI/CD**: Robust (lint, test, benchmarks, release, auto-labeling)
- **Documentation**: README, AGENTS.md (detailed dev guidelines), inline docs
- **Contribution**: Clear conventions, conventional commits enforced
- **Activity**: Very high (daily releases, active issue/PR churn)