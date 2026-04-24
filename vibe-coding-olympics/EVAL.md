# `eval/` — Vibe Coding Olympics Judge

Single-site website scoring CLI + async library. LLM-as-judge across 6 axes plus one deterministic accessibility axis from axe-core. Caller parallelizes for multi-site rounds and owns aggregation.

## Install

```bash
cd eval
uv sync && .venv/bin/playwright install chromium
```

## CLI

```bash
.venv/bin/python judge.py \
  --url https://alice.fly.dev \
  --name Alice \
  --prompt "build a cat shrine" \
  --round 1
# optional: --out results  --no-axe  --no-langsmith
```

## Library

```python
import asyncio
from judge import evaluate_site
from aggregate import aggregate

async def run():
    results = await asyncio.gather(
        evaluate_site(url=u1, site_name="Alice", prompt=p, round_num=1),
        evaluate_site(url=u2, site_name="Bob",   prompt=p, round_num=1),
    )
    for score, _capture in results:
        print(score.site_name, aggregate(score.axes))
```

## Inputs

| Arg | Type | Notes |
| --- | --- | --- |
| `url` | `str` | Site to judge |
| `site_name` | `str` | Display/filename key (unique per parallel call) |
| `prompt` | `str` | Contestant's original prompt |
| `round_num` | `int \| None` | Drives filenames + LangSmith experiment prefix |
| `run_axe` | `bool` | Toggle a11y audit; default `True` |

## Outputs

`evaluate_site()` returns `(SiteScore, Capture)`.

```python
SiteScore(
    site_name="Alice",
    url="https://…",
    prompt="build a cat shrine",
    axes={
        "color": 0.82, "typography": 0.71, "layout": 0.65,
        "content_completeness": 0.78, "creativity": 0.55,
        "interpretation_quality": 0.80,
        "accessibility": 0.93,   # None if axe failed or --no-axe
    },
    metadata={
        "accessibility_audit": "ok",
        "accessibility_violations": 1,
        "accessibility_serious_violations": 0,
        "accessibility_passes": 42,
    },
    round_num=1,
)

Capture(
    html: str,
    screenshot_png: bytes,
    accessibility: AccessibilityReport | None,
)
```

All axis scores are `float ∈ [0.0, 1.0]`. `None` means the axis failed and is skipped by `aggregate()`.

## CLI side effects

- `results/round-{n}-{name}.json` — serialized `SiteScore`
- `screenshots/round-{n}-{name}.png` — raw PNG
- LangSmith experiment `round-{n}-{name}` — one feedback key per axis (best-effort; swallows errors)

Library calls are pure — no files written, no LangSmith calls. Caller opts in.

## Axes

| Axis | Source | What it measures |
| --- | --- | --- |
| `color` | LLM + vision | Palette cohesion, contrast, restraint |
| `typography` | LLM + vision | Readability, hierarchy, consistency |
| `layout` | LLM + vision | Composition, alignment, spacing rhythm |
| `content_completeness` | LLM + HTML | Meaningful copy, expected sections present |
| `creativity` | LLM + vision | Novelty, memorability, clever technique |
| `interpretation_quality` | LLM + both | Defensible reading of the (often vague) prompt |
| `accessibility` | axe-core | WCAG 2.0 A/AA violations, serious-impact doubled |

Judge model: `openai:gpt-5.4` (hard-coded in `evaluators.py`).

## Aggregation

Aggregation lives in `aggregate.py` so callers can reweight per round without touching the judge.

```python
from aggregate import aggregate, DEFAULT_WEIGHTS, score_accessibility

aggregate(score.axes)                       # default weights
aggregate(score.axes, {"creativity": 0.5,   # round-specific reweight
                       "interpretation_quality": 0.3,
                       "layout": 0.1, "color": 0.1})
```

Default weights (sum to 1.0):

| Axis | Weight |
| --- | --- |
| layout | 0.20 |
| content_completeness | 0.20 |
| creativity | 0.15 |
| interpretation_quality | 0.15 |
| color | 0.10 |
| typography | 0.10 |
| accessibility | 0.10 |

Weights renormalize across present axes, so `None` axes don't punish the total.

## What the caller can do

### 1. One-shot: score one site

```python
score, capture = await evaluate_site(url=..., site_name=..., prompt=..., round_num=1)
overall = aggregate(score.axes)
worst_axis = min(score.axes, key=lambda k: score.axes[k] or 1.0)
```
Use for: dashboards, single leaderboard entry, CI pass/fail gate.

### 2. Tournament: parallel + rank

```python
scores = await asyncio.gather(*[
    evaluate_site(url=u, site_name=n, prompt=p, round_num=r)
    for (n, u) in contestants
])
ranked = sorted(
    [(s.site_name, aggregate(s.axes)) for s, _ in scores],
    key=lambda x: x[1], reverse=True,
)
```
Use for: head-to-head rounds, live Olympics scoreboard.

### 3. Round-specific rubrics

```python
CREATIVITY_ROUND = {"creativity": 0.5, "interpretation_quality": 0.3,
                    "layout": 0.1, "color": 0.1}
total = aggregate(score.axes, CREATIVITY_ROUND)
```

## What `Capture` unlocks

- `capture.html` — diffing, scraping, content audits, DQ checks (`<meta viewport>`, etc.)
- `capture.screenshot_png` — attach to Slack/Linear/PR comments, archive, feed a custom vision check
- `capture.accessibility` — inspect raw axe violations (`report.violations[i]["help"]`, `["nodes"]`) to surface contestant-facing feedback
- Pair with LangSmith attachments to re-score the same capture under a new rubric — no re-fetch

## Layer-on ideas (not in `eval/` today)

- Narrator LLM that turns `score.axes` + `capture.html` into judge commentary
- SQLite/Postgres/Gsheet persistence for cross-round analytics
- Live scoreboard re-scoring every N seconds during the 5-min timer
- Golden-screenshot SSIM baselines per contestant
- DQ gates (e.g., `accessibility.serious_violation_count > 5`)

## Known limits

- Judge model is hard-coded — change in `evaluators.py` and restart
- Single viewport (1280×900) — no mobile scoring
- axe-core loads from CDN — audit cleanly skips (axis = `None`) when offline
- Axes are a fixed set — editing requires a code change; not runtime-configurable
- No `score_capture(capture, prompt)` helper yet — rescoring a cached capture under a new rubric means re-fetching
