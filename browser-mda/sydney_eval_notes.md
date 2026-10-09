# Judge notes: Opus vs Jev

Measured on this project's WebVoyager runs, October 2026. Everything below with a
number next to it was measured here unless labelled as published.

## Headline

| | Opus (`claude-opus-5-5`) | Jev (`typesafe/jev-latest`) |
| --- | --- | --- |
| latency, median | **2.53 - 3.32 s** | **0.30 - 0.31 s** |
| cost per call | **$0.0070 - $0.0080** | **$0.000078 - $0.000127** |
| ratio | — | **~8-10x faster, ~55-100x cheaper** |
| output | free text to parse | typed verdict + calibrated probability |

Two measurement passes (20 answers, then 8 answers x 5 repeats) gave consistent
figures; the ranges above are those two passes.

### What that means at our scale

A 30-task x 3-repetition suite is 90 answers per variant, 180 across both.

| | Opus | Jev |
| --- | --- | --- |
| 180 answers, 1 sample | ~$1.35 | ~$0.02 |
| 180 answers, 3-sample vote | ~$4.05 | ~$0.05 |
| grading wall-clock, serial | ~9 min | ~55 s |

Negligible either way against the agent runs themselves (~$0.08/task, so ~$14 per
suite). The reason to care is not the bill.

## Why it actually matters: the failure mode, not the price

Opus returns prose that has to be parsed into a verdict. That parsing step is
where this project lost a full experiment:

- the rubric made the model reason before answering
- `max_tokens=8` truncated it mid-thought, returning an **empty string**
- the parser's fallback was `FAILURE`

Result: **106 of 180 grades wrong**, every one of them in the direction that looks
like a real finding. A judge that silently defaults to "the agent failed" is worse
than a slow one.

Jev cannot fail this way. The answer is typed: a `noul` is a number in [0,1], a
`choice` is one of the options you defined. There is nothing to parse and no
fallback to get wrong.

## Variance

The published LangChain benchmark reports Jev matching a human reviewer on every
decision with **92-913x lower variance** than LLM judges, at 0.44 s and
$0.00035/call.

**I did not reproduce that here.** Over 8 answers x 5 repeats both judges were
perfectly stable: 0/8 verdict flips each. The instability I saw earlier in this
project came from (a) the truncation bug above and (b) single-sample Sonnet with
a short rubric -- not from Opus-at-proper-settings. So from this data the honest
claim is **equal stability, 8-10x faster, 55-100x cheaper**, not lower variance.

## Where they disagreed, and who was right

One genuine, stable disagreement in 20 answers: **ESPN--23**, Jev FAILURE 5/5 vs
Opus SUCCESS 5/5.

The task asked for the final score *and the top rebounder*. The agent's own
closing line:

> "I matched Ware's 19 rebounds to him by the order of the starters in the box
> score, because the stat rows don't carry player names in the page text."

The score was right (37+25+37+26 = 125, 30+36+39+27 = 132). The rebounder --
the thing the task asked for -- was an admitted positional guess. **Jev was
right.** Opus accepted a confident-sounding answer whose key fact the agent had
flagged as inferred.

This is the argument for asking several atomic questions rather than one. A
dedicated "are the claims read from the page or inferred?" question caught it; a
single correctness verdict did not. We have since reduced to `correct` + `blocked`
for simplicity, which trades that catch away -- worth remembering if
over-confident answers start slipping through.

## Question design (what the docs say)

One endpoint, one request can carry several questions, and extra questions barely
change latency.

| type | returns | use for |
| --- | --- | --- |
| `noul` | probability in [0,1] | yes/no properties. 0.5 means *unsure*, not *medium* |
| `choice` | chosen option + distribution + confidence | unordered categories, 2-255 |
| `score` | level 0..n-1, may land between levels, + confidence | ordered scales, 2-10 levels |

Gotchas found by probing, none of them documented in the LangSmith pages:

- model name must be provider-qualified: `typesafe/jev-latest`
- it is **not** a chat endpoint. `/v1/chat/completions` returns
  *"use /v1/systemone for this model"*
- `questions` is a **dict** keyed by your own names, not a list
- `criteria` for a `noul` takes **polarity keys** -- `{"true": ..., "false": ...}`.
  An arbitrary key returns *"Noul question must have criteria or instructions"*
- state should carry the data only; grading instructions belong in `criteria`

Feedback mapping: `noul` -> feedback `score`, `choice` -> feedback `value`,
`score` -> feedback `score` (0..n-1).

## Current setup

Two questions, two feedback keys:

- `correct` -- did the agent accomplish the task (binary in spirit, like
  WebVoyager's own SUCCESS / NOT SUCCESS)
- `blocked` -- did the site prevent it

Separate on purpose: a bot wall says nothing about agent quality, and folding it
into correctness is what made a correctly-reported Allrecipes block read as a
failure.

## Known gap

WebVoyager's official judge is **vision-based**: GPT-4V over end-of-run
screenshots plus the response, `temperature=0`, `seed=42`, verdict
SUCCESS / NOT SUCCESS. Its rule is that when response and screenshot conflict,
**the screenshot wins**.

We grade text only, so we cannot catch an answer the page never supported --
exactly the ESPN case. Our numbers are therefore an upper bound and not directly
comparable to published WebVoyager scores.
