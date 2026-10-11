# Are the three arms comparing what we think they are?

The three-way run is meant to vary **the interface to the browser** and hold
everything else still. Audited 2026-10-10 against the live tool surfaces; three
things were varying that should not have been.

| | CLI (`browser-mda`) | MCP (`browser-mda-mcp`) | REPL (`browser-mda-repl`) |
|---|---|---|---|
| how the model acts | `execute` -> `browse` CLI over a shell | 6 MCP tools, called directly | 4 functions inside a QuickJS REPL |
| navigate | yes | yes | yes |
| act / observe / extract | n/a (shell verbs) | yes | yes |
| session start / end | implicit | **explicit tools** | implicit, per thread |
| JS eval in page | yes, via the shell | **none — server has no such tool** | ~~`evaluateJs`~~ now off by default |
| read current URL | yes | buried in the navigate payload | ~~`currentUrl`~~ now off by default |
| Stagehand model | n/a | `google/gemini-2.5-flash-lite` | ~~`anthropic/claude-sonnet-4-6`~~ now matched |
| viewport | sandbox Chrome default | 1024x768 (hosted default) | 1280x800 |

## 1. The Stagehand reasoning model differed -- fixed

The hosted Browserbase MCP endpoint runs `act`/`observe`/`extract` on
**`google/gemini-2.5-flash-lite`**. The REPL arm ran the same verbs on
**`anthropic/claude-sonnet-4-6`**, so the MCP-vs-REPL comparison was varying the
model *and* the interface at once.

Verified from the endpoint's own responses rather than from the docs alone: in
274 sampled `browserbase_*` results from the 2026-10-09 run, 72 carry

```
"response": { "id": "aiobj-...", "modelId": "gemini-2.5-flash-lite", ... }
```

The archived server source agrees -- `defaultConfig.modelName` is
`"google/gemini-2.5-flash-lite"`, commented "matches hosted MCP".

This is very likely not a small effect. `act` is a structured-output call, and
**82 of the MCP arm's 88 `act` failures were the single string
`AI_NoObjectGeneratedError: No object generated: response did not match
schema`** -- a 37.8% hard failure rate, and exactly the failure a small model
produces when asked for schema-conforming output. A meaningful share of what
`MCP_CORRECTNESS_ANALYSIS.md` attributes to "the MCP surface" may be attributable
to flash-lite.

Fixed by matching **downward**: `browser-mda-repl/tools/browser.py` now
defaults `STAGEHAND_MODEL` to `google/gemini-2.5-flash-lite`, the hosted
endpoint's model.

The other direction -- pinning MCP up to Sonnet -- needs a `modelApiKey`
Browserbase can call directly, and the LangSmith gateway key in
`ANTHROPIC_API_KEY` is not one (Browserbase does not know to send it to
`ANTHROPIC_BASE_URL`). That plumbing is in `browser-mda-mcp/tools/mcp.py` and
stays, as a no-op until `STAGEHAND_MODEL_KEY` is set, because the model is worth
varying deliberately once a usable key exists.

> Unverified: that Stagehand in the REPL arm serves `google/...` through
> Browserbase's Model Gateway without a provider key, the way it does today for
> the Anthropic model. Confirm on the first run that `act` still works.

> **Worth doing next, separately.** Both arms now run the browser verbs on
> flash-lite, and that model is a prime suspect for the 37.8% `act` failure
> rate -- 82 of 88 failures were `AI_NoObjectGeneratedError: ... did not match
> schema`, the signature of a small model missing structured output. Parity
> makes the arms comparable; it does not make either good. A two-point run at a
> stronger Stagehand model would separate "MCP surface is costly" from
> "flash-lite cannot drive it".

## 2. The REPL arm had two tools MCP cannot match -- fixed

The Browserbase MCP server exposes exactly six tools -- `start`, `end`,
`navigate`, `act`, `observe`, `extract` -- and **has no JavaScript-evaluation
tool at all**. The REPL arm shipped `evaluateJs` (arbitrary JS in the page) and
`currentUrl` on top of the four shared verbs, so it was being handed a strictly
larger capability, not just a different interface.

`evaluateJs` was not incidental: the instructions called it "the cheap read
path" and told the agent to prefer it. Removing it is the honest comparison and
it will probably cost the REPL arm some score.

Fixed in `browser-mda-repl/tools/browser.py`: the surface is now
`navigate, extract, act, observe`. `REPL_BROWSER_EXTRAS=1` restores the other
two for a separate "what can code mode actually do" run.
`browser-mda-repl/instructions.md` no longer references either.

`start`/`end` have no counterpart to remove -- they are session plumbing the
REPL handles per-thread, not a capability.

## 3. Viewport still differs -- not fixed

REPL launches 1280x800; the hosted MCP endpoint uses 1024x768 and exposes
`browserWidth`/`browserHeight` only as STDIO flags, not as query parameters. No
way to equalise without self-hosting the server. Low impact relative to the
first two, recorded rather than fixed.

## What this does not fix

The CLI arm is not capability-matched to either and cannot be: a shell with
`curl` and `grep` is strictly more powerful than four natural-language verbs.
That is the point of including it -- it is the unconstrained reference, not a
controlled third arm. CLI-vs-MCP and CLI-vs-REPL differences are "shell beats
natural-language verbs"; only MCP-vs-REPL isolates interface shape.
