---
name: multi-model-review
description: "Scaffold and dispatch a subagent backed by a different or additional model (e.g. a 'kimi' subagent), and build a reusable multi-agent code-review workflow. Use when the user asks to: (1) run or review something with a different/additional model, (2) set up a kimi (or openrouter/baseten/ollama) subagent, (3) override a subagent's model, (4) get a second-model opinion or cross-model review, or (5) package a multi-agent code-review workflow. Trigger on phrases like \"review with another model\", \"kimi subagent\", \"use a different model for review\", \"second opinion from X model\", or \"multi-agent review\"."
license: MIT
compatibility: designed for deepagents-code
---

# Multi-Model Review

## Overview

A subagent can run on a different model than the main agent. Its model is
overridden with a `model:` field in the subagent's
`.deepagents/agents/{name}/AGENTS.md` frontmatter (loaded by
`deepagents_code/subagents.py`). This skill scaffolds a review subagent on the
chosen model and dispatches it via the `task` tool.

**Do the work — don't just describe it.** When this skill triggers, actually
write the subagent `AGENTS.md` file and demonstrate a dispatch. Do not stop at
explaining the approach or asking "would you like me to build this?".

## Model Override

Set `model:` in the subagent frontmatter using `provider:model-name` format.
Kimi example (Fireworks):

```yaml
---
name: kimi-reviewer
description: Reviews code changes for bugs, edge cases, and style issues.
model: fireworks:accounts/fireworks/models/kimi-k2p7-code
---
```

Other providers for the same model:

- `openrouter:moonshotai/kimi-k2` (OpenRouter)
- `baseten:moonshotai/Kimi-K2-Instruct` (Baseten)
- `ollama:kimi-k2` (local Ollama)

Any `provider:model-name` string the model registry accepts works; the value is
passed through unchanged by `subagents.py`. Ensure the relevant provider API key
is set in the environment.

## Process

1. **Pick the model.** Confirm the provider/model (default to the kimi Fireworks
   example above unless the user named another) and that its API key is available.
2. **Scaffold the review subagent.** Create
   `.deepagents/agents/{name}/AGENTS.md` (folder name = subagent name). Put the
   chosen `model:` in the frontmatter and a review-prompt template in the body:

   ```markdown
   ---
   name: kimi-reviewer
   description: Reviews code changes for bugs, edge cases, and style issues.
   model: fireworks:accounts/fireworks/models/kimi-k2p7-code
   ---

   You are a focused code reviewer.

   Review the provided diff or files and report:
   1. Correctness bugs and edge cases (highest priority)
   2. Security issues
   3. Readability and style problems

   For each finding give: file:line, severity, and a concrete fix. Be specific;
   skip praise. End with a short overall verdict (ship / needs changes).
   ```

3. **Dispatch it.** Use the `task` tool, targeting the subagent by its `name`,
   with the concrete files/diff to review. For a multi-model review, dispatch the
   same review task to two subagents on different models in parallel and compare.
4. **Synthesize.** Merge the subagent findings, deduplicate, and present the
   combined review with a single verdict.

## Reusable Workflow

To make cross-model review repeatable, keep the review subagent(s) committed
under `.deepagents/agents/` so every session can dispatch them without
re-deriving the setup. To review with N models, create one
`agents/{name}/AGENTS.md` per model (same review-prompt body, different `model:`)
and dispatch them together.

## Common Pitfalls

- Placing the file at `agents/{name}.md` instead of `agents/{name}/AGENTS.md` —
  `subagents.py` only loads `{name}/AGENTS.md`.
- Missing provider API key — the model override resolves but the dispatch fails
  at call time.
- Only explaining the setup or asking for permission instead of writing the
  files and running a dispatch.
