---
name: pr-reviewer-response
description: Drafts a code-reviewer's response to a pull request and emits a merge-readiness decision. Builds on the pr-review-core workflow (use the same get_pr / list_open_prs tools and PR-loading conventions) and adds two strict output contracts the agent must follow: an ACK/CHANGE/VALIDATION/RISK reviewer reply, and a ship/no-ship merge-readiness summary with the top two risks and follow-up tasks owned by a person at a priority. Use when the user asks for a reviewer response, a reviewer reply, code review feedback to the author, an approval/merge decision, "should we ship this PR", or any PR-comment-shaped output beyond a plain explanation.
---

# pr-reviewer-response

This skill extends `pr-review-core`. Load `pr-review-core/SKILL.md` first if you have not already — that skill defines how to fetch PRs (`get_pr`, `list_open_prs`), how to read diffs and metadata, and the tool sequencing. This skill layers two strict output contracts on top of that workflow.

If the preflight tells you to put `MISSING_SKILL:pr-reviewer-response` on line 1, only do so when this file truly is unreadable. If you are reading this, the skill is loaded — do NOT emit that token.

## When to use

- The user asks for a reviewer's *response* / *reply* / *comment* on a PR (not just an explanation of the diff).
- The user asks whether a PR is ready to merge / ship.
- The user asks for required follow-ups, risks, or blockers before approval.

For plain "what does this PR do?" explanations, stay in `pr-review-core` — do not force this skill's templates onto a summary request.

## Workflow

1. Load the PR with `get_pr` (or pick from `list_open_prs` if no PR is specified). Follow `pr-review-core` for parsing the diff, files touched, tests, and CI status.
2. Identify, in order:
   - The single highest-leverage code-level change required before merge (becomes `CHANGE`).
   - The minimum proof the author can produce to unblock approval (becomes `VALIDATION`).
   - The most likely failure mode if the PR ships as-is (becomes `RISK`).
3. Decide ship / no-ship using the rubric below.
4. Emit the two outputs in order: **Reviewer reply** first, then **Merge-readiness decision**. Do not interleave them.

## Output 1: Reviewer reply (ACK / CHANGE / VALIDATION / RISK)

Exactly four labeled sections, in this order, each one line unless a code-level detail requires a short follow-up line. Each label is followed by a colon and the content.

```
ACK: <one sentence acknowledging what the PR does well or what intent is correct — concrete, not flattery>
CHANGE: <the single highest-leverage code-level fix required before merge — name the file/symbol when possible>
VALIDATION: <the minimal proof the author should produce — a test name, a command output, a screenshot, a benchmark — that would unblock approval>
RISK: <the most likely failure mode if this ships as-is, framed for the author, not the reader>
```

Field semantics — keep these tight:

- `ACK` is *specific*. "Nice refactor" is not specific; "Splitting the retry loop out of `send_batch` makes the timeout path testable" is.
- `CHANGE` is *code-level and singular*. If multiple changes are needed, pick the highest-leverage one and put the rest in follow-up tasks in Output 2. CHANGE is never "add more tests" — that belongs in `VALIDATION`.
- `VALIDATION` is *minimal and concrete*. Name the test, command, or artifact. "Add tests" is not VALIDATION; "Add a test that covers `send_batch` returning after `max_retries=0`" is.
- `RISK` is *one failure mode*, not a list. Pick the one most likely to actually bite in production.

## Output 2: Merge-readiness decision

```
Merge readiness: SHIP | NO-SHIP
Top risks:
  1. <risk one — what breaks, where>
  2. <risk two — what breaks, where>
Follow-ups:
  - [P0|P1|P2] <task> — owner: <@handle or role>
  - [P0|P1|P2] <task> — owner: <@handle or role>
  ...
```

Rubric:

- `SHIP` only if: no P0 follow-ups, `CHANGE` is non-blocking polish OR already addressed, CI is green or its failure is unrelated and acknowledged.
- `NO-SHIP` if any of: `CHANGE` is required before merge, P0 follow-up exists, tests are missing for a behavior change, or `RISK` is a data-loss / security / regression class issue.

Priority semantics:

- `P0` — blocks merge. Must be resolved in this PR.
- `P1` — must land in a follow-up PR before the next release; tracked, not blocking.
- `P2` — nice-to-have / cleanup; safe to defer.

Owner semantics: prefer the PR author for code-level follow-ups, the reviewer or a domain owner for infra/operational follow-ups. Use a GitHub handle when known, otherwise a role (e.g., `owner: @author`, `owner: platform-on-call`). Never leave an owner blank.

Top risks: always exactly two entries. If there is genuinely only one risk, the second slot is "None — happy path is well-covered by <test or check>" so the shape stays consistent.

## Tone

Reviewer-to-author, not reviewer-to-PM. Direct, specific, actionable. No hedging adverbs ("maybe", "perhaps", "might want to consider"). Past-tense observations of the diff, present-tense asks.
