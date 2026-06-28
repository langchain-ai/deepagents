---
name: bracket-tournament
description: "Runs a single-elimination bracket or round-robin over N items the agent generated, using one batched `js_eval` to produce items and one helper to judge each match. Use when the user asks to: (1) run a tournament, (2) build a bracket, (3) hold a single-elimination or head-to-head competition, (4) generate N items and rank/compare/pick the best, (5) round-robin or pairwise compare N items, or (6) seed and play out matches between generated items. Trigger on phrases like \"tournament\", \"bracket\", \"single-elimination\", \"head-to-head\", \"round-robin\", \"compare N <items>\", \"rank N <items>\", or \"generate N <items> and pick the best\"."
---

# Bracket Tournament

Standardized N-item tournament/aggregate workflow. The defaults below are the *only* shape — do not redefine the schema, the accumulator, or the bracket helper per session.

## Defaults (do not redefine)

- **Accumulator:** `globalThis.__items` (array of items), `globalThis.__bracket` (rounds), `globalThis.__results` (winners per round). Persists across `js_eval` calls.
- **Judge schema:** `{ winner: "a" | "b", reason: string }`. Do not invent additional fields.
- **Item shape:** `{ id: number, content: string }` where `id` is the seed (1-indexed).

## Workflow

Three `js_eval` calls total. Do not split steps further.

### 1. Generate items in one batched loop

One `js_eval` containing one `for` loop that issues `await task({...})` per item and pushes to `globalThis.__items`. Do **not** issue one `js_eval` per `task()` call.

```js
globalThis.__items = [];
for (let i = 1; i <= N; i++) {
  const content = await task({
    description: `Generate item ${i} about ${TOPIC}. Return only the item content, no preamble.`,
  });
  globalThis.__items.push({ id: i, content });
}
globalThis.__items.length;
```

### 2. Build the bracket

```js
function buildBracket(items) {
  // Pad to next power of two with byes (null).
  const n = items.length;
  const size = 1 << Math.ceil(Math.log2(Math.max(n, 2)));
  const padded = items.slice();
  while (padded.length < size) padded.push(null);
  const rounds = [];
  let current = padded;
  while (current.length > 1) {
    const pairs = [];
    for (let i = 0; i < current.length; i += 2) {
      pairs.push([current[i], current[i + 1]]);
    }
    rounds.push(pairs);
    current = new Array(current.length / 2).fill(null);
  }
  return rounds;
}
globalThis.__bracket = buildBracket(globalThis.__items);
globalThis.__bracket.length;
```

### 3. Play rounds with the shared `judgeMatch` helper

```js
async function judgeMatch(a, b) {
  if (a == null) return b;
  if (b == null) return a;
  const verdict = await task({
    description:
      `Judge which item wins this head-to-head match. Return JSON matching ` +
      `{"winner": "a" | "b", "reason": string}.\n\n` +
      `A (id=${a.id}): ${a.content}\n\nB (id=${b.id}): ${b.content}`,
  });
  const parsed = JSON.parse(verdict);
  return parsed.winner === "a" ? a : b;
}

globalThis.__results = [];
let winners = globalThis.__items.slice();
for (const round of globalThis.__bracket) {
  const next = [];
  for (const [a, b] of round) next.push(await judgeMatch(a, b));
  globalThis.__results.push(next);
  winners = next;
}
winners[0];
```

## Final summary (fixed format)

Render the result with this layout — no markdown reinvention:

```
# Tournament results: <topic>

**Winner:** #<id> — <content>

## Bracket
- Round 1: <winner ids>
- Round 2: <winner ids>
- ...
- Final: #<id>

## Entries
1. <content>
2. <content>
...
```

## Rules

- One `js_eval` for generation, one for bracket construction, one for play. Three calls total.
- Never redefine `judgeMatch` or the judge schema — the shape above is the contract.
- Never issue a separate `js_eval` per `task()` call. Loop inside one `js_eval`.
- Use `globalThis.__items` / `__bracket` / `__results` for state. Do not reintroduce items as locals between calls.
- For round-robin instead of single-elimination, replace step 2/3 with a pairwise loop but keep the same accumulator names and `judgeMatch` helper.
