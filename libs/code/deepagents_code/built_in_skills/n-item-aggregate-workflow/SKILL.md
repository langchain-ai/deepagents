---
name: n-item-aggregate-workflow
description: "Generate, score, and aggregate N items (haikus, headlines, candidate answers) in a single workflow — covers bracket tournaments, ranked lists, side-by-side comparisons, and other N-item aggregate tasks. Use when the user asks to: (1) generate N of something and pick the best, (2) run a tournament / bracket / playoff, (3) rank or score a batch of items, (4) produce a comparison table over N items."
license: MIT
compatibility: designed for deepagents-code
---

# N-item aggregate workflow

## Overview

N-item aggregate tasks ("generate 32 haikus and run a tournament", "rank these 20 headlines", "score 16 candidate answers") fail in a characteristic way when the entire batch is requested in a single structured-output call: the model's structured-output token budget is exceeded, the JSON truncates, and the tool surfaces `StructuredOutputValidationError: Failed to parse structured output for tool 'response_format'`. Retrying the same call shape burns tokens with no progress. Chunking the generation phase avoids the failure entirely, and halving the chunk on failure provides a deterministic recovery path.

## Best Practices

- **Default chunk size is 8.** For the generation phase, never request more than 8 items per structured-output call. For N > 8, issue ⌈N / 8⌉ sequential calls and concatenate results.
- **On `StructuredOutputValidationError`, halve the chunk size.** Re-issue the *remaining* work at the smaller chunk (8 → 4 → 2 → 1). Do not retry the same chunk size against the same error.
- **Stop at chunk size 1.** If a single-item call still fails structured-output validation, surface the failure to the user as a hard error — the problem is no longer batch shape.
- **Never retry the same batch shape more than once on the same error class.** A second identical attempt is wasted tokens.
- **Bracket / tournament phases are independent.** The pairwise comparison phase is small per call and is not subject to the same constraint — only the generation phase needs chunking.

## Process

1. Parse `N` from the user request (number of items to generate / rank / compare).
2. Pick a chunk size `C = min(8, N)`.
3. Generate items in ⌈N / C⌉ chunks. Concatenate.
4. If any chunk raises `StructuredOutputValidationError` referencing `response_format`:
   - Set `C = max(1, C // 2)`.
   - Re-issue the *failed chunk only* at the new size.
   - If `C == 1` and the call still fails, stop and report the failure.
5. Run the aggregate phase (bracket / tournament / ranking / scoring) on the concatenated list. This phase is per-pair or per-item and does not need chunking.
6. Return the final aggregate result.

## Common Pitfalls

- **Re-issuing the full N-item call after a structured-output error.** This is the failure mode this skill exists to prevent. The same call shape will fail the same way.
- **Treating user-initiated retries as a recovery strategy.** If the user re-sends the same prompt after a structured-output failure, the agent must still chunk — re-running the same oversized call from a fresh trace produces the same cascade.
- **Chunking the comparison/bracket phase unnecessarily.** Pairwise calls are small; chunking them adds latency with no benefit.
- **Picking a chunk size larger than 8 because "the model can probably handle it".** The structured-output budget is provider-controlled and not visible to the agent; 8 is the safe default.
