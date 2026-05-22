---
name: github-triage
description: Deterministic interpreter workflow for GitHub triage and clustering, exposed through one `triage("<org>/<repo>", options)` entrypoint.
module: ./index.ts
---

# GitHub Triage Interpreter

Use this skill when you need a reproducible pull + classify + cluster pipeline for GitHub issues/PRs/discussions.

## Hard Rules

1. Use the `triage(...)` API in `index.ts` as the orchestration entrypoint.
2. Import via the exact alias path: `@/skills/github-triage`.
3. Do not read skill source files and reconstruct orchestration logic in interpreter code.
4. Do not invent API routes outside `github.ts`; source pull is code-defined.
5. Do not infer missing source items from memory or previous runs.
6. Keep all batch processing bounded with configured concurrency.
7. Do not use unbounded `Promise.all` over large source lists.
8. For internal skill modules, use explicit file extensions in relative imports (for example `./github.ts`).

If the `@/skills/...` import fails, stop and report the import failure. Do not rebuild this pipeline ad hoc.

## Entrypoint

```ts
const { triage } = await import("@/skills/github-triage");

const result = await triage("langchain-ai/deepagents", {
  issues: true,
  prs: true,
  discussions: true,
  state: "open",
});
```

Do not replace this import with `read_file(...)` over `github.ts` / `triage.ts` / `cluster.ts` / `condense.ts`.

## Options

```ts
type TriageOptions = {
  issues?: boolean;                 // default true
  prs?: boolean;                    // default true
  discussions?: boolean;            // default true
  state?: "open" | "closed" | "all"; // default "open"
  fetch?: {
    fetch_concurrency?: number;     // default 2
    page_delay_ms?: number;         // default 150
    max_retries?: number;           // default 5
    retry_base_delay_ms?: number;   // default 800
  };
  max_concurrency?: number;         // default 12
  cluster?: {
    review_every?: number;          // default 10
    reconcile_with_classifier?: boolean; // default false
    similarity_threshold?: number;  // default 0.65
    merge_threshold?: number;       // default 0.75
    max_cluster_refs_in_prompt?: number; // default 8
  };
};
```

## Pipeline Behavior

1. Pull selected sources from GitHub via `listIssues`, `listPullRequests`, `listDiscussions`.
2. Run classifier over each source item to produce normalized triage records.
3. Start from an empty cluster set.
4. Schedule triage records through `ClusterSet` with bounded concurrency.
5. Cluster assignment is classifier-only:
   - no lexical fallback
   - cluster creation comes only from classifier-proposed themes
   - low-confidence/invalid assignments become `forgotten`
6. Optional periodic cluster reconciliation (brief refresh + merge planning) is classifier-based and disabled by default.
7. Await `ClusterSet` completion and return a result object with data + renderers.

## Result Contract

`triage(...)` returns:

- `repo`, `scope`
- `source_items`
- `triage_records`
- `assignments`
- `snapshot` (cluster state + counters)
- `forgotten_items`
- `toMarkdown(): string`
- `toString(): string` (same as markdown)

## Markdown Output Shape

`toMarkdown()` includes:

1. Report header + run stats.
2. One section per cluster with:
   - cluster ID
   - brief
   - item count
3. A table per cluster:
   - typed item identifier (`issue#123`, `pr#456`, `discussion#789`)
   - triage action
   - URL
   - context blurb
4. A `Forgotten Items` section when present.
