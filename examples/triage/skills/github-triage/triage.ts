import {
  listDiscussions,
  listIssues,
  listPullRequests,
  type FetchRunOptions,
  type RepoScope,
  type SourceItem,
  type StateFilter,
} from "./github.ts";
import {
  ClusterSet,
  type ClusterAssignment,
  type ClusterSetOptions,
  type ClusterSetSnapshot,
} from "./cluster.ts";
import {
  condenseGitHubItem,
  type TriageRecord,
} from "./condense.ts";
import { mapWithConcurrency } from "./utils.ts";

export type { FetchRunOptions, RepoScope, SourceItem, StateFilter } from "./github.ts";
export type {
  ClusterAssignment,
  ClusterSetOptions,
  ClusterSetSnapshot,
} from "./cluster.ts";
export type { TriageAction, TriageRecord } from "./condense.ts";

export type TriageSourceSelection = {
  issues?: boolean;
  prs?: boolean;
  discussions?: boolean;
};

export type TriageOptions = TriageSourceSelection & {
  state?: StateFilter;
  fetch?: FetchRunOptions;
  max_concurrency?: number;
  cluster?: Omit<ClusterSetOptions, "initial_clusters">;
};

export type TriageResult = {
  repo: string;
  scope: RepoScope;
  source_items: SourceItem[];
  triage_records: TriageRecord[];
  assignments: ClusterAssignment[];
  snapshot: ClusterSetSnapshot;
  forgotten_items: TriageRecord[];
  toMarkdown: () => string;
  toString: () => string;
};

const DEFAULT_MAX_CONCURRENCY = 12;
const DEFAULT_SOURCE_SELECTION: Required<TriageSourceSelection> = {
  issues: true,
  prs: true,
  discussions: true,
};

function sourceKey(item: Pick<SourceItem, "type" | "id">): string {
  return `${item.type}:${item.id}`;
}

function triageKey(record: Pick<TriageRecord, "type" | "id">): string {
  return `${record.type}:${record.id}`;
}

function parseRepoKey(repoKey: string): { org: string; repo: string } {
  const value = repoKey.trim();
  const [org, repo, ...rest] = value.split("/");
  if (!org || !repo || rest.length > 0) {
    throw new Error(
      `Invalid repo key "${repoKey}". Expected "<org>/<repo>", e.g. "langchain-ai/deepagents".`
    );
  }
  return { org, repo };
}

function normalizeSelection(
  options: TriageSourceSelection | undefined
): Required<TriageSourceSelection> {
  const selection = {
    issues: options?.issues ?? DEFAULT_SOURCE_SELECTION.issues,
    prs: options?.prs ?? DEFAULT_SOURCE_SELECTION.prs,
    discussions: options?.discussions ?? DEFAULT_SOURCE_SELECTION.discussions,
  };
  if (!selection.issues && !selection.prs && !selection.discussions) {
    throw new Error("At least one source selector must be true: issues, prs, discussions.");
  }
  return selection;
}

async function pullSourceItems(
  scope: RepoScope,
  selection: Required<TriageSourceSelection>,
  fetchOptions: FetchRunOptions | undefined
): Promise<SourceItem[]> {
  const loaders: Array<() => Promise<SourceItem[]>> = [];
  if (selection.issues) {
    loaders.push(() => listIssues(scope, fetchOptions));
  }
  if (selection.prs) {
    loaders.push(() => listPullRequests(scope, fetchOptions));
  }
  if (selection.discussions) {
    loaders.push(() => listDiscussions(scope, fetchOptions));
  }

  const fetchFanout = Math.max(1, Math.floor(fetchOptions?.fetch_concurrency ?? 2));
  const batches = await mapWithConcurrency(
    loaders,
    Math.min(fetchFanout, loaders.length),
    (loader) => loader()
  );
  const dedup = new Map<string, SourceItem>();
  for (const batch of batches) {
    for (const item of batch) {
      dedup.set(sourceKey(item), item);
    }
  }

  return [...dedup.values()].sort((a, b) => {
    if (a.id !== b.id) {
      return a.id - b.id;
    }
    return a.type.localeCompare(b.type);
  });
}

function escapePipe(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\n+/g, " ").trim();
}

function createMarkdown(
  repo: string,
  sourceItems: SourceItem[],
  records: TriageRecord[],
  assignments: ClusterAssignment[],
  snapshot: ClusterSetSnapshot
): string {
  const sourceByKey = new Map<string, SourceItem>();
  for (const item of sourceItems) {
    sourceByKey.set(sourceKey(item), item);
  }
  const recordByKey = new Map<string, TriageRecord>();
  for (const record of records) {
    recordByKey.set(triageKey(record), record);
  }

  const lines: string[] = [];
  lines.push(`# Triage Report — ${repo}`);
  lines.push("");
  lines.push(`- Source items: ${sourceItems.length}`);
  lines.push(`- Triaged items: ${records.length}`);
  lines.push(`- Clusters: ${snapshot.clusters.length}`);
  lines.push(`- Forgotten items: ${snapshot.forgotten_count}`);
  lines.push(`- Review runs: ${snapshot.review_runs}`);
  lines.push("");

  for (const cluster of snapshot.clusters) {
    lines.push(`## ${cluster.name}`);
    lines.push("");
    lines.push(`- Cluster ID: \`${cluster.cluster_id}\``);
    lines.push(`- Brief: ${cluster.brief || "_No brief_"}`);
    lines.push(`- Items: ${cluster.references.length}`);
    lines.push("");
    lines.push("| Item | Action | URL | Context |");
    lines.push("| --- | --- | --- | --- |");
    for (const ref of cluster.references) {
      const key = `${ref.type}:${ref.id}`;
      const source = sourceByKey.get(key);
      const record = recordByKey.get(key);
      const urlCell = source?.url ? `[link](${source.url})` : "";
      const context = escapePipe(record?.context_blurb || ref.context_blurb);
      lines.push(`| ${ref.type}#${ref.id} | ${ref.action} | ${urlCell} | ${context} |`);
    }
    lines.push("");
  }

  const forgottenAssignments = assignments.filter((assignment) => assignment.forgotten);
  if (forgottenAssignments.length > 0) {
    lines.push("## Forgotten Items");
    lines.push("");
    lines.push("| Item | Reason |");
    lines.push("| --- | --- |");
    for (const assignment of forgottenAssignments) {
      const key = `${assignment.item_type}:${assignment.item_id}`;
      const record = recordByKey.get(key);
      lines.push(
        `| ${(record?.type ?? assignment.item_type)}#${assignment.item_id} | ${escapePipe(
          assignment.rationale
        )} |`
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Single-entrypoint triage orchestrator.
 *
 * Pipeline:
 * 1. Pull selected source items from GitHub.
 * 2. Condense each source item into a normalized triage record.
 * 3. Start `ClusterSet` empty and let classifier theme decisions form clusters.
 * 4. Use one bounded-concurrency loop for `condense -> enqueue ClusterSet`.
 * 5. Await cluster completion and return a report object.
 */
export async function triage(
  repo: string,
  options: TriageOptions = {}
): Promise<TriageResult> {
  const { org, repo: name } = parseRepoKey(repo);
  const selection = normalizeSelection(options);
  const scope: RepoScope = {
    org,
    repo: name,
    state: options.state ?? "open",
  };

  const sourceItems = await pullSourceItems(scope, selection, options.fetch);
  const clusterSet = new ClusterSet({
    ...(options.cluster ?? {}),
    initial_clusters: [],
  });

  const maxConcurrency = options.max_concurrency ?? DEFAULT_MAX_CONCURRENCY;
  const pipelineResults = await mapWithConcurrency(
    sourceItems,
    maxConcurrency,
    async (item) => {
      const record = await condenseGitHubItem(item);
      const assignment = await clusterSet.schedule(record);
      return { record, assignment };
    }
  );
  const triageRecords = pipelineResults.map((row) => row.record);
  const assignments = pipelineResults.map((row) => row.assignment);

  const snapshot = await clusterSet;
  const forgottenKeys = new Set(
    assignments
      .filter((assignment) => assignment.forgotten)
      .map((assignment) => `${assignment.item_type}:${assignment.item_id}`)
  );
  const forgottenItems = triageRecords.filter((record) =>
    forgottenKeys.has(triageKey(record))
  );

  const markdown = createMarkdown(repo, sourceItems, triageRecords, assignments, snapshot);

  return {
    repo,
    scope,
    source_items: sourceItems,
    triage_records: triageRecords,
    assignments,
    snapshot,
    forgotten_items: forgottenItems,
    toMarkdown: () => markdown,
    toString: () => markdown,
  };
}
