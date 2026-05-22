import type { TriageRecord } from "./condense.ts";

/**
 * Queue-driven clustering engine for triage records.
 *
 * Architecture overview:
 * - `ClusterSet` accepts triage records incrementally through `schedule(...)`.
 * - Processing is single-flight: one queue consumer loop (`pump`) mutates state.
 * - Each record is assigned strictly by classifier output.
 * - Clusters store only references to source records, not copies of full artifacts.
 * - Every N processed records, a periodic review pass refreshes briefs and merges
 *   high-overlap clusters.
 *
 * Concurrency model:
 * - Scheduling is "fire-and-forget" from the caller perspective.
 * - Internal processing remains deterministic (one-at-a-time assignment).
 * - `ClusterSet` is `PromiseLike`, so callers can `await clusterSet` to wait for
 *   all in-flight and newly-added work to settle.
 *
 * Tool integration:
 * - Primary cluster placement uses `classifier` when available.
 * - Optional periodic review and merge planning also use `classifier`.
 * - Classifier can be injected explicitly or discovered from `globalThis.tools`.
 */

/**
 * Minimal immutable reference shape persisted inside a cluster.
 *
 * This keeps cluster state compact and stable even if upstream triage schemas
 * evolve. References are sufficient for later "double click" inspection through
 * additional tool calls.
 */
export type ClusterReference = {
  id: number;
  type: string;
  action: string;
  context_blurb: string;
  reason: string;
  duplicate_of: number | null;
};

/**
 * Runtime cluster object maintained by `ClusterSet`.
 *
 * Notes:
 * - `brief` is mutable and refreshed during periodic review.
 * - `references` grows as new records are assigned.
 * - timestamps use epoch milliseconds from `options.now` (or `Date.now`).
 */
export type ClusterNode = {
  cluster_id: string;
  name: string;
  brief: string;
  references: ClusterReference[];
  created_at: number;
  updated_at: number;
};

/**
 * Normalized classifier routing decision.
 *
 * Decision carries classifier-selected theme metadata and optional resolved cluster.
 */
export type ClusterDecision = {
  cluster_id: string | null;
  theme_slug: string;
  theme_name: string;
  theme_brief: string;
  confidence: number;
  rationale: string;
};

/**
 * Result returned to callers for each scheduled record.
 *
 * This is intentionally operational (what happened) rather than semantic
 * (whether the clustering was "correct").
 */
export type ClusterAssignment = {
  item_id: number;
  item_type: string;
  cluster_id: string | null;
  created_cluster: boolean;
  confidence: number;
  rationale: string;
  forgotten: boolean;
  error?: string;
};

/**
 * Snapshot view returned by `ClusterSet#snapshot()` and by awaiting the set.
 */
export type ClusterSetSnapshot = {
  processed_count: number;
  forgotten_count: number;
  pending_count: number;
  review_runs: number;
  clusters: ClusterNode[];
};

/**
 * Configuration surface for `ClusterSet`.
 */
export type ClusterSetOptions = {
  /**
   * Number of processed records between review/merge passes.
   * Defaults to 10. Only used when `reconcile_with_classifier` is true.
   */
  review_every?: number;
  /**
   * Minimum confidence required to accept classifier theme routing.
   * Values are clamped to [0,1]. Defaults to 0.65.
   */
  similarity_threshold?: number;
  /**
   * Minimum merge confidence required during periodic review.
   * Values are clamped to [0,1]. Defaults to 0.75.
   * Only used when `reconcile_with_classifier` is true.
   */
  merge_threshold?: number;
  /**
   * Cap on references embedded in classifier/review prompts per cluster.
   * Defaults to 8.
   */
  max_cluster_refs_in_prompt?: number;
  /**
   * Optional explicit classifier function override.
   * If omitted, `globalThis.tools.classifier` is used when available.
   */
  classifier_tool?: (input: any) => Promise<any>;
  /**
   * Enable classifier-based periodic cluster review/merge.
   * Defaults to false.
   */
  reconcile_with_classifier?: boolean;
  /**
   * Deprecated alias for `reconcile_with_classifier`.
   */
  reconcile_with_task?: boolean;
  /**
   * Clock override for deterministic testing.
   */
  now?: () => number;
  /**
   * Optional pre-seeded clusters. Useful when auto-creation is disabled.
   */
  initial_clusters?: ClusterNode[];
  /**
   * Whether new clusters may be created from classifier theme output.
   * Defaults to true.
   */
  allow_new_clusters?: boolean;
};

type QueueEntry = {
  record: TriageRecord;
  resolve: (value: ClusterAssignment) => void;
};

type MergePlan = {
  merges: Array<{
    from_cluster_id: string;
    into_cluster_id: string;
    confidence: number;
    rationale: string;
  }>;
};

const DEFAULT_REVIEW_EVERY = 10;
const DEFAULT_SIMILARITY_THRESHOLD = 0.65;
const DEFAULT_MERGE_THRESHOLD = 0.75;
const DEFAULT_MAX_CLUSTER_REFS_IN_PROMPT = 8;
const DEFAULT_RECONCILE_WITH_CLASSIFIER = false;

const CLASSIFIER_JSON_SCHEMA = {
  name: "cluster_theme_decision",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["theme_slug", "theme_name", "theme_brief", "confidence", "rationale"],
    properties: {
      theme_slug: { type: "string", minLength: 1, maxLength: 80 },
      theme_name: { type: "string", minLength: 1, maxLength: 160 },
      theme_brief: { type: "string", minLength: 1, maxLength: 500 },
      confidence: { type: "number", minimum: 0, maximum: 1 },
      rationale: { type: "string", minLength: 1, maxLength: 500 },
    },
  },
  strict: true,
} as const;

const CLUSTER_BRIEF_JSON_SCHEMA = {
  name: "cluster_brief",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["brief"],
    properties: {
      brief: { type: "string", minLength: 1, maxLength: 500 },
    },
  },
  strict: true,
} as const;

const CLUSTER_MERGE_PLAN_JSON_SCHEMA = {
  name: "cluster_merge_plan",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["merges"],
    properties: {
      merges: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["from_cluster_id", "into_cluster_id", "confidence", "rationale"],
          properties: {
            from_cluster_id: { type: "string", minLength: 1, maxLength: 200 },
            into_cluster_id: { type: "string", minLength: 1, maxLength: 200 },
            confidence: { type: "number", minimum: 0, maximum: 1 },
            rationale: { type: "string", minLength: 1, maxLength: 500 },
          },
        },
      },
    },
  },
  strict: true,
} as const;

/**
 * Dynamic promise aggregator that stays awaitable while new promises are added.
 *
 * Why this exists:
 * - Native `Promise.all([...])` snapshots a fixed list.
 * - `ClusterSet` needs "open set" semantics where work may be scheduled while
 *   previous work is still running.
 *
 * Stability condition:
 * - `then(...)` resolves only after one full await cycle observes no new
 *   additions (`version` unchanged).
 */
export class PromiseSet implements PromiseLike<void> {
  private current: Promise<void> = Promise.resolve();
  private version = 0;

  /**
   * Adds a promise to the active set.
   *
   * The chain is extended so any current or future awaiter waits on this work.
   */
  add(promise: PromiseLike<any>): void {
    this.version += 1;
    const next = Promise.all([this.current, Promise.resolve(promise)]).then(
      () => undefined
    );
    this.current = next;
  }

  private async awaitStable(): Promise<void> {
    let seenVersion = this.version;
    while (true) {
      await this.current;
      if (seenVersion === this.version) {
        return;
      }
      seenVersion = this.version;
    }
  }

  /**
   * PromiseLike implementation that resolves when the set becomes stable.
   */
  then<TResult1 = void, TResult2 = never>(
    onfulfilled?:
      | ((value: void) => TResult1 | PromiseLike<TResult1>)
      | null,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | null
  ): Promise<TResult1 | TResult2> {
    return this.awaitStable().then(
      onfulfilled ?? undefined,
      onrejected ?? undefined
    );
  }
}

/**
 * Stateful clustering queue that supports incremental scheduling and deferred await.
 *
 * Core guarantees:
 * - Assignment loop is serialized to avoid racey cluster mutations.
 * - New items can be scheduled while processing is ongoing.
 * - Cluster creation policy is controlled by `allow_new_clusters`.
 * - Unassignable items are explicitly marked as forgotten.
 * - Awaiting the instance waits for the queue to drain *and* for any newly
 *   scheduled work observed during that drain.
 */
export class ClusterSet implements PromiseLike<ClusterSetSnapshot> {
  private readonly reviewEvery: number;
  private readonly similarityThreshold: number;
  private readonly mergeThreshold: number;
  private readonly maxClusterRefsInPrompt: number;
  private readonly reconcileWithClassifier: boolean;
  private readonly allowNewClusters: boolean;
  private readonly classifierTool?: (input: any) => Promise<any>;
  private readonly now: () => number;

  private readonly queue: QueueEntry[] = [];
  private readonly pending = new PromiseSet();
  private readonly clusterMap = new Map<string, ClusterNode>();
  private processing = false;
  private processedCount = 0;
  private forgottenCount = 0;
  private reviewRuns = 0;

  /**
   * @param options Tuning and tool injection controls for clustering behavior.
   */
  constructor(options: ClusterSetOptions = {}) {
    this.reviewEvery = Math.max(1, options.review_every ?? DEFAULT_REVIEW_EVERY);
    this.similarityThreshold = Math.max(
      0,
      Math.min(1, options.similarity_threshold ?? DEFAULT_SIMILARITY_THRESHOLD)
    );
    this.mergeThreshold = Math.max(
      0,
      Math.min(1, options.merge_threshold ?? DEFAULT_MERGE_THRESHOLD)
    );
    this.maxClusterRefsInPrompt = Math.max(
      1,
      options.max_cluster_refs_in_prompt ?? DEFAULT_MAX_CLUSTER_REFS_IN_PROMPT
    );
    this.reconcileWithClassifier =
      options.reconcile_with_classifier ??
      options.reconcile_with_task ??
      DEFAULT_RECONCILE_WITH_CLASSIFIER;
    this.allowNewClusters = options.allow_new_clusters ?? true;
    this.classifierTool = options.classifier_tool;
    this.now = options.now ?? (() => Date.now());
    if (Array.isArray(options.initial_clusters)) {
      for (const cluster of options.initial_clusters) {
        this.clusterMap.set(cluster.cluster_id, {
          ...cluster,
          references: [...cluster.references],
        });
      }
    }
  }

  /**
   * Adds or replaces a cluster definition in the active set.
   *
   * This is the primary mechanism to seed clusters when auto-creation is disabled.
   */
  upsertCluster(cluster: ClusterNode): void {
    this.clusterMap.set(cluster.cluster_id, {
      ...cluster,
      references: [...cluster.references],
    });
  }

  /**
   * Enqueues one triage record for asynchronous clustering.
   *
   * Processing starts automatically if not already running.
   *
   * @param record Triage record to assign.
   * @returns Per-item assignment promise.
   */
  schedule(record: TriageRecord): Promise<ClusterAssignment> {
    const work = new Promise<ClusterAssignment>((resolve) => {
      this.queue.push({ record, resolve });
      void this.pump();
    });

    this.pending.add(work.then(() => undefined));
    return work;
  }

  /**
   * Convenience bulk enqueue wrapper over {@link schedule}.
   */
  scheduleMany(records: TriageRecord[]): Promise<ClusterAssignment[]> {
    return Promise.all(records.map((record) => this.schedule(record)));
  }

  /**
   * Returns a defensive copy of current clusters and references.
   */
  getClusters(): ClusterNode[] {
    return [...this.clusterMap.values()].map((cluster) => ({
      ...cluster,
      references: cluster.references.map((ref) => ({ ...ref })),
    }));
  }

  /**
   * Returns an immutable runtime summary of queue + cluster state.
   */
  snapshot(): ClusterSetSnapshot {
    return {
      processed_count: this.processedCount,
      forgotten_count: this.forgottenCount,
      pending_count: this.queue.length,
      review_runs: this.reviewRuns,
      clusters: this.getClusters(),
    };
  }

  /**
   * PromiseLike implementation.
   *
   * `await clusterSet` resolves to a stable snapshot after all currently known
   * and newly-added in-flight work has finished.
   */
  then<TResult1 = ClusterSetSnapshot, TResult2 = never>(
    onfulfilled?:
      | ((value: ClusterSetSnapshot) => TResult1 | PromiseLike<TResult1>)
      | null,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | null
  ): Promise<TResult1 | TResult2> {
    return this.pending.then(() => this.snapshot()).then(
      onfulfilled ?? undefined,
      onrejected ?? undefined
    );
  }

  /**
   * Main single-consumer loop.
   *
   * Design rationale:
   * - Centralizes all state mutation (`queue`, `clusterMap`, counters).
   * - Prevents concurrent writes and keeps assignment order deterministic.
   * - If new work arrives during shutdown, loop re-enters automatically.
   */
  private async pump(): Promise<void> {
    if (this.processing) {
      return;
    }
    this.processing = true;

    try {
      while (this.queue.length > 0) {
        const entry = this.queue.shift();
        if (!entry) {
          continue;
        }

        const assignment = await this.processRecord(entry.record);
        entry.resolve(assignment);

        this.processedCount += 1;
        if (
          this.reconcileWithClassifier &&
          this.processedCount % this.reviewEvery === 0
        ) {
          await this.runPeriodicReview();
        }
      }
    } finally {
      this.processing = false;
      if (this.queue.length > 0) {
        void this.pump();
      }
    }
  }

  /**
   * Processes one triage record through classify -> resolve target -> append reference.
   *
   * Failure policy:
   * - Never throws to caller.
   * - On classifier/tool/parse failure, returns a forgotten assignment.
   */
  private async processRecord(record: TriageRecord): Promise<ClusterAssignment> {
    try {
      const clusters = [...this.clusterMap.values()];
      const decision = await this.classifyAgainstClusters(record, clusters);
      const belowThreshold = decision.confidence < this.similarityThreshold;
      let createdCluster = false;
      let target =
        !belowThreshold && decision.cluster_id != null
          ? this.clusterMap.get(decision.cluster_id)
          : undefined;
      if (!target && this.allowNewClusters) {
        target = this.createClusterFromDecision(decision);
        createdCluster = true;
      }
      if (!target) {
        this.forgottenCount += 1;
        return {
          item_id: record.id,
          item_type: record.type,
          cluster_id: null,
          created_cluster: false,
          confidence: decision.confidence,
          rationale: belowThreshold
            ? "Forgotten: classifier confidence below threshold and new clusters are disabled."
            : "Forgotten: classifier returned unknown theme and new clusters are disabled.",
          forgotten: true,
        };
      }

      target.references.push(this.toReference(record));
      target.updated_at = this.now();

      if (!target.brief) {
        target.brief = this.buildFallbackBrief(target);
      }

      return {
        item_id: record.id,
        item_type: record.type,
        cluster_id: target.cluster_id,
        created_cluster: createdCluster,
        confidence: decision.confidence,
        rationale: decision.rationale,
        forgotten: false,
      };
    } catch (err) {
      this.forgottenCount += 1;
      return {
        item_id: record.id,
        item_type: record.type,
        cluster_id: null,
        created_cluster: false,
        confidence: 0,
        rationale: "Forgotten: clustering error.",
        forgotten: true,
        error: this.toErrorMessage(err),
      };
    }
  }

  /**
   * Converts full triage record into compact reference stored in clusters.
   */
  private toReference(record: TriageRecord): ClusterReference {
    return {
      id: record.id,
      type: record.type,
      action: record.action,
      context_blurb: record.context_blurb,
      reason: record.reason,
      duplicate_of: record.duplicate_of,
    };
  }

  /**
   * Runs strict classifier-based routing.
   *
   * Invalid classifier output throws and is handled by caller policy.
   */
  private async classifyAgainstClusters(
    record: TriageRecord,
    clusters: ClusterNode[]
  ): Promise<ClusterDecision> {
    const prompt = this.buildClassifierPrompt(record, clusters);
    const raw = await this.callClassifierTool({
      description: `classify triage item ${record.id} into product theme`,
      prompt,
      json_schema: CLASSIFIER_JSON_SCHEMA,
      record,
    });
    return this.parseClassifierDecision(raw, clusters);
  }

  /**
   * Builds a theme-classification prompt from one triage record.
   */
  private buildClassifierPrompt(
    record: TriageRecord,
    clusters: ClusterNode[]
  ): string {
    const allowedThemes = clusters
      .map((cluster) => {
        const themeSlug = this.clusterThemeFromNode(cluster);
        if (!themeSlug) {
          return null;
        }
        return {
          theme_slug: themeSlug,
          theme_name: cluster.name,
          theme_brief:
            cluster.brief || this.buildFallbackBrief(cluster),
          item_count: cluster.references.length,
        };
      })
      .filter((theme): theme is {
        theme_slug: string;
        theme_name: string;
        theme_brief: string;
        item_count: number;
      } => theme != null);

    return [
      "Classify this triage record into one product theme slug.",
      "Return strict JSON: theme_slug, theme_name, theme_brief, confidence, rationale.",
      "theme_slug must be slug-like and concise (example: agent-reliability).",
      "Prefer existing themes from ALLOWED_THEMES over creating new ones.",
      `If no existing theme fits well (confidence < ${this.similarityThreshold.toFixed(2)}), propose a new one.`,
      "confidence must be in [0,1].",
      "",
      "TRIAGE_RECORD:",
      JSON.stringify(record, null, 2),
      "",
      "ALLOWED_THEMES:",
      JSON.stringify(allowedThemes, null, 2),
    ].join("\n");
  }

  /**
   * Invokes classifier tool with strict structured output requirements.
   */
  private async callClassifierTool(input: {
    description: string;
    prompt: string;
    json_schema: Record<string, unknown>;
    record?: TriageRecord;
    cluster?: ClusterNode;
    clusters?: ClusterNode[];
  }): Promise<unknown> {
    const globalTools = (globalThis as any).tools ?? {};
    const classifier = this.classifierTool ?? globalTools.classifier;
    if (!classifier) {
      throw new Error("classifier tool is unavailable");
    }
    return classifier(input);
  }

  /**
   * Parses classifier payload into normalized decision shape.
   *
   * Throws when payload cannot be interpreted safely.
   */
  private parseClassifierDecision(
    raw: unknown,
    clusters: ClusterNode[]
  ): ClusterDecision {
    const payload = this.parseObject(raw, "classifier output");
    const rawTheme = this.readRequiredString(
      payload,
      "theme_slug",
      "classifier output.theme_slug"
    );
    const normalizedTheme = this.slugify(rawTheme);
    if (!normalizedTheme) {
      throw new Error("classifier output.theme_slug is invalid");
    }
    const themeName = this.readRequiredString(
      payload,
      "theme_name",
      "classifier output.theme_name"
    );
    const themeBrief = this.readRequiredString(
      payload,
      "theme_brief",
      "classifier output.theme_brief"
    );
    const confidence = this.readRequiredNumber(
      payload,
      "confidence",
      "classifier output.confidence"
    );
    const rationale = this.readRequiredString(
      payload,
      "rationale",
      "classifier output.rationale"
    );
    const clusterId = this.resolveClusterIdFromTheme(normalizedTheme, clusters);

    return {
      cluster_id: clusterId,
      theme_slug: normalizedTheme,
      theme_name: themeName.slice(0, 160),
      theme_brief: themeBrief.slice(0, 500),
      confidence: Math.max(0, Math.min(1, confidence)),
      rationale,
    };
  }

  private resolveClusterIdFromTheme(
    theme: string,
    clusters: ClusterNode[]
  ): string | null {
    const normalizedTheme = this.slugify(theme);
    if (!normalizedTheme) {
      return null;
    }

    const directId = clusters.find(
      (cluster) => cluster.cluster_id === normalizedTheme
    );
    if (directId) {
      return directId.cluster_id;
    }

    const prefixedId = clusters.find(
      (cluster) => cluster.cluster_id === `cluster-${normalizedTheme}`
    );
    if (prefixedId) {
      return prefixedId.cluster_id;
    }

    const byDerivedTheme = clusters.find((cluster) => {
      const clusterTheme = this.clusterThemeFromNode(cluster);
      return clusterTheme === normalizedTheme;
    });
    if (byDerivedTheme) {
      return byDerivedTheme.cluster_id;
    }

    return null;
  }

  private clusterThemeFromNode(cluster: ClusterNode): string {
    const id = cluster.cluster_id.trim().toLowerCase();
    if (id.startsWith("cluster-") && id.length > "cluster-".length) {
      return this.slugify(id.slice("cluster-".length));
    }
    return this.slugify(cluster.name);
  }

  private titleizeSlug(value: string): string {
    return value
      .split("-")
      .map((part) => (part ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
      .join(" ");
  }

  private createClusterFromDecision(decision: ClusterDecision): ClusterNode {
    const baseId = `cluster-${decision.theme_slug}`;
    let clusterId = baseId;
    let suffix = 2;
    while (this.clusterMap.has(clusterId)) {
      clusterId = `${baseId}-${suffix}`;
      suffix += 1;
    }

    const now = this.now();
    const cluster: ClusterNode = {
      cluster_id: clusterId,
      name: decision.theme_name.slice(0, 160),
      brief: decision.theme_brief.slice(0, 500),
      references: [],
      created_at: now,
      updated_at: now,
    };
    this.clusterMap.set(cluster.cluster_id, cluster);
    return cluster;
  }

  private slugify(value: string): string {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80);
  }

  /**
   * Periodic cluster maintenance pass.
   *
   * Steps:
   * 1. Refresh each cluster brief via classifier.
   * 2. Ask classifier for merge candidates.
   * 3. Apply merges above confidence threshold.
   */
  private async runPeriodicReview(): Promise<void> {
    if (!this.reconcileWithClassifier) {
      return;
    }
    const clusters = [...this.clusterMap.values()];
    if (clusters.length <= 1) {
      return;
    }

    this.reviewRuns += 1;

    for (const cluster of clusters) {
      cluster.brief = await this.refreshClusterBrief(cluster);
      cluster.updated_at = this.now();
    }

    const mergePlan = await this.proposeClusterMerges(clusters);
    this.applyMergePlan(mergePlan);
  }

  /**
   * Regenerates one cluster brief from current references.
   *
   * Falls back to deterministic brief when classifier output is invalid.
   */
  private async refreshClusterBrief(cluster: ClusterNode): Promise<string> {
    const prompt = [
      `Write a concise brief for cluster ${cluster.cluster_id}.`,
      "Focus on shared user need and why these references belong together.",
      "Return strict JSON with key: brief.",
      "",
      "CLUSTER:",
      JSON.stringify(
        {
          cluster_id: cluster.cluster_id,
          name: cluster.name,
          references: cluster.references.slice(0, this.maxClusterRefsInPrompt),
        },
        null,
        2
      ),
    ].join("\n");

    try {
      const raw = await this.callClassifierTool({
        description: `refresh brief ${cluster.cluster_id}`,
        prompt,
        json_schema: CLUSTER_BRIEF_JSON_SCHEMA,
        cluster,
      });
      const payload = this.parseObject(raw, "cluster brief output");
      const brief = this.readRequiredString(
        payload,
        "brief",
        "cluster brief output.brief"
      );
      return brief.slice(0, 500);
    } catch {
      return this.buildFallbackBrief(cluster);
    }
  }

  /**
   * Deterministic local brief used when classifier output is missing/invalid.
   */
  private buildFallbackBrief(cluster: ClusterNode): string {
    const sample = cluster.references
      .slice(0, 2)
      .map((ref) => `#${ref.id} ${ref.context_blurb}`)
      .join(" ");
    return `Cluster for ${cluster.name}. ${sample}`.trim();
  }

  /**
   * Requests merge candidates from classifier.
   *
   * Output is parsed defensively and normalized into `MergePlan`.
   */
  private async proposeClusterMerges(clusters: ClusterNode[]): Promise<MergePlan> {
    const prompt = [
      "Given these clusters, propose merges for highly overlapping clusters only.",
      "Return strict JSON with key `merges` as array of objects:",
      "`{ from_cluster_id, into_cluster_id, confidence, rationale }`",
      "confidence must be in [0,1].",
      "",
      "CLUSTERS:",
      JSON.stringify(
        clusters.map((cluster) => ({
          cluster_id: cluster.cluster_id,
          name: cluster.name,
          brief: cluster.brief,
          references: cluster.references.slice(0, this.maxClusterRefsInPrompt).map((ref) => ({
            id: ref.id,
            context_blurb: ref.context_blurb,
          })),
        })),
        null,
        2
      ),
    ].join("\n");

    try {
      const raw = await this.callClassifierTool({
        description: "cluster merge planning",
        prompt,
        json_schema: CLUSTER_MERGE_PLAN_JSON_SCHEMA,
        clusters,
      });
      const payload = this.parseObject(raw, "cluster merge output");
      const mergesRaw = payload.merges;
      if (!Array.isArray(mergesRaw)) {
        throw new Error("cluster merge output.merges is not an array");
      }

      const merges: MergePlan["merges"] = [];
      for (const row of mergesRaw) {
        if (!row || typeof row !== "object" || Array.isArray(row)) {
          throw new Error("cluster merge row is not an object");
        }
        const merge = row as Record<string, unknown>;
        const fromClusterId = this.readRequiredString(
          merge,
          "from_cluster_id",
          "cluster merge output.from_cluster_id"
        );
        const intoClusterId = this.readRequiredString(
          merge,
          "into_cluster_id",
          "cluster merge output.into_cluster_id"
        );
        const confidence = this.readRequiredNumber(
          merge,
          "confidence",
          "cluster merge output.confidence"
        );
        const rationale = this.readRequiredString(
          merge,
          "rationale",
          "cluster merge output.rationale"
        );
        if (fromClusterId === intoClusterId) {
          continue;
        }
        merges.push({
          from_cluster_id: fromClusterId,
          into_cluster_id: intoClusterId,
          confidence: Math.max(0, Math.min(1, confidence)),
          rationale,
        });
      }
      return { merges };
    } catch {
      return { merges: [] };
    }
  }

  /**
   * Applies merge plan in-place and deduplicates merged references.
   *
   * Only merges with confidence >= configured merge threshold are applied.
   */
  private applyMergePlan(plan: MergePlan): void {
    for (const merge of plan.merges) {
      if (merge.confidence < this.mergeThreshold) {
        continue;
      }

      const from = this.clusterMap.get(merge.from_cluster_id);
      const into = this.clusterMap.get(merge.into_cluster_id);
      if (!from || !into) {
        continue;
      }

      const mergedRefs = new Map<string, ClusterReference>();
      for (const ref of into.references) {
        mergedRefs.set(`${ref.type}:${ref.id}`, ref);
      }
      for (const ref of from.references) {
        mergedRefs.set(`${ref.type}:${ref.id}`, ref);
      }

      into.references = [...mergedRefs.values()];
      into.updated_at = this.now();
      into.brief = `${into.brief} Merged ${from.cluster_id}: ${merge.rationale}`.trim();
      this.clusterMap.delete(from.cluster_id);
    }
  }

  private parseObject(raw: unknown, label: string): Record<string, unknown> {
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      return raw as Record<string, unknown>;
    }
    if (typeof raw === "string") {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          return parsed as Record<string, unknown>;
        }
      } catch {
        throw new Error(`${label} was not valid JSON object output`);
      }
    }
    throw new Error(`${label} was not an object`);
  }

  private readRequiredString(
    obj: Record<string, unknown>,
    key: string,
    label: string
  ): string {
    const value = obj[key];
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(`${label} must be a non-empty string`);
    }
    return value.trim();
  }

  private readRequiredNumber(
    obj: Record<string, unknown>,
    key: string,
    label: string
  ): number {
    const value = Number(obj[key]);
    if (!Number.isFinite(value)) {
      throw new Error(`${label} must be a finite number`);
    }
    return value;
  }

  /**
   * Normalizes unknown errors into loggable message strings.
   */
  private toErrorMessage(err: unknown): string {
    if (err instanceof Error) {
      return err.message;
    }
    return String(err ?? "unknown error");
  }
}
