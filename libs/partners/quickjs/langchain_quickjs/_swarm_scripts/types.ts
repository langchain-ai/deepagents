/**
 * Lightweight handle returned by `create()`.
 *
 * Contains only metadata — actual row data stays on the backend.
 * The agent uses this handle to reference the table in subsequent
 * `run()` and `rows()` calls. Handles are stable across evals.
 */
export interface SwarmHandle {
  /**
   * Unique table identifier (e.g. `"t_a1b2c3"`).
   */
  id: string;

  /**
   * Number of rows in the table at creation time.
   */
  count: number;

  /**
   * Column names present in the first row (e.g. `["id", "file"]`).
   */
  columns: string[];
}

/**
 * Source specification for `create()`.
 *
 * Exactly one of `glob`, `filePaths`, or `tasks` must be set.
 * Providing zero or more than one source throws an error.
 */
export interface CreateSource {
  /**
   * Glob pattern(s) to match files. Each match becomes a row with
   * `{ id: <basename>, file: <full path> }` columns. Requires a
   * `glob` tool in the PTC configuration.
   */
  glob?: string | string[];

  /**
   * Explicit list of file paths. Same row structure as `glob`
   * (`{ id, file }`) but skips pattern resolution.
   */
  filePaths?: string[];

  /**
   * Custom row data. Each object must include a string `id` field.
   * All other fields become table columns.
   */
  tasks?: Array<Record<string, unknown>>;
}

/**
 * Per-row batch size function.
 *
 * Returns the desired batch size for a given row. Rows that return
 * the same batch size are grouped together, then chunked into
 * batches of that size.
 */
export type BatchFn = (
  row: Record<string, unknown>,
  rowCount: number,
) => number;

/**
 * Options for `run()`.
 *
 * Controls how rows are selected, how instructions are templated,
 * and how subagent dispatch is configured.
 */
export interface RunOptions {
  /**
   * Instruction template with `{column}` placeholders that are
   * interpolated per-row (e.g. `"Review {file} for security issues"`).
   */
  instruction: string;

  /**
   * Context prose prepended to every subagent prompt. Use for shared
   * background that applies to all rows (e.g. project description).
   */
  context?: string;

  /**
   * Filter clause to select a subset of rows. Rows that don't match
   * are skipped (counted in `RunResult.skipped`).
   */
  filter?: SwarmFilter;

  /**
   * Name of the subagent type to dispatch to. When set, each dispatch
   * runs a full agentic loop with tools. When omitted, each dispatch
   * is a direct model call with structured output (no tools, no iteration).
   */
  subagentType?: string;

  /**
   * JSON Schema (type: "object") for structured output. Each property
   * in the schema becomes a top-level column on the row.
   */
  responseSchema: Record<string, unknown>;

  /**
   * Controls how rows are grouped into subagent calls.
   *
   * - **Number**: uniform batch size for all rows.
   * - **Function**: called per-row, returns desired batch size. Rows with
   *   the same batch size are grouped together, then chunked.
   *
   * Batch sizes are clamped to [1, MAX_BATCH_SIZE] after evaluation.
   *
   * @default auto-batch based on table size to cap total dispatches.
   */
  batchSize?: number | BatchFn;

  /**
   * Maximum concurrent subagent dispatches. Clamped to [1, MAX_SUBAGENTS].
   * Defaults to MAX_SUBAGENTS (10) when omitted.
   */
  concurrency?: number;
}

/**
 * Summary returned by `run()`.
 *
 * Contains counts and deduplicated failure groups. The agent uses
 * this to decide whether to retry, inspect, or proceed.
 */
export interface RunResult {
  /**
   * Number of rows where the subagent succeeded and a result was merged.
   */
  completed: number;

  /**
   * Number of rows where the subagent failed or interpolation failed.
   */
  failed: number;

  /**
   * Number of rows excluded by the filter (not dispatched).
   */
  skipped: number;

  /**
   * Failures grouped by error message, sorted by count descending.
   */
  failures: FailureGroup[];
}

/**
 * A group of rows that failed with the same error message.
 *
 * Deduplication keeps the failure list compact even when hundreds of
 * rows hit the same error (e.g. rate limiting).
 */
export interface FailureGroup {
  /**
   * The error message shared by all rows in this group.
   */
  error: string;

  /**
   * Number of rows that hit this error.
   */
  count: number;

  /**
   * IDs of all rows that hit this error.
   */
  ids: string[];
}

/**
 * Options for `rows()`.
 *
 * Controls filtering, column projection, and row limiting when
 * retrieving table data for inspection or aggregation.
 */
export interface RowsOptions {
  /**
   * Filter clause — only rows matching the filter are returned.
   */
  filter?: SwarmFilter;

  /**
   * Project to specific columns. Omit to return all columns.
   */
  columns?: string[];

  /**
   * Maximum number of rows to return. Omit for no limit.
   */
  limit?: number;
}

/**
 * Filter clause for selecting rows. Can be a leaf predicate or a
 * combinator (`and`/`or`) composing multiple clauses.
 *
 * Leaf predicates operate on a single column (supports dot-paths
 * for nested access, e.g. `"meta.score"`).
 */
export type SwarmFilter =
  | {
      /**
       * Column path to compare.
       */
      column: string;

      /**
       * Row matches if column value deeply equals this value.
       */
      equals: unknown;
    }
  | {
      /**
       * Column path to compare.
       */
      column: string;

      /**
       * Row matches if column value does NOT deeply equal this value.
       */
      notEquals: unknown;
    }
  | {
      /** Column path to compare. */
      column: string;

      /**
       * Row matches if column value deeply equals any item in this array.
       */
      in: unknown[];
    }
  | {
      /**
       * Column path to compare.
       */
      column: string;

      /**
       * When true, matches non-null/non-undefined. When false, matches null/undefined.
       */
      exists: boolean;
    }
  | {
      /**
       * All sub-filters must match for the row to match.
       */
      and: SwarmFilter[];
    }
  | {
      /**
       * At least one sub-filter must match for the row to match.
       */
      or: SwarmFilter[];
    };

/**
 * A single dispatch unit for the executor.
 *
 * Represents one subagent call — either a single row's interpolated
 * prompt or a batch prompt covering multiple rows.
 */
export interface TaskSpec {
  /**
   * Row ID (single dispatch) or batch ID (batched dispatch).
   */
  id: string;

  /**
   * Fully interpolated prompt to send to the subagent.
   */
  prompt: string;

  /**
   * Name of the subagent type to dispatch to. When omitted, the
   * dispatch is a direct model call (invoke mode).
   */
  subagentType?: string;

  /**
   * Optional JSON Schema to constrain the subagent's response.
   */
  responseSchema?: Record<string, unknown>;

  /**
   * Dispatch mode for this task.
   *
   * - `"agent"` — Full agentic loop with tools and middleware.
   * - `"invoke"` — Direct model call, no tools or iteration.
   *
   * @default "agent"
   */
  mode?: "agent" | "invoke";
}

/**
 * Result of a single subagent dispatch.
 *
 * Returned by the executor in the same order as the input `TaskSpec[]`.
 */
export interface TaskResult {
  /**
   * Row ID or batch ID that this result corresponds to.
   */
  id: string;

  /**
   * Whether the dispatch succeeded or failed.
   */
  status: "completed" | "failed";

  /**
   * The subagent's response string (present when `status` is `"completed"`).
   */
  result?: string;

  /**
   * Error message (present when `status` is `"failed"`).
   */
  error?: string;
}
