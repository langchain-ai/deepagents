import { createTable, loadTable, saveTable } from "./table.js";
import { interpolate, extractPlaceholders } from "./interpolate.js";
import { readColumn } from "./utils.js";
import { evaluateFilter } from "./filter.js";
import { dispatch, deduplicateFailures, mergeResult } from "./executor.js";
import {
  resolveBatchGroups,
  wrapSchema,
  buildBatchPrompt,
  unpackBatchResults,
} from "./batching.js";
import type {
  CreateSource,
  SwarmHandle,
  RunOptions,
  RunResult,
  RowsOptions,
  TaskSpec,
  TaskResult,
} from "./types.js";

/**
 * Maximum concurrent subagent dispatches per `run()` call.
 *
 * When matched rows exceed this and no explicit `batchSize` is set,
 * auto-batching groups rows to stay within this concurrency budget.
 */
const MAX_SUBAGENTS = 10;

/**
 * A dispatch unit is a single task for the executor. It tracks
 * whether it covers one row (single) or multiple (batch) so the
 * merge step knows how to unpack the result.
 */
interface DispatchUnit {
  /**
   * The task to dispatch to the executor.
   */
  task: TaskSpec;

  /**
   * Row IDs covered by this task. Single: length 1. Batch: length > 1.
   */
  rowIds: string[];
}

/**
 * Build dispatch units from pre-grouped batches.
 *
 * Single-row batches produce interpolated per-row prompts with the
 * user's responseSchema. Multi-row batches produce batch prompts
 * with a wrapped schema.
 */
function buildDispatchUnits(
  batches: Record<string, unknown>[][],
  opts: {
    instruction: string;
    context?: string;
    subagentType?: string;
    responseSchema: Record<string, unknown>;
    mode: "agent" | "invoke";
  },
): { units: DispatchUnit[]; errors: TaskResult[] } {
  const units: DispatchUnit[] = [];
  const errors: TaskResult[] = [];

  let batchIndex = 0;
  for (const batch of batches) {
    if (batch.length === 1) {
      // Single-row dispatch: interpolate instruction, use schema directly
      const row = batch[0];
      const rowId = String(row.id);

      try {
        let prompt = interpolate(opts.instruction, row);
        if (opts.context) {
          prompt = `${opts.context}\n\n${prompt}`;
        }

        units.push({
          task: {
            id: rowId,
            prompt,
            subagentType: opts.subagentType,
            responseSchema: opts.responseSchema,
            mode: opts.mode,
          },
          rowIds: [rowId],
        });
      } catch (err) {
        errors.push({
          id: rowId,
          status: "failed",
          error: (err as Error).message,
        });
      }
    } else {
      // Multi-row batch: build batch prompt, wrap schema
      const rowIds = batch.map((r) => String(r.id));
      units.push({
        task: {
          id: `batch_${batchIndex}`,
          prompt: buildBatchPrompt(opts.instruction, batch, opts.context),
          subagentType: opts.subagentType,
          responseSchema: wrapSchema(opts.responseSchema, batch.length),
          mode: opts.mode,
        },
        rowIds,
      });
      batchIndex++;
    }
  }

  return { units, errors };
}

/**
 * Normalize dispatch results into per-row results.
 *
 * Single-row units pass through directly. Batch units are unpacked
 * into one result per row — missing rows become failures.
 */
function unpackDispatchResults(
  units: DispatchUnit[],
  results: TaskResult[],
): TaskResult[] {
  const rowResults: TaskResult[] = [];

  for (let idx = 0; idx < units.length; idx++) {
    const unit = units[idx];
    const result = results[idx];

    if (unit.rowIds.length === 1) {
      rowResults.push(result);
      continue;
    }

    if (result.status === "failed") {
      for (const rowId of unit.rowIds) {
        rowResults.push({ id: rowId, status: "failed", error: result.error });
      }
      continue;
    }

    const { results: unpacked } = unpackBatchResults(
      result.result ?? "",
      unit.rowIds,
    );
    for (const rowId of unit.rowIds) {
      const value = unpacked.get(rowId);
      if (value !== undefined) {
        rowResults.push({
          id: rowId,
          status: "completed",
          result: typeof value === "string" ? value : JSON.stringify(value),
        });
      } else {
        rowResults.push({
          id: rowId,
          status: "failed",
          error: "Missing from batch response",
        });
      }
    }
  }

  return rowResults;
}

/**
 * Parse and merge per-row results into table rows.
 *
 * Each completed result is JSON-parsed and spread onto the
 * corresponding row via `mergeResult`.
 */
function mergeRowResults(
  rowResults: TaskResult[],
  rowById: Map<string, Record<string, unknown>>,
): { completed: number; failed: number } {
  let completed = 0;
  let failed = 0;

  for (const result of rowResults) {
    const row = rowById.get(result.id);
    if (!row) {
      failed++;
      continue;
    }

    if (result.status === "completed" && result.result != null) {
      try {
        mergeResult(row, JSON.parse(result.result));
        completed++;
      } catch {
        failed++;
      }
    } else {
      failed++;
    }
  }

  return { completed, failed };
}

/**
 * Verify every `{column}` reference in `instruction` resolves on at
 * least one matched row. Throws with a list of unresolved paths.
 */
function validatePlaceholders(
  instruction: string,
  rows: Record<string, unknown>[],
): void {
  const placeholders = extractPlaceholders(instruction);
  if (placeholders.length === 0) {
    return;
  }
  const unresolved = placeholders.filter(
    (p) => !rows.some((r) => readColumn(r, p) !== undefined),
  );
  if (unresolved.length > 0) {
    throw new Error(
      `instruction references unknown column(s): ${unresolved.join(", ")}`,
    );
  }
}

/**
 * Create a table from a source specification and persist it to the backend.
 *
 * Thin wrapper around `createTable` — validates the source, builds rows,
 * runs eviction if necessary, and persists the table as JSONL.
 *
 * @param source - Exactly one of `glob`, `filePaths`, or `tasks`.
 * @returns A lightweight handle with the table's ID, row count, and columns.
 */
export async function create(source: CreateSource): Promise<SwarmHandle> {
  return createTable(source);
}

/**
 * Dispatch work across table rows and update the table in place.
 *
 * Loads the table, partitions rows by filter, interpolates the
 * instruction template per-row (or builds batch prompts), dispatches
 * to subagents via `tools.swarm_task()`, merges results into rows,
 * and persists the updated table.
 *
 * @param handle - A table handle or object with an `id` field.
 * @param options - Dispatch configuration (instruction, filter, schema, etc.).
 * @returns A summary with completion counts and deduplicated failure groups.
 */
export async function run(
  tableId: string,
  options: RunOptions,
): Promise<RunResult> {
  const allRows = await loadTable(tableId);
  const {
    instruction,
    context,
    filter,
    subagentType,
    responseSchema,
    batchSize,
    concurrency,
  } = options;
  const mode = subagentType != null ? "agent" : "invoke";

  const effectiveConcurrency = Math.max(
    1,
    Math.min(concurrency ?? MAX_SUBAGENTS, MAX_SUBAGENTS),
  );

  // -----------------------------------------------------------------------
  // 1. Partition rows into matched (dispatched) and skipped (filtered out)
  // -----------------------------------------------------------------------

  const matched: Record<string, unknown>[] = [];
  let skippedCount = 0;

  for (const row of allRows) {
    if (!filter || evaluateFilter(filter, row)) {
      matched.push(row);
    } else {
      skippedCount++;
    }
  }

  if (matched.length === 0) {
    return {
      completed: 0,
      failed: 0,
      skipped: allRows.length,
      failures: [],
    };
  }

  validatePlaceholders(instruction, matched);

  // -----------------------------------------------------------------------
  // 2. Resolve batches and build dispatch units
  // -----------------------------------------------------------------------

  const batches = resolveBatchGroups(matched, effectiveConcurrency, batchSize);

  const { units, errors: interpolationErrors } = buildDispatchUnits(batches, {
    instruction,
    context,
    subagentType,
    responseSchema,
    mode,
  });

  // -----------------------------------------------------------------------
  // 3. Dispatch
  // -----------------------------------------------------------------------

  const dispatchResults = await dispatch(
    units.map((u) => u.task),
    { concurrency: effectiveConcurrency },
  );

  // -----------------------------------------------------------------------
  // 4. Unpack and merge results into rows
  // -----------------------------------------------------------------------

  const rowById = new Map<string, Record<string, unknown>>();
  for (const row of matched) {
    rowById.set(String(row.id), row);
  }

  const rowResults = unpackDispatchResults(units, dispatchResults);
  const { completed, failed: mergeFailed } = mergeRowResults(
    rowResults,
    rowById,
  );
  const failed = mergeFailed + interpolationErrors.length;
  const allRowResults = [...interpolationErrors, ...rowResults];

  // -----------------------------------------------------------------------
  // 5. Persist and return summary
  // -----------------------------------------------------------------------

  await saveTable(tableId, allRows);

  return {
    completed,
    failed,
    skipped: skippedCount,
    failures: deduplicateFailures(allRowResults),
  };
}

/**
 * Retrieve rows from a table, optionally filtered and projected.
 *
 * Loads the table and applies filter, column projection, and row
 * limiting in that order. Use for inspection and JS-based aggregation
 * — the heavy data stays in the sandbox and only the computed result
 * (via `console.log`) goes back to the agent's context.
 *
 * @param handle - A table handle or object with an `id` field.
 * @param options - Optional filtering, projection, and limiting.
 * @returns Array of row objects matching the criteria.
 */
export async function rows(
  tableId: string,
  options?: RowsOptions,
): Promise<Record<string, unknown>[]> {
  let result = await loadTable(tableId);

  if (options?.filter) {
    const f = options.filter;
    result = result.filter((row) => evaluateFilter(f, row));
  }

  if (options?.columns) {
    const cols = options.columns;
    result = result.map((row) => {
      const projected: Record<string, unknown> = {};
      for (const col of cols) {
        if (col in row) projected[col] = row[col];
      }
      return projected;
    });
  }

  if (options?.limit != null && options.limit >= 0) {
    result = result.slice(0, options.limit);
  }

  return result;
}
