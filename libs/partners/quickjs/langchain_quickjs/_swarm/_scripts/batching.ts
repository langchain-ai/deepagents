import { extractPlaceholders } from "./interpolate.js";
import type { BatchFn } from "./types.js";
import { readColumn } from "./utils.js";

/**
 * Maximum rows per batch when auto-batching.
 */
export const MAX_BATCH_SIZE = 50;

/**
 * Group an array of items into batches of a given size.
 *
 * The last batch may be smaller than `batchSize` if the total count
 * is not evenly divisible.
 *
 * @param items - Array of items to batch.
 * @param batchSize - Maximum number of items per batch.
 * @returns Array of batches (each batch is an array of items).
 */
export function createBatches<T>(items: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize));
  }
  return batches;
}

/**
 * Clamp a batch size to [1, MAX_BATCH_SIZE].
 */
function clampBatchSize(n: number): number {
  return Math.max(1, Math.min(Math.round(n), MAX_BATCH_SIZE));
}

/**
 * Resolve batch sizes and group rows into dispatch-ready batches.
 *
 * Handles all three modes:
 * - **Auto** (`batchSize` undefined): computes a uniform size from row
 *   count and `maxSubagents` to stay within the concurrency budget.
 * - **Uniform** (`batchSize` is a number): all rows use that size.
 * - **Per-row** (`batchSize` is a function): evaluates per row, groups
 *   rows sharing the same batch size, then chunks each group.
 *
 * Every batch size is clamped to [1, MAX_BATCH_SIZE].
 *
 * @param rows - Matched rows to dispatch.
 * @param batchSize - Batch strategy: undefined (auto), number, or function.
 * @param maxSubagents - Concurrency cap used for auto-batch calculation.
 * @returns Array of row batches, each ready for dispatch as a single task.
 */
export function resolveBatchGroups(
  rows: Record<string, unknown>[],
  maxSubagents: number,
  batchSize?: number | BatchFn,
): Record<string, unknown>[][] {
  if (rows.length === 0) {
    return [];
  }

  if (batchSize === undefined) {
    // Auto: keep total dispatches under maxSubagents
    const auto =
      rows.length > maxSubagents
        ? Math.min(Math.ceil(rows.length / maxSubagents), MAX_BATCH_SIZE)
        : 1;
    return createBatches(rows, auto);
  }

  if (typeof batchSize === "number") {
    return createBatches(rows, clampBatchSize(batchSize));
  }

  const groups = new Map<number, Record<string, unknown>[]>();
  for (const row of rows) {
    const size = clampBatchSize(batchSize(row, rows.length));
    let group = groups.get(size);
    if (!group) {
      group = [];
      groups.set(size, group);
    }
    group.push(row);
  }

  const batches: Record<string, unknown>[][] = [];
  for (const [size, group] of groups) {
    for (const batch of createBatches(group, size)) {
      batches.push(batch);
    }
  }

  return batches;
}

/**
 * Wrap a per-item JSON Schema into a batch-level response schema.
 *
 * Produces a schema of the form:
 * ```json
 * { "results": [{ "id": "...", ...itemProps }] }
 * ```
 *
 * The item schema's properties are merged with an `id` field so each
 * batch entry can be matched back to its row.
 *
 * @param itemSchema - Per-item JSON Schema.
 * @returns Batch-level JSON Schema wrapping items in a `results` array.
 */
export function wrapSchema(
  itemSchema: Record<string, unknown>,
  count?: number,
): Record<string, unknown> {
  const props = (itemSchema.properties as Record<string, unknown>) ?? {};
  const req = (itemSchema.required as string[]) ?? [];
  const itemProperties: Record<string, unknown> = {
    id: { type: "string" },
    ...props,
  };
  const itemRequired: string[] = ["id", ...req];

  const resultsArray: Record<string, unknown> = {
    type: "array",
    items: {
      type: "object",
      additionalProperties: false,
      properties: itemProperties,
      required: itemRequired,
    },
  };

  if (count != null) {
    resultsArray.minItems = count;
    resultsArray.maxItems = count;
  }

  return {
    type: "object",
    additionalProperties: false,
    properties: {
      results: resultsArray,
    },
    required: ["results"],
  };
}

/**
 * Format a single column value for inclusion in a batch prompt.
 *
 * Strings are inserted verbatim; numbers/booleans are stringified;
 * objects/arrays are JSON-serialized; `undefined` and `null` become
 * the empty string so the row still renders with its id.
 */
function formatValue(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

/**
 * Rewrite `{col}` placeholders in the author's instruction to
 * `` `col` `` (backtick-quoted column name, no braces). The model
 * sees a column name as a name, never as template syntax.
 */
function renderTaskBlock(instruction: string): string {
  return instruction.replace(
    /\{([^}]+)\}/g,
    (_m, raw) => `\`${String(raw).trim()}\``,
  );
}

/**
 * Render the items section.
 *
 * - 0 placeholders → `[id]` per row (no values, degenerate).
 * - 1 placeholder  → `[id] <value>` per row (flat).
 * - 2+ placeholders → labeled block:
 *     [id]
 *       col1: <value>
 *       col2: <value>
 */
function renderItemsBlock(
  rows: Array<Record<string, unknown>>,
  placeholders: string[],
): string {
  const lines: string[] = [];

  for (const row of rows) {
    const id = String(row.id);

    if (placeholders.length === 0) {
      lines.push(`[${id}]`);
      continue;
    }

    if (placeholders.length === 1) {
      const value = readColumn(row, placeholders[0]);
      lines.push(`[${id}] ${formatValue(value)}`);
      continue;
    }

    lines.push(`[${id}]`);
    for (const col of placeholders) {
      const value = readColumn(row, col);
      lines.push(`  ${col}: ${formatValue(value)}`);
    }
  }

  return lines.join("\n");
}

/**
 * Build a single prompt for a batch of rows.
 *
 * The instruction is rewritten to drop template-syntax braces — every
 * `{col}` becomes `` `col` `` so the model sees column names as names,
 * not as slots it must fill in. Items are rendered as either a flat
 * list (single-column case) or a labeled per-column block, so the
 * binding from row id to column value is structural and explicit.
 *
 * @param instruction - Instruction template with `{column}` placeholders.
 * @param rows - Array of row objects to include in the batch.
 * @param context - Optional context prose prepended to the prompt.
 * @returns A single prompt string covering all rows in the batch.
 */
export function buildBatchPrompt(
  instruction: string,
  rows: Array<Record<string, unknown>>,
  context?: string,
): string {
  const placeholders = extractPlaceholders(instruction);
  const taskBlock = renderTaskBlock(instruction);
  const itemsBlock = renderItemsBlock(rows, placeholders);

  const parts: string[] = [];

  if (context) {
    parts.push(context);
    parts.push("");
  }

  parts.push("# Task");
  parts.push(taskBlock);
  parts.push("");

  parts.push(`# Items (${rows.length})`);
  if (placeholders.length === 1) {
    parts.push(`Each item below is the value of \`${placeholders[0]}\`.`);
    parts.push("");
  } else if (placeholders.length > 1) {
    const cols = placeholders.map((p) => `\`${p}\``).join(", ");
    parts.push(`Each item below provides ${cols}.`);
    parts.push("");
  }
  parts.push(itemsBlock);
  parts.push("");

  parts.push(
    `Return a JSON object with a 'results' array of exactly ${rows.length} ` +
      "entries, each including the item's 'id' exactly as shown above.",
  );

  return parts.join("\n");
}

/**
 * Unpack a batch response string into per-row results.
 *
 * Parses the JSON response expecting `{ results: [{ id, ...fields }] }`.
 * Maps each item's `id` to its remaining fields. IDs present in
 * `expectedIds` but absent from the response are returned in `missing`.
 *
 * @param response - Raw JSON string from the subagent.
 * @param expectedIds - List of row IDs the batch was supposed to cover.
 * @returns Map of ID → result fields, plus a list of IDs missing from
 *          the response.
 */
export function unpackBatchResults(
  response: string,
  expectedIds: string[],
): { results: Map<string, unknown>; missing: string[] } {
  const resultsMap = new Map<string, unknown>();
  const missing: string[] = [];

  try {
    const parsed = JSON.parse(response);
    const items: Array<Record<string, unknown>> = parsed?.results ?? [];

    for (const item of items) {
      if (item && typeof item.id === "string") {
        const { id, ...fields } = item;
        resultsMap.set(id, fields);
      }
    }
  } catch {
    // Parse failure — all IDs are missing
  }

  for (const id of expectedIds) {
    if (!resultsMap.has(id)) {
      missing.push(id);
    }
  }

  return { results: resultsMap, missing };
}
