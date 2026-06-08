// table.ts — row storage helpers. Pure data, no PTC calls.
//
// The agent reads this file and uses these helpers inline in a single eval block.
// No import needed — copy the function bodies into your eval and call them directly.

interface Row extends Record<string, unknown> {
  id: string;
}

/** Build a row array from plain objects. Each object must include a string id. */
function create(tasks: Array<{ id: string; [key: string]: unknown }>): Row[] {
  return tasks.map((t) => ({ ...t }));
}

/** Return all rows, optionally filtered. */
function rows(table: Row[], filter?: (r: Row) => boolean): Row[] {
  return filter ? table.filter(filter) : [...table];
}

/** Merge dispatch results back into rows (by id). Mutates table in place. */
function mergeResults(
  table: Row[],
  results: Array<{ id: string; [key: string]: unknown }>,
): void {
  const byId = new Map(results.map((r) => [r.id, r]));
  for (const row of table) {
    const result = byId.get(row.id);
    if (result) Object.assign(row, result);
  }
}
