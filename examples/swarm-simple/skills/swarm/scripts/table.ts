// table.ts — row storage helpers.
//
// Pure data manipulation except createFromGlob(), which calls tools.glob (PTC).
// The agent reads this file and defines these helpers inline in its eval block.

declare const tools: {
  glob: (args: { pattern: string }) => Promise<string | string[]>;
};

interface Row extends Record<string, unknown> {
  id: string;
}

/** Build rows from plain task objects. Each must include a string id. */
function create(tasks: Array<{ id: string; [key: string]: unknown }>): Row[] {
  return tasks.map((t) => ({ ...t }));
}

/** Build rows from a glob pattern — one file = one row with { id, file }. */
async function createFromGlob(pattern: string | string[]): Promise<Row[]> {
  const patterns = Array.isArray(pattern) ? pattern : [pattern];
  const all: Row[] = [];
  for (const p of patterns) {
    const raw = await tools.glob({ pattern: p });
    const files: string[] = typeof raw === "string" ? JSON.parse(raw) : raw;
    for (const file of files) {
      const id = file.replace(/\//g, "_").replace(/[^a-zA-Z0-9_.-]/g, "");
      all.push({ id, file });
    }
  }
  // deduplicate by id, last occurrence wins
  const seen = new Map<string, Row>();
  for (const r of all) seen.set(r.id, r);
  return [...seen.values()];
}

/** Build rows from an explicit file list. */
function createFromFiles(filePaths: string[]): Row[] {
  return filePaths.map((file) => {
    const id = file.replace(/\//g, "_").replace(/[^a-zA-Z0-9_.-]/g, "");
    return { id, file };
  });
}

/**
 * Return rows, optionally filtered. Pass any JS predicate.
 *
 * Common patterns:
 *   rows(table, r => !r.result)        // not yet processed
 *   rows(table, r => !!r.error)        // failed rows
 *   rows(table, r => r.sentiment === "negative")
 */
function rows(table: Row[], filter?: (r: Row) => boolean): Row[] {
  return filter ? table.filter(filter) : [...table];
}
