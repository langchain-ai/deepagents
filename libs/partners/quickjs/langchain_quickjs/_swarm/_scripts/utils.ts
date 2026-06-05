/**
 * Read a value from a row by dot-separated column path.
 *
 * Traverses nested objects segment by segment (e.g. `"meta.score"`
 * reads `row.meta.score`). Returns `undefined` if any intermediate
 * segment is missing or not an object.
 *
 * @param row - The table row to read from.
 * @param path - Dot-separated column path (e.g. `"file"` or `"meta.score"`).
 * @returns The resolved value, or `undefined` if the path is invalid.
 */
export function readColumn(
  row: Record<string, unknown>,
  path: string,
): unknown {
  const segments = path.split(".");

  let current = row;
  for (let idx = 0; idx < segments.length - 1; idx++) {
    const next = current[segments[idx]];
    if (next == null || typeof next !== "object" || Array.isArray(next)) {
      return undefined;
    }
    current = next as Record<string, unknown>;
  }

  return current[segments[segments.length - 1]];
}
