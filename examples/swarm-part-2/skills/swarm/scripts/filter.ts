import type { SwarmFilter } from "./types.js";
import { readColumn } from "./utils.js";

/**
 * Compare two values for deep equality.
 *
 * Handles primitives via `===` and objects/arrays via JSON
 * serialization. `null` and `undefined` only equal themselves.
 *
 * @param a - First value.
 * @param b - Second value.
 * @returns `true` if the values are deeply equal.
 */
function deepEquals(a: unknown, b: unknown): boolean {
  if (a === b) {
    return true;
  }

  if (a == null || b == null) {
    return false;
  }

  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Evaluate a filter clause against a single table row.
 *
 * Supports leaf predicates (`equals`, `notEquals`, `in`, `exists`)
 * and recursive combinators (`and`, `or`). Column paths support
 * dot notation for nested access.
 *
 * @param filter - The filter clause to evaluate.
 * @param row - The table row to test against.
 * @returns `true` if the row matches the filter.
 */
export function evaluateFilter(
  filter: SwarmFilter,
  row: Record<string, unknown>,
): boolean {
  if (filter == null || typeof filter !== "object") {
    throw new Error(
      `evaluateFilter: expected a filter object, got ${JSON.stringify(filter)}`,
    );
  }

  if ("and" in filter) {
    return filter.and.every((f) => evaluateFilter(f, row));
  }

  if ("or" in filter) {
    return filter.or.some((f) => evaluateFilter(f, row));
  }

  const value = readColumn(row, filter.column);

  if ("equals" in filter) {
    return deepEquals(value, filter.equals);
  }

  if ("notEquals" in filter) {
    return !deepEquals(value, filter.notEquals);
  }

  if ("in" in filter) {
    return filter.in.some((item) => deepEquals(value, item));
  }

  if ("exists" in filter) {
    return filter.exists ? value != null : value == null;
  }

  return false;
}
