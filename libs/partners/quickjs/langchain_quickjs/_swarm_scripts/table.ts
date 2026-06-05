import type { CreateSource, SwarmHandle } from "./types.js";

/**
 * PTC tool declarations for file operations.
 *
 * At runtime in QuickJS, `tools` is an ambient global injected by the
 * PTC layer. For vitest, set up `globalThis.tools` in `beforeEach`.
 */
declare const tools: {
  glob?: (args: { pattern: string }) => Promise<string>;
  readFile?: (args: { file_path: string }) => Promise<string>;
  writeFile?: (args: { file_path: string; content: string }) => Promise<string>;
  editFile?: (args: {
    file_path: string;
    old_string: string;
    new_string: string;
  }) => Promise<string>;
};

/**
 * Session ID injected by the QuickJS middleware as a global.
 * Scopes table files to the current conversation thread.
 */
declare const __sessionId__: string | undefined;

/**
 * Sanitize a session ID for use as a directory name component.
 * Replaces any character that isn't alphanumeric, hyphen, or underscore
 * with an underscore, and caps length to prevent excessively long paths.
 */
function sanitizeSessionId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

/**
 * Directory prefix for all table JSONL files, scoped to the session.
 */
function getTableDir(): string {
  const id = typeof __sessionId__ !== "undefined" ? __sessionId__ : "default";
  return `/tmp/.swarm/${sanitizeSessionId(id)}`;
}

/**
 * Maximum number of tables before oldest are evicted.
 */
const MAX_TABLES = 5;

/**
 * A table's rows and backend file path, cached in memory to avoid
 * redundant PTC reads within the same session.
 */
interface CachedTable {
  /**
   * The table's row data. Mutated in place during `run()`.
   */
  rows: Record<string, unknown>[];

  /**
   * Backend file path (e.g. `".swarm/003-t_a1b2c3.jsonl"`).
   */
  path: string;

  /**
   * The JSONL content from the most recent successful write.
   * Used as `old_string` when falling back to editFile for overwrites.
   */
  lastWritten: string;
}

/**
 * In-memory table cache keyed by table ID.
 */
const cache = new Map<string, CachedTable>();

/**
 * Monotonic counter for table file sequence numbers.
 */
let sequenceCounter = 0;

/**
 * Reset all module-level state for testing.
 *
 * Clears the in-memory cache and resets the sequence counter.
 */
export function _resetForTesting(): void {
  cache.clear();
  sequenceCounter = 0;
}

/**
 * Generate a random 6-hex-char table ID prefixed with `t_`.
 *
 * @returns A string like `"t_a1b2c3"`.
 */
export function generateId(): string {
  const hex = Math.floor(Math.random() * 0xffffff)
    .toString(16)
    .padStart(6, "0");
  return `t_${hex}`;
}

/**
 * Build the backend file path for a table.
 *
 * @param sequence - Zero-padded monotonic sequence number.
 * @param id - Table ID (e.g. `"t_a1b2c3"`).
 * @returns Path like `".swarm/003-t_a1b2c3.jsonl"`.
 */
export function tablePath(sequence: number, id: string): string {
  const padded = String(sequence).padStart(3, "0");
  return `${getTableDir()}/${padded}-${id}.jsonl`;
}

/**
 * Serialize an array of row objects to JSONL format.
 * One JSON object per line, no trailing newline.
 *
 * @param rows - Array of row objects to serialize.
 * @returns JSONL string.
 */
export function serializeJsonl(rows: Record<string, unknown>[]): string {
  return rows.map((r) => JSON.stringify(r)).join("\n");
}

/**
 * Parse a JSONL string into an array of row objects.
 * Validates that each line parses to a non-null, non-array object.
 *
 * @param content - Raw JSONL content from the backend.
 * @returns Array of parsed row objects.
 * @throws Error with line number if any line is malformed.
 */
export function parseJsonl(content: string): Record<string, unknown>[] {
  if (!content.trim()) {
    return [];
  }

  const parseLine = (line: string, idx: number): Record<string, unknown> => {
    try {
      const parsed = JSON.parse(line);
      if (
        typeof parsed !== "object" ||
        parsed === null ||
        Array.isArray(parsed)
      ) {
        throw new Error(`expected object`);
      }
      return parsed as Record<string, unknown>;
    } catch (e) {
      throw new Error(
        `JSONL parse error at line ${idx + 1}: ${(e as Error).message}`,
        { cause: e },
      );
    }
  };

  return content
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map(parseLine);
}

/**
 * Extract a table ID from a `.swarm/NNN-t_XXXXXX.jsonl` filename.
 *
 * @param filePath - Full path to a table JSONL file.
 * @returns The table ID (e.g. `"t_a1b2c3"`), or `undefined` if the
 *          filename doesn't match the expected pattern.
 */
export function extractIdFromPath(filePath: string): string | undefined {
  const filename = filePath.split("/").pop() || "";
  const match = filename.match(/^\d+-(t_[a-f0-9]+)\.jsonl$/);
  return match ? match[1] : undefined;
}

/**
 * Extract the sequence number from a `.swarm/NNN-t_XXXXXX.jsonl` filename.
 *
 * @param filePath - Full path to a table JSONL file.
 * @returns The sequence number, or `0` if the filename doesn't match.
 */
export function extractSeqFromPath(filePath: string): number {
  const filename = filePath.split("/").pop() || "";
  const match = filename.match(/^(\d+)-/);
  return match ? parseInt(match[1], 10) : 0;
}

/**
 * Build `{ id, file }` rows from a list of file paths.
 *
 * Uses the basename (last path segment) as the row ID. When multiple
 * paths share the same basename, disambiguates by prepending the
 * parent directory name (e.g. `"routes-index.ts"` vs `"handlers-index.ts"`).
 *
 * @param paths - List of file paths.
 * @returns Array of `{ id, file }` row objects.
 */
export function pathsToRows(
  paths: string[],
): Array<{ id: string; file: string }> {
  const basenames = paths.map((p) => {
    const parts = p.split("/");
    return parts[parts.length - 1] || p;
  });

  const counts = new Map<string, number>();
  for (const basename of basenames) {
    counts.set(basename, (counts.get(basename) ?? 0) + 1);
  }

  return paths.map((filePath, idx) => {
    let id = basenames[idx];
    if ((counts.get(id) ?? 0) > 1) {
      const parts = filePath.split("/");
      if (parts.length >= 2) {
        id = `${parts[parts.length - 2]}-${id}`;
      }
    }
    return { id, file: filePath };
  });
}

/**
 * Find duplicate `id` values in a row array.
 *
 * @param rows - Row objects to scan.
 * @returns Array of duplicate ids, in first-seen order, deduplicated.
 */
function findDuplicateIds(rows: Record<string, unknown>[]): string[] {
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const row of rows) {
    const id = String(row.id);
    if (seen.has(id)) {
      dupes.add(id);
    } else {
      seen.add(id);
    }
  }
  return [...dupes];
}

/**
 * Resolve a glob pattern to a list of file paths via the PTC `glob` tool.
 *
 * Handles both `string[]` and `{ path: string }[]` return formats
 * from different glob tool implementations.
 *
 * @param pattern - Glob pattern to resolve.
 * @returns Array of matching file paths.
 * @throws Error if the `glob` PTC tool is not configured.
 *
 * @internal
 */
export async function globFiles(pattern: string): Promise<string[]> {
  if (typeof tools.glob !== "function") {
    throw new Error(`Swarm requires a 'glob' tool in the PTC configuration`);
  }

  const raw = await tools.glob({ pattern });
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    return [];
  }

  const paths: string[] = [];
  for (const item of parsed) {
    if (typeof item === "string") {
      paths.push(item);
    } else if (item && typeof item.path === "string") {
      paths.push(item.path);
    }
  }

  return paths;
}

/**
 * Read a file's content from the backend via the PTC `readFile` tool.
 *
 * @param path - Backend file path.
 * @returns The file content as a string.
 * @throws Error if the `readFile` PTC tool is not configured.
 *
 * @internal
 */
export async function readFile(path: string): Promise<string> {
  if (typeof tools.readFile !== "function") {
    throw new Error(
      `Swarm requires a 'readFile' tool in the PTC configuration`,
    );
  }
  return tools.readFile({ file_path: path });
}

/**
 * Write string content to a backend file via the PTC `writeFile` tool.
 *
 * If the file already exists, falls back to `editFile` for a full
 * replacement — the backend's `write` rejects overwrites by design.
 * When `previousContent` is provided it is used as the `old_string`
 * for the edit, avoiding an unreliable round-trip through readFile.
 *
 * @param path - Backend file path. Created if it doesn't exist.
 * @param content - String content to write.
 * @param previousContent - The last-known content of the file, used
 *   as `old_string` for the editFile fallback.
 * @throws Error if the `writeFile` PTC tool is not configured.
 *
 * @internal
 */
export async function writeFile(
  path: string,
  content: string,
  previousContent?: string,
): Promise<void> {
  if (typeof tools.writeFile !== "function") {
    throw new Error(
      `Swarm requires a 'writeFile' tool in the PTC configuration`,
    );
  }
  const result = await tools.writeFile({ file_path: path, content });
  if (typeof result === "string" && result.includes("already exists")) {
    if (typeof tools.editFile !== "function") {
      throw new Error(
        "Swarm requires an 'edit_file' PTC tool to update existing tables",
      );
    }
    if (previousContent == null) {
      throw new Error(
        `Cannot overwrite ${path}: file already exists and no previous content available`,
      );
    }
    await tools.editFile({
      file_path: path,
      old_string: previousContent,
      new_string: content,
    });
  }
}

/**
 * List all table JSONL files in the `.swarm/` directory, sorted by
 * filename (which encodes creation order via the sequence prefix).
 *
 * @returns Sorted array of file paths, or empty array on failure.
 */
async function listTableFiles(): Promise<string[]> {
  try {
    const files = await globFiles(`${getTableDir()}/*.jsonl`);
    return files.sort();
  } catch {
    return [];
  }
}

/**
 * Evict the oldest tables when the count meets or exceeds `MAX_TABLES`.
 *
 * Clears evicted entries from the in-memory cache and overwrites
 * backend files with empty content (no delete_file tool available).
 * Empty files are treated as evicted by `loadTable`.
 */
async function evict(): Promise<void> {
  const files = await listTableFiles();
  if (files.length < MAX_TABLES) {
    return;
  }

  const toEvict = files.slice(0, files.length - MAX_TABLES + 1);
  for (const filePath of toEvict) {
    const id = extractIdFromPath(filePath);
    const prev = id ? cache.get(id)?.lastWritten : undefined;
    if (id) {
      cache.delete(id);
    }
    try {
      await writeFile(filePath, "", prev);
    } catch {
      // Best-effort eviction — non-fatal if overwrite fails
    }
  }
}

/**
 * Determine the next sequence number for a new table file.
 *
 * Reads existing files on the backend to avoid sequence collisions
 * across runs (same thread, new session). The counter only advances
 * forward — it never reuses a sequence number.
 *
 * @returns The next available sequence number.
 */
async function nextSequence(): Promise<number> {
  const files = await listTableFiles();
  if (files.length > 0) {
    const lastSequence = extractSeqFromPath(files[files.length - 1]);
    if (lastSequence >= sequenceCounter) {
      sequenceCounter = lastSequence + 1;
    }
  }
  return sequenceCounter++;
}

/**
 * Resolve one or more glob patterns into a deduplicated, sorted list
 * of file paths.
 *
 * @param pattern - A single glob string or array of glob strings.
 * @returns Sorted, deduplicated array of matching file paths.
 * @throws Error if no files match any of the provided patterns.
 */
async function resolveGlob(pattern: string | string[]): Promise<string[]> {
  const patterns = Array.isArray(pattern) ? pattern : [pattern];
  const allPaths: string[] = [];
  for (const p of patterns) {
    const paths = await globFiles(p);
    allPaths.push(...paths);
  }

  const unique = [...new Set(allPaths)].sort();
  if (unique.length === 0) {
    throw new Error(`No files matched pattern: ${JSON.stringify(pattern)}`);
  }

  return unique;
}

/**
 * Create a table from a source spec.
 *
 * Validates the source, builds rows, runs eviction if the table count
 * is at capacity, persists the new table to the backend as JSONL, and
 * returns a lightweight handle.
 *
 * @param source - Exactly one of `glob`, `filePaths`, or `tasks`.
 * @returns A handle with the table's ID, row count, and column names.
 * @throws Error if the source is invalid, empty, or missing required PTC tools.
 */
export async function createTable(source: CreateSource): Promise<SwarmHandle> {
  const sourceCount = [source.glob, source.filePaths, source.tasks].filter(
    (s) => s != null,
  ).length;

  if (sourceCount === 0) {
    throw new Error(
      "create() requires exactly one source: glob, filePaths, or tasks",
    );
  }

  if (sourceCount > 1) {
    throw new Error("create() accepts only one source type at a time");
  }

  let rows: Record<string, unknown>[];

  if (source.glob != null) {
    const paths = await resolveGlob(source.glob);
    rows = pathsToRows(paths);
  } else if (source.filePaths != null) {
    if (source.filePaths.length === 0) {
      throw new Error("filePaths array is empty");
    }
    rows = pathsToRows(source.filePaths);
  } else {
    const tasks = source.tasks ?? [];
    if (tasks.length === 0) {
      throw new Error("tasks array is empty");
    }

    for (let idx = 0; idx < tasks.length; idx++) {
      if (typeof tasks[idx].id !== "string") {
        throw new Error(`tasks[${idx}] is missing string 'id' field`);
      }
    }

    rows = tasks;
  }

  const dupes = findDuplicateIds(rows);
  if (dupes.length > 0) {
    throw new Error(`create() received duplicate row ids: ${dupes.join(", ")}`);
  }

  await evict();

  const id = generateId();
  const seq = await nextSequence();
  const path = tablePath(seq, id);

  const content = serializeJsonl(rows);
  await writeFile(path, content);
  cache.set(id, { rows, path, lastWritten: content });

  return {
    id,
    count: rows.length,
    columns: Object.keys(rows[0] ?? {}),
  };
}

/**
 * Load a table's rows by ID.
 *
 * Checks the in-memory cache first. On a cache miss (e.g. cross-run
 * resume), globs the backend to locate the JSONL file, reads and
 * parses it, and populates the cache.
 *
 * @param id - The table ID from a `SwarmHandle`.
 * @returns The table's row array (by reference — mutations are visible).
 * @throws Error if the table is not found (evicted or never created).
 */
export async function loadTable(
  id: string,
): Promise<Record<string, unknown>[]> {
  const cached = cache.get(id);
  if (cached) {
    return cached.rows;
  }

  const files = await listTableFiles();
  const match = files.find((f) => f.endsWith(`-${id}.jsonl`));
  if (!match) {
    throw new Error(`Table "${id}" not found. It may have been evicted`);
  }

  const content = await readFile(match);
  if (!content.trim()) {
    throw new Error(`Table "${id}" not found. It may have been evicted`);
  }

  const rows = parseJsonl(content);
  cache.set(id, { rows, path: match, lastWritten: serializeJsonl(rows) });

  return rows;
}

/**
 * Persist a table's current rows to the backend.
 *
 * Updates both the in-memory cache and the backend JSONL file.
 * The table must have been previously loaded via `loadTable` so
 * that its backend file path is known.
 *
 * @param id - The table ID from a `SwarmHandle`.
 * @param rows - The updated row array to persist.
 * @throws Error if the table has not been loaded into cache.
 */
export async function saveTable(
  id: string,
  rows: Record<string, unknown>[],
): Promise<void> {
  const cached = cache.get(id);
  if (!cached) {
    throw new Error(`Table "${id}" is not loaded - call loadTable first`);
  }
  cached.rows = rows;
  const content = serializeJsonl(rows);
  await writeFile(cached.path, content, cached.lastWritten);
  cached.lastWritten = content;
}
