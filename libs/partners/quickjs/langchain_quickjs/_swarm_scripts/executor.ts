import type { FailureGroup, TaskResult, TaskSpec } from "./types.js";

/**
 * PTC tool declaration for swarm subagent dispatch.
 *
 * At runtime in QuickJS, `tools` is an ambient global injected by the
 * PTC layer. The swarm skill uses `swarm_task` (a PTC-only tool) rather
 * than the general `task` tool, so that response_schema and mode support
 * are decoupled from the main agent's task tool.
 *
 * For vitest, set up `globalThis.tools` in `beforeEach`.
 */
declare const tools: {
  swarmTask?: (args: {
    description: string;
    subagent_type?: string;
    response_schema?: Record<string, unknown>;
    mode?: "agent" | "invoke";
  }) => Promise<string>;
};

/**
 * Column names that must not be overwritten by structured output merging.
 */
const RESERVED_COLUMNS = new Set(["id", "file"]);

/**
 * Call the PTC `swarm_task` tool.
 *
 * @internal Exported for testing — not part of the public API.
 * @param args - Task arguments forwarded to the swarm task tool.
 * @returns The subagent's response as a string.
 * @throws Error if the `swarm_task` PTC tool is not configured.
 */
export async function callTask(args: {
  description: string;
  subagent_type?: string;
  response_schema?: Record<string, unknown>;
  mode?: "agent" | "invoke";
}): Promise<string> {
  if (typeof tools.swarmTask !== "function") {
    throw new Error(
      "Swarm requires a 'swarm_task' tool in the PTC configuration.",
    );
  }
  return tools.swarmTask(args);
}

/**
 * Dispatch an array of task specs to subagents with bounded concurrency.
 *
 * Spawns up to `concurrency` workers that pull from the task queue.
 * Each worker calls the task function and records the result (or error)
 * at the same index as the input spec, preserving order.
 *
 * @param tasks - Task specs to dispatch.
 * @param options - Dispatch options (currently just `concurrency`).
 * @returns Results in the same order as the input tasks.
 */
export async function dispatch(
  tasks: TaskSpec[],
  options: { concurrency: number },
): Promise<TaskResult[]> {
  const results = new Array<TaskResult>(tasks.length);

  let idx = 0;
  async function worker(): Promise<void> {
    while (idx < tasks.length) {
      const i = idx++;
      const spec = tasks[i];
      try {
        const output = await callTask({
          description: spec.prompt,
          ...(spec.subagentType != null && {
            subagent_type: spec.subagentType,
          }),
          ...(spec.responseSchema != null && {
            response_schema: spec.responseSchema,
          }),
          ...(spec.mode != null && { mode: spec.mode }),
        });
        results[i] = {
          id: spec.id,
          status: "completed",
          result: String(output),
        };
      } catch (err: unknown) {
        const msg =
          err != null && typeof (err as Error).message === "string"
            ? (err as Error).message
            : String(err);
        results[i] = { id: spec.id, status: "failed", error: msg };
      }
    }
  }

  const workers: Promise<void>[] = [];
  for (let w = 0; w < Math.min(options.concurrency, tasks.length); w++) {
    workers.push(worker());
  }
  await Promise.all(workers);

  return results;
}

/**
 * Group failed task results by error message.
 *
 * Produces deduplicated failure groups sorted by count descending,
 * each containing the shared error message, the count of affected
 * rows, and the full list of affected row IDs.
 *
 * @param results - Array of task results (may include completed results).
 * @returns Deduplicated failure groups, sorted by count descending.
 */
export function deduplicateFailures(results: TaskResult[]): FailureGroup[] {
  const groups = new Map<string, string[]>();

  for (const r of results) {
    if (r.status !== "failed" || !r.error) {
      continue;
    }

    const ids = groups.get(r.error);
    if (ids) {
      ids.push(r.id);
    } else {
      groups.set(r.error, [r.id]);
    }
  }

  const out: FailureGroup[] = [];
  for (const [error, ids] of groups) {
    out.push({ error, count: ids.length, ids });
  }
  out.sort((a, b) => b.count - a.count);

  return out;
}

/**
 * Merge a subagent result into a table row.
 *
 * Each property of the parsed structured output is spread onto the
 * row as a top-level column — except reserved columns (`id`, `file`)
 * which are never overwritten.
 *
 * @param row - The table row to update (mutated in place).
 * @param value - The subagent's parsed structured output.
 */
export function mergeResult(
  row: Record<string, unknown>,
  value: Record<string, unknown>,
): void {
  for (const [k, v] of Object.entries(value)) {
    if (!RESERVED_COLUMNS.has(k)) {
      row[k] = v;
    }
  }
}
