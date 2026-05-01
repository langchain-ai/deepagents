// Swarm skill — fan out subagent calls with bounded concurrency.
//
// Uses `tools.task(...)` (the Deep Agents `task` tool surfaced via the
// REPL's PTC layer) so this skill works on any agent whose subagent
// config exposes `task`. No host bindings are required beyond what
// REPLMiddleware installs — `tools` is the only global we touch.

/**
 * One task to dispatch. `subagentType` is optional — when omitted the
 * caller's default (or "general-purpose") is used.
 */
export interface SwarmTask {
  description: string;
  subagentType?: string;
}

export interface SwarmResult {
  id: number;
  status: "completed" | "failed";
  output?: string;
  error?: string;
}

export interface SwarmSummary {
  total: number;
  completed: number;
  failed: number;
  results: SwarmResult[];
}

export interface RunSwarmOptions {
  tasks: SwarmTask[];
  concurrency?: number;
  subagentType?: string;
}

const DEFAULT_CONCURRENCY = 5;
const MAX_CONCURRENCY = 10;

/**
 * Dispatch `tasks` to the agent's `task` tool with at most
 * `concurrency` calls in flight at once. Preserves input order in the
 * returned `results` array.
 */
export async function runSwarm(opts: RunSwarmOptions): Promise<SwarmSummary> {
  const tasks = opts.tasks ?? [];
  const concurrency = Math.max(
    1,
    Math.min(opts.concurrency ?? DEFAULT_CONCURRENCY, MAX_CONCURRENCY),
  );
  const defaultSubagent = opts.subagentType ?? "general-purpose";

  const results: SwarmResult[] = new Array(tasks.length);
  let nextIndex = 0;

  // Simple worker-pool pattern. We spawn `concurrency` promises that
  // each pull the next task index off a shared counter until nothing's
  // left. Preserves input order because we write into a fixed-size
  // array at the task's original index.
  const worker = async (): Promise<void> => {
    while (true) {
      const idx = nextIndex++;
      if (idx >= tasks.length) return;
      const task = tasks[idx];
      const subagentType = task.subagentType ?? defaultSubagent;
      try {
        const out = await tools.task({
          description: task.description,
          subagent_type: subagentType,
        });
        results[idx] = { id: idx, status: "completed", output: String(out) };
      } catch (err: any) {
        results[idx] = {
          id: idx,
          status: "failed",
          error: err?.message ?? String(err),
        };
      }
    }
  };

  const workers: Promise<void>[] = [];
  for (let i = 0; i < concurrency; i++) workers.push(worker());
  await Promise.all(workers);

  let completed = 0;
  let failed = 0;
  for (const r of results) {
    if (r.status === "completed") completed++;
    else failed++;
  }

  return { total: tasks.length, completed, failed, results };
}

// Ambient declaration so TypeScript is happy about `tools.task`. The
// REPL's PTC layer injects `tools` as a global — oxidase strips this
// at install time, so it never actually ships to QuickJS.
declare const tools: {
  task: (args: { description: string; subagent_type?: string }) => Promise<unknown>;
};
