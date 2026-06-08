// executor.ts — fan out via tools.task() with bounded concurrency.
//
// tools.task is injected by CodeInterpreterMiddleware PTC — no import needed.
// The agent reads this file and defines run() inline in its eval block.
// run() mutates the table in place (merges results onto rows) and returns a summary.

declare const tools: {
  task: (args: {
    description: string;
    subagent_type?: string;
  }) => Promise<string>;
};

interface RunOpts {
  /** Per-row prompt template. {column} placeholders resolved from row fields. */
  instruction: string;
  /** Prose prepended to every subagent prompt. Use for shared background. */
  context?: string;
  /** Ask subagents to return JSON matching this shape. Used for prompt hint + validation. */
  responseSchema?: Record<string, unknown>;
  /** Named subagent type. Omit for the default general-purpose subagent. */
  subagentType?: string;
  /** Max parallel dispatches. Default 10. */
  concurrency?: number;
}

interface FailureGroup {
  error: string;
  count: number;
  ids: string[];
}

interface RunSummary {
  completed: number;
  failed: number;
  failures: FailureGroup[];
}

/** Validate a parsed object against a JSON Schema subset (required, type, enum). */
function validate(
  obj: Record<string, unknown>,
  schema: Record<string, unknown>,
): string[] {
  const errors: string[] = [];
  const props = (schema.properties ?? {}) as Record<string, { type?: string; enum?: unknown[] }>;
  const required = (schema.required ?? []) as string[];
  for (const key of required) {
    if (!(key in obj)) errors.push(`missing required field: ${key}`);
  }
  for (const [key, def] of Object.entries(props)) {
    if (!(key in obj)) continue;
    const val = obj[key];
    if (def.type && typeof val !== def.type)
      errors.push(`${key}: expected ${def.type}, got ${typeof val}`);
    if (def.enum && !def.enum.includes(val))
      errors.push(`${key}: ${JSON.stringify(val)} not in enum ${JSON.stringify(def.enum)}`);
  }
  return errors;
}

/**
 * Dispatch each row to tools.task() in parallel chunks.
 * Merges results onto rows in place. Returns a run summary.
 *
 * Retry failed rows by filtering the table and calling run() again:
 *   const failed = rows(table, r => !!r.error);
 *   await run(failed, opts);
 */
async function run(
  table: Array<{ id: string; [key: string]: unknown }>,
  opts: RunOpts,
): Promise<RunSummary> {
  const { instruction, context, responseSchema, subagentType, concurrency = 10 } = opts;
  const schemaHint = responseSchema
    ? `\n\nRespond with JSON matching: ${JSON.stringify(responseSchema)}`
    : "";
  const contextPrefix = context ? `${context}\n\n` : "";

  let completed = 0;
  let failed = 0;
  const failureMap = new Map<string, string[]>();

  for (let i = 0; i < table.length; i += concurrency) {
    const chunk = table.slice(i, i + concurrency);
    await Promise.all(
      chunk.map(async (row) => {
        const prompt =
          contextPrefix +
          instruction.replace(/\{(\w+)\}/g, (_, k) => String(row[k] ?? "")) +
          schemaHint;
        try {
          const raw = await tools.task({
            description: prompt,
            ...(subagentType != null && { subagent_type: subagentType }),
          });
          const match = raw.match(/\{[\s\S]*\}/);
          const result: Record<string, unknown> = match
            ? JSON.parse(match[0])
            : { output: raw };
          const errs = responseSchema ? validate(result, responseSchema) : [];
          if (errs.length > 0) {
            const msg = `validation: ${errs.join("; ")}`;
            row.error = msg;
            failureMap.set(msg, [...(failureMap.get(msg) ?? []), row.id]);
            failed++;
          } else {
            Object.assign(row, result);
            delete row.error;
            completed++;
          }
        } catch (err) {
          const msg = String(err);
          row.error = msg;
          failureMap.set(msg, [...(failureMap.get(msg) ?? []), row.id]);
          failed++;
        }
      }),
    );
  }

  const failures: FailureGroup[] = [...failureMap.entries()]
    .map(([error, ids]) => ({ error, count: ids.length, ids }))
    .sort((a, b) => b.count - a.count);

  return { completed, failed, failures };
}
