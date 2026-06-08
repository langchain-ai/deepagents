// executor.ts — fan out via tools.task() with bounded concurrency.
//
// tools.task is injected by CodeInterpreterMiddleware PTC — no import needed.
// The agent reads this file and uses run() inline in a single eval block.

declare const tools: {
  task: (args: {
    description: string;
    subagent_type?: string;
  }) => Promise<string>;
};

interface RunOpts {
  /** Per-row prompt template. {column} placeholders are resolved from row fields. */
  instruction: string;
  /** Ask subagents to return JSON matching this shape; included as a prompt hint. */
  responseSchema?: Record<string, unknown>;
  /** Named subagent type. Omit to use the default general-purpose subagent. */
  subagentType?: string;
  /** Max parallel dispatches. Default 10. */
  concurrency?: number;
}

interface RowResult {
  id: string;
  result?: Record<string, unknown>;
  error?: string;
  validationErrors?: string[];
}

/**
 * Validate a parsed object against a JSON Schema subset.
 * Checks required fields, primitive types, and enum membership.
 * Returns an empty array when valid.
 */
function validate(
  obj: Record<string, unknown>,
  schema: Record<string, unknown>,
): string[] {
  const errors: string[] = [];
  const props = (schema.properties ?? {}) as Record<
    string,
    { type?: string; enum?: unknown[] }
  >;
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

/** Dispatch each row to tools.task() in parallel chunks, return results by id. */
async function run(
  table: Array<{ id: string; [key: string]: unknown }>,
  opts: RunOpts,
): Promise<RowResult[]> {
  const { instruction, responseSchema, subagentType, concurrency = 10 } = opts;
  const schemaHint = responseSchema
    ? `\n\nRespond with JSON matching: ${JSON.stringify(responseSchema)}`
    : "";
  const out: RowResult[] = [];

  for (let i = 0; i < table.length; i += concurrency) {
    const chunk = table.slice(i, i + concurrency);
    const results = await Promise.all(
      chunk.map(async (row) => {
        const prompt =
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
          const validationErrors =
            responseSchema ? validate(result, responseSchema) : [];
          return validationErrors.length > 0
            ? { id: row.id, result, validationErrors }
            : { id: row.id, result };
        } catch (err) {
          return { id: row.id, error: String(err) };
        }
      }),
    );
    out.push(...results);
  }
  return out;
}
