import type { SourceItem } from "./github.ts";

export type TriageAction = "fix" | "close" | "ask_for_info";

export type TriageRecord = {
  id: number;
  type: "issue" | "pr" | "discussion";
  context_blurb: string;
  action: TriageAction;
  reason: string;
  ambiguity_flag: boolean;
  duplicate_of: number | null;
};

export type CondenseOptions = {
  classifier_tool?: (input: any) => Promise<any>;
};

const TRIAGE_ACTIONS: TriageAction[] = ["fix", "close", "ask_for_info"];

export const TRIAGE_CLASSIFY_JSON_SCHEMA = {
  name: "github_triage_record",
  schema: {
    type: "object",
    additionalProperties: false,
    required: [
      "context_blurb",
      "action",
      "reason",
      "ambiguity_flag",
      "duplicate_of",
    ],
    properties: {
      context_blurb: { type: "string", minLength: 1, maxLength: 1200 },
      action: { type: "string", enum: TRIAGE_ACTIONS },
      reason: { type: "string", minLength: 1, maxLength: 1200 },
      ambiguity_flag: { type: "boolean" },
      duplicate_of: { anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }] },
    },
  },
  strict: true,
} as const;

function toObject(raw: unknown): Record<string, unknown> | null {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return null;
    } catch {
      return null;
    }
  }
  return null;
}

function normalizeAction(value: unknown): TriageAction | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  return TRIAGE_ACTIONS.includes(normalized as TriageAction)
    ? (normalized as TriageAction)
    : null;
}

function normalizeTriageRecord(item: SourceItem, raw: unknown): TriageRecord {
  const payload = toObject(raw);
  if (!payload) {
    throw new Error("Classifier payload was not a JSON object.");
  }

  const action = normalizeAction(payload.action);
  const contextBlurb =
    typeof payload.context_blurb === "string" ? payload.context_blurb.trim() : "";
  const reason = typeof payload.reason === "string" ? payload.reason.trim() : "";
  const ambiguityFlag =
    typeof payload.ambiguity_flag === "boolean" ? payload.ambiguity_flag : null;

  const duplicateRaw = payload.duplicate_of;
  const duplicateOf =
    duplicateRaw === null
      ? null
      : Number.isInteger(duplicateRaw) && Number(duplicateRaw) > 0
        ? Number(duplicateRaw)
        : null;

  if (!action) {
    throw new Error("Classifier output `action` was invalid.");
  }
  if (!contextBlurb) {
    throw new Error("Classifier output `context_blurb` was empty.");
  }
  if (!reason) {
    throw new Error("Classifier output `reason` was empty.");
  }
  if (ambiguityFlag == null) {
    throw new Error("Classifier output `ambiguity_flag` was invalid.");
  }

  return {
    id: item.id,
    type: item.type,
    context_blurb: contextBlurb.slice(0, 1200),
    action,
    reason: reason.slice(0, 1200),
    ambiguity_flag: ambiguityFlag,
    duplicate_of: duplicateOf,
  };
}

export function buildClassifierPrompt(item: SourceItem): string {
  return [
    `Condense GitHub ${item.type} #${item.id} into a strict triage record.`,
    "Do not include clustering or pillar fields. Identity fields are managed by runtime.",
    "Output must follow the provided json_schema exactly.",
    "",
    "SOURCE_ITEM:",
    JSON.stringify(
      {
        id: item.id,
        type: item.type,
        title: item.title,
        state: item.state,
        url: item.url,
        labels: item.labels,
        body: item.body?.slice(0, 8000),
      },
      null,
      2
    ),
  ].join("\n");
}

async function callClassifier(item: SourceItem, options: CondenseOptions): Promise<unknown> {
  const classifier = options.classifier_tool ?? (globalThis as any).tools?.classifier;
  if (!classifier) {
    throw new Error("classifier tool is unavailable");
  }

  const prompt = buildClassifierPrompt(item);
  const input = {
    description: `triage-condense source item ${item.type} #${item.id}`,
    prompt,
    json_schema: TRIAGE_CLASSIFY_JSON_SCHEMA,
    item,
  };

  return classifier(input);
}

export async function condenseGitHubItem(
  item: SourceItem,
  options: CondenseOptions = {}
): Promise<TriageRecord> {
  const raw = await callClassifier(item, options);
  return normalizeTriageRecord(item, raw);
}
