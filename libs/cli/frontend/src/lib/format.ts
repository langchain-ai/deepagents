function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function extOf(path: string): string {
  return path.split(".").pop()?.toLowerCase() ?? "";
}

export function parseDisplayContent(rawContent: unknown): string {
  if (rawContent == null) return "";
  if (typeof rawContent === "string") {
    const trimmed = rawContent.trimStart();
    if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return rawContent;
    try {
      return JSON.stringify(JSON.parse(rawContent), null, 2);
    } catch {
      return rawContent;
    }
  }

  if (Array.isArray(rawContent)) {
    return rawContent
      .map((item) => parseDisplayContent(item))
      .filter(Boolean)
      .join("\n");
  }

  if (!isRecord(rawContent)) return String(rawContent);

  const content = rawContent.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((item) => parseDisplayContent(item)).join("\n");
  }
  return JSON.stringify(rawContent, null, 2);
}

export function getErrorMessage(error: unknown, fallback = "Something went wrong."): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return fallback;
}
