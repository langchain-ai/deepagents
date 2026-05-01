import { useStream, type UseDeepAgentStreamOptions, type UseStream } from "@langchain/react";
import type { DefaultToolCall } from "@langchain/react";
import type { AgentState } from "../types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isImageContentBlock(
  block: unknown,
): block is {
  type: "image" | "image_url";
  image_url?: string | { url: string };
  source?: { data?: string };
  url?: string;
} {
  return (
    isRecord(block) &&
    typeof block.type === "string" &&
    (block.type === "image_url" || block.type === "image")
  );
}

function getImageUrl(block: {
  image_url?: string | { url: string };
  source?: { data?: string };
  url?: string;
}): string {
  if (typeof block.image_url === "string") return block.image_url;
  if (isRecord(block.image_url) && typeof block.image_url.url === "string") {
    return block.image_url.url;
  }
  if (isRecord(block.source) && typeof block.source.data === "string") {
    return block.source.data;
  }
  return typeof block.url === "string" ? block.url : "";
}

export function useAgentStream(
  options: UseDeepAgentStreamOptions<AgentState>,
): UseStream<AgentState> {
  // UseDeepAgentStreamOptions extends UseStreamOptions with subagent support,
  // but the return type is the same UseStream shape at runtime.
  return useStream<AgentState>(options) as unknown as UseStream<AgentState>;
}

export function getImageBlocks(content: unknown): { url: string }[] {
  if (!Array.isArray(content)) return [];

  return content
    .filter(isImageContentBlock)
    .map((block) => ({ url: getImageUrl(block) }))
    .filter((block) => block.url.length > 0);
}

export function getElapsedTime(
  startedAt: Date | null,
  completedAt: Date | null,
): string | null {
  if (!startedAt) return null;

  const end = completedAt ?? new Date();
  const seconds = Math.round((end.getTime() - startedAt.getTime()) / 1000);

  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
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

  if (!isRecord(rawContent)) {
    return String(rawContent);
  }

  const content = rawContent.content;

  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((item) => parseDisplayContent(item)).join("\n");
  }

  return JSON.stringify(rawContent, null, 2);
}

export function getErrorMessage(
  error: unknown,
  fallback = "Something went wrong.",
): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return fallback;
}

export function getToolSummary(args: DefaultToolCall["args"] | string | undefined): string {
  if (typeof args === "string") return args.slice(0, 60);
  if (!isRecord(args)) return "";

  const fileArg = [args.file_name, args.filename, args.path].find(
    (value) => typeof value === "string" && value.length > 0,
  );
  if (typeof fileArg === "string") return fileArg;

  if (Array.isArray(args.todos)) return `${args.todos.length} items`;
  if (typeof args.query === "string") return args.query;

  const firstString = Object.values(args).find(
    (value): value is string => typeof value === "string" && value.length > 0,
  );

  return firstString?.slice(0, 60) ?? "";
}
