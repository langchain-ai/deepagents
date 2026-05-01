import { useState, type FC } from "react";
import { getToolSummary } from "../lib/stream";
import type { AgentToolCallResult } from "../types";
import { getToolRenderer, getToolIcon } from "./toolcalls";

function formatDetails(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

const ToolIcon: FC<{ name: string }> = ({ name }) => {
  const info = getToolIcon(name);
  if (!info) return null;

  switch (info.icon) {
    case "file-plus":
    case "file-edit":
    case "file-text":
      return (
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
          <path d="M14 2v4a2 2 0 0 0 2 2h4" />
        </svg>
      );
    case "list-checks":
      return (
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 20h9" />
          <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838.838-2.872a2 2 0 0 1 .506-.855z" />
        </svg>
      );
    case "folder":
      return (
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2" />
        </svg>
      );
    case "search":
      return (
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
      );
    case "brain":
      return (
        <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2a4 4 0 0 0-4 4v1a3 3 0 0 0-3 3v1a3 3 0 0 0 3 3h1a4 4 0 0 0 0 8h2a4 4 0 0 0 0-8h1a3 3 0 0 0 3-3v-1a3 3 0 0 0-3-3V6a4 4 0 0 0-4-4z" />
          <path d="M9 10h6" />
          <path d="M9 14h6" />
        </svg>
      );
    default:
      return null;
  }
};

const ToolCallCard: FC<{ toolCall: AgentToolCallResult }> = ({ toolCall }) => {
  const [expanded, setExpanded] = useState(false);
  const isPending = toolCall.state === "pending";
  const isError = toolCall.state === "error";
  const summary = getToolSummary(toolCall.call.args);
  const argsStr = formatDetails(toolCall.call.args);
  const resultStr = formatDetails(toolCall.result);

  const Renderer = getToolRenderer(toolCall.call.name);
  const hasToolIcon = getToolIcon(toolCall.call.name) !== null;

  return (
    <div className="anim-fade-in max-w-[90%] overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] text-xs">
      <button
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-[var(--muted)]/30"
      >
        {isPending ? (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-amber-100 text-amber-700">
            <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          </span>
        ) : isError ? (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-red-100 text-red-700">
            <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" x2="9" y1="9" y2="15" />
              <line x1="9" x2="15" y1="9" y2="15" />
            </svg>
          </span>
        ) : hasToolIcon ? (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-blue-50 text-blue-600">
            <ToolIcon name={toolCall.call.name} />
          </span>
        ) : (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-emerald-100 text-emerald-700">
            <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </span>
        )}
        <span className="font-medium text-[var(--foreground)]">
          {toolCall.call.name}
        </span>
        {summary && !expanded && (
          <span className="truncate text-[var(--muted-foreground)]">{summary}</span>
        )}
        <span className="ml-auto flex items-center gap-1.5">
          {isPending && (
            <span className="text-[var(--muted-foreground)]">running...</span>
          )}
          {isError && <span className="text-red-600">error</span>}
          <svg
            className={`h-3 w-3 text-[var(--muted-foreground)] transition-transform ${
              expanded ? "rotate-180" : ""
            }`}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </span>
      </button>
      {Renderer ? (
        <Renderer toolCall={toolCall} expanded={expanded} />
      ) : expanded && (argsStr || resultStr) ? (
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap border-t border-[var(--border)] px-3 py-2 text-[var(--muted-foreground)]">
          {resultStr ?? argsStr}
        </pre>
      ) : null}
    </div>
  );
};

export default ToolCallCard;
