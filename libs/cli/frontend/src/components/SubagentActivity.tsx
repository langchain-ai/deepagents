import { useEffect, useRef, useState, type FC } from "react";
import { Streamdown } from "streamdown";
import type { SubagentStatus } from "@langchain/react";
import type { AgentSubagent } from "../types";
import { getElapsedTime } from "../lib/stream";

const StatusIcon: FC<{ status: SubagentStatus }> = ({ status }) => {
  switch (status) {
    case "pending":
      return (
        <span className="text-gray-400">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
          </svg>
        </span>
      );
    case "running":
      return (
        <span className="animate-spin text-[var(--accent)]">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
          </svg>
        </span>
      );
    case "complete":
      return (
        <span className="text-emerald-500">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </span>
      );
    case "error":
      return (
        <span className="text-red-500">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" x2="9" y1="9" y2="15" />
            <line x1="9" x2="15" y1="9" y2="15" />
          </svg>
        </span>
      );
  }
};

const StatusBadge: FC<{ status: SubagentStatus }> = ({ status }) => {
  const styles: Record<SubagentStatus, string> = {
    pending: "bg-gray-100 text-gray-600",
    running: "bg-[var(--accent-bg)] text-[var(--accent-foreground)]",
    complete: "bg-emerald-100 text-emerald-700",
    error: "bg-red-100 text-red-700",
  };

  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${styles[status]}`}>
      {status}
    </span>
  );
};

const SubagentCard: FC<{
  subagent: AgentSubagent;
  autoCollapse?: boolean;
}> = ({ subagent, autoCollapse = false }) => {
  const [expanded, setExpanded] = useState(
    !autoCollapse || subagent.status === "running",
  );
  const scrollRef = useRef<HTMLDivElement>(null);
  const typeName =
    typeof subagent.toolCall.args.subagent_type === "string"
      ? subagent.toolCall.args.subagent_type
      : null;
  const description =
    typeof subagent.toolCall.args.description === "string"
      ? subagent.toolCall.args.description
      : "";
  const title = typeName ?? `Agent ${subagent.id.slice(0, 8)}`;
  const elapsed = getElapsedTime(subagent.startedAt, subagent.completedAt);
  const lastAIMessage = subagent.messages.filter((m) => m.type === "ai").at(-1);
  const isStreaming = subagent.status === "running";
  const lastAIContent = lastAIMessage?.content;
  const lastAIText =
    typeof lastAIContent === "string"
      ? lastAIContent
      : Array.isArray(lastAIContent)
        ? lastAIContent
            .filter((b): b is { type: "text"; text: string } =>
              typeof b === "object" && b !== null && "type" in b && b.type === "text" && "text" in b,
            )
            .map((b) => b.text)
            .join("")
        : "";
  const displayContent =
    subagent.status === "complete" ? subagent.result ?? "" : lastAIText;

  useEffect(() => {
    if (!isStreaming || !scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [displayContent, isStreaming]);

  return (
    <div className="anim-fade-in overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-sm">
      <button
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left hover:bg-[var(--muted)]/30"
      >
        <div className="flex min-w-0 items-center gap-2.5">
          <StatusIcon status={subagent.status} />
          <div className="min-w-0">
            <h4 className="truncate text-xs font-semibold capitalize">{title}</h4>
            {description && (
              <p className="truncate text-[11px] text-[var(--muted-foreground)]">
                {description}
              </p>
            )}
          </div>
        </div>
        <div className="ml-2 flex shrink-0 items-center gap-2">
          {elapsed && (
            <span className="text-[10px] text-[var(--muted-foreground)]">
              {elapsed}
            </span>
          )}
          <StatusBadge status={subagent.status} />
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
        </div>
      </button>
      {expanded && displayContent && (
        <div className="border-t border-[var(--border)] px-3 py-2.5">
          <div ref={scrollRef} className="max-h-64 overflow-y-auto">
            <div className="markdown-body prose prose-sm max-w-none text-xs leading-relaxed">
              <Streamdown animated isAnimating={isStreaming} parseIncompleteMarkdown>
                {displayContent}
              </Streamdown>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const SubagentProgress: FC<{ subagents: AgentSubagent[] }> = ({
  subagents,
}) => {
  const completed = subagents.filter((subagent) => subagent.status === "complete").length;
  const total = subagents.length;
  const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] text-[var(--muted-foreground)]">
        <span>Subagent progress</span>
        <span>
          {completed}/{total} complete
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-[var(--accent-bg)]">
        <div
          className="h-full rounded-full bg-[var(--accent)] transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export const SubagentPipeline: FC<{ subagents: AgentSubagent[] }> = ({
  subagents,
}) => {
  if (subagents.length === 0) return null;

  return (
    <div className="w-full max-w-[90%] space-y-2 border-l-2 border-[var(--accent)] pl-3">
      <SubagentProgress subagents={subagents} />
      {subagents.map((subagent) => (
        <SubagentCard
          key={subagent.id}
          subagent={subagent}
          autoCollapse={subagents.length >= 5 && subagent.status === "complete"}
        />
      ))}
    </div>
  );
};

export const SynthesisIndicator: FC<{
  subagents: AgentSubagent[];
  isLoading: boolean;
}> = ({ subagents, isLoading }) => {
  const allDone =
    subagents.length >= 2 &&
    subagents.every(
      (subagent) =>
        subagent.status === "complete" || subagent.status === "error",
    );

  if (!allDone || !isLoading) return null;

  return (
    <div className="anim-fade-in flex items-center gap-2 rounded-lg bg-[var(--accent-bg)] px-3 py-2 text-xs text-[var(--foreground)]">
      <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
      Synthesizing results from {subagents.length} subagent
      {subagents.length !== 1 ? "s" : ""}...
    </div>
  );
};
