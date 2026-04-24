import type { FC } from "react";
import type { ToolMessage } from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import { TOOL_RENDERERS } from "./tools";

type Props = {
  toolCall: ToolCall;
  toolResult?: ToolMessage;
  status?: "running" | "success" | "error";
};

const STATUS_DOT: Record<NonNullable<Props["status"]> | "default", string> = {
  running: "bg-yellow-400",
  success: "bg-green-500",
  error: "bg-red-500",
  default: "bg-[var(--muted-foreground)]",
};

const ToolCallPart: FC<Props> = ({ toolCall, toolResult, status }) => {
  const Renderer = TOOL_RENDERERS[toolCall.name];
  if (Renderer) {
    return (
      <Renderer
        toolCall={toolCall}
        toolResult={toolResult}
        status={status ?? "running"}
      />
    );
  }

  // Generic pill fallback
  const dotClass = STATUS_DOT[status ?? "default"];

  const resultContent =
    toolResult != null
      ? typeof (toolResult as any).content === "string"
        ? (toolResult as any).content
        : JSON.stringify((toolResult as any).content, null, 2)
      : null;

  return (
    <div className="w-full max-w-[90%] rounded-xl border border-[var(--border)] bg-[var(--muted)] px-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        <span className="font-mono text-[var(--foreground)]">{toolCall.name}</span>
        <span className={`h-2 w-2 shrink-0 rounded-full ${dotClass}`} aria-hidden="true" />
      </div>
      <details className="mt-1.5">
        <summary className="cursor-pointer text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
          Details
        </summary>
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--surface)] p-2 text-[11px] text-[var(--muted-foreground)]">
          {JSON.stringify(toolCall.args, null, 2)}
        </pre>
        {resultContent != null && (
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--surface)] p-2 text-[11px] text-[var(--muted-foreground)]">
            {resultContent}
          </pre>
        )}
      </details>
    </div>
  );
};

export default ToolCallPart;
