import { useState, type FC } from "react";
import type { ToolRendererProps } from "./index";

function parseArgs(args: unknown): Record<string, unknown> {
  if (typeof args === "string") {
    try { return JSON.parse(args); } catch { return {}; }
  }
  return (args && typeof args === "object") ? args as Record<string, unknown> : {};
}

function formatResult(result: unknown): string | null {
  if (result == null) return null;
  if (typeof result === "string") return result;
  return JSON.stringify(result, null, 2);
}

const MAX_PREVIEW_LINES = 20;

const CodePreview: FC<{ content: string; label?: string }> = ({ content, label }) => {
  const lines = content.split("\n");
  const [showAll, setShowAll] = useState(false);
  const truncated = !showAll && lines.length > MAX_PREVIEW_LINES;
  const displayContent = truncated ? lines.slice(0, MAX_PREVIEW_LINES).join("\n") : content;

  return (
    <div>
      {label && (
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
          {label}
        </div>
      )}
      <pre className="tool-code-block">{displayContent}</pre>
      {truncated && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-1 text-[10px] text-[var(--accent)] hover:underline"
        >
          Show all {lines.length} lines
        </button>
      )}
    </div>
  );
};

const FileToolCard: FC<ToolRendererProps> = ({ toolCall, expanded }) => {
  const args = parseArgs(toolCall.call.args);
  const toolName = toolCall.call.name;
  const fileName = (args.file_name ?? args.filename ?? args.path ?? "") as string;
  const content = (args.content ?? "") as string;
  const oldStr = (args.old_str ?? args.old_string ?? "") as string;
  const newStr = (args.new_str ?? args.new_string ?? "") as string;
  const result = formatResult(toolCall.result);

  if (!expanded) return null;

  if (toolName === "edit_file" && (oldStr || newStr)) {
    return (
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {fileName && (
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
            <svg className="h-3 w-3 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
              <path d="M14 2v4a2 2 0 0 0 2 2h4" />
            </svg>
            {fileName}
          </div>
        )}
        {oldStr && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-red-500">Removed</div>
            <pre className="tool-diff-old">{oldStr}</pre>
          </div>
        )}
        {newStr && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-600">Added</div>
            <pre className="tool-diff-new">{newStr}</pre>
          </div>
        )}
      </div>
    );
  }

  if (toolName === "read_file") {
    return (
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {fileName && (
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
            <svg className="h-3 w-3 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
              <path d="M14 2v4a2 2 0 0 0 2 2h4" />
            </svg>
            {fileName}
          </div>
        )}
        {result ? (
          <CodePreview content={result} />
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Reading...</div>
        )}
      </div>
    );
  }

  // write_file
  return (
    <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
      {fileName && (
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
          <svg className="h-3 w-3 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
            <path d="M14 2v4a2 2 0 0 0 2 2h4" />
          </svg>
          {fileName}
        </div>
      )}
      {content ? (
        <CodePreview content={content} />
      ) : result ? (
        <CodePreview content={result} />
      ) : null}
    </div>
  );
};

export default FileToolCard;
