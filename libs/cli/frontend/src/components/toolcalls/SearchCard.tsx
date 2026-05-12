import type { FC } from "react";
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
  if (Array.isArray(result)) return result.map(String).join("\n");
  return JSON.stringify(result, null, 2);
}

const FileEntry: FC<{ name: string }> = ({ name }) => (
  <div className="flex items-center gap-1.5 py-0.5 text-xs text-[var(--foreground)]">
    <svg className="h-3 w-3 shrink-0 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
    </svg>
    <span className="truncate font-mono text-[11px]">{name}</span>
  </div>
);

const SearchCard: FC<ToolRendererProps> = ({ toolCall, expanded }) => {
  const args = parseArgs(toolCall.call.args);
  const toolName = toolCall.call.name;
  const result = formatResult(toolCall.result);

  if (!expanded) return null;

  if (toolName === "ls") {
    const path = (args.path ?? args.directory ?? ".") as string;
    const lines = result?.split("\n").filter(Boolean) ?? [];

    return (
      <div className="border-t border-[var(--border)] px-3 py-2">
        <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
          <svg className="h-3 w-3 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2" />
          </svg>
          {path}
        </div>
        {lines.length > 0 ? (
          <div className="max-h-48 space-y-0.5 overflow-y-auto">
            {lines.map((line, i) => (
              <FileEntry key={i} name={line} />
            ))}
          </div>
        ) : result ? (
          <pre className="tool-code-block">{result}</pre>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Loading...</div>
        )}
      </div>
    );
  }

  if (toolName === "glob") {
    const pattern = (args.pattern ?? args.glob ?? "") as string;
    const lines = result?.split("\n").filter(Boolean) ?? [];

    return (
      <div className="border-t border-[var(--border)] px-3 py-2">
        {pattern && (
          <div className="mb-1.5 font-mono text-[11px] text-[var(--muted-foreground)]">
            Pattern: {pattern}
          </div>
        )}
        {lines.length > 0 ? (
          <div className="max-h-48 space-y-0.5 overflow-y-auto">
            {lines.map((line, i) => (
              <FileEntry key={i} name={line} />
            ))}
          </div>
        ) : result ? (
          <pre className="tool-code-block">{result}</pre>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Searching...</div>
        )}
      </div>
    );
  }

  // grep
  const query = (args.pattern ?? args.query ?? "") as string;

  return (
    <div className="border-t border-[var(--border)] px-3 py-2">
      {query && (
        <div className="mb-1.5 font-mono text-[11px] text-[var(--muted-foreground)]">
          Search: <span className="text-[var(--foreground)]">{query}</span>
        </div>
      )}
      {result ? (
        <pre className="tool-code-block">{result}</pre>
      ) : (
        <div className="text-[11px] text-[var(--muted-foreground)]">Searching...</div>
      )}
    </div>
  );
};

export default SearchCard;
