import { useState, type FC, type ReactElement, type ReactNode } from "react";
import type { ToolCall } from "@langchain/core/messages/tool";
import type { ToolMessage } from "@langchain/core/messages";
import { useChat, type AgentState } from "../../ChatProvider";

export type ToolCallRenderer = (props: {
  toolCall: ToolCall;
  toolResult?: ToolMessage;
  status: "running" | "success" | "error";
}) => ReactElement | null;

// ---------------------------------------------------------------------------
// Helpers (ported from src/components/tools.tsx)
// ---------------------------------------------------------------------------

type ArgsLike = Record<string, unknown>;

function asObject(value: unknown): ArgsLike {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as ArgsLike;
  }
  return {};
}

function getArgs(args: unknown): ArgsLike {
  return asObject(args);
}

function formatResult(result: unknown): string | null {
  if (result == null) return null;
  if (typeof result === "string") return result;
  if (Array.isArray(result)) return result.map(String).join("\n");
  return JSON.stringify(result, null, 2);
}

function getSummary(args: ArgsLike): string {
  const fileArg = [args.file_name, args.filename, args.path].find(
    (v) => typeof v === "string" && v.length > 0,
  );
  if (typeof fileArg === "string") return fileArg;
  if (Array.isArray(args.todos)) return `${args.todos.length} items`;
  if (typeof args.query === "string") return args.query;
  if (typeof args.pattern === "string") return args.pattern;
  const firstString = Object.values(args).find(
    (v): v is string => typeof v === "string" && v.length > 0,
  );
  return firstString?.slice(0, 60) ?? "";
}

function getResultString(toolResult: ToolMessage | undefined): string | null {
  if (!toolResult) return null;
  const content = toolResult.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(String).join("\n");
  return formatResult(content);
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

const SpinnerIcon = () => (
  <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
  </svg>
);
const ErrIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="15" x2="9" y1="9" y2="15" />
    <line x1="9" x2="15" y1="9" y2="15" />
  </svg>
);
const FileIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
    <path d="M14 2v4a2 2 0 0 0 2 2h4" />
  </svg>
);
const FolderIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2" />
  </svg>
);
const SearchIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);
const TodosIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 20h9" />
    <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838.838-2.872a2 2 0 0 1 .506-.855z" />
  </svg>
);

// ---------------------------------------------------------------------------
// ToolCard
// ---------------------------------------------------------------------------

type CardProps = {
  toolName: string;
  icon: ReactNode;
  isPending: boolean;
  isError: boolean;
  summary?: string;
  children?: ReactNode;
};

const ToolCard: FC<CardProps> = ({ toolName, icon, isPending, isError, summary, children }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="anim-fade-in mt-2 max-w-[90%] overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] text-xs">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-[var(--muted)]/30"
      >
        {isPending ? (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-amber-100 text-amber-700">
            <SpinnerIcon />
          </span>
        ) : isError ? (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-red-100 text-red-700">
            <ErrIcon />
          </span>
        ) : (
          <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded bg-blue-50 text-blue-600">
            {icon}
          </span>
        )}
        <span className="font-medium text-[var(--foreground)]">{toolName}</span>
        {summary && !expanded && (
          <span className="truncate text-[var(--muted-foreground)]">{summary}</span>
        )}
        <span className="ml-auto flex items-center gap-1.5">
          {isPending && <span className="text-[var(--muted-foreground)]">running...</span>}
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
      {expanded && children}
    </div>
  );
};

// ---------------------------------------------------------------------------
// CodePreview / FileLine
// ---------------------------------------------------------------------------

const MAX_PREVIEW_LINES = 20;
const CodePreview: FC<{ content: string; label?: string }> = ({ content, label }) => {
  const lines = content.split("\n");
  const [showAll, setShowAll] = useState(false);
  const truncated = !showAll && lines.length > MAX_PREVIEW_LINES;
  const display = truncated ? lines.slice(0, MAX_PREVIEW_LINES).join("\n") : content;
  return (
    <div>
      {label && (
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
          {label}
        </div>
      )}
      <pre className="tool-code-block">{display}</pre>
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

const FileLine: FC<{ name: string }> = ({ name }) => (
  <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
    <span className="h-3 w-3 text-[var(--muted-foreground)]">
      <FileIcon />
    </span>
    {name}
  </div>
);

// ---------------------------------------------------------------------------
// FileTool
// ---------------------------------------------------------------------------

export const FileTool: ToolCallRenderer = ({ toolCall, toolResult, status }) => {
  const a = getArgs(toolCall.args);
  const toolName = toolCall.name;
  const fileName = (a.file_name ?? a.filename ?? a.path ?? "") as string;
  const content = (a.content ?? "") as string;
  const oldStr = (a.old_str ?? a.old_string ?? "") as string;
  const newStr = (a.new_str ?? a.new_string ?? "") as string;
  const resultStr = getResultString(toolResult);
  const isPending = status === "running";
  const isError = status === "error";

  let body: ReactNode = null;
  if (toolName === "edit_file" && (oldStr || newStr)) {
    body = (
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {fileName && <FileLine name={fileName} />}
        {oldStr && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-red-500">
              Removed
            </div>
            <pre className="tool-diff-old">{oldStr}</pre>
          </div>
        )}
        {newStr && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-600">
              Added
            </div>
            <pre className="tool-diff-new">{newStr}</pre>
          </div>
        )}
      </div>
    );
  } else if (toolName === "read_file") {
    body = (
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {fileName && <FileLine name={fileName} />}
        {resultStr ? (
          <CodePreview content={resultStr} />
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Reading...</div>
        )}
      </div>
    );
  } else {
    body = (
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {fileName && <FileLine name={fileName} />}
        {content ? <CodePreview content={content} /> : resultStr ? <CodePreview content={resultStr} /> : null}
      </div>
    );
  }

  return (
    <ToolCard
      toolName={toolName}
      icon={<FileIcon />}
      isPending={isPending}
      isError={isError}
      summary={fileName || getSummary(a)}
    >
      {body}
    </ToolCard>
  );
};

// ---------------------------------------------------------------------------
// SearchTool
// ---------------------------------------------------------------------------

export const SearchTool: ToolCallRenderer = ({ toolCall, toolResult, status }) => {
  const a = getArgs(toolCall.args);
  const toolName = toolCall.name;
  const resultStr = getResultString(toolResult);
  const isPending = status === "running";
  const isError = status === "error";

  let body: ReactNode = null;
  if (toolName === "ls") {
    const path = (a.path ?? a.directory ?? ".") as string;
    const lines = resultStr?.split("\n").filter(Boolean) ?? [];
    body = (
      <div className="border-t border-[var(--border)] px-3 py-2">
        <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium text-[var(--foreground)]">
          <span className="h-3 w-3 text-[var(--muted-foreground)]">
            <FolderIcon />
          </span>
          {path}
        </div>
        {lines.length > 0 ? (
          <div className="max-h-48 space-y-0.5 overflow-y-auto">
            {lines.map((line, i) => (
              <FileLine key={i} name={line} />
            ))}
          </div>
        ) : resultStr ? (
          <pre className="tool-code-block">{resultStr}</pre>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Loading...</div>
        )}
      </div>
    );
  } else if (toolName === "glob") {
    const pattern = (a.pattern ?? a.glob ?? "") as string;
    const lines = resultStr?.split("\n").filter(Boolean) ?? [];
    body = (
      <div className="border-t border-[var(--border)] px-3 py-2">
        {pattern && (
          <div className="mb-1.5 font-mono text-[11px] text-[var(--muted-foreground)]">
            Pattern: {pattern}
          </div>
        )}
        {lines.length > 0 ? (
          <div className="max-h-48 space-y-0.5 overflow-y-auto">
            {lines.map((line, i) => (
              <FileLine key={i} name={line} />
            ))}
          </div>
        ) : resultStr ? (
          <pre className="tool-code-block">{resultStr}</pre>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Searching...</div>
        )}
      </div>
    );
  } else {
    const query = (a.pattern ?? a.query ?? "") as string;
    body = (
      <div className="border-t border-[var(--border)] px-3 py-2">
        {query && (
          <div className="mb-1.5 font-mono text-[11px] text-[var(--muted-foreground)]">
            Search: <span className="text-[var(--foreground)]">{query}</span>
          </div>
        )}
        {resultStr ? (
          <pre className="tool-code-block">{resultStr}</pre>
        ) : (
          <div className="text-[11px] text-[var(--muted-foreground)]">Searching...</div>
        )}
      </div>
    );
  }

  return (
    <ToolCard
      toolName={toolName}
      icon={toolName === "ls" ? <FolderIcon /> : <SearchIcon />}
      isPending={isPending}
      isError={isError}
      summary={getSummary(a)}
    >
      {body}
    </ToolCard>
  );
};

// ---------------------------------------------------------------------------
// ThinkTool
// ---------------------------------------------------------------------------

const PREVIEW_LINES = 3;
function parseThought(args: ArgsLike): string {
  const fromArgs = args.thought ?? args.content ?? args.reflection ?? args.text;
  if (typeof fromArgs === "string") return fromArgs;
  const firstString = Object.values(args).find(
    (v): v is string => typeof v === "string" && v.length > 0,
  );
  return firstString ?? "";
}

export const ThinkTool: ToolCallRenderer = ({ toolCall }) => {
  const a = getArgs(toolCall.args);
  const thought = parseThought(a);
  const lines = thought.split("\n");
  const [showAll, setShowAll] = useState(false);
  const needsTrunc = lines.length > PREVIEW_LINES;
  const preview = lines.slice(0, PREVIEW_LINES).join("\n");
  return (
    <div className="anim-fade-in mt-2 max-w-[90%] rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      <div className="tool-think-block mx-3 my-2">
        <div className="whitespace-pre-wrap text-xs leading-relaxed">
          {showAll ? thought : preview}
          {needsTrunc && !showAll && (
            <>
              {"..."}
              <button
                onClick={() => setShowAll(true)}
                className="ml-1 text-violet-600 hover:underline"
              >
                more
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// TodosTool — reads live state from ChatProvider
// ---------------------------------------------------------------------------

export const TodosTool: ToolCallRenderer = ({ toolCall, status }) => {
  const { stream } = useChat();
  const todos = (stream.values as unknown as AgentState | undefined)?.todos ?? [];
  const toolName = toolCall.name;
  const isPending = status === "running";
  const isError = status === "error";
  return (
    <ToolCard
      toolName={toolName}
      icon={<TodosIcon />}
      isPending={isPending}
      isError={isError}
      summary={`${todos.length} items`}
    >
      <div className="space-y-1 border-t border-[var(--border)] px-3 py-2">
        {todos.map((todo: any, i: number) => (
          <div key={todo.id ?? `${todo.content}-${i}`} className="flex items-start gap-2 rounded-md px-1 py-1 text-xs">
            {todo.status === "completed" ? (
              <span className="mt-0.5 text-emerald-600">
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </span>
            ) : todo.status === "in_progress" ? (
              <span className="mt-0.5 text-blue-600">
                <SpinnerIcon />
              </span>
            ) : (
              <span className="mt-0.5 text-[var(--muted-foreground)]">
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                </svg>
              </span>
            )}
            <span className={todo.status === "completed" ? "line-through text-[var(--muted-foreground)]" : "text-[var(--foreground)]"}>
              {todo.content ?? "Untitled task"}
            </span>
          </div>
        ))}
      </div>
    </ToolCard>
  );
};

// ---------------------------------------------------------------------------
// SubagentTool
// ---------------------------------------------------------------------------

const SubagentIcon = () => (
  <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="8" r="4" />
    <path d="M20 21a8 8 0 0 0-16 0" />
  </svg>
);

// Retained for future HITL / task-approval UI. Currently the `task` key
// in TOOL_RENDERERS maps to NoopTool and subagent progress is rendered
// via SubagentPipeline.
export const SubagentTool: ToolCallRenderer = ({ toolCall, toolResult, status }) => {
  const a = getArgs(toolCall.args);
  const subagentType = (a.subagent_type ?? a.type ?? "subagent") as string;
  const description = (a.description ?? "") as string;
  const prompt = (a.prompt ?? "") as string;
  const resultStr = getResultString(toolResult);
  const isPending = status === "running";
  const isError = status === "error";

  return (
    <ToolCard
      toolName={`task → ${subagentType}`}
      icon={<SubagentIcon />}
      isPending={isPending}
      isError={isError}
      summary={description.slice(0, 80)}
    >
      <div className="space-y-2 border-t border-[var(--border)] px-3 py-2">
        {description && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
              Description
            </div>
            <div className="text-[11px] text-[var(--foreground)]">{description}</div>
          </div>
        )}
        {prompt && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
              Prompt
            </div>
            <pre className="tool-code-block">{prompt}</pre>
          </div>
        )}
        {resultStr && (
          <div>
            <div className="mb-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-600">
              Result
            </div>
            <CodePreview content={resultStr} />
          </div>
        )}
      </div>
    </ToolCard>
  );
};

// ---------------------------------------------------------------------------
// NoopTool — `task` tool calls render in SubagentPipeline, not inline
// ---------------------------------------------------------------------------

const NoopTool: ToolCallRenderer = () => null;

// ---------------------------------------------------------------------------
// TOOL_RENDERERS dispatch table
// ---------------------------------------------------------------------------

export const TOOL_RENDERERS: Record<string, ToolCallRenderer> = {
  read_file: FileTool,
  write_file: FileTool,
  edit_file: FileTool,
  ls: SearchTool,
  glob: SearchTool,
  grep: SearchTool,
  think: ThinkTool,
  think_tool: ThinkTool,
  write_todos: TodosTool,
  task: NoopTool,
};
