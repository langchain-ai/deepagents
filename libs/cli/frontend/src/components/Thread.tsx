import { useMemo, useRef, useState, type FC } from "react";
import {
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
  useMessagePartText,
  type EmptyMessagePartComponent,
  type FileMessagePartComponent,
  type ImageMessagePartComponent,
  type ReasoningMessagePartComponent,
  type ToolCallMessagePartComponent,
} from "@assistant-ui/react";
import { StreamdownTextPrimitive } from "@assistant-ui/react-streamdown";
import { APP_NAME } from "../constants";
import { useGraphValues } from "../RuntimeProvider";
import { useTheme } from "../ThemeProvider";
import { TOOL_RENDERERS } from "./tools";
import {
  SubagentPipeline,
  SynthesisIndicator,
  type DisplaySubagent,
  type DisplaySubagentStatus,
} from "./SubagentActivity";
import TodosPanel from "./TodosPanel";
import FilesPanel from "./FilePanels";

const SUGGESTIONS = [
  "What can you help me with today?",
  "Walk me through what you can do.",
  "Draft a plan for a task.",
];

export const ThreadView: FC = () => {
  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col bg-[var(--background)]">
      <ThreadPrimitive.Viewport className="relative flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-4 sm:px-4 sm:py-6">
        <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col">
          <ThreadEmpty />
          <div className="flex flex-col gap-3">
            <ThreadPrimitive.Messages>
              {() => <ThreadMessage />}
            </ThreadPrimitive.Messages>
          </div>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadEmpty: FC = () => {
  const { theme } = useTheme();
  const logoSrc = theme === "dark" ? "/app/logo-dark.svg" : "/app/logo-light.svg";
  return (
    <ThreadPrimitive.Empty>
      <div className="flex flex-1 flex-col items-center justify-center py-24 text-center">
        <img
          src={logoSrc}
          alt={APP_NAME}
          className="anim-stagger-1 mb-4 h-14 w-14 rounded-xl"
          width={56}
          height={56}
        />
        <h2 className="anim-stagger-2 text-2xl font-semibold lc-gradient-text">
          {APP_NAME}
        </h2>
        <p className="anim-stagger-2 mt-2 text-[var(--muted-foreground)]">
          How can I help you today?
        </p>
        <div className="anim-stagger-3 mt-6 flex flex-wrap justify-center gap-2">
          {SUGGESTIONS.map((s) => (
            <ThreadPrimitive.Suggestion key={s} prompt={s} asChild>
              <button className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-xs font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)] hover:text-[var(--foreground)]">
                {s}
              </button>
            </ThreadPrimitive.Suggestion>
          ))}
        </div>
      </div>
    </ThreadPrimitive.Empty>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  if (role === "user") return <UserMessage />;
  return <AssistantMessage />;
};

const UserMessage: FC = () => (
  <MessagePrimitive.Root
    data-role="user"
    className="anim-msg flex justify-end"
  >
    <div className="max-w-[90%] rounded-2xl rounded-br-sm bg-[var(--primary)] px-4 py-2.5 text-sm leading-relaxed text-[var(--primary-foreground)]">
      <MessagePrimitive.Parts />
    </div>
  </MessagePrimitive.Root>
);

const AssistantMessage: FC = () => {
  // Derive in useMemo, not inside the selector — useAuiState compares by
  // identity and a fresh array each call loops forever (React #185).
  const parts = useAuiState((s) => s.message.parts);
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const timingsRef = useRef<Map<string, { startedAt: number; completedAt?: number }>>(
    new Map(),
  );
  const subagents = useMemo<DisplaySubagent[]>(() => {
    type Part = {
      type: string;
      toolName?: string;
      toolCallId?: string;
      args?: { subagent_type?: unknown; description?: unknown };
      result?: unknown;
      status?: { type?: string };
      isError?: boolean;
    };
    const out: DisplaySubagent[] = [];
    const timings = timingsRef.current;
    for (const p of parts as readonly Part[]) {
      if (p.type !== "tool-call" || p.toolName !== "task") continue;
      const id = p.toolCallId ?? "";
      const stType = p.status?.type;
      let status: DisplaySubagentStatus;
      if (p.isError) status = "error";
      else if (stType === "running" || stType === "incomplete") status = "running";
      else if (p.result !== undefined) status = "complete";
      else status = "pending";

      const existing = timings.get(id);
      if (status !== "pending" && !existing) {
        timings.set(id, { startedAt: Date.now() });
      }
      if ((status === "complete" || status === "error") && existing && !existing.completedAt) {
        existing.completedAt = Date.now();
      }
      const t = timings.get(id);

      const result =
        p.result == null
          ? null
          : typeof p.result === "string"
            ? p.result
            : JSON.stringify(p.result);
      out.push({
        id,
        subagentType:
          typeof p.args?.subagent_type === "string" ? p.args.subagent_type : null,
        description:
          typeof p.args?.description === "string" ? p.args.description : "",
        status,
        result,
        startedAt: t ? new Date(t.startedAt) : null,
        completedAt: t?.completedAt ? new Date(t.completedAt) : null,
      });
    }
    return out;
  }, [parts]);
  return (
    <MessagePrimitive.Root
      data-role="assistant"
      className="anim-msg flex flex-col items-start gap-2"
    >
      <MessagePrimitive.Parts
        unstable_showEmptyOnNonTextEnd={false}
        components={{
          Empty: PulsingDotsPart,
          Text: AssistantTextPart,
          Image: ImagePart,
          File: FilePart,
          Reasoning: ReasoningPart,
          tools: { by_name: TOOL_RENDERERS, Fallback: ToolFallback },
        }}
      />
      <SubagentPipeline subagents={subagents} />
      <SynthesisIndicator subagents={subagents} isRunning={isRunning} />
      <MessageError />
    </MessagePrimitive.Root>
  );
};

const AssistantTextPart: FC = () => {
  const part = useMessagePartText();
  // Skip empty text bubbles — tool-call parts have their own running indicator.
  if (!part.text) return null;
  return (
    <div className="ai-bubble max-w-[90%] rounded-2xl rounded-bl-sm border border-[var(--border)] bg-[var(--surface)] px-4 py-2.5 text-sm leading-relaxed shadow-sm">
      <div className="markdown-body">
        <StreamdownTextPrimitive />
      </div>
    </div>
  );
};

const PulsingDotsPart: EmptyMessagePartComponent = () => (
  <div className="rounded-2xl rounded-bl-sm border border-[var(--border)] bg-[var(--surface)] px-4 py-2.5 text-sm text-[var(--muted-foreground)] shadow-sm">
    <span className="inline-flex items-center gap-1">
      <span className="animate-pulse">●</span>
      <span className="animate-pulse" style={{ animationDelay: "150ms" }}>●</span>
      <span className="animate-pulse" style={{ animationDelay: "300ms" }}>●</span>
    </span>
  </div>
);

const ImagePart: ImageMessagePartComponent = ({ image }) => (
  <div className="max-w-[90%] overflow-hidden rounded-xl border border-[var(--border)]">
    <img src={image} alt="" className="max-h-96 w-auto object-contain" />
  </div>
);

const FilePart: FileMessagePartComponent = ({ filename, mimeType }) => (
  <div className="flex max-w-[90%] items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs">
    <svg className="h-3.5 w-3.5 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
      <path d="M14 2v4a2 2 0 0 0 2 2h4" />
    </svg>
    <span className="font-medium">{filename ?? "file"}</span>
    {mimeType && <span className="text-[var(--muted-foreground)]">{mimeType}</span>}
  </div>
);

const ReasoningPart: ReasoningMessagePartComponent = ({ text }) => {
  if (!text) return null;
  return (
    <details className="max-w-[90%] rounded-xl border border-[var(--border)] bg-[var(--muted)]/40 px-3 py-2 text-xs text-[var(--muted-foreground)]">
      <summary className="cursor-pointer font-medium">Reasoning</summary>
      <pre className="mt-2 whitespace-pre-wrap">{text}</pre>
    </details>
  );
};

const ToolFallback: ToolCallMessagePartComponent = ({
  toolName,
  argsText,
  result,
}) => (
  <div className="w-full max-w-[90%] rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs">
    <div className="flex items-center gap-2">
      <span className="rounded bg-[var(--accent-bg)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--foreground)]">
        tool
      </span>
      <span className="font-medium">{toolName}</span>
    </div>
    {argsText && (
      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--muted)] p-2 text-[11px] text-[var(--muted-foreground)]">
        {argsText}
      </pre>
    )}
    {result !== undefined && (
      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--muted)] p-2 text-[11px] text-[var(--muted-foreground)]">
        {typeof result === "string" ? result : JSON.stringify(result, null, 2)}
      </pre>
    )}
  </div>
);

const MessageError: FC = () => (
  <MessagePrimitive.Error>
    <ErrorPrimitive.Root className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
      <ErrorPrimitive.Message />
    </ErrorPrimitive.Root>
  </MessagePrimitive.Error>
);

type ComposerProps = {
  todoCount: number;
  fileCount: number;
  showTodos: boolean;
  showFiles: boolean;
  onToggleTodos: () => void;
  onToggleFiles: () => void;
};

export const Composer: FC<ComposerProps> = ({
  todoCount,
  fileCount,
  showTodos,
  showFiles,
  onToggleTodos,
  onToggleFiles,
}) => (
  <footer className="bg-[var(--background)] px-2 py-3 sm:px-4 sm:py-4">
    <ComposerPrimitive.Root className="mx-auto max-w-4xl">
      <div className="composer">
        <ComposerPrimitive.Input
          placeholder="Send a message..."
          rows={1}
          autoFocus
          submitMode="enter"
          className="min-h-[44px] max-h-[200px] w-full resize-none bg-transparent px-4 pt-3 pb-1 text-sm outline-none placeholder:text-[var(--muted-foreground)]"
        />
        <div className="flex items-center justify-between px-4 pb-3">
          <div className="flex items-center gap-1.5">
            {todoCount > 0 && (
              <button
                type="button"
                onClick={onToggleTodos}
                className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  showTodos
                    ? "bg-[var(--accent-bg)] text-[var(--foreground)]"
                    : "text-[var(--muted-foreground)] hover:bg-[var(--muted)]"
                }`}
              >
                <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 20h9" />
                  <path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838.838-2.872a2 2 0 0 1 .506-.855z" />
                </svg>
                Tasks
                <span className="rounded-full bg-[var(--primary)] px-1.5 py-0.5 text-[9px] font-medium text-[var(--primary-foreground)]">
                  {todoCount}
                </span>
              </button>
            )}
            {fileCount > 0 && (
              <button
                type="button"
                onClick={onToggleFiles}
                className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  showFiles
                    ? "bg-[var(--accent-bg)] text-[var(--foreground)]"
                    : "text-[var(--muted-foreground)] hover:bg-[var(--muted)]"
                }`}
              >
                <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
                  <path d="M14 2v4a2 2 0 0 0 2 2h4" />
                </svg>
                Files
                <span className="rounded-full bg-[var(--primary)] px-1.5 py-0.5 text-[9px] font-medium text-[var(--primary-foreground)]">
                  {fileCount}
                </span>
              </button>
            )}
          </div>
          <ComposerSendCancel />
        </div>
      </div>
    </ComposerPrimitive.Root>
  </footer>
);

const Thread: FC = () => {
  const { todos, files } = useGraphValues();
  const [showTodos, setShowTodos] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const fileCount = Object.keys(files).length;

  return (
    <>
      <ThreadView />
      {showTodos && todos.length > 0 && (
        <div className="border-t border-[var(--border)]">
          <TodosPanel todos={todos} />
        </div>
      )}
      {showFiles && fileCount > 0 && (
        <div className="border-t border-[var(--border)]">
          <FilesPanel files={files} />
        </div>
      )}
      <Composer
        todoCount={todos.length}
        fileCount={fileCount}
        showTodos={showTodos}
        showFiles={showFiles}
        onToggleTodos={() => setShowTodos((v) => !v)}
        onToggleFiles={() => setShowFiles((v) => !v)}
      />
    </>
  );
};

export default Thread;

const ComposerSendCancel: FC = () => {
  const isRunning = useAuiState((s) => s.thread.isRunning);
  if (isRunning) {
    return (
      <ComposerPrimitive.Cancel asChild>
        <button
          type="button"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-red-500 text-white transition-all hover:bg-red-600"
          aria-label="Stop"
        >
          <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="1" />
          </svg>
        </button>
      </ComposerPrimitive.Cancel>
    );
  }
  return (
    <ComposerPrimitive.Send asChild>
      <button
        type="submit"
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)] text-[var(--accent-foreground)] transition-all hover:bg-[var(--accent-hover)] disabled:opacity-30"
        aria-label="Send"
      >
        <svg
          className="h-3.5 w-3.5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m5 12 7-7 7 7" />
          <path d="M12 19V5" />
        </svg>
      </button>
    </ComposerPrimitive.Send>
  );
};
