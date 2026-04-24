import { useState } from "react";
import { useChat } from "../ChatProvider";
import { useTheme } from "../ThemeProvider";
import { APP_NAME } from "../constants";
import MessageList from "./chat/MessageList";
import Composer from "./chat/Composer";
import SubagentPipeline from "./chat/SubagentPipeline";
import TodosPanel from "./TodosPanel";
import FilesPanel from "./FilePanels";
import type { BaseMessage } from "@langchain/core/messages";

function getMessageType(message: BaseMessage): string {
  if (typeof (message as any).getType === "function") {
    return (message as any).getType();
  }
  return (message as any).type ?? "unknown";
}

const SUGGESTIONS = [
  "What can you help me with today?",
  "Walk me through what you can do.",
  "Draft a plan for a task.",
];

function EmptyState() {
  const { theme } = useTheme();
  const logoSrc = theme === "dark" ? "/app/logo-dark.svg" : "/app/logo-light.svg";
  return (
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
          <span
            key={s}
            className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-xs font-medium text-[var(--muted-foreground)]"
          >
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function NewThread() {
  const { stream } = useChat();
  const [showTodos, setShowTodos] = useState(false);
  const [showFiles, setShowFiles] = useState(false);

  const todos = (stream.values as any)?.todos ?? [];
  const files = (stream.values as any)?.files ?? {};
  const fileCount = Object.keys(files).length;

  const handleSubmit = (text: string) => {
    void stream.submit(
      { messages: [{ type: "human", content: text }] },
      { streamSubgraphs: true },
    );
  };

  const hasMessages = stream.messages.length > 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-[var(--background)]">
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {!hasMessages ? (
          <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-4 sm:px-4 sm:py-6">
            <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col">
              <EmptyState />
            </div>
          </div>
        ) : (
          <MessageList
            messages={stream.messages}
          >
            {(msg) => {
              const type = getMessageType(msg);
              if (type !== "ai") return null;
              const msgId = (msg as any).id;
              if (!msgId) return null;
              const subagents = stream.getSubagentsByMessage(msgId);
              if (subagents.length === 0) return null;
              return (
                <div className="ml-4 mt-1">
                  <SubagentPipeline subagents={subagents} />
                </div>
              );
            }}
          </MessageList>
        )}
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
      </div>
      <footer className="bg-[var(--background)] px-2 py-3 sm:px-4 sm:py-4">
        <div className="mx-auto max-w-4xl">
          <div className="composer">
            <Composer
              onSubmit={handleSubmit}
              onStop={() => stream.stop()}
              isLoading={stream.isLoading}
            />
            <div className="flex items-center gap-1.5 px-4 pb-3">
              {todos.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowTodos((v) => !v)}
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
                    {todos.length}
                  </span>
                </button>
              )}
              {fileCount > 0 && (
                <button
                  type="button"
                  onClick={() => setShowFiles((v) => !v)}
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
          </div>
        </div>
      </footer>
    </div>
  );
}
