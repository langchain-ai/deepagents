import { useCallback, useEffect, useState, type FC } from "react";
import type { Thread } from "@langchain/langgraph-sdk";
import { useChat } from "../../ChatProvider";
import { getErrorMessage } from "../../lib/format";

type ThreadSummary = Thread<Record<string, unknown>>;

const ThreadPicker: FC = () => {
  const { currentThreadId, switchToThread, newThread, createClient } = useChat();

  const [open, setOpen] = useState(false);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const handleSelect = useCallback(
    (id: string | null) => {
      if (id === null) newThread();
      else switchToThread(id);
    },
    [newThread, switchToThread],
  );

  const loadThreads = useCallback(async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const result = await createClient().threads.search<ThreadSummary["values"]>({
        limit: 20,
      });
      setThreads(result);
    } catch (error) {
      setLoadError(getErrorMessage(error, "Failed to load threads."));
    } finally {
      setIsLoading(false);
    }
  }, [createClient]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  const handleToggleOpen = () => {
    setOpen((prev) => {
      const next = !prev;
      if (next) void loadThreads();
      return next;
    });
  };

  const label = currentThreadId
    ? `${currentThreadId.slice(0, 8)}...`
    : "Threads";

  return (
    <div className="relative">
      <button
        onClick={handleToggleOpen}
        className="flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted-foreground)] hover:bg-[var(--muted)]"
      >
        <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        {label}
        <svg
          className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="anim-slide-down absolute right-0 top-full z-50 mt-1 w-[calc(100vw-2rem)] rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-lg sm:w-72">
            <div className="border-b border-[var(--border)] px-3 py-2">
              <button
                onClick={() => {
                  handleSelect(null);
                  setOpen(false);
                }}
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs font-medium text-[var(--foreground)] hover:bg-[var(--muted)]"
              >
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14" />
                  <path d="M12 5v14" />
                </svg>
                New Thread
              </button>
            </div>
            <div className="max-h-64 overflow-y-auto py-1">
              {isLoading && (
                <p className="px-3 py-4 text-center text-xs text-[var(--muted-foreground)]">
                  Loading threads...
                </p>
              )}
              {loadError && (
                <p className="px-3 py-4 text-center text-xs text-red-600">
                  {loadError}
                </p>
              )}
              {!isLoading && !loadError && threads.length === 0 && (
                <p className="px-3 py-4 text-center text-xs text-[var(--muted-foreground)]">
                  No threads yet
                </p>
              )}
              {!isLoading &&
                !loadError &&
                threads.map((thread) => (
                  <button
                    key={thread.thread_id}
                    onClick={() => {
                      handleSelect(thread.thread_id);
                      setOpen(false);
                    }}
                    className={`flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-[var(--muted)] ${
                      thread.thread_id === currentThreadId
                        ? "bg-[var(--muted)] font-medium"
                        : ""
                    }`}
                  >
                    <span className="truncate">
                      {(typeof thread.metadata?.title === "string"
                        ? thread.metadata.title
                        : "") || `${thread.thread_id.slice(0, 12)}…`}
                    </span>
                    <span className="ml-auto shrink-0 text-[var(--muted-foreground)]">
                      {new Date(thread.created_at).toLocaleDateString()}
                    </span>
                  </button>
                ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ThreadPicker;
