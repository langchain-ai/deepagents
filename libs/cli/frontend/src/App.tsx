import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAuthAdapter } from "./auth/loader";
import type { AuthAdapter } from "./auth/types";
import { ASSISTANT_ID } from "./constants";
import RuntimeProvider from "./RuntimeProvider";
import Thread from "./components/Thread";
import AppHeader from "./components/AppHeader";
import ThreadPicker from "./components/ThreadPicker";
import MessageList from "./components/MessageList";
import FilesPanel from "./components/FilePanels";
import TodosPanel from "./components/TodosPanel";
import { useAgentStream } from "./lib/stream";

const USE_NEW_CHAT = import.meta.env.VITE_NEW_CHAT === "1";

export default function App() {
  return (
    <Suspense fallback={<SplashScreen />}>
      <AppWithAdapter />
    </Suspense>
  );
}

function AppWithAdapter() {
  const adapter = useAuthAdapter();
  return (
    <adapter.Provider>
      <Gate adapter={adapter} />
    </adapter.Provider>
  );
}

function Gate({ adapter }: { adapter: AuthAdapter }) {
  const session = adapter.useSession();
  if (session.status === "loading") return <SplashScreen />;
  if (session.status === "signed-out") {
    const { AuthUI } = adapter;
    return <AuthUI />;
  }
  return (
    <AuthenticatedApp
      accessToken={session.accessToken}
      userEmail={session.userEmail}
      onSignOut={session.signOut}
    />
  );
}

function SplashScreen() {
  return <div className="min-h-dvh bg-[var(--background)]" />;
}

function AuthenticatedApp({
  accessToken,
  userEmail,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  if (USE_NEW_CHAT) {
    return (
      <NewChatApp
        accessToken={accessToken}
        userEmail={userEmail}
        onSignOut={onSignOut}
      />
    );
  }

  return (
    <RuntimeProvider accessToken={accessToken} assistantId={ASSISTANT_ID}>
      <div className="flex h-dvh flex-col bg-[var(--background)]">
        <AppHeader userEmail={userEmail} onSignOut={onSignOut} threadPicker={null} />
        <Thread />
      </div>
    </RuntimeProvider>
  );
}

function NewChatApp({
  accessToken,
  userEmail,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [showTodos, setShowTodos] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const mainRef = useRef<HTMLDivElement | null>(null);
  const isNearBottom = useRef(true);
  const rafId = useRef(0);

  const defaultHeaders = useMemo(
    () => ({ Authorization: `Bearer ${accessToken}` }),
    [accessToken],
  );

  const stream = useAgentStream({
    apiUrl: window.location.origin,
    assistantId: ASSISTANT_ID,
    messagesKey: "messages",
    threadId,
    onThreadId: setThreadId,
    filterSubagentMessages: true,
    defaultHeaders,
  });

  const { messages, isLoading, error, values } = stream;
  const files = (values.files ?? {}) as Record<string, unknown>;
  const todos = values.todos ?? [];
  const todoCount = todos.length;
  const fileCount = Object.keys(files).length;

  const handleScroll = useCallback(() => {
    const element = mainRef.current;
    if (!element) return;
    isNearBottom.current =
      element.scrollHeight - element.scrollTop - element.clientHeight < 80;
  }, []);

  useEffect(() => {
    if (!isNearBottom.current) return;

    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(() => {
      const element = mainRef.current;
      if (element) element.scrollTop = element.scrollHeight;
    });
  }, [messages]);

  useEffect(() => () => cancelAnimationFrame(rafId.current), []);

  const submitInput = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    await stream.submit(
      { messages: [{ type: "human", content: text }] },
      { streamSubgraphs: true },
    );
  }, [input, isLoading, stream]);

  const threadPicker = (
    <ThreadPicker
      currentThreadId={threadId}
      onSelect={setThreadId}
      accessToken={accessToken}
    />
  );

  return (
    <div className="flex h-dvh flex-col bg-[var(--background)]">
      <AppHeader
        userEmail={userEmail}
        onSignOut={onSignOut}
        threadPicker={threadPicker}
      />

      <MessageList
        bottomRef={bottomRef}
        error={error}
        isLoading={isLoading}
        mainRef={mainRef}
        messages={messages}
        onScroll={handleScroll}
        onSuggestionSelect={setInput}
        stream={stream}
      />

      {showTodos && todoCount > 0 && (
        <div className="border-t border-[var(--border)]">
          <TodosPanel todos={todos} />
        </div>
      )}
      {showFiles && fileCount > 0 && (
        <div className="border-t border-[var(--border)]">
          <FilesPanel files={files} />
        </div>
      )}

      <footer className="bg-[var(--background)] px-2 py-3 sm:px-4 sm:py-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submitInput();
          }}
          className="mx-auto max-w-4xl"
        >
          <div className="composer">
            <textarea
              value={input}
              onChange={(event) => {
                setInput(event.target.value);
                event.target.style.height = "auto";
                event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`;
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitInput();
                }
              }}
              placeholder="Send a message..."
              rows={1}
              autoFocus
              className="min-h-[44px] max-h-[200px] w-full resize-none bg-transparent px-4 pt-3 pb-1 text-sm outline-none placeholder:text-[var(--muted-foreground)]"
            />
            <div className="flex items-center justify-between px-4 pb-3">
              <div className="flex items-center gap-1.5">
                {todoCount > 0 && (
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
                      {todoCount}
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
              <button
                type={isLoading ? "button" : "submit"}
                onClick={isLoading ? () => void stream.stop() : undefined}
                disabled={!isLoading && !input.trim()}
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-all disabled:opacity-30 ${
                  isLoading
                    ? "bg-red-500 text-white hover:bg-red-600"
                    : "bg-[var(--accent)] text-[var(--accent-foreground)] hover:bg-[var(--accent-hover)]"
                }`}
              >
                {isLoading ? (
                  <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                ) : (
                  <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m5 12 7-7 7 7" />
                    <path d="M12 19V5" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </form>
      </footer>
    </div>
  );
}
