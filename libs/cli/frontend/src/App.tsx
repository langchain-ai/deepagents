import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Client } from "@langchain/langgraph-sdk";
import type { BaseMessage } from "@langchain/core/messages";

import { useAuthAdapter } from "./auth/loader";
import type { AuthAdapter } from "./auth/types";
import { ASSISTANT_ID, SANDBOX_SCOPE, UPLOADS_ENABLED } from "./constants";
import AppHeader from "./components/AppHeader";
import ThreadPicker from "./components/ThreadPicker";
import MessageList from "./components/MessageList";
import FilesPanel from "./components/FilePanels";
import TodosPanel from "./components/TodosPanel";
import { useAgentStream } from "./lib/stream";

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
      userIdentity={session.userIdentity}
      isAnonymous={session.isAnonymous}
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
  userIdentity,
  isAnonymous,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  userIdentity: string;
  isAnonymous: boolean;
  onSignOut: () => Promise<void>;
}) {
  return (
    <NewChatApp
      accessToken={accessToken}
      userEmail={userEmail}
      userIdentity={userIdentity}
      isAnonymous={isAnonymous}
      onSignOut={onSignOut}
    />
  );
}

type UploadResponse = {
  path?: string;
  sandbox?: SandboxRecord;
  error?: string;
};

type SandboxRecord = {
  provider: string;
  sandbox_id: string;
  cache_key: string;
  image: string;
  scope: "thread" | "assistant";
};

type EnsureSandboxResponse = {
  sandbox?: SandboxRecord;
  error?: string;
};

function isSandboxRecord(value: unknown): value is SandboxRecord {
  if (typeof value !== "object" || value == null) return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.provider === "string" &&
    typeof record.sandbox_id === "string" &&
    typeof record.cache_key === "string" &&
    typeof record.image === "string" &&
    (record.scope === "thread" || record.scope === "assistant")
  );
}

function isSupportedUpload(file: File) {
  if (file.type.startsWith("text/") || file.type.startsWith("image/")) {
    return true;
  }
  return /\.(csv|gif|jpe?g|jsonl?|md|png|tsv|txt|webp)$/i.test(file.name);
}

function uploadedFilesPrompt(paths: string[]) {
  if (paths.length === 1) {
    return `I uploaded a file and it is available at ${paths[0]}.`;
  }
  return `I uploaded files and they are available at:\n${paths
    .map((path) => `- ${path}`)
    .join("\n")}`;
}

function NewChatApp({
  accessToken,
  userEmail,
  userIdentity,
  isAnonymous,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  userIdentity: string;
  isAnonymous: boolean;
  onSignOut: () => Promise<void>;
}) {
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [showTodos, setShowTodos] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const mainRef = useRef<HTMLDivElement | null>(null);
  const isNearBottom = useRef(true);
  const rafId = useRef(0);
  const sandboxRecordRef = useRef<SandboxRecord | null>(null);
  // When the user submits to a not-yet-created thread, stash the first
  // message's text here. The SDK assigns a thread_id asynchronously via
  // onThreadId — once we have it, we write the title as thread metadata so
  // the picker shows "What can you help..." instead of "b6cde91f...".
  const pendingTitleRef = useRef<string | null>(null);

  const defaultHeaders = useMemo(
    () => ({ Authorization: `Bearer ${accessToken}` }),
    [accessToken],
  );

  const client = useMemo(
    () =>
      new Client({
        apiUrl: window.location.origin,
        defaultHeaders,
      }),
    [defaultHeaders],
  );

  const handleThreadId = useCallback(
    (id: string | null) => {
      setThreadId(id);
      if (id == null) return;
      const metadata: Record<string, string> = {};
      if (pendingTitleRef.current) {
        metadata.title = pendingTitleRef.current;
        pendingTitleRef.current = null;
      }
      if (isAnonymous) {
        metadata.dap_anon_id = userIdentity;
      }
      if (Object.keys(metadata).length > 0) {
        client.threads.update(id, { metadata }).catch((err) => {
          console.warn("Failed to write thread metadata", err);
        });
      }
    },
    [client, isAnonymous, userIdentity],
  );

  const ensureThreadId = useCallback(async () => {
    if (threadId != null) return threadId;
    const thread = await client.threads.create();
    const id = thread.thread_id;
    handleThreadId(id);
    return id;
  }, [client, handleThreadId, threadId]);

  const rememberSandboxRecord = useCallback(
    async (id: string | null, record: SandboxRecord) => {
      sandboxRecordRef.current = record;
      try {
        if (SANDBOX_SCOPE === "assistant") {
          const assistant = await client.assistants.get(ASSISTANT_ID);
          await client.assistants.update(ASSISTANT_ID, {
            metadata: { ...(assistant.metadata ?? {}), deepagents_sandbox: record },
          });
        } else if (id) {
          const thread = await client.threads.get(id);
          await client.threads.update(id, {
            metadata: { ...(thread.metadata ?? {}), deepagents_sandbox: record },
          });
        }
      } catch (err) {
        console.warn("Failed to write sandbox metadata", err);
      }
    },
    [client],
  );

  const ensureSandbox = useCallback(
    async (id: string | null) => {
      if (sandboxRecordRef.current) return sandboxRecordRef.current;

      let existingRecord: SandboxRecord | null = null;
      try {
        const target =
          SANDBOX_SCOPE === "assistant"
            ? await client.assistants.get(ASSISTANT_ID)
            : id
              ? await client.threads.get(id)
              : null;
        const metadataRecord = target?.metadata?.deepagents_sandbox;
        if (isSandboxRecord(metadataRecord)) {
          existingRecord = metadataRecord;
          sandboxRecordRef.current = metadataRecord;
        }
      } catch (err) {
        console.warn("Failed to read sandbox metadata", err);
      }

      const params = new URLSearchParams({ assistant_id: ASSISTANT_ID });
      if (SANDBOX_SCOPE === "thread" && id) {
        params.set("thread_id", id);
      }
      if (existingRecord) {
        params.set("sandbox_id", existingRecord.sandbox_id);
        params.set("provider", existingRecord.provider);
        params.set("scope", existingRecord.scope);
        params.set("image", existingRecord.image);
      }

      const response = await fetch(
        `/sandboxes?${params.toString()}`,
        { method: "POST", headers: defaultHeaders },
      );
      const payload = (await response.json().catch(() => ({}))) as EnsureSandboxResponse;
      if (!response.ok || !payload.sandbox) {
        throw new Error(payload.error || "Failed to prepare sandbox");
      }

      await rememberSandboxRecord(id, payload.sandbox);
      return payload.sandbox;
    },
    [client, defaultHeaders, rememberSandboxRecord],
  );

  const stream = useAgentStream({
    apiUrl: window.location.origin,
    assistantId: ASSISTANT_ID,
    messagesKey: "messages",
    threadId,
    onThreadId: handleThreadId,
    filterSubagentMessages: true,
    // Coalesce token deltas so Streamdown's word-level fade animation has
    // time to play between updates. Dropping this to 0/false re-renders
    // per token (jumpy); pushing above ~150 makes the stream feel laggy.
    throttle: 100,
    defaultHeaders,
  });

  const { messages, isLoading, error, values } = stream;
  const files = values.files ?? {};
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

  const uploadFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList || fileList.length === 0 || uploadingFiles) return;

      const filesToUpload = Array.from(fileList);
      const unsupported = filesToUpload.find((file) => !isSupportedUpload(file));
      if (unsupported) {
        setUploadError(`Unsupported file type: ${unsupported.name}`);
        return;
      }

      setUploadingFiles(true);
      setUploadError(null);
      try {
        const id = SANDBOX_SCOPE === "thread" ? await ensureThreadId() : threadId;
        const record = await ensureSandbox(id);
        const uploadedPaths: string[] = [];

        for (const file of filesToUpload) {
          const params = new URLSearchParams({
            filename: file.name,
            assistant_id: ASSISTANT_ID,
            sandbox_id: record.sandbox_id,
            provider: record.provider,
            scope: record.scope,
            image: record.image,
          });
          if (id) {
            params.set("thread_id", id);
          }
          const response = await fetch(
            `/sandboxes/${encodeURIComponent(record.sandbox_id)}/files?${params.toString()}`,
            {
              method: "POST",
              headers: {
                ...defaultHeaders,
                "content-type": file.type || "application/octet-stream",
              },
              body: file,
            },
          );
          const payload = (await response.json().catch(() => ({}))) as UploadResponse;
          if (!response.ok || !payload.path) {
            throw new Error(payload.error || `Failed to upload ${file.name}`);
          }
          if (payload.sandbox) {
            await rememberSandboxRecord(id, payload.sandbox);
          }
          uploadedPaths.push(payload.path);
        }

        const prompt = uploadedFilesPrompt(uploadedPaths);
        setInput((current) => {
          const trimmed = current.trim();
          return trimmed ? `${trimmed}\n\n${prompt}` : prompt;
        });
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploadingFiles(false);
      }
    },
    [defaultHeaders, ensureSandbox, ensureThreadId, rememberSandboxRecord, threadId, uploadingFiles],
  );

  const submitInput = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading || uploadingFiles) return;

    // Capture the first-message text before we submit so the picker shows
    // the question instead of a UUID. The thread may already exist if the
    // user uploaded a file first (uploadFiles → ensureThreadId creates it
    // up-front so the upload has somewhere to land); in that case write
    // the title directly. Otherwise stash it for handleThreadId to write
    // once the SDK assigns the id on stream.submit.
    if (messages.length === 0) {
      const title = text.slice(0, 60);
      if (threadId == null) {
        pendingTitleRef.current = title;
      } else {
        client.threads.update(threadId, { metadata: { title } }).catch((err) => {
          console.warn("Failed to write thread title", err);
        });
      }
    }

    const record =
      UPLOADS_ENABLED
        ? sandboxRecordRef.current ??
          (SANDBOX_SCOPE === "assistant" || threadId != null
            ? await ensureSandbox(threadId)
            : null)
        : null;

    setInput("");
    // The wire protocol expects {type, content} dicts, not langchain-serialized
    // class instances; cast through BaseMessage so AgentState.messages typing
    // (BaseMessage[]) is satisfied without sending the lc/id/kwargs envelope.
    await stream.submit(
      {
        messages: [
          { type: "human", content: text } as unknown as BaseMessage,
        ],
      },
      {
        streamSubgraphs: true,
        config: record
          ? { configurable: { deepagents_sandbox: record } }
          : undefined,
      },
    );
  }, [client, ensureSandbox, input, isLoading, messages.length, stream, threadId, uploadingFiles]);

  const threadPicker = (
    <ThreadPicker
      currentThreadId={threadId}
      onSelect={setThreadId}
      accessToken={accessToken}
      userIdentity={userIdentity}
      isAnonymous={isAnonymous}
    />
  );

  return (
    <div className="flex h-dvh flex-col bg-[var(--background)]">
      <AppHeader
        userEmail={userEmail}
        isAnonymous={isAnonymous}
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
            {uploadError && (
              <div className="border-b border-[var(--border)] px-4 py-2 text-[11px] text-red-500">
                {uploadError}
              </div>
            )}
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
                {UPLOADS_ENABLED && (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept="text/*,image/*,.csv,.json,.jsonl,.md,.tsv,.txt"
                      className="hidden"
                      onChange={(event) => {
                        const files = event.currentTarget.files;
                        void uploadFiles(files);
                        event.currentTarget.value = "";
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploadingFiles || isLoading}
                      className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--muted)] disabled:opacity-50"
                    >
                      <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21.44 11.05 12.25 20.24a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                      </svg>
                      {uploadingFiles ? "Uploading..." : "Upload"}
                    </button>
                  </>
                )}
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
                disabled={!isLoading && (!input.trim() || uploadingFiles)}
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
