import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import {
  useLangGraphRuntime,
  type LangChainMessage,
} from "@assistant-ui/react-langgraph";
import { Client } from "@langchain/langgraph-sdk";
import {
  createThread,
  getCheckpointId,
  getThreadState,
  sendMessage,
  updateThreadMetadata,
  type ChatApiContext,
} from "./lib/chatApi";
import type { TodoItem } from "./types";

type GraphValues = {
  files: Record<string, unknown>;
  todos: TodoItem[];
};

const EMPTY_VALUES: GraphValues = { files: {}, todos: [] };
const GraphValuesContext = createContext<GraphValues>(EMPTY_VALUES);
export const useGraphValues = () => useContext(GraphValuesContext);

type ThreadActions = {
  currentExternalId: string | null;
  switchToExistingThread: (externalId: string) => void;
  newThread: () => void;
};

const ThreadActionsContext = createContext<ThreadActions>({
  currentExternalId: null,
  switchToExistingThread: () => {},
  newThread: () => {},
});
export const useThreadActions = () => useContext(ThreadActionsContext);

type LangGraphClientFactory = () => Client;

const LangGraphClientContext = createContext<LangGraphClientFactory>(() => {
  throw new Error("useLangGraphClient must be used inside RuntimeProvider");
});
export const useLangGraphClient = () => useContext(LangGraphClientContext);

type Props = {
  accessToken: string;
  assistantId: string;
  children: ReactNode;
};

export function RuntimeProvider({ accessToken, assistantId, children }: Props) {
  const ctx = useMemo<ChatApiContext>(
    () => ({
      apiUrl: window.location.origin,
      accessToken,
      assistantId,
    }),
    [accessToken, assistantId],
  );

  const [values, setValues] = useState<GraphValues>(EMPTY_VALUES);
  const [currentExternalId, setCurrentExternalId] = useState<string | null>(null);

  const untitledThreadsRef = useRef<Set<string>>(new Set());

  const runtime = useLangGraphRuntime({
    unstable_allowCancellation: true,
    stream: async function* (messages, { initialize, ...config }) {
      const { externalId } = await initialize();
      if (!externalId) throw new Error("Thread not found");

      if (untitledThreadsRef.current.has(externalId)) {
        untitledThreadsRef.current.delete(externalId);
        // assistant-ui-langgraph may pass either LangChain (`type: "human"`)
        // or OpenAI-style (`role: "user"`) message shapes.
        const firstUser = messages.find((m) => {
          const mm = m as { type?: string; role?: string };
          return mm.type === "human" || mm.role === "user";
        });
        const content = firstUser?.content;
        let raw = "";
        if (typeof content === "string") {
          raw = content;
        } else if (Array.isArray(content)) {
          raw = content
            .map((p) => {
              if (typeof p === "string") return p;
              const pp = p as { text?: string; type?: string };
              return pp.type === "text" && typeof pp.text === "string"
                ? pp.text
                : "";
            })
            .join(" ");
        }
        const title = raw.trim().slice(0, 60);
        if (title) {
          updateThreadMetadata(ctx, externalId, { title }).catch((err) => {
            console.warn("Failed to write thread title", err);
          });
        }
      }

      yield* sendMessage(ctx, {
        threadId: externalId,
        messages,
        config,
      });
    },
    create: async () => {
      const { thread_id } = await createThread(ctx);
      untitledThreadsRef.current.add(thread_id);
      setCurrentExternalId(thread_id);
      return { externalId: thread_id };
    },
    load: async (externalId) => {
      setCurrentExternalId(externalId);
      const state = await getThreadState(ctx, externalId);
      const stateValues = state.values as {
        messages?: LangChainMessage[];
        files?: Record<string, unknown>;
        todos?: TodoItem[];
      };
      setValues({
        files: stateValues.files ?? {},
        todos: stateValues.todos ?? [],
      });
      return {
        messages: stateValues.messages ?? [],
        interrupts: state.tasks[0]?.interrupts ?? [],
      };
    },
    getCheckpointId: (threadId, parentMessages) =>
      getCheckpointId(ctx, threadId, parentMessages),
    eventHandlers: {
      onValues: (next) => {
        const v = next as { files?: Record<string, unknown>; todos?: TodoItem[] };
        setValues({ files: v.files ?? {}, todos: v.todos ?? [] });
      },
    },
  });

  // Patch the default InMemoryThreadListAdapter (which rejects `fetch`) so
  // `switchToThread(externalId)` can register threads we discovered via our
  // own picker. The library recreates the adapter on every render, so we
  // re-patch after every commit.
  useEffect(() => {
    const core = (runtime.threads as unknown as { _core?: unknown })._core as
      | { _options?: { adapter?: Record<string, unknown> } }
      | undefined;
    const adapter = core?._options?.adapter;
    if (!adapter) return;
    adapter.fetch = async (threadId: string) => ({
      status: "regular" as const,
      remoteId: threadId,
      externalId: threadId,
      title: undefined,
    });
  });

  const switchToExistingThread = useCallback(
    (externalId: string) => {
      setCurrentExternalId(externalId);
      setValues(EMPTY_VALUES);
      runtime.threads.switchToThread(externalId).catch((err) => {
        console.warn("Failed to switch thread", err);
      });
    },
    [runtime],
  );

  const newThread = useCallback(() => {
    setCurrentExternalId(null);
    setValues(EMPTY_VALUES);
    runtime.threads.switchToNewThread().catch((err) => {
      console.warn("Failed to start new thread", err);
    });
  }, [runtime]);

  const threadActions = useMemo<ThreadActions>(
    () => ({ currentExternalId, switchToExistingThread, newThread }),
    [currentExternalId, switchToExistingThread, newThread],
  );

  const clientFactory = useCallback<LangGraphClientFactory>(
    () =>
      new Client({
        apiUrl: ctx.apiUrl,
        defaultHeaders: ctx.accessToken
          ? { Authorization: `Bearer ${ctx.accessToken}` }
          : undefined,
      }),
    [ctx],
  );

  // Changing assistant mid-conversation starts a fresh thread — old history
  // belongs to a different graph.
  const prevAssistantIdRef = useRef(assistantId);
  useEffect(() => {
    if (prevAssistantIdRef.current !== assistantId) {
      prevAssistantIdRef.current = assistantId;
      newThread();
    }
  }, [assistantId, newThread]);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <GraphValuesContext.Provider value={values}>
        <ThreadActionsContext.Provider value={threadActions}>
          <LangGraphClientContext.Provider value={clientFactory}>
            {children}
          </LangGraphClientContext.Provider>
        </ThreadActionsContext.Provider>
      </GraphValuesContext.Provider>
    </AssistantRuntimeProvider>
  );
}

export default RuntimeProvider;
