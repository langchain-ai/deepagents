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
// Returns a factory; consumers call it once (e.g. in a `useMemo`) to get a
// stable `Client` instance configured with the current auth headers.
export const useLangGraphClient = () => useContext(LangGraphClientContext);

type Props = {
  accessToken: string;
  assistantId: string;
  children: ReactNode;
};

export function RuntimeProvider({ accessToken, assistantId, children }: Props) {
  const ctxRef = useRef<ChatApiContext>({
    apiUrl: window.location.origin,
    accessToken,
    assistantId,
  });
  ctxRef.current = {
    apiUrl: window.location.origin,
    accessToken,
    assistantId,
  };

  const [values, setValues] = useState<GraphValues>(EMPTY_VALUES);
  const [currentExternalId, setCurrentExternalId] = useState<string | null>(null);

  // When set, the next `create()` returns this externalId so the runtime picks
  // up an existing LangGraph thread. UI history hydrates on the next stream.
  const pendingExternalIdRef = useRef<string | null>(null);

  const runtime = useLangGraphRuntime({
    unstable_allowCancellation: true,
    stream: async function* (messages, { initialize, ...config }) {
      const { externalId } = await initialize();
      if (!externalId) throw new Error("Thread not found");
      yield* sendMessage(ctxRef.current, {
        threadId: externalId,
        messages,
        config,
      });
    },
    create: async () => {
      const pending = pendingExternalIdRef.current;
      if (pending) {
        pendingExternalIdRef.current = null;
        setCurrentExternalId(pending);
        return { externalId: pending };
      }
      const { thread_id } = await createThread(ctxRef.current);
      setCurrentExternalId(thread_id);
      return { externalId: thread_id };
    },
    load: async (externalId) => {
      const state = await getThreadState(ctxRef.current, externalId);
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
      getCheckpointId(ctxRef.current, threadId, parentMessages),
    eventHandlers: {
      onValues: (next) => {
        const v = next as { files?: Record<string, unknown>; todos?: TodoItem[] };
        setValues({ files: v.files ?? {}, todos: v.todos ?? [] });
      },
    },
  });

  const switchToExistingThread = useCallback(
    (externalId: string) => {
      pendingExternalIdRef.current = externalId;
      setCurrentExternalId(externalId);
      setValues(EMPTY_VALUES);
      void runtime.threads.switchToNewThread();
    },
    [runtime],
  );

  const newThread = useCallback(() => {
    pendingExternalIdRef.current = null;
    setCurrentExternalId(null);
    setValues(EMPTY_VALUES);
    void runtime.threads.switchToNewThread();
  }, [runtime]);

  const threadActions = useMemo<ThreadActions>(
    () => ({ currentExternalId, switchToExistingThread, newThread }),
    [currentExternalId, switchToExistingThread, newThread],
  );

  const clientFactory = useCallback<LangGraphClientFactory>(
    () =>
      new Client({
        apiUrl: ctxRef.current.apiUrl,
        defaultHeaders: ctxRef.current.accessToken
          ? { Authorization: `Bearer ${ctxRef.current.accessToken}` }
          : undefined,
      }),
    [],
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
