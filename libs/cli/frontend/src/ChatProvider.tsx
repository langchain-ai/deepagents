import {
  createContext,
  useContext,
  useState,
  useMemo,
  useCallback,
  type ReactNode,
} from "react";
import { useStream, type DeepAgentTypeConfigLike } from "@langchain/react";
import type { BaseMessage } from "@langchain/core/messages";
import { ASSISTANT_ID } from "./constants";
import type { TodoItem } from "./types";

interface AgentState {
  messages: BaseMessage[];
  files?: Record<string, unknown>;
  todos?: TodoItem[];
}

// Branded state for useStream's deep-agent overload. The phantom
// property exists only in the type system — never present at runtime.
type DeepAgentState = AgentState & {
  "~deepAgentTypes": DeepAgentTypeConfigLike;
};

type ChatContextValue = {
  stream: ReturnType<typeof useStream<DeepAgentState>>;
  currentThreadId: string | null;
  switchToThread: (id: string) => void;
  newThread: () => void;
};

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

type ChatProviderProps = {
  accessToken: string;
  children: ReactNode;
};

export function ChatProvider({ accessToken, children }: ChatProviderProps) {
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);

  const stream = useStream<DeepAgentState>({
    apiUrl: window.location.origin,
    assistantId: ASSISTANT_ID,
    threadId: currentThreadId,
    onThreadId: setCurrentThreadId,
    filterSubagentMessages: true,
    defaultHeaders: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
  });

  const switchToThread = useCallback((id: string) => {
    setCurrentThreadId(id);
  }, []);

  const newThread = useCallback(() => {
    setCurrentThreadId(null);
  }, []);

  const value = useMemo<ChatContextValue>(
    () => ({
      stream,
      currentThreadId,
      switchToThread,
      newThread,
    }),
    [stream, currentThreadId, switchToThread, newThread]
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used inside ChatProvider");
  }
  return context;
}
