import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { useStream, type DeepAgentTypeConfigLike } from "@langchain/react";
import type { BaseMessage } from "@langchain/core/messages";
import { Client } from "@langchain/langgraph-sdk";
import { ASSISTANT_ID } from "./constants";
import type { TodoItem } from "./types";

export interface AgentState {
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
  accessToken: string | undefined;
  createClient: () => Client;
};

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

type ChatProviderProps = {
  accessToken?: string;
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

  const createClient = useCallback(
    () =>
      new Client({
        apiUrl: window.location.origin,
        defaultHeaders: accessToken
          ? { Authorization: `Bearer ${accessToken}` }
          : undefined,
      }),
    [accessToken],
  );

  return (
    <ChatContext.Provider
      value={{ stream, currentThreadId, switchToThread, newThread, accessToken, createClient }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatContextValue {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used inside ChatProvider");
  }
  return context;
}
