import type { UseStream } from "@langchain/react";
import type {
  DefaultToolCall,
  Message,
  Thread,
  ToolCallWithResult as SdkToolCallWithResult,
} from "@langchain/langgraph-sdk";
import type { SubagentStreamInterface } from "@langchain/langgraph-sdk/ui";

export type TodoStatus = "pending" | "in_progress" | "completed" | string;

export interface TodoItem {
  id?: string;
  content: string;
  status: TodoStatus;
}

export interface AgentState extends Record<string, unknown> {
  messages: Message<DefaultToolCall>[];
  files?: Record<string, unknown>;
  todos?: TodoItem[];
}

export type AgentStream = UseStream<AgentState>;
export type AgentSubagent = SubagentStreamInterface;
export type AgentToolCallResult = SdkToolCallWithResult<DefaultToolCall>;
export type ThreadSummary = Thread<AgentState>;
