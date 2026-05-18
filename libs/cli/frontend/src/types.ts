import type { UseStream, DefaultToolCall, SubagentStreamInterface } from "@langchain/react";
import type { BaseMessage } from "@langchain/core/messages";
import type { Thread, ToolCallWithResult as SdkToolCallWithResult } from "@langchain/langgraph-sdk";

export type TodoStatus = "pending" | "in_progress" | "completed" | string;

export interface TodoItem {
  id?: string;
  content: string;
  status: TodoStatus;
}

export interface AgentState extends Record<string, unknown> {
  messages: BaseMessage[];
  files?: Record<string, unknown>;
  todos?: TodoItem[];
}

export type AgentStream = UseStream<AgentState>;
export type AgentSubagent = SubagentStreamInterface;
export type AgentToolCallResult = SdkToolCallWithResult<DefaultToolCall>;
export type ThreadSummary = Thread<AgentState>;
