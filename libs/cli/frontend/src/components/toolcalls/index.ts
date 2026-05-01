import type { FC } from "react";
import type { AgentToolCallResult } from "../../types";
import FileToolCard from "./FileToolCard";
import TodosCard from "./TodosCard";
import SearchCard from "./SearchCard";
import ThinkCard from "./ThinkCard";

export type ToolRendererProps = {
  toolCall: AgentToolCallResult;
  expanded: boolean;
};

const TOOL_RENDERERS: Record<string, FC<ToolRendererProps>> = {
  write_file: FileToolCard,
  edit_file: FileToolCard,
  read_file: FileToolCard,
  write_todos: TodosCard,
  ls: SearchCard,
  glob: SearchCard,
  grep: SearchCard,
  think: ThinkCard,
  think_tool: ThinkCard,
};

export function getToolRenderer(toolName: string): FC<ToolRendererProps> | null {
  return TOOL_RENDERERS[toolName] ?? null;
}

export function getToolIcon(toolName: string): { icon: string; label: string } | null {
  switch (toolName) {
    case "write_file":
      return { icon: "file-plus", label: "Write" };
    case "edit_file":
      return { icon: "file-edit", label: "Edit" };
    case "read_file":
      return { icon: "file-text", label: "Read" };
    case "write_todos":
      return { icon: "list-checks", label: "Todos" };
    case "ls":
      return { icon: "folder", label: "List" };
    case "glob":
      return { icon: "search", label: "Glob" };
    case "grep":
      return { icon: "search", label: "Search" };
    case "think":
    case "think_tool":
      return { icon: "brain", label: "Think" };
    default:
      return null;
  }
}
