import type { FC } from "react";
import type { ToolRendererProps } from "./index";

type TodoArg = { content?: string; status?: string; id?: string };

function parseTodos(args: unknown): TodoArg[] {
  if (typeof args === "string") {
    try { args = JSON.parse(args); } catch { return []; }
  }
  if (!args || typeof args !== "object") return [];
  const record = args as Record<string, unknown>;
  const todos = record.todos;
  if (Array.isArray(todos)) return todos as TodoArg[];
  return [];
}

const TodosCard: FC<ToolRendererProps> = ({ toolCall, expanded }) => {
  const todos = parseTodos(toolCall.call.args);

  if (!expanded || todos.length === 0) return null;

  return (
    <div className="space-y-1 border-t border-[var(--border)] px-3 py-2">
      {todos.map((todo, index) => (
        <div
          key={todo.id ?? `${todo.content}-${index}`}
          className="flex items-start gap-2 rounded-md px-1 py-1 text-xs"
        >
          {todo.status === "completed" ? (
            <span className="mt-0.5 text-emerald-600">
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </span>
          ) : todo.status === "in_progress" ? (
            <span className="mt-0.5 text-blue-600">
              <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            </span>
          ) : (
            <span className="mt-0.5 text-[var(--muted-foreground)]">
              <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
              </svg>
            </span>
          )}
          <span
            className={
              todo.status === "completed"
                ? "line-through text-[var(--muted-foreground)]"
                : "text-[var(--foreground)]"
            }
          >
            {todo.content ?? "Untitled task"}
          </span>
        </div>
      ))}
    </div>
  );
};

export default TodosCard;
