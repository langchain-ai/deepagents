import type { FC } from "react";
import type { TodoItem } from "../types";

const TodosPanel: FC<{ todos: TodoItem[] }> = ({ todos }) => {
  if (todos.length === 0) return null;

  const inProgress = todos.filter((todo) => todo.status === "in_progress");
  const completed = todos.filter((todo) => todo.status === "completed");
  const pending = todos.filter((todo) => todo.status === "pending");

  return (
    <div className="max-h-48 overflow-y-auto px-4 py-2">
      <div className="space-y-1">
        {[...inProgress, ...pending, ...completed].map((todo, index) => (
          <div
            key={todo.id ?? `${todo.content}-${index}`}
            className="flex items-start gap-2 rounded-md px-2 py-1.5 text-xs"
          >
            {todo.status === "completed" ? (
              <span className="mt-0.5 text-emerald-600">
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </span>
            ) : todo.status === "in_progress" ? (
              <span className="mt-0.5 animate-spin text-blue-600">
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
                  : ""
              }
            >
              {todo.content}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TodosPanel;
