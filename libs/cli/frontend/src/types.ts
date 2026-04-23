export type TodoStatus = "pending" | "in_progress" | "completed" | string;

export interface TodoItem {
  id?: string;
  content: string;
  status: TodoStatus;
}
