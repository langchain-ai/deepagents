import type { BaseMessage } from "@langchain/core/messages";

export function getMessageType(message: BaseMessage): string | undefined {
  const m = message as { getType?: () => string; type?: string };
  if (typeof m.getType === "function") return m.getType();
  return m.type;
}
