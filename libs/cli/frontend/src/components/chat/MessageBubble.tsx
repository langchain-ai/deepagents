import type { FC } from "react";
import type { BaseMessage } from "@langchain/core/messages";
import type { ToolMessage } from "@langchain/core/messages";
import type { ToolCall } from "@langchain/core/messages/tool";
import TextPart from "./TextPart";
import ToolCallPart from "./ToolCallPart";
import { getMessageType } from "../../lib/messages";

type Props = {
  message: BaseMessage;
  toolResultsByCallId: Map<string, ToolMessage>;
};

type ContentPart = { type: string; text?: string };

const MessageBubble: FC<Props> = ({ message, toolResultsByCallId }) => {
  const type = getMessageType(message);

  if (type === "tool") {
    return null;
  }

  const content = message.content;

  const textNodes: React.ReactNode[] = [];

  if (typeof content === "string") {
    if (content) {
      textNodes.push(<TextPart key="text" text={content} />);
    }
  } else if (Array.isArray(content)) {
    content.forEach((part, i) => {
      const p = part as ContentPart;
      if (p.type === "text" && p.text) {
        textNodes.push(<TextPart key={i} text={p.text} />);
      }
      // other part types deferred
    });
  }

  const toolCallNodes: React.ReactNode[] = [];

  if (type === "ai") {
    const toolCalls: ToolCall[] = (message as { tool_calls?: ToolCall[] }).tool_calls ?? [];
    toolCalls.forEach((tc) => {
      const toolResult = toolResultsByCallId.get(tc.id ?? "");
      const status: "running" | "success" | "error" =
        toolResult === undefined
          ? "running"
          : (toolResult as { status?: string }).status === "error" ||
              (toolResult as { additional_kwargs?: { error?: unknown } }).additional_kwargs?.error
            ? "error"
            : "success";
      toolCallNodes.push(
        <ToolCallPart key={tc.id} toolCall={tc} toolResult={toolResult} status={status} />,
      );
    });
  }

  if (type === "system") {
    return (
      <div className="anim-msg px-2 py-1 text-xs text-[var(--muted-foreground)]">
        {textNodes}
      </div>
    );
  }

  if (type === "human") {
    return (
      <div className="anim-msg flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[var(--muted)] px-4 py-2.5 text-sm leading-relaxed">
          {textNodes}
        </div>
      </div>
    );
  }

  // ai (and any unknown role) — left-aligned, no background
  return (
    <div className="anim-msg flex flex-col items-start gap-2">
      <div className="ai-bubble max-w-[90%] px-4 py-2.5 text-sm leading-relaxed">
        {textNodes}
      </div>
      {toolCallNodes.length > 0 && (
        <div className="flex flex-col gap-1.5">{toolCallNodes}</div>
      )}
    </div>
  );
};

export default MessageBubble;
