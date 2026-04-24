import { useEffect, useRef, useMemo, type FC, type ReactNode } from "react";
import type { BaseMessage } from "@langchain/core/messages";
import type { ToolMessage } from "@langchain/core/messages";
import MessageBubble from "./MessageBubble";
import { getMessageType } from "../../lib/messages";

type Props = {
  messages: BaseMessage[];
  /** Optional render-prop appended after each message — used by later tasks to inject subagent pipelines. */
  children?: (msg: BaseMessage) => ReactNode;
};

const BOTTOM_THRESHOLD = 50;

const MessageList: FC<Props> = ({ messages, children }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isAtBottomRef.current = distanceFromBottom <= BOTTOM_THRESHOLD;
  };

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      if (isAtBottomRef.current && scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const toolResultsByCallId = useMemo(() => {
    const map = new Map<string, ToolMessage>();
    for (const msg of messages) {
      const mType = getMessageType(msg);
      if (mType === "tool") {
        const toolMsg = msg as ToolMessage;
        const id = (toolMsg as { tool_call_id?: string }).tool_call_id;
        if (typeof id === "string") map.set(id, toolMsg);
      }
    }
    return map;
  }, [messages]);

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-4 sm:px-4 sm:py-6"
    >
      <div ref={contentRef} className="mx-auto flex w-full max-w-4xl flex-col gap-3">
        {messages.map((msg, i) => (
          <div key={(msg as { id?: string }).id ?? i}>
            <MessageBubble message={msg} toolResultsByCallId={toolResultsByCallId} />
            {children?.(msg)}
          </div>
        ))}
      </div>
    </div>
  );
};

export default MessageList;
