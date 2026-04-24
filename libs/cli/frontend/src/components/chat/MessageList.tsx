import { useEffect, useRef, type FC, type ReactNode } from "react";
import type { BaseMessage } from "@langchain/core/messages";
import MessageBubble from "./MessageBubble";

type Props = {
  messages: BaseMessage[];
  children?: (msg: BaseMessage) => ReactNode;
};

const BOTTOM_THRESHOLD = 50;

const MessageList: FC<Props> = ({ messages, children }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isAtBottomRef.current = distanceFromBottom <= BOTTOM_THRESHOLD;
  };

  useEffect(() => {
    if (!isAtBottomRef.current) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="flex min-h-0 flex-1 flex-col overflow-y-auto px-2 py-4 sm:px-4 sm:py-6"
    >
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-3">
        {messages.map((msg, i) => (
          <div key={(msg as any).id ?? i}>
            <MessageBubble message={msg} />
            {children?.(msg)}
          </div>
        ))}
      </div>
    </div>
  );
};

export default MessageList;
