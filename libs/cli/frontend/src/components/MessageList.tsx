import type { FC, RefObject } from "react";
import { Streamdown } from "streamdown";
import { APP_NAME } from "../constants";
import {
  getErrorMessage,
  getImageBlocks,
} from "../lib/stream";
import type { AgentStream } from "../types";
import type { AIMessage } from "@langchain/langgraph-sdk";
import { SubagentPipeline, SynthesisIndicator } from "./SubagentActivity";
import ToolCallCard from "./ToolCallCard";

type MessageListProps = {
  bottomRef: RefObject<HTMLDivElement | null>;
  error: unknown;
  isLoading: boolean;
  mainRef: RefObject<HTMLDivElement | null>;
  messages: AgentStream["messages"];
  onScroll: () => void;
  onSuggestionSelect: (prompt: string) => void;
  stream: AgentStream;
};

const EMPTY_STATE_SUGGESTIONS = [
  "What can you help me research?",
  "Research a topic for me",
  "Help me write a report",
];

const MessageList: FC<MessageListProps> = ({
  bottomRef,
  error,
  isLoading,
  mainRef,
  messages,
  onScroll,
  onSuggestionSelect,
  stream,
}) => {
  const allSubagents = Array.from(stream.subagents.values());

  return (
    <main
      ref={mainRef}
      onScroll={onScroll}
      className="flex-1 overflow-y-auto px-2 py-4 sm:px-4 sm:py-6"
    >
      <div className="mx-auto max-w-4xl space-y-3">
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <svg className="anim-stagger-1 mb-4 h-12 w-12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
              <path d="M12 2L2 12l10 10 10-10L12 2z" />
              <path d="M12 8L8 12l4 4 4-4-4-4z" />
            </svg>
            <h2 className="anim-stagger-2 text-2xl font-semibold lc-gradient-text">
              {APP_NAME}
            </h2>
            <p className="anim-stagger-2 mt-2 text-[var(--muted-foreground)]">
              How can I help with your research today?
            </p>
            <div className="anim-stagger-3 mt-6 flex flex-wrap justify-center gap-2">
              {EMPTY_STATE_SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => onSuggestionSelect(suggestion)}
                  className="rounded-full border border-[var(--border)] bg-[var(--surface)] px-4 py-2 text-xs font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)] hover:text-[var(--foreground)]"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => {
          if (message.type === "tool") return null;

          const content = message.text;
          const isLastMessage = index === messages.length - 1;

          if (message.getType() === "human") {
            return (
              <div key={message.id ?? index} className="anim-msg flex justify-end">
                <div className="max-w-[90%] rounded-2xl rounded-br-sm bg-[var(--primary)] px-4 py-2.5 text-sm leading-relaxed text-[var(--primary-foreground)]">
                  {content}
                </div>
              </div>
            );
          }

          if (message.getType() === "ai") {
            const toolCalls = stream.getToolCalls(message as AIMessage);
            const messageSubagents = message.id
              ? stream.getSubagentsByMessage(message.id)
              : [];
            const subagentIds = new Set(messageSubagents.map((subagent) => subagent.id));
            const nonSubagentToolCalls = toolCalls.filter(
              (toolCall) => !subagentIds.has(toolCall.id),
            );
            const imageBlocks = getImageBlocks(message.content);
            const hasSubagents = messageSubagents.length > 0;
            const isStreaming = isLastMessage && isLoading;

            return (
              <div key={message.id ?? index} className="anim-msg flex flex-col items-start gap-2">
                {content ? (
                  <div className="ai-bubble max-w-[90%] rounded-2xl rounded-bl-sm border border-[var(--border)] bg-[var(--surface)] px-4 py-2.5 text-sm leading-relaxed shadow-sm">
                    <div className="markdown-body">
                      <Streamdown
                        animated={{ animation: "fadeIn", sep: "word", duration: 300 }}
                        isAnimating={isStreaming}
                        parseIncompleteMarkdown
                      >
                        {content}
                      </Streamdown>
                    </div>
                  </div>
                ) : !toolCalls.length && !hasSubagents ? (
                  <div className="rounded-2xl rounded-bl-sm border border-[var(--border)] bg-[var(--surface)] px-4 py-2.5 text-sm text-[var(--muted-foreground)] shadow-sm">
                    <span className="inline-flex items-center gap-1">
                      <span className="animate-pulse">●</span>
                      <span className="animate-pulse" style={{ animationDelay: "150ms" }}>
                        ●
                      </span>
                      <span className="animate-pulse" style={{ animationDelay: "300ms" }}>
                        ●
                      </span>
                    </span>
                  </div>
                ) : null}

                {imageBlocks.map((image, imageIndex) => (
                  <div
                    key={`${message.id ?? index}-image-${imageIndex}`}
                    className="max-w-[90%] overflow-hidden rounded-xl border border-[var(--border)]"
                  >
                    <img
                      src={image.url}
                      alt="Generated content"
                      className="max-h-96 w-auto object-contain"
                    />
                  </div>
                ))}

                <SubagentPipeline subagents={messageSubagents} />

                {nonSubagentToolCalls.map((toolCall) => (
                  <ToolCallCard key={toolCall.id} toolCall={toolCall} />
                ))}
              </div>
            );
          }

          if (!content) return null;

          return (
            <div key={message.id ?? index} className="flex justify-start">
              <div className="max-w-[90%] rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--muted-foreground)]">
                <span className="font-mono">[{message.type}]</span>{" "}
                {content.slice(0, 300)}
              </div>
            </div>
          );
        })}

        <SynthesisIndicator subagents={allSubagents} isLoading={isLoading} />

        {error != null && (
          <div className="anim-fade-in rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            Error: {getErrorMessage(error)}
          </div>
        )}

        <div ref={bottomRef} className="scroll-anchor" />
      </div>
    </main>
  );
};

export default MessageList;
