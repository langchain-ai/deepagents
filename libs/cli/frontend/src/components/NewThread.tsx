import { useChat } from "../ChatProvider";
import MessageList from "./chat/MessageList";
import Composer from "./chat/Composer";

export default function NewThread() {
  const { stream } = useChat();

  const handleSubmit = (text: string) => {
    void stream.submit(
      { messages: [{ type: "human", content: text }] },
      { streamSubgraphs: true },
    );
  };

  return (
    <div className="flex h-full flex-col bg-[var(--background)]">
      <div className="flex-1 min-h-0">
        <MessageList messages={stream.messages} />
      </div>
      <div className="border-t border-[var(--border)] p-3">
        <Composer
          onSubmit={handleSubmit}
          onStop={() => stream.stop()}
          isLoading={stream.isLoading}
        />
      </div>
    </div>
  );
}
