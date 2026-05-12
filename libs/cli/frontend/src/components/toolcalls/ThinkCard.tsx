import { useState, type FC } from "react";
import type { ToolRendererProps } from "./index";

function parseThought(args: unknown): string {
  if (typeof args === "string") {
    try {
      const parsed = JSON.parse(args);
      return parseThought(parsed);
    } catch {
      return args;
    }
  }
  if (args && typeof args === "object") {
    const record = args as Record<string, unknown>;
    const thought = record.thought ?? record.content ?? record.reflection ?? record.text;
    if (typeof thought === "string") return thought;
    const firstString = Object.values(record).find(
      (v): v is string => typeof v === "string" && v.length > 0,
    );
    return firstString ?? JSON.stringify(args, null, 2);
  }
  return String(args ?? "");
}

const PREVIEW_LINES = 3;

const ThinkCard: FC<ToolRendererProps> = ({ toolCall, expanded }) => {
  const thought = parseThought(toolCall.call.args);
  const lines = thought.split("\n");
  const [showAll, setShowAll] = useState(false);
  const needsTruncation = lines.length > PREVIEW_LINES;
  const preview = lines.slice(0, PREVIEW_LINES).join("\n");

  // Always show a preview even when collapsed
  return (
    <div className="border-t border-[var(--border)]">
      <div className="tool-think-block mx-3 my-2">
        <div className="whitespace-pre-wrap text-xs leading-relaxed">
          {expanded || showAll ? thought : preview}
          {!expanded && needsTruncation && !showAll && (
            <>
              {"..."}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowAll(true);
                }}
                className="ml-1 text-violet-600 hover:underline"
              >
                more
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ThinkCard;
