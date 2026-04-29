import { useCallback, useEffect, useMemo, useState, type FC } from "react";
import { createPortal } from "react-dom";
import { Streamdown } from "streamdown";
import { extOf, parseDisplayContent } from "../lib/stream";

const FileViewDialog: FC<{
  fileName: string;
  content: unknown;
  onClose: () => void;
}> = ({ fileName, content: rawContent, onClose }) => {
  const content = useMemo(() => parseDisplayContent(rawContent), [rawContent]);
  const extension = extOf(fileName);
  const isMarkdown = ["md", "markdown"].includes(extension);
  const isHtml = ["html", "htm"].includes(extension);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content).catch((err) => {
      console.warn("Clipboard write failed", err);
    });
  }, [content]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = fileName.split("/").pop() ?? "file";
    link.click();
    URL.revokeObjectURL(url);
  }, [content, fileName]);

  return createPortal(
    <div
      className="dialog-backdrop fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="dialog-content flex max-h-[80vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl bg-[var(--surface)] shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <svg className="h-4 w-4 shrink-0 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
              <path d="M14 2v4a2 2 0 0 0 2 2h4" />
            </svg>
            <span className="truncate text-sm font-medium">{fileName}</span>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <button onClick={handleCopy} className="rounded-md p-1.5 text-[var(--muted-foreground)] hover:bg-[var(--muted)]" title="Copy">
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect width="14" height="14" x="8" y="8" rx="2" />
                <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
              </svg>
            </button>
            <button onClick={handleDownload} className="rounded-md p-1.5 text-[var(--muted-foreground)] hover:bg-[var(--muted)]" title="Download">
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" x2="12" y1="15" y2="3" />
              </svg>
            </button>
            <button onClick={onClose} className="rounded-md p-1.5 text-[var(--muted-foreground)] hover:bg-[var(--muted)]" title="Close">
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-5">
          {isHtml && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              HTML preview is disabled in the open-source build. Inspect the escaped
              content below or download the file locally.
            </div>
          )}
          {content ? (
            isMarkdown && !isHtml ? (
              <div className="markdown-body text-sm">
                <Streamdown mode="static">{content}</Streamdown>
              </div>
            ) : (
              <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-[var(--foreground)]">
                {content}
              </pre>
            )
          ) : (
            <p className="text-center text-sm text-[var(--muted-foreground)]">
              File is empty
            </p>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
};

const FilesPanel: FC<{ files: Record<string, unknown> }> = ({ files }) => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const fileNames = Object.keys(files);

  if (fileNames.length === 0) return null;

  return (
    <>
      <div className="max-h-48 overflow-y-auto px-4 py-2">
        <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2">
          {fileNames.map((name) => (
            <button
              key={name}
              onClick={() => setSelectedFile(name)}
              className="flex flex-col items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-3 text-center transition-colors hover:bg-[var(--muted)]"
            >
              <svg className="h-6 w-6 text-[var(--muted-foreground)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
                <path d="M14 2v4a2 2 0 0 0 2 2h4" />
              </svg>
              <span className="w-full truncate text-xs font-medium">
                {name.split("/").pop()}
              </span>
            </button>
          ))}
        </div>
      </div>

      {selectedFile && (
        <FileViewDialog
          fileName={selectedFile}
          content={files[selectedFile] ?? ""}
          onClose={() => setSelectedFile(null)}
        />
      )}
    </>
  );
};

export default FilesPanel;
