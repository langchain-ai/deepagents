"use strict";

function isDocsTitle(title) {
  return /^docs(?:\([^)]*\))?!?:\s/.test(title || "");
}

function addedMarkdownFiles(files) {
  return files
    .filter(
      (file) =>
        file.status === "added" && file.filename.toLowerCase().endsWith(".md"),
    )
    .map((file) => file.filename);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function fileLines(files, limit = 50) {
  const lines = files
    .slice(0, limit)
    .map((file) => `- <code>${escapeHtml(file)}</code>`);
  if (files.length > limit) {
    lines.push(`- _and ${files.length - limit} more_`);
  }
  return lines;
}

module.exports = { addedMarkdownFiles, fileLines, isDocsTitle };
