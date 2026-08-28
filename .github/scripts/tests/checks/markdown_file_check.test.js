"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  addedMarkdownFiles,
  fileLines,
  isDocsTitle,
} = require("../../checks/markdown_file_check.js");

test("recognizes only docs Conventional Commit titles", () => {
  assert.equal(isDocsTitle("docs(sdk): add a guide"), true);
  assert.equal(isDocsTitle("docs!: replace the guide"), true);
  assert.equal(isDocsTitle("fix(docs): repair rendering"), false);
  assert.equal(isDocsTitle("documentation: add a guide"), false);
});

test("returns only newly added Markdown files", () => {
  const files = [
    { filename: "README.md", status: "added" },
    { filename: "guides/UPPER.MD", status: "added" },
    { filename: "old.md", status: "modified" },
    { filename: "removed.md", status: "removed" },
    { filename: "notes.mdx", status: "added" },
  ];

  assert.deepEqual(addedMarkdownFiles(files), ["README.md", "guides/UPPER.MD"]);
});

test("escapes and bounds file paths rendered in the sticky comment", () => {
  assert.deepEqual(fileLines(["<script>&.md", "second.md"], 1), [
    "- <code>&lt;script&gt;&amp;.md</code>",
    "- _and 1 more_",
  ]);
});
