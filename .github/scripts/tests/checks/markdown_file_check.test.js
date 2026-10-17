"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  ACKNOWLEDGMENT_LABEL,
  FILE_LIST_LIMIT,
  RELEASE_PLEASE_BRANCH_PREFIX,
  STICKY_MARKER,
  WORKFLOW_BOT_LOGIN,
  addedMarkdownFiles,
  buildBody,
  fileLines,
  isDocsTitle,
  isReleasePleasePr,
  run,
} = require("../../checks/markdown_file_check.js");

// --- Fakes ------------------------------------------------------------------

function makeCore() {
  return {
    failed: null,
    infos: [],
    notices: [],
    warnings: [],
    errors: [],
    summaryWrites: [],
    summaryError: null,
    info(message) {
      this.infos.push(message);
    },
    notice(message) {
      this.notices.push(message);
    },
    warning(message) {
      this.warnings.push(message);
    },
    error(message) {
      this.errors.push(message);
    },
    setFailed(message) {
      this.failed = message;
    },
    summary: {
      pending: [],
      addHeading(text) {
        this.pending.push(text);
        return this;
      },
      addRaw(text) {
        this.pending.push(text);
        return this;
      },
      async write() {
        const owner = this.owner;
        if (owner.summaryError) {
          throw owner.summaryError;
        }
        owner.summaryWrites.push(this.pending.join("\n"));
        this.pending = [];
      },
    },
  };
}

function newCore() {
  const core = makeCore();
  core.summary.owner = core;
  return core;
}

function httpError(message, status) {
  const error = new Error(message);
  error.status = status;
  return error;
}

const DEFAULT_PR = {
  number: 7,
  title: "feat(code): add a thing",
  changed_files: 1,
  user: { login: "contributor", type: "User" },
  head: {
    ref: "feature-branch",
    repo: { full_name: "langchain-ai/deepagents" },
  },
};

function makeGithub({
  // What `pulls.get` returns. Every gate input is read from here rather than
  // the event payload, so tests set the PR's real state here and use
  // makeContext only to model what the (possibly stale) event carried.
  pr = {},
  files = [],
  labels = [],
  comments = [],
  pullsGetError = null,
  listFilesError = null,
  listLabelsError = null,
  listCommentsError = null,
  createCommentError = null,
  updateCommentError = null,
  deleteCommentError = null,
} = {}) {
  const livePr = { ...DEFAULT_PR, ...pr };
  const calls = {
    createComment: [],
    updateComment: [],
    deleteComment: [],
    pullsGet: [],
    listFiles: [],
    listLabels: [],
  };
  const github = {
    rest: {
      issues: {
        listComments: "listComments",
        listLabelsOnIssue: async (params) => {
          calls.listLabels.push(params);
          if (listLabelsError) throw listLabelsError;
          return { data: labels };
        },
        createComment: async (params) => {
          calls.createComment.push(params);
          if (createCommentError) throw createCommentError;
        },
        updateComment: async (params) => {
          calls.updateComment.push(params);
          if (updateCommentError) throw updateCommentError;
        },
        deleteComment: async (params) => {
          calls.deleteComment.push(params);
          if (deleteCommentError) throw deleteCommentError;
        },
      },
      pulls: {
        listFiles: "listFiles",
        get: async (params) => {
          calls.pullsGet.push(params);
          if (pullsGetError) throw pullsGetError;
          return { data: livePr };
        },
      },
    },
    async paginate(route, params) {
      if (route === "listFiles") {
        calls.listFiles.push(params);
        if (listFilesError) throw listFilesError;
        return files;
      }
      if (route === "listComments") {
        if (listCommentsError) throw listCommentsError;
        return comments;
      }
      throw new Error(`unexpected paginate route: ${route}`);
    },
  };
  return { github, calls };
}

// The event payload contributes only the PR number; everything the gate
// decides on is re-read live. `stale` models what an out-of-order event still
// claims, so a test can prove the payload's version is never consulted.
function makeContext({ number = 7, stale = {} } = {}) {
  return {
    repo: { owner: "langchain-ai", repo: "deepagents" },
    payload: {
      pull_request: { number, ...stale },
    },
  };
}

function botComment({ id = 1, body = `${STICKY_MARKER}\nold` } = {}) {
  return { id, body, user: { login: WORKFLOW_BOT_LOGIN } };
}

// --- Pure helpers -----------------------------------------------------------

test("recognizes only docs Conventional Commit titles", () => {
  assert.equal(isDocsTitle("docs(sdk): add a guide"), true);
  assert.equal(isDocsTitle("docs!: replace the guide"), true);
  assert.equal(isDocsTitle("docs: add a guide"), true);
  assert.equal(isDocsTitle("fix(docs): repair rendering"), false);
  assert.equal(isDocsTitle("documentation: add a guide"), false);
  // Lowercase-only, matching _TITLE_RE in check_pr_scope_files.py.
  assert.equal(isDocsTitle("Docs: add a guide"), false);
  assert.equal(isDocsTitle("DOCS: add a guide"), false);
  // The space after the colon is required by the Conventional Commit grammar.
  assert.equal(isDocsTitle("docs:add a guide"), false);
  assert.equal(isDocsTitle(undefined), false);
  assert.equal(isDocsTitle(""), false);
});

test("returns only newly added Markdown files", () => {
  const files = [
    { filename: "README.md", status: "added" },
    { filename: "guides/UPPER.MD", status: "added" },
    { filename: "old.md", status: "modified" },
    { filename: "removed.md", status: "removed" },
    { filename: "notes.mdx", status: "added" },
    { filename: "unchanged.md", status: "unchanged" },
  ];

  assert.deepEqual(addedMarkdownFiles(files), ["README.md", "guides/UPPER.MD"]);
});

test("treats a rename or copy into Markdown as a new Markdown file", () => {
  const files = [
    // A new Markdown document that did not exist before: must be caught.
    { filename: "notes.md", status: "renamed", previous_filename: "notes.txt" },
    { filename: "copied.md", status: "copied", previous_filename: "source.txt" },
    // Moving an existing .md adds no new document to review: must be ignored.
    { filename: "docs/moved.md", status: "renamed", previous_filename: "moved.md" },
    // Case-only rename: the source was already Markdown, so nothing is new.
    { filename: "docs/CASE.md", status: "renamed", previous_filename: "CASE.MD" },
    // A rename whose destination is not Markdown is irrelevant either way.
    { filename: "notes.txt", status: "renamed", previous_filename: "notes.md" },
  ];

  assert.deepEqual(addedMarkdownFiles(files), ["notes.md", "copied.md"]);
});

test("a rename with no previous_filename counts as new Markdown", () => {
  // Fail closed: without provenance we cannot prove the source was Markdown.
  assert.deepEqual(
    addedMarkdownFiles([{ filename: "new.md", status: "renamed" }]),
    ["new.md"],
  );
});

test("escapes and bounds file paths rendered in the sticky comment", () => {
  assert.deepEqual(fileLines(["<script>&.md", "second.md"], 1), [
    "- <code>&lt;script&gt;&amp;.md</code>",
    "- _and 1 more_",
  ]);
});

test("the default file-list limit truncates at exactly one over", () => {
  const exact = Array.from({ length: FILE_LIST_LIMIT }, (_, i) => `f${i}.md`);
  const overflow = [...exact, "extra.md"];

  const atLimit = fileLines(exact);
  assert.equal(atLimit.length, FILE_LIST_LIMIT);
  assert.ok(!atLimit.some((line) => line.includes("more")));

  const past = fileLines(overflow);
  assert.equal(past.length, FILE_LIST_LIMIT + 1);
  assert.equal(past.at(-1), "- _and 1 more_");
});

test("recognizes release-please PRs only on full provenance", () => {
  const repo = { owner: "langchain-ai", repo: "deepagents" };
  const genuine = {
    authorLogin: "github-actions[bot]",
    authorType: "Bot",
    headRef: `${RELEASE_PLEASE_BRANCH_PREFIX}deepagents`,
    headRepo: "langchain-ai/deepagents",
  };
  assert.equal(isReleasePleasePr(genuine, repo), true);
  // Case-insensitive repo comparison, since GitHub is inconsistent about case.
  assert.equal(
    isReleasePleasePr({ ...genuine, headRepo: "LangChain-AI/DeepAgents" }, repo),
    true,
  );

  // Each signal alone must be insufficient.
  assert.equal(
    isReleasePleasePr({ ...genuine, authorLogin: "attacker" }, repo),
    false,
  );
  assert.equal(isReleasePleasePr({ ...genuine, authorType: "User" }, repo), false);
  assert.equal(
    isReleasePleasePr({ ...genuine, headRef: "release-please-ish" }, repo),
    false,
  );
  // The prefix with no component suffix is not a fanout release branch.
  assert.equal(
    isReleasePleasePr({ ...genuine, headRef: RELEASE_PLEASE_BRANCH_PREFIX }, repo),
    false,
  );
  // A fork cannot borrow the exemption.
  assert.equal(
    isReleasePleasePr({ ...genuine, headRepo: "attacker/deepagents" }, repo),
    false,
  );
  assert.equal(isReleasePleasePr({ ...genuine, headRepo: undefined }, repo), false);
});

test("acknowledged and blocking bodies are distinguishable and carry the marker", () => {
  const files = ["a.md"];
  const blocking = buildBody({ acknowledged: false, markdownFiles: files });
  const acked = buildBody({ acknowledged: true, markdownFiles: files });

  assert.ok(blocking.startsWith(STICKY_MARKER));
  assert.ok(acked.startsWith(STICKY_MARKER));
  assert.ok(blocking.includes("⛔"));
  assert.ok(acked.includes("acknowledged"));
  assert.ok(blocking.includes(ACKNOWLEDGMENT_LABEL));
  assert.ok(blocking.includes("<code>a.md</code>"));
});

// --- run(): the gate decision ----------------------------------------------

test("blocks a non-docs PR that adds Markdown and has no label", async () => {
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [{ name: "size:S" }],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
  assert.equal(calls.createComment.length, 1);
  assert.ok(calls.createComment[0].body.includes("⛔"));
});

test("passes a non-docs PR that adds Markdown once the label is applied", async () => {
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [{ name: ACKNOWLEDGMENT_LABEL }],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.failed, null);
  assert.equal(calls.createComment.length, 1);
  assert.ok(calls.createComment[0].body.includes("acknowledged"));
});

test("reads every label so acknowledgment is never missed", async () => {
  // The REST default is 30 per page; an issue caps at 100 labels. Asking for
  // fewer than the cap would drop the acknowledgment label on a busy PR and
  // fail it while telling the maintainer to apply a label already present.
  const labels = Array.from({ length: 99 }, (_, i) => ({ name: `label-${i}` }));
  labels.push({ name: ACKNOWLEDGMENT_LABEL });
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels,
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(calls.listLabels[0].per_page, 100);
  assert.equal(core.failed, null);
});

test("blocks a PR that lands Markdown by renaming a non-Markdown file", async () => {
  const { github } = makeGithub({
    files: [
      { filename: "DESIGN.md", status: "renamed", previous_filename: "DESIGN.txt" },
    ],
    labels: [],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
});

test("passes a docs PR without consulting files or labels", async () => {
  const { github, calls } = makeGithub({
    pr: { title: "docs: add a guide" },
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    comments: [botComment()],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.failed, null);
  assert.equal(calls.listFiles.length, 0);
  assert.equal(calls.listLabels.length, 0);
  // A stale block notice from before the retitle is cleared.
  assert.equal(calls.deleteComment.length, 1);
});

test("passes a release-please PR that adds a new component CHANGELOG", async () => {
  const { github } = makeGithub({
    pr: {
      title: "release(newpkg): 0.1.0",
      user: { login: "github-actions[bot]", type: "Bot" },
      head: {
        ref: `${RELEASE_PLEASE_BRANCH_PREFIX}newpkg`,
        repo: { full_name: "langchain-ai/deepagents" },
      },
    },
    files: [{ filename: "libs/newpkg/CHANGELOG.md", status: "added" }],
    labels: [],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.failed, null);
});

test("blocks a PR imitating a release-please branch from a fork", async () => {
  const { github } = makeGithub({
    pr: {
      title: "release(newpkg): 0.1.0",
      user: { login: "attacker", type: "User" },
      head: {
        ref: `${RELEASE_PLEASE_BRANCH_PREFIX}newpkg`,
        repo: { full_name: "attacker/deepagents" },
      },
    },
    files: [{ filename: "sneaky.md", status: "added" }],
    labels: [],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
});

test("passes and clears the sticky comment when no Markdown is added", async () => {
  const { github, calls } = makeGithub({
    files: [{ filename: "main.py", status: "added" }],
    comments: [botComment({ id: 42 })],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.failed, null);
  assert.deepEqual(
    calls.deleteComment.map((c) => c.comment_id),
    [42],
  );
});

// --- run(): the payload is never trusted ------------------------------------

test("a stale docs: payload title cannot pass a PR now titled feat:", async () => {
  // Two rapid title edits produce two runs against the same head SHA and check
  // name. Ordering is not guaranteed, so the older `docs:` event can land last
  // — and its result is the one branch protection reads.
  const { github } = makeGithub({
    pr: { title: "feat(code): add a thing" },
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
  });
  const core = newCore();
  const context = makeContext({ stale: { title: "docs: add a guide" } });

  await run({ github, context, core });

  assert.match(core.failed, /adds 1 Markdown file/);
});

test("a stale feat: payload title cannot block a PR now titled docs:", async () => {
  const { github } = makeGithub({
    pr: { title: "docs: add a guide" },
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
  });
  const core = newCore();
  const context = makeContext({ stale: { title: "feat(code): add a thing" } });

  await run({ github, context, core });

  assert.equal(core.failed, null);
});

test("release-please provenance is read live, not from the payload", async () => {
  // Otherwise a stale payload claiming bot provenance would carry the
  // exemption for a PR that is no longer a release PR.
  const { github } = makeGithub({
    pr: { title: "feat(code): add a thing" },
    files: [{ filename: "sneaky.md", status: "added" }],
    labels: [],
  });
  const core = newCore();
  const context = makeContext({
    stale: {
      title: "release(newpkg): 0.1.0",
      user: { login: "github-actions[bot]", type: "Bot" },
      head: {
        ref: `${RELEASE_PLEASE_BRANCH_PREFIX}newpkg`,
        repo: { full_name: "langchain-ai/deepagents" },
      },
    },
  });

  await run({ github, context, core });

  assert.match(core.failed, /adds 1 Markdown file/);
});

test("the truncation guard compares live files against the live total", async () => {
  // A live file list measured against a stale payload total would fail honest
  // PRs whenever a push landed between event delivery and this run.
  const { github } = makeGithub({
    pr: { changed_files: 2 },
    files: [
      { filename: "main.py", status: "modified" },
      { filename: "other.py", status: "modified" },
    ],
  });
  const core = newCore();
  const context = makeContext({ stale: { changed_files: 1 } });

  await run({ github, context, core });

  assert.equal(core.failed, null);
});

test("fails closed when the live pull request cannot be read", async () => {
  // Falling back to the payload here would quietly reinstate the whole race.
  const { github, calls } = makeGithub({
    pullsGetError: httpError("upstream boom", 502),
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /Could not read the pull request.*status=502/s);
  assert.equal(calls.listFiles.length, 0);
  assert.equal(calls.createComment.length, 0);
});

// --- run(): fail-closed guards ---------------------------------------------

test("fails closed when the changed-file list is truncated", async () => {
  const { github, calls } = makeGithub({
    pr: { changed_files: 3000 },
    files: [{ filename: "main.py", status: "modified" }],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /incomplete \(1 of 3000\)/);
  // The guard must block, not merely warn, and must not proceed to the label read.
  assert.equal(core.warnings.length, 0);
  assert.equal(calls.listLabels.length, 0);
});

test("fails closed when changed_files is missing from the payload", async () => {
  const { github } = makeGithub({ pr: { changed_files: undefined }, files: [] });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /missing numeric changed_files/);
});

test("fails closed when the file listing errors", async () => {
  const { github } = makeGithub({ listFilesError: httpError("boom", 502) });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /Could not list PR files.*status=502/s);
});

test("fails closed when the label read errors", async () => {
  const { github } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    listLabelsError: httpError("nope", 403),
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /Could not read PR labels.*status=403/s);
});

test("fails closed when there is no pull_request payload", async () => {
  const { github } = makeGithub();
  const core = newCore();

  await run({ github, context: { repo: { owner: "o", repo: "r" }, payload: {} }, core });

  assert.match(core.failed, /No pull_request payload/);
});

// --- run(): sticky comment handling ----------------------------------------

test("ignores a sticky marker posted by someone other than the bot", async () => {
  // A PR author who pre-posts the marker must not capture the slot, or the
  // real explanation is never posted and their forged notice stands.
  const impostor = {
    id: 99,
    body: `${STICKY_MARKER}\nℹ️ **New Markdown files acknowledged**`,
    user: { login: "contributor" },
  };
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    comments: [impostor],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
  assert.equal(calls.updateComment.length, 0);
  assert.equal(calls.createComment.length, 1);
});

test("updates an existing bot comment instead of posting a second one", async () => {
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    comments: [botComment({ id: 5 })],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(
    calls.updateComment.map((c) => c.comment_id),
    [5],
  );
});

test("skips a no-op edit so re-runs do not re-notify subscribers", async () => {
  const body = buildBody({ acknowledged: false, markdownFiles: ["NOTES.md"] });
  const { github, calls } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    comments: [botComment({ id: 5, body })],
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(calls.updateComment.length, 0);
  assert.equal(calls.createComment.length, 0);
  assert.match(core.failed, /adds 1 Markdown file/);
});

test("a failed comment write cannot turn a block into a pass", async () => {
  const { github } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    createCommentError: httpError("rate limited", 429),
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
  assert.equal(core.warnings.length, 1);
  // The file list still reaches the log and the summary.
  assert.ok(core.infos.some((m) => m.includes("NOTES.md")));
  assert.equal(core.summaryWrites.length, 1);
});

test("a 403 on the comment write is escalated above a warning", async () => {
  // Either the slot was captured or pull-requests: write was dropped. Both are
  // permanent, so they must not read as a transient blip.
  const { github } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    createCommentError: httpError("Resource not accessible by integration", 403),
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.errors.length, 1);
  assert.match(core.errors[0], /status=403/);
  assert.equal(core.warnings.length, 0);
});

test("the log fallback survives a job-summary failure", async () => {
  const { github } = makeGithub({
    files: [{ filename: "NOTES.md", status: "added" }],
    labels: [],
    createCommentError: httpError("boom", 500),
  });
  const core = newCore();
  core.summaryError = new Error("summary unwritable");

  await run({ github, context: makeContext(), core });

  assert.match(core.failed, /adds 1 Markdown file/);
  // core.info runs before the failable summary, so the file list survives.
  assert.ok(core.infos.some((m) => m.includes("NOTES.md")));
  assert.equal(core.summaryWrites.length, 0);
});

test("a failed cleanup notices rather than reddening a passing PR", async () => {
  const { github } = makeGithub({
    files: [{ filename: "main.py", status: "modified" }],
    comments: [botComment({ id: 3 })],
    deleteCommentError: httpError("gone", 404),
  });
  const core = newCore();

  await run({ github, context: makeContext(), core });

  assert.equal(core.failed, null);
  assert.equal(core.notices.length, 1);
  assert.match(core.notices[0], /stale block notice/);
});

// --- Cross-file constant pinning -------------------------------------------

const fs = require("node:fs");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "../../../..");

// RELEASE_PLEASE_BRANCH_PREFIX is a hardcoded literal duplicated across six
// workflows, and nothing notices when the config that determines the real
// branch name changes. Here the consequence is a stalled release pipeline: the
// exemption stops matching, a new component's first CHANGELOG.md is blocked,
// and the bot cannot label its way out. Mirrors the pin in close-old-prs.test.js.
test("the release branch prefix matches release-please-config.json", () => {
  const config = JSON.parse(
    fs.readFileSync(path.join(REPO_ROOT, "release-please-config.json"), "utf8"),
  );

  assert.equal(
    config["separate-pull-requests"],
    true,
    "separate-pull-requests drives the --components-- branch segment",
  );
  assert.equal(
    config["target-branch"],
    undefined,
    "a target-branch override changes the <base> segment of the branch name",
  );
  assert.equal(
    RELEASE_PLEASE_BRANCH_PREFIX,
    "release-please--branches--main--components--",
  );
});

// The label name is a literal in three places: this module, the workflow's
// bypass documentation, and LAYOUT.md. If it drifts from the label that
// actually exists on the repo the gate becomes permanently unbypassable, so
// pin the documentation to the constant rather than trusting three copies.
test("the acknowledgment label is documented consistently", () => {
  const workflow = fs.readFileSync(
    path.join(REPO_ROOT, ".github/workflows/markdown_file_check.yml"),
    "utf8",
  );
  const layout = fs.readFileSync(
    path.join(REPO_ROOT, ".github/LAYOUT.md"),
    "utf8",
  );

  assert.ok(
    workflow.includes(ACKNOWLEDGMENT_LABEL),
    `markdown_file_check.yml should document the '${ACKNOWLEDGMENT_LABEL}' label`,
  );
  assert.ok(
    layout.includes(ACKNOWLEDGMENT_LABEL),
    `LAYOUT.md should document the '${ACKNOWLEDGMENT_LABEL}' label`,
  );
});
