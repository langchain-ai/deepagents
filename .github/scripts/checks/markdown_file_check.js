"use strict";

// Detector for the "new Markdown files need acknowledgment" gate.
//
// The workflow (.github/workflows/markdown_file_check.yml) is a thin shim that
// requires `run` from this file, matching the repo pattern established by
// .github/scripts/labeling/close-old-prs.js. Everything the gate decides lives
// here so it can be executed against a fake octokit in
// .github/scripts/tests/checks/markdown_file_check.test.js — a decision that
// only appears inside the workflow's YAML `script:` string can be asserted on
// as text but never actually run, which lets an inverted condition ship green.

const STICKY_MARKER = "<!-- markdown-file-check -->";
const ACKNOWLEDGMENT_LABEL = "markdown-added: acknowledged";

// Who authors this workflow's own sticky comments. Matched on when finding the
// comment to update so a PR author cannot pre-post the marker and capture the
// slot; see findStickyComment.
const WORKFLOW_BOT_LOGIN = "github-actions[bot]";

// Distinct identity from WORKFLOW_BOT_LOGIN despite the same value today:
// this is "who opens release PRs" (release-please runs on GITHUB_TOKEN).
// Kept in step with RELEASE_PLEASE_AUTHOR in close-old-prs.js.
const RELEASE_PLEASE_AUTHOR = "github-actions[bot]";

// Mirrors the branch shape `separate-pull-requests: true` produces in
// release-please-config.json. The same literal is independently hardcoded in
// close-old-prs.js, release-notes.js, check_sdk_pin.yml, check_partner_bounds.yml,
// release_please_fanout_watch.yml, and release-please.yml — keep them in
// lockstep. A test pins this against release-please-config.json.
const RELEASE_PLEASE_BRANCH_PREFIX =
  "release-please--branches--main--components--";

// How many file paths the sticky comment lists before collapsing the rest into
// a count. Bounds an author-controlled list against GitHub's 65536-character
// comment limit.
const FILE_LIST_LIMIT = 50;

function isDocsTitle(title) {
  // Lowercase-only on purpose: this mirrors `_TITLE_RE` in
  // .github/scripts/labeling/check_pr_scope_files.py, and pr_lint.yml enforces
  // the Conventional Commit type list case-sensitively. "Docs: ..." is not a
  // valid title in this repo, so it must not earn the bypass either.
  return /^docs(?:\([^)]*\))?!?:\s/.test(title || "");
}

function isMarkdown(filename) {
  return typeof filename === "string" && filename.toLowerCase().endsWith(".md");
}

// Files this PR causes to exist as Markdown when they did not before.
//
// `status === "added"` alone is not enough: GitHub reports a `git mv
// notes.txt notes.md` as `renamed` and a copy as `copied`, both of which land
// a brand new Markdown document at a path the repo did not have. Keying on
// "the destination is Markdown and the source was not" closes that bypass
// while still ignoring a pure move of an existing `.md` file, which adds no
// new document to review.
function addedMarkdownFiles(files) {
  return files
    .filter((file) => {
      if (!isMarkdown(file.filename)) {
        return false;
      }
      if (file.status === "added" || file.status === "copied") {
        return true;
      }
      if (file.status === "renamed") {
        return !isMarkdown(file.previous_filename ?? "");
      }
      return false;
    })
    .map((file) => file.filename);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function fileLines(files, limit = FILE_LIST_LIMIT) {
  const lines = files
    .slice(0, limit)
    .map((file) => `- <code>${escapeHtml(file)}</code>`);
  if (files.length > limit) {
    lines.push(`- _and ${files.length - limit} more_`);
  }
  return lines;
}

// Release PRs add a package's first CHANGELOG.md when a new component is
// onboarded, under a `release(<pkg>): ...` title that this gate would
// otherwise block — and a bot PR cannot apply the acknowledgment label to
// unblock itself, so the release pipeline would stall until a human noticed.
//
// Provenance, not the title, drives the exemption: author identity, branch
// prefix, and same-repo head must all match. An outside contributor can name
// a branch anything, but cannot push it to this repository nor author a PR as
// the bot, so no single spoofable signal carries the bypass. This mirrors
// isReleasePr in close-old-prs.js.
function isReleasePleasePr(
  { authorLogin, authorType, headRef, headRepo },
  { owner, repo },
) {
  if (authorLogin !== RELEASE_PLEASE_AUTHOR || authorType !== "Bot") {
    return false;
  }
  if (
    typeof headRef !== "string" ||
    !headRef.startsWith(RELEASE_PLEASE_BRANCH_PREFIX) ||
    headRef.length <= RELEASE_PLEASE_BRANCH_PREFIX.length
  ) {
    return false;
  }
  return (
    typeof headRepo === "string" &&
    headRepo.toLowerCase() === `${owner}/${repo}`.toLowerCase()
  );
}

// A thrown non-Error (or an octokit error without `.status`) otherwise logs as
// "undefined", which is worse than no message at all — it reads as a transient
// blip when it may be a permanent misconfiguration.
function describeError(error) {
  const status = error?.status ?? "n/a";
  const message = error?.message ?? String(error);
  return `[status=${status}] ${message}`;
}

// Matched on the marker AND the bot login: matching the marker alone lets a PR
// author pre-post a comment starting with the marker and capture the sticky
// slot. The subsequent update/delete then 403s, so the genuine explanation is
// never posted while an author-authored — and author-editable — notice sits on
// the PR looking official. Same reasoning as ripgrep_timeout_comment.yml.
// If the posting identity ever changes, update WORKFLOW_BOT_LOGIN; a missed
// match only costs one extra comment, a hijacked slot costs the explanation.
async function findStickyComment({ github, owner, repo, number }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: number,
    per_page: 100,
  });
  return comments.find(
    (comment) =>
      comment.user?.login === WORKFLOW_BOT_LOGIN &&
      (comment.body ?? "").startsWith(STICKY_MARKER),
  );
}

// Called when the PR no longer needs the gate (retitled to `docs:`, or the
// Markdown file dropped). Failure here cannot turn a clean result red, but it
// does leave a stale block notice on a now-passing PR, so the warning goes out
// as core.notice: a warning inside a *successful* step is log-only, and nobody
// opens the log of a green check.
async function clearStickyComment({ github, owner, repo, number, core }) {
  try {
    const existing = await findStickyComment({ github, owner, repo, number });
    if (existing) {
      await github.rest.issues.deleteComment({
        owner,
        repo,
        comment_id: existing.id,
      });
    }
  } catch (error) {
    core.notice(
      "Could not remove the prior Markdown-file comment; a stale block notice " +
        `may still be shown on this now-passing PR: ${describeError(error)}`,
    );
  }
}

async function upsertStickyComment({ github, owner, repo, number, core, body }) {
  try {
    const existing = await findStickyComment({ github, owner, repo, number });
    if (existing) {
      // Skip a no-op edit so re-runs (a label toggle, a rebase) do not
      // re-notify every subscriber with identical content.
      if (existing.body !== body) {
        await github.rest.issues.updateComment({
          owner,
          repo,
          comment_id: existing.id,
          body,
        });
      }
    } else {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: number,
        body,
      });
    }
    return;
  } catch (error) {
    // 403 is not a blip. It means either the sticky slot was captured by
    // another author, or `pull-requests: write` was dropped from the workflow
    // — a permanent regression that silently degrades every future run to
    // log-only. core.error surfaces it as a Checks annotation; anything else
    // is plausibly transient (rate limit, 5xx) and stays a warning.
    const report = error?.status === 403 ? core.error : core.warning;
    report.call(
      core,
      `Could not post the Markdown-file comment: ${describeError(error)}`,
    );
  }

  // The comment was the user-facing channel and it failed, so fall back —
  // unconditionally first, because core.info cannot throw and the file list is
  // the only actionable part of this check. The job summary is nicer to read
  // but is itself failable (unwritable summary, size limit), so it comes
  // second and its failure costs nothing.
  core.info(`Markdown-file check details:\n${body}`);
  try {
    await core.summary
      .addHeading("New Markdown files require acknowledgment")
      .addRaw(body)
      .write();
  } catch (summaryError) {
    core.warning(
      `Could not write the job summary fallback: ${describeError(summaryError)}`,
    );
  }
}

function buildBody({ acknowledged, markdownFiles }) {
  const shownFiles = fileLines(markdownFiles);
  if (acknowledged) {
    return [
      STICKY_MARKER,
      `ℹ️ **New Markdown files acknowledged** via the \`${ACKNOWLEDGMENT_LABEL}\` label.`,
      "",
      ...shownFiles,
      "",
      "Remove the label to re-enable the block.",
    ].join("\n");
  }
  return [
    STICKY_MARKER,
    "⛔ **This non-docs PR adds new Markdown files.**",
    "",
    ...shownFiles,
    "",
    "Change the PR type to `docs` if it only contains documentation changes. Otherwise, review the new files and apply the acknowledgment label:",
    "",
    `\`${ACKNOWLEDGMENT_LABEL}\``,
  ].join("\n");
}

async function run({ github, context, core }) {
  const { owner, repo } = context.repo;
  const pullRequest = context.payload.pull_request;
  if (!pullRequest) {
    core.setFailed(
      "No pull_request payload; this workflow only supports pull_request_target events.",
    );
    return;
  }

  const { number } = pullRequest;

  // Every input to the decision is read live, never from the event payload.
  // The payload is a snapshot taken at delivery, and two rapid title edits
  // produce two runs against the same head SHA and the same check name.
  // `cancel-in-progress: false` lets both finish and GitHub does not promise
  // FIFO ordering, so the older event can land last: a payload still reading
  // `docs:` would clear the sticky comment and publish a green required check
  // over a PR whose title is now `feat:`. Re-running the check cannot repair
  // that, because the stale run is the most recent result. One `pulls.get`
  // removes the race for the title and, at no extra cost, for the
  // release-please provenance and changed-file total alongside it.
  let livePullRequest;
  try {
    ({ data: livePullRequest } = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: number,
    }));
  } catch (error) {
    // A gate input, so it fails closed like listFiles and listLabelsOnIssue.
    // Falling back to the payload would reinstate exactly the race above, and
    // silently — the run would look like any other pass.
    core.setFailed(`Could not read the pull request: ${describeError(error)}`);
    return;
  }

  const title = livePullRequest.title;

  if (isDocsTitle(title)) {
    await clearStickyComment({ github, owner, repo, number, core });
    core.info("PR type is docs; no acknowledgment required.");
    return;
  }

  if (isReleasePleasePr(
    {
      authorLogin: livePullRequest.user?.login,
      authorType: livePullRequest.user?.type,
      headRef: livePullRequest.head?.ref,
      headRepo: livePullRequest.head?.repo?.full_name,
    },
    { owner, repo },
  )) {
    await clearStickyComment({ github, owner, repo, number, core });
    core.info(
      "Release-please PR (verified by author, branch, and head repo); " +
        "a new component's first CHANGELOG.md needs no acknowledgment.",
    );
    return;
  }

  let files;
  try {
    files = await github.paginate(github.rest.pulls.listFiles, {
      owner,
      repo,
      pull_number: number,
      per_page: 100,
    });
  } catch (error) {
    core.setFailed(`Could not list PR files: ${describeError(error)}`);
    return;
  }

  // listFiles is hard-capped by GitHub at 3000 files regardless of pagination.
  // A truncated list would hide newly added Markdown, so verify the collected
  // count against the PR's own total and fail closed otherwise. Split into two
  // checks with distinct messages — as pr_scope_file_check.yml does — because
  // a missing payload field and a truncated list are different root causes and
  // a maintainer needs to tell them apart.
  const expected = livePullRequest.changed_files;
  if (typeof expected !== "number") {
    core.setFailed(
      `PR payload missing numeric changed_files (got ${JSON.stringify(expected)}); ` +
        "cannot verify the changed-file list is complete. Failing closed.",
    );
    return;
  }
  if (files.length !== expected) {
    core.setFailed(
      `Changed-file list is incomplete (${files.length} of ${expected}); cannot ` +
        "determine whether this PR adds Markdown files. Failing closed. A PR " +
        "over GitHub's 3000-file listing cap cannot pass this check — split it.",
    );
    return;
  }

  const markdownFiles = addedMarkdownFiles(files);
  if (markdownFiles.length === 0) {
    await clearStickyComment({ github, owner, repo, number, core });
    core.info("No newly added Markdown files.");
    return;
  }

  let labels;
  try {
    // per_page: 100 is the whole story, not a page size — GitHub caps an issue
    // at 100 labels, so one request always sees every label. The REST default
    // of 30 would drop the acknowledgment label on a heavily labeled PR and
    // fail it while telling the maintainer to apply a label they can see is
    // already there.
    ({ data: labels } = await github.rest.issues.listLabelsOnIssue({
      owner,
      repo,
      issue_number: number,
      per_page: 100,
    }));
  } catch (error) {
    core.setFailed(`Could not read PR labels: ${describeError(error)}`);
    return;
  }

  const acknowledged = labels.some(
    (label) => label.name === ACKNOWLEDGMENT_LABEL,
  );
  const body = buildBody({ acknowledged, markdownFiles });

  // setFailed before the comment write: upsertStickyComment swallows its own
  // errors so the red check is the load-bearing signal and must be set first.
  if (!acknowledged) {
    core.setFailed(
      `Non-docs PR adds ${markdownFiles.length} Markdown file(s); apply ` +
        `'${ACKNOWLEDGMENT_LABEL}' to acknowledge.`,
    );
  }
  await upsertStickyComment({ github, owner, repo, number, core, body });
}

module.exports = {
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
};
