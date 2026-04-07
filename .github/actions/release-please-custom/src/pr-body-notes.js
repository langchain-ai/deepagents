// Custom ChangelogNotes implementation that extracts release notes from PR
// bodies. Falls back to formatted conventional-commit entries when the PR body
// has no "## Release Notes" section, the PR cannot be fetched, or no PR is
// associated with the commit.

const core = require("@actions/core");

const RELEASE_NOTES_RE =
  /^##\s+Release\s+Notes\s*\n([\s\S]*?)(?=\n##\s|\n---|\s*$)/im;

const DEFAULT_HOST = "https://github.com";

class PRBodyChangelogNotes {
  /**
   * @param {import('release-please').GitHub} github
   * @param {import('release-please').ChangelogSection[] | undefined} changelogSections
   */
  constructor(github, changelogSections) {
    this.github = github;
    this.changelogSections = changelogSections;
  }

  /**
   * Implements the ChangelogNotes.buildNotes interface. Unlike the default
   * implementation, this builds a custom markdown structure using PR body
   * release notes and ignores the targetBranch option.
   *
   * @param {import('release-please').ConventionalCommit[]} commits
   * @param {import('release-please').BuildNotesOptions} options
   * @returns {Promise<string>}
   */
  async buildNotes(commits, options) {
    const { owner, repository, version, previousTag, currentTag } = options;
    const host = options.host || DEFAULT_HOST;
    const sections = this.changelogSections || options.changelogSections || [];

    // Build lookup tables from changelog-sections config
    const sectionHeading = new Map();
    const hiddenTypes = new Set();
    for (const s of sections) {
      if (s.hidden) {
        hiddenTypes.add(s.type);
      } else {
        sectionHeading.set(s.type, s.section);
      }
    }

    // Bucket commits by section heading
    /** @type {Map<string, string[]>} */
    const buckets = new Map();
    /** @type {string[]} */
    const breaking = [];

    for (const commit of commits) {
      // Collect BREAKING CHANGE notes regardless of hidden status
      if (commit.breaking) {
        for (const note of commit.notes) {
          if (note.title === "BREAKING CHANGE") {
            breaking.push(note.text);
          }
        }
      }

      if (hiddenTypes.has(commit.type)) continue;
      const heading = sectionHeading.get(commit.type);
      if (!heading) continue;

      const entry = await this._buildEntry(commit, {
        owner,
        repository,
        host,
      });

      if (!buckets.has(heading)) buckets.set(heading, []);
      buckets.get(heading).push(entry);
    }

    const lines = [];

    const versionLink = previousTag
      ? `[${version}](${host}/${owner}/${repository}/compare/${previousTag}...${currentTag})`
      : version;
    lines.push(
      `## ${versionLink} (${new Date().toISOString().split("T")[0]})`
    );

    if (breaking.length > 0) {
      lines.push("", "### ⚠ BREAKING CHANGES", "");
      for (const text of breaking) {
        lines.push(`* ${text}`);
      }
    }

    for (const [heading, entries] of buckets) {
      lines.push("", `### ${heading}`, "");
      for (const entry of entries) {
        // Multi-line entries (from PR body) get indented continuation
        const [first, ...rest] = entry.split("\n");
        lines.push(`* ${first}`);
        for (const line of rest) {
          if (line.trim()) lines.push(`  ${line}`);
        }
      }
    }

    return lines.join("\n");
  }

  /**
   * Build a single changelog entry for a commit.
   *
   * Tries to extract a "## Release Notes" section from the PR body first.
   * Falls back to a formatted conventional commit entry (scope prefix, PR
   * link, SHA link).
   */
  async _buildEntry(commit, { owner, repository, host }) {
    const prNumber = this._prNumber(commit);

    if (prNumber) {
      try {
        const { data: pr } = await this.github.octokit.rest.pulls.get({
          owner,
          repo: repository,
          pull_number: prNumber,
        });
        const custom = this._parseReleaseNotes(pr.body);
        if (custom) return custom;
      } catch (err) {
        const status = err?.status ?? err?.response?.status;
        if (status === 404 || status === 410) {
          core.debug(
            `PR #${prNumber} not found (${status}), falling back to commit message`
          );
        } else if (status === 401 || status === 403 || status === 429) {
          throw new Error(
            `Cannot fetch PR #${prNumber}: HTTP ${status}. ` +
              `Check token permissions and rate limits. Original: ${err.message}`
          );
        } else {
          core.warning(
            `Failed to fetch PR #${prNumber} for changelog entry: ${err.message}`
          );
        }
      }
    }

    // Fallback: conventional commit message with links
    const scope = commit.scope ? `**${commit.scope}:** ` : "";
    const prLink = prNumber
      ? ` ([#${prNumber}](${host}/${owner}/${repository}/pull/${prNumber}))`
      : "";
    const sha = commit.sha.substring(0, 7);
    const shaLink = ` ([${sha}](${host}/${owner}/${repository}/commit/${commit.sha}))`;
    return `${scope}${commit.bareMessage}${prLink}${shaLink}`;
  }

  /**
   * Extract PR number from a commit. Checks pullRequest.number first, then
   * references, then trailing (#NNN) in the commit message.
   */
  _prNumber(commit) {
    if (commit.pullRequest?.number) return commit.pullRequest.number;
    for (const ref of commit.references || []) {
      if (ref.issue) {
        const num = parseInt(ref.issue, 10);
        if (!Number.isNaN(num)) return num;
      }
    }
    const match = commit.message?.match(/\(#(\d+)\)\s*$/);
    return match ? parseInt(match[1], 10) : null;
  }

  /** Extract the body under "## Release Notes" from a PR description. */
  _parseReleaseNotes(body) {
    if (!body) return null;
    const match = body.match(RELEASE_NOTES_RE);
    if (!match) return null;
    const notes = match[1].trim();
    return notes || null;
  }
}

module.exports = { PRBodyChangelogNotes };
