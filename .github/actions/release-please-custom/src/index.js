// Entry point for the custom release-please action.
//
// Registers a "pr-body" ChangelogNotes type that pulls release notes from
// PR bodies, then runs a minimal release-please manifest flow (modeled on
// the upstream googleapis/release-please-action but supporting only token,
// config-file, manifest-file, and target-branch inputs).

const core = require("@actions/core");
const {
  GitHub,
  Manifest,
  registerChangelogNotes,
  VERSION,
} = require("release-please");
const { PRBodyChangelogNotes } = require("./pr-body-notes");

// ---------------------------------------------------------------------------
// Register our custom changelog type before any manifest is loaded.
// Config must set `"changelog-type": "pr-body"` to activate it.
// Factory receives ChangelogNotesFactoryOptions; we only use github +
// changelogSections.
// ---------------------------------------------------------------------------
registerChangelogNotes(
  "pr-body",
  ({ github, changelogSections }) =>
    new PRBodyChangelogNotes(github, changelogSections)
);

// ---------------------------------------------------------------------------
// Action I/O helpers (mirrored from release-please-action)
// ---------------------------------------------------------------------------

function setPathOutput(path, key, value) {
  if (path === ".") {
    core.setOutput(key, value);
  } else {
    core.setOutput(`${path}--${key}`, value);
  }
}

function outputReleases(releases) {
  releases = releases.filter(Boolean);
  const pathsReleased = [];
  core.setOutput("releases_created", releases.length > 0);
  for (const release of releases) {
    const path = release.path || ".";
    pathsReleased.push(path);
    setPathOutput(path, "release_created", true);
    for (const [rawKey, value] of Object.entries(release)) {
      let key = rawKey;
      if (key === "tagName") key = "tag_name";
      if (key === "uploadUrl") key = "upload_url";
      if (key === "notes") key = "body";
      if (key === "url") key = "html_url";
      setPathOutput(path, key, value);
    }
  }
  core.setOutput("paths_released", JSON.stringify(pathsReleased));
}

function outputPRs(prs) {
  prs = prs.filter(Boolean);
  core.setOutput("prs_created", prs.length > 0);
  if (prs.length) {
    core.setOutput("pr", prs[0]);
    core.setOutput("prs", JSON.stringify(prs));
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  core.info(`Running release-please ${VERSION} with custom pr-body changelog`);

  const token = core.getInput("token", { required: true });
  const configFile =
    core.getInput("config-file") || "release-please-config.json";
  const manifestFile =
    core.getInput("manifest-file") || ".release-please-manifest.json";
  const targetBranch = core.getInput("target-branch") || undefined;

  const githubRepository = process.env.GITHUB_REPOSITORY;
  if (!githubRepository || !githubRepository.includes("/")) {
    throw new Error(
      `GITHUB_REPOSITORY env var is missing or malformed: "${githubRepository ?? ""}". ` +
        "Expected format: owner/repo. This action must run inside GitHub Actions."
    );
  }
  const idx = githubRepository.indexOf("/");
  const owner = githubRepository.substring(0, idx);
  const repo = githubRepository.substring(idx + 1);

  const github = await GitHub.create({
    owner,
    repo,
    token,
  });

  const branch = targetBranch || github.repository.defaultBranch;

  // Create GitHub Releases for merged release PRs.
  // Returns no releases when skip-github-release is set in the config;
  // actual release creation is handled separately (see RELEASING.md).
  const errors = [];
  {
    try {
      const manifest = await Manifest.fromManifest(
        github,
        branch,
        configFile,
        manifestFile
      );
      core.debug("Creating releases");
      outputReleases(await manifest.createReleases());
    } catch (err) {
      errors.push(`Releases: ${err.message}`);
      core.error(`Failed to create releases: ${err.message}`);
      if (err.stack) core.debug(err.stack);
    }
  }

  // Create or update release PRs
  {
    try {
      const manifest = await Manifest.fromManifest(
        github,
        branch,
        configFile,
        manifestFile
      );
      core.debug("Creating pull requests");
      outputPRs(await manifest.createPullRequests());
    } catch (err) {
      errors.push(`Pull requests: ${err.message}`);
      if (err.stack) core.debug(err.stack);
    }
  }

  if (errors.length) {
    core.setFailed(`release-please failed:\n${errors.join("\n")}`);
  }
}

main().catch((err) => {
  const message = err instanceof Error ? err.message : String(err);
  const stack = err instanceof Error ? err.stack : undefined;
  if (stack) core.error(stack);
  core.setFailed(`release-please failed: ${message}`);
});
