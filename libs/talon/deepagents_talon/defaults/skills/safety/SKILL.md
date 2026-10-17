---
name: safety
description: Check secret handling, staged changes and push targets, destructive Git operations, Docker image hygiene, and new dependencies before consequential coding actions. Use for commits, pushes, container builds or publishes, dependency additions, and explicit safety reviews.
---

# Safety preflight

Use these checks before the relevant action, not as a session-wide prerequisite. This is guidance, not a tool permission or an automatic approval gate. Respect the repository's instructions and existing approval controls; if a check cannot be completed, explain the uncertainty and do not claim it passed.

## Secrets

- Treat credentials, private keys, tokens, and secret-bearing files as sensitive, regardless of their names. Do not display values in chat, commands, logs, diffs, or tool arguments. Use environment-variable names, secret-store references, or file paths where the tool supports them. Do not print the environment or decode secrets just to inspect them.
- A workflow may need to consume a secret: use its intended secret-store, environment, or file interface without copying the value into source, history, or a public artifact. If a tool requires a literal value in a logged command or other exposed input, stop and seek a safer interface. Check presence or metadata without exposing contents where possible.
- Before committing or publishing, inspect changed paths and diffs for accidental secrets. Avoid dumping suspected values while investigating; if exposed, contain the leak and recommend rotation through the appropriate owner.

## Git writes and destructive operations

- Before committing, inspect `git status`, staged paths (`git diff --cached --name-only`), and the staged diff using a local scanner or redacted review if it may contain secrets. Confirm only intended files are staged and no credentials or local secret files are included. Unstage unwanted files without discarding local changes; do not rely on `.gitignore` to protect already tracked files.
- Before pushing, inspect the actual destination (`git remote get-url <remote>`), branch and refspec. Specify the intended remote and branch explicitly. Stop for clarification when the destination is unexpected or its ownership is unclear; do not assume a particular organization or forbid legitimate forks.
- Before force-pushing, resetting, cleaning, deleting branches or stashes, or other work-losing operations, inspect what would be overwritten or removed and identify the exact target. Prefer reversible alternatives and `--force-with-lease` where appropriate. Obtain explicit confirmation before discarding another person's work or unrecoverable local changes; never treat a preview as authorization.

## Docker

- Before a build, check the build context, `.dockerignore`, Dockerfile and Compose configuration for secret-bearing files and values. Keep secrets out of `COPY`/`ADD`, `ARG`/`ENV`, command lines, and image layers; `.dockerignore` is not a substitute for checking explicit copies or external build contexts.
- Use runtime secret injection or mounts for runtime needs and BuildKit secret mounts for build-time needs. Do not include secret values in Compose files or build arguments. Before publishing, verify the image destination and review relevant build history/configuration without printing suspected secrets. If exposure is suspected, do not publish; rebuild cleanly and rotate affected credentials.

## Dependencies and supply chain

- Prefer existing or standard-library functionality. Before adding a package, verify its source and registry identity, maintenance, licensing, known advisories, and why the existing dependencies do not suffice. Pin or bound versions according to the repository's conventions and review lockfile changes, lifecycle scripts, and unexpected transitive additions.
- Use the project's declared package manager and available audit or registry-protection tools. A firewall such as `sfw` is optional: if an expected tool is unavailable, do not install it automatically or bypass a repository-mandated check. Report the missing protection and use approved alternatives or ask how to proceed when it is required. Never claim an unrun scan passed.
