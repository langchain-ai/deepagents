#!/usr/bin/env bash
# Provisioned once at bake time; the result is saved as the sandbox snapshot.
# Runs with `bash -e`; a non-zero exit fails the snapshot (deploy keeps the last good one).
#
# IMPORTANT: do NOT write secrets or a `link-cli auth login` session here — anything on
# disk becomes part of every thread's image. Auth happens per-thread at runtime.
set -euo pipefail

# Node.js (Link CLI needs Node 20+). Install from NodeSource on the Debian-family base.
if ! command -v node >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list
  apt-get update
  apt-get install -y nodejs
fi

# The Stripe Link CLI — the agent calls this via the sandbox `execute` tool.
npm install -g @stripe/link-cli

node --version
link-cli --version

# A scratch dir for one-time-use card files (written 0600, deleted after checkout).
mkdir -p /workspace
