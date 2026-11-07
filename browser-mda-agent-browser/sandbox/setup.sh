#!/usr/bin/env bash
# Baked once into the sandbox snapshot at `mda deploy` / `mda dev` time, then
# cloned by every thread. Reruns only when this file changes.
#
# Base image is Ubuntu 26.04 (amd64) running as root, with Node 24 and npm
# already installed -- so this script installs a browser and the CLI, nothing more.
#
# Secrets: .env values are visible here as environment variables during the bake,
# but threads cloning the snapshot do not get them -- and anything written to disk
# lands in every thread's image. So nothing here writes a credential to a file.
# agent-browser drives a local Chrome and needs no Browserbase API key at all.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  fonts-liberation \
  fonts-noto-color-emoji

# Ubuntu's `chromium` package has no candidate and `chromium-browser` is only a
# snap shim -- it exits telling you to `snap install chromium`, which cannot work
# in a container. Install Google's real .deb instead; apt resolves its library
# deps, and it lands at /usr/bin/google-chrome-stable, a name `browse` probes.
curl -fsSL -o /tmp/chrome.deb \
  https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install -y --no-install-recommends /tmp/chrome.deb
rm -f /tmp/chrome.deb

npm install -g agent-browser

CHROME_BIN="$(command -v google-chrome-stable || command -v google-chrome)"
for alias_name in chromium chromium-browser chrome; do
  ln -sf "${CHROME_BIN}" "/usr/local/bin/${alias_name}"
done

# agent-browser is headless by default (`--headed` is the opt-in), so the
# wrapper exists only to pin the container-safe Chrome flags and the metadata
# deny rules. --no-sandbox because Chrome runs as root here and its own sandbox
# cannot initialize; the isolation boundary is the MDA sandbox, not Chrome's.
# --disable-dev-shm-usage avoids crashes on a container's small /dev/shm.
export CHROME_PATH="${CHROME_BIN}"
cat > /etc/profile.d/agent-browser.sh <<PROFILE
export CHROME_PATH="${CHROME_BIN}"
export AGENT_BROWSER_CHROME_ARGS="--no-sandbox --disable-dev-shm-usage --host-resolver-rules=MAP 169.254.169.254 ~NOTFOUND,MAP metadata.google.internal ~NOTFOUND,MAP metadata ~NOTFOUND"
PROFILE
chmod 0644 /etc/profile.d/agent-browser.sh

mkdir -p /workspace

# Fail the bake now, loudly, rather than at the agent's first browse command.
"${CHROME_BIN}" --version
agent-browser --version
# Prove the browser actually drives before any thread clones this snapshot.
CHROME_PATH="${CHROME_BIN}" agent-browser open https://example.com >/dev/null
echo "setup.sh OK: ${CHROME_BIN} + $(agent-browser --version)"
