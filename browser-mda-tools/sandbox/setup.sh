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
# The browser runs in --local mode, which needs no Browserbase API key at all.
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

npm install -g browse

CHROME_BIN="$(command -v google-chrome-stable || command -v google-chrome)"
for alias_name in chromium chromium-browser chrome; do
  ln -sf "${CHROME_BIN}" "/usr/local/bin/${alias_name}"
done

# One canonical way to start a page, so the agent cannot drift off the hardened
# flags. --no-sandbox because Chrome runs as root in a container and its own
# sandbox cannot initialize there; the isolation boundary is the MDA sandbox, not
# Chrome's. --disable-dev-shm-usage avoids crashes on the small /dev/shm a
# container gets. The resolver rules are defence in depth on the cloud metadata
# endpoint -- the real network control is proxy_config in sandbox/__init__.py.
cat > /usr/local/bin/web-open <<WEBOPEN
#!/usr/bin/env bash
set -euo pipefail
url="\${1:?usage: web-open <url>}"
export CHROME_PATH="${CHROME_BIN}"
exec browse open "\$url" \\
  --local \\
  --headless \\
  --chrome-arg=--no-sandbox \\
  --chrome-arg=--disable-dev-shm-usage \\
  --chrome-arg="--host-resolver-rules=MAP 169.254.169.254 ~NOTFOUND,MAP metadata.google.internal ~NOTFOUND,MAP metadata ~NOTFOUND"
WEBOPEN
chmod 0755 /usr/local/bin/web-open

mkdir -p /workspace

# Fail the bake now, loudly, rather than at the agent's first browse command.
"${CHROME_BIN}" --version
browse --version
echo "setup.sh OK: ${CHROME_BIN}"
