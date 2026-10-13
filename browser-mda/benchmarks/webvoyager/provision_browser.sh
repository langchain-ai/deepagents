#!/usr/bin/env bash
# Install and configure the browser one arm needs.
#
# A script rather than inline YAML so it can be run before it is pushed. The
# first version of this lived in the workflow, had an indented heredoc
# terminator, and failed every job -- which is exactly the class of mistake
# that only shows up when you execute the thing.
#
#   ./provision_browser.sh browser-use
set -euo pipefail

ARM="${1:?usage: provision_browser.sh <arm>}"
CONFIG_DIR="${BROWSER_USE_CONFIG_DIR:-$HOME/.config/browseruse}"

case "$ARM" in
  playwright)
    # Playwright MCP brings its own browser and already passes --no-sandbox.
    npx -y playwright@latest install --with-deps chromium
    ;;

  agent-browser)
    # Must be on PATH: a smoke task answered that the CLI "is not available in
    # this sandbox environment".
    npm install -g agent-browser@latest
    agent-browser --version
    ;;

  browser-use)
    # Drives Playwright's own Chromium, not the system Chrome, and reads its
    # flags from a config file rather than argv.
    npx -y playwright@latest install --with-deps chromium
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/config.json" <<'JSON'
{
  "browser_profile": {
    "ci": {
      "id": "ci",
      "default": true,
      "headless": true,
      "chromium_sandbox": false,
      "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
    }
  }
}
JSON
    python3 -c "import json,sys; json.load(open('$CONFIG_DIR/config.json'))"
    echo "wrote $CONFIG_DIR/config.json"
    ;;

  wdio)
    # Resolves its own Chrome for Testing; flags go through start_session
    # capabilities, which the prompt pins.
    npx -y playwright@latest install --with-deps chromium
    ;;

  *)
    echo "unknown arm: $ARM" >&2
    exit 1
    ;;
esac
