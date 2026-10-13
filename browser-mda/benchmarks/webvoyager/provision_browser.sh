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
export BROWSER_USE_CONFIG_PATH="${BROWSER_USE_CONFIG_PATH:-$HOME/.config/browseruse/config.json}"

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
    # Drives Playwright's own Chromium and builds its own BrowserProfile, so
    # neither the PATH shim nor a hand-written config reaches it: the loader
    # treats an unfamiliar file as "old format" and silently replaces it with
    # a fresh default, which is how three env vars all appeared to be ignored.
    # Let it mint the config, then patch the default profile in place.
    npx -y playwright@latest install --with-deps chromium
    mkdir -p "$(dirname "$BROWSER_USE_CONFIG_PATH")"
    uv run --with browser-use python - <<'PYEOF'
import json, os
from pathlib import Path
from browser_use.config import load_browser_use_config, get_default_profile
from browser_use.browser.profile import BrowserProfile

load_browser_use_config()
path = Path(os.environ["BROWSER_USE_CONFIG_PATH"])
config = json.loads(path.read_text())
for entry in config["browser_profile"].values():
    if entry.get("default"):
        entry.update(
            headless=True,
            chromium_sandbox=False,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
path.write_text(json.dumps(config, indent=2))

profile = BrowserProfile(**get_default_profile(load_browser_use_config()))
assert profile.chromium_sandbox is False, "chromium_sandbox did not stick"
assert "--no-sandbox" in (profile.args or []), "args did not stick"
print(f"browser-use profile ready: headless={profile.headless} args={profile.args}")

# Start it here. Every previous fix was a guess checked by reading what the
# agent said three steps later; this puts the real traceback in the job log.
import asyncio, traceback
from browser_use.browser.session import BrowserSession


async def smoke() -> None:
    session = BrowserSession(browser_profile=profile)
    try:
        await session.start()
        print("browser-use: session started OK")
    except BaseException:
        print("browser-use: SESSION FAILED TO START")
        traceback.print_exc()
        raise
    finally:
        try:
            await session.stop()
        except BaseException:
            pass


asyncio.run(smoke())
PYEOF
    ;;

  wdio)
    # Resolves its own Chrome for Testing; the prompt pins
    # goog:chromeOptions.binary at the system Chrome instead. Prove that
    # binary works headless here rather than inferring it from a dead session.
    npx -y playwright@latest install --with-deps chromium
    BIN="$(command -v google-chrome)"
    echo "wdio will drive: $BIN"
    "$BIN" --headless --disable-gpu --dump-dom about:blank >/dev/null \
      && echo "wdio: pinned Chrome runs headless OK"
    npx -y chromedriver@latest --version || echo "note: chromedriver resolved at session time"
    ;;

  *)
    echo "unknown arm: $ARM" >&2
    exit 1
    ;;
esac
