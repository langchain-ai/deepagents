# `vibe-players` — test runbook

End-to-end smoke test for the player-dispatch CLI. Both `vibe-players` (CLI) and `vibe-control` (web UI) delegate to the same iTerm2 session-discovery helpers in `control/control_server/iterm_ctrl.py`, so passing this runbook validates the web panel's Players section too.

Run commands from `vibe-coding-olympics/control/` unless noted.

## Prereqs

- iTerm2 running; Prefs → General → Magic → **Enable Python API** is **on**.
- `uv sync` has been run in `control/` (installs `iterm2` + `fastapi`).
- `deepagents` CLI is on your `$PATH`. Install it locally as a `uv` tool from `libs/cli`, e.g.:

  ```bash
  uv tool install -e /Users/mdrxy/oss/deepagents/libs/cli \
    --with-editable ~/lc/libs/langchain_v1 \
    --with-editable ~/lc/libs/core \
    --with-editable ~/lc/libs/partners/anthropic \
    --with-editable ~/lc/libs/partners/openai \
    --with-editable ~/lc/libs/partners/groq \
    --with-editable ~/lc/libs/partners/ollama \
    --with-editable ~/oss/langchain-google/libs/genai \
    --with-editable ~/oss/langchain/libs/partners/openrouter \
    --with-editable ~/oss/langchain-nvidia/libs/ai-endpoints \
    --with-editable ~/oss/langchain-baseten/libs/baseten \
    --with-editable ~/oss/langchain/libs/partners/fireworks \
    --with langchain-daytona \
    --with langchain-modal \
    --with langchain-runloop \
    -U
  ```

## 1. Launch a player

From `vibe-coding-olympics/`:

```bash
./play.sh "a website for a taco truck" 3001
```

Expect: a new iTerm2 window, tab 1 running the CLI (splash → skill → first turn), tab 2 running `tail -f /tmp/vite.log`. iTerm2's title bar for tab 1 should read **`vibe-player-3001`** and the launch shell should print a socket path like `/tmp/deepagents-vibe-3001.sock`. If the title tag is missing, every command below will miss; if the socket path is missing, `clear` cannot reach the CLI.

## 2. Confirm discovery

```bash
uv run vibe-players list
```

Expect:

```
vibe-player-3001
```

Launch a second player on another port and rerun `list` to confirm both show up:

```bash
./play.sh "a minimalist timer" 3002
uv run vibe-players list
# vibe-player-3001
# vibe-player-3002
```

## 3. Test socket `clear`

Let the CLI on port 3001 build up a few turns of conversation first, then:

```bash
uv run vibe-players clear --port 3001
```

Expect in that iTerm2 tab:

- No slash command is typed into the input.
- The controller sends a `force-clear` signal over the player's
  `DEEPAGENTS_CLI_EXTERNAL_EVENT_SOCKET_PATH`.
- The CLI resets to a fresh thread.
- The 3002 session is untouched.

Stdout on the controlling shell:

```
cleared vibe-player-3001
```

Fan-out variant:

```bash
uv run vibe-players clear --all
# cleared vibe-player-3001
# cleared vibe-player-3002
```

## 4. Test `reset`

```bash
uv run vibe-players reset --port 3001
```

Expect:

1. `/quit` appears in the CLI input and runs — the Textual app exits.
2. ~1 s pause.
3. `deepagents -y` is typed at the now-bare shell prompt and launches a
   fresh CLI — splash screen visible, no skill auto-invoked, no prompt.

Stdout:

```
reset vibe-player-3001
```

Re-verify with `list` — the session still shows up (the shell still owns the tag; `deepagents -y` started by `reset` runs inside it).

Fan-out variant: `reset --all`.

## 5. Failure / edge cases to spot-check

| Scenario | Command | Expected |
| --- | --- | --- |
| No players running | `vibe-players list` | `No active player sessions found.` exit 0 |
| Target a non-existent port | `vibe-players clear --port 9999` | `No matching player sessions.` exit 1 |
| `reset` while the CLI is mid-turn | `vibe-players reset --port 3001` | `/quit` still works (ALWAYS-bypass tier) |
| Close the player window manually, rerun `list` | — | It disappears from the output |

## 6. Cleanup

```bash
uv run vibe-players reset --all   # optional
# then close the iTerm2 windows by hand
```

`play.sh` also auto-frees the port next time you relaunch, so leaking a window doesn't block a rerun.

## Troubleshooting

- **`list` shows nothing after `play.sh`**: confirm iTerm2's Python API is
  enabled; restart iTerm2 if you just toggled it. Double-check the tab
  title reads `vibe-player-<port>`.
- **`clear` runs but nothing happens in the CLI**: confirm the player was
  launched by the updated `play.sh` and that `/tmp/deepagents-vibe-<port>.sock`
  exists while the CLI is running. Old player sessions do not advertise an
  external event socket.
- **`reset` sends `deepagents -y` before the CLI has exited**: bump
  `RESET_QUIT_GRACE_SECS` in `control/control_server/iterm_ctrl.py`.
- **Stale tag on a dead window**: close the window; the tag goes with it.
