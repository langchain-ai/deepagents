# `control.py` — test runbook

End-to-end smoke test for the player-dispatch script. Run from
`vibe-coding-olympics/`.

## Prereqs

- iTerm2 running; Prefs → General → Magic → **Enable Python API** is **on**.
- `uv sync` has been run in this directory (installs `iterm2`).
- `deepagents` CLI is on your `$PATH`.

## 1. Launch a player

```bash
./play.sh "a website for a taco truck" 3001
```

Expect: a new iTerm2 window, tab 1 running the CLI (splash → skill → first
turn), tab 2 running `tail -f /tmp/vite.log`. iTerm2's title bar for tab 1
should read **`vibe-player-3001`**. If it doesn't, the tag isn't set and
every command below will miss.

## 2. Confirm discovery

```bash
uv run --project . python control.py list
```

Expect:

```
vibe-player-3001
```

Launch a second player on another port and rerun `list` to confirm both
show up:

```bash
./play.sh "a minimalist timer" 3002
uv run --project . python control.py list
# vibe-player-3001
# vibe-player-3002
```

## 3. Test `clear`

Let the CLI on port 3001 build up a few turns of conversation first, then:

```bash
uv run --project . python control.py clear --port 3001
```

Expect in that iTerm2 tab:

- `/clear` appears in the input.
- The CLI runs it, conversation resets, you're on a fresh thread.
- The 3002 session is untouched.

Stdout on the controlling shell:

```
cleared vibe-player-3001
```

Fan-out variant:

```bash
uv run --project . python control.py clear --all
# cleared vibe-player-3001
# cleared vibe-player-3002
```

## 4. Test `reset`

```bash
uv run --project . python control.py reset --port 3001
```

Expect:

1. `/quit` appears in the CLI input and runs — the Textual app exits.
2. ~1 s pause.
3. `deepagents` is typed at the now-bare shell prompt and launches a
   fresh CLI — splash screen visible, no skill auto-invoked, no prompt.

Stdout:

```
reset vibe-player-3001
```

Re-verify with `list` — the session still shows up (the shell still owns
the tag; `deepagents` started by `reset` runs inside it).

Fan-out variant: `reset --all`.

## 5. Failure / edge cases to spot-check

| Scenario | Command | Expected |
| --- | --- | --- |
| No players running | `control.py list` | `No active player sessions found.` exit 0 |
| Target a non-existent port | `control.py clear --port 9999` | `No matching player sessions.` exit 1 |
| `reset` while the CLI is mid-turn | `control.py reset --port 3001` | `/quit` still works (ALWAYS-bypass tier) |
| Close the player window manually, rerun `list` | — | It disappears from the output |

## 6. Cleanup

```bash
uv run --project . python control.py reset --all   # optional
# then close the iTerm2 windows by hand
```

`play.sh` also auto-frees the port next time you relaunch, so leaking a
window doesn't block a rerun.

## Troubleshooting

- **`list` shows nothing after `play.sh`**: confirm iTerm2's Python API is
  enabled; restart iTerm2 if you just toggled it. Double-check the tab
  title reads `vibe-player-<port>`.
- **`clear` runs but nothing happens in the CLI**: the focused pane in the
  tagged tab probably isn't the CLI one. `control.py` targets every
  session whose tag matches — if you manually split the pane, both get
  the keystrokes.
- **`reset` sends `deepagents` before the CLI has exited**: bump
  `RESET_QUIT_GRACE_SECS` in `control.py`.
- **Stale tag on a dead window**: close the window; the tag goes with it.
