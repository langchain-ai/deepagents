# Vibe Coding Olympics

Day-of runbook for the live two-player setup.

## Start The Game

Start OBS first and make sure the websocket server is enabled. See
`obs/README.md` for the one-time OBS scene/source setup.

On the controller machine:

```bash
# terminal 1
cd vibe-coding-olympics/obs
uv run vibe-obs

# terminal 2
cd vibe-coding-olympics/control
uv run vibe-control
```

Open the controller UI:

```txt
http://localhost:8766
```

On the two player computers:

```bash
cd vibe-coding-olympics
./play.sh 3001
```

```bash
cd vibe-coding-olympics
./play.sh 3002
```

If the controller is not reachable from player laptops at
`http://localhost:8766`, set `VIBE_CONTROL_API` before launching `play.sh`:

```bash
export VIBE_CONTROL_API=http://<controller-host>:8766
./play.sh 3001
```

## Live Round Flow

1. Players enter their names and select models in the CLI.
2. The controller UI should show both players and both model-ready badges.
3. The controller chooses or draws a prompt, then clicks **Start**.
4. At the end of the timer, enter scores and click **End**.
5. If the round must stop before the timer, click **End early**.
6. Between rounds, click **Reset round all**. This clears player CLI readiness
   without restarting Vite or reopening browser previews.

## Useful Commands

From `vibe-coding-olympics/control`:

```bash
uv run vibe-players list
uv run vibe-players times-up --all
uv run vibe-players clear --all
uv run vibe-players reset --all
```

Use `clear` between normal rounds. Use `reset` only when a player CLI needs to
quit and relaunch.

## More Detail

- `control/README.md` documents the controller API and player dispatch tools.
- `obs/README.md` documents the OBS runner, scene setup, and transition API.
- `CONTROL.md` is a smoke-test runbook for the `vibe-players` CLI and iTerm2
  session discovery.
