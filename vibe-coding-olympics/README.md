# Vibe Coding Olympics

Day-of runbook for the live two-player setup.

## Start The Game

Start OBS and make sure the websocket server is enabled. (See `obs/README.md` for the one-time OBS scene/source setup.)

Ensure the controller and player laptops are on the same LAN. Plug in ethernet cables via USB-C dongles.

On the controller machine, from the root of the [`deepagents`](https://github.com/langchain-ai/deepagents) repo:

```bash
# terminal 1
cd vibe-coding-olympics/obs
uv run vibe-obs
```

```bash
# terminal 2
cd vibe-coding-olympics/control
uv run vibe-control
```

```txt
# (We set these in the .zshrc)
export VIBE_PLAYER_TOKEN=<shared-token>

export VIBE_PLAYER_3001_RELAY=http://<player-1-static-ip>:9771
export VIBE_PLAYER_3002_RELAY=http://<player-2-static-ip>:9771

export VIBE_CONTROL_HOST=0.0.0.0
export VIBE_CONTROL_PORT=8766
export VIBE_OBS_API="http://localhost:8765"
```

Open the controller UI from the controller machine:

```txt
http://localhost:8766
```

On the two player computers, after cloning the repo, from the root:

```bash
cd vibe-coding-olympics

# Player 1
./play.sh 3001

# Player 2
./play.sh 3002
```

```txt
# (We set these in .zshrc)
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_CONTROL_API=http://<controller-static-ip>:8766

export VIBE_RELAY_HOST=0.0.0.0
export VIBE_RELAY_HOST=9771
export VIBE_LAUNCH_RELAY=1

export model api keys...
```

`play.sh` instruments a Deep Agents CLI hook that reports the player's name and model-ready status back to `vibe-control` using `VIBE_CONTROL_API`.

It also starts a heartbeat loop and launches `vibe-player-relay` in a separate iTerm tab so the controller can send prompt, times-up, and clear commands back to each player laptop CLI instance over the LAN.

## Live Round Flow

1. Players enter their names and select models in the CLI.
2. The controller UI should show both players and both model-ready badges.
3. The controller chooses or draws a prompt, then clicks **Start**.
4. If the round must stop before the timer, click **End early**.
   1. (This option is available only if a round is ongoing)
5. When the timer reaches zero, judging begins
6. Between rounds, click **Reset round all** to blank player pages and reset readiness.

## More Detail

- `control/README.md` documents the controller API and player dispatch path.
- `obs/README.md` documents the OBS runner, scene setup, and transition API.
