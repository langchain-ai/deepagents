# Vibe Coding Olympics

Day-of runbook for the live two-player setup.

## Start The Game

Start OBS first and make sure the websocket server is enabled. See
`obs/README.md` for the one-time OBS scene/source setup.

The live event setup assumes the controller and player laptops are on the same LAN. Prefer the dedicated event VLAN/switch and ask the venue for a static IP for the controller machine.

On the controller machine, from the root of the [`deepagents`](https://github.com/langchain-ai/deepagents) repo:

```bash
# terminal 1
cd vibe-coding-olympics/obs
uv run vibe-obs

# terminal 2
cd vibe-coding-olympics/control
VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

Open the controller UI from the controller machine:

```txt
http://localhost:8766
```

Use the controller's LAN IP when opening it from another machine:

```txt
http://<controller-static-ip>:8766
```

On the two player computers, after cloning the repo, from the root:

```bash
cd vibe-coding-olympics
export VIBE_CONTROL_API=http://<controller-static-ip>:8766
./play.sh 3001
```

```bash
cd vibe-coding-olympics
export VIBE_CONTROL_API=http://<controller-static-ip>:8766
./play.sh 3002
```

If port `8766` is unavailable on the controller machine, override both the control server bind port and the URL used by player laptops:

```bash
cd vibe-coding-olympics/control
VIBE_CONTROL_HOST=0.0.0.0 VIBE_CONTROL_PORT=8876 uv run vibe-control
```

```bash
cd vibe-coding-olympics
export VIBE_CONTROL_API=http://<controller-static-ip>:8876
./play.sh 3001
```

If port `8765` is unavailable for the OBS runner, override the OBS runner bind port and point the control server at that URL:

```bash
# terminal 1
cd vibe-coding-olympics/obs
VIBE_OBS_API_PORT=8875 uv run vibe-obs

# terminal 2
cd vibe-coding-olympics/control
VIBE_OBS_API=http://localhost:8875 VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

## Live Round Flow

1. Players enter their names and select models in the CLI.
2. The controller UI should show both players and both model-ready badges.
3. The controller chooses or draws a prompt, then clicks **Start**.
4. If the round must stop before the timer, click **End early**.
5. At the end of the timer, enter scores and click **End**.
6. Between rounds, click **Reset round all**.

## More Detail

- `control/README.md` documents the controller API and player dispatch path.
- `control/LAN_COMMAND_CHANNEL.md` drafts the LAN relay needed for controller-to-player commands on separate laptops.
- `obs/README.md` documents the OBS runner, scene setup, and transition API.
