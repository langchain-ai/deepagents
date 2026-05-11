# LAN command channel

The current player command path only works when the controller can access the
player CLI's local iTerm2 session and Unix-domain event socket. That is fine
for a same-machine smoke test, but separate player laptops on the event LAN
need a network command path.

## Goal

Allow the controller to send the existing player commands over the event LAN:

- Prompt injection: `/skill:web-vibe Prompt: ...`
- Signals: `players-ready`, `times-up`, `force-clear`
- Optional hard reset: `/quit` followed by `deepagents -y`

Keep the Deep Agents CLI external-event socket protocol unchanged. The network
layer should be a relay that accepts controller commands over HTTP and forwards
them to the local Unix socket on each player laptop.

## Non-goals

- Do not expose the Unix socket directly over TCP.
- Do not require controller-side access to player iTerm2 sessions.
- Do not add a persistent database.
- Do not replace the existing local iTerm2 path immediately. Keep it for
  local development and smoke tests until the LAN path is proven.

## Proposed architecture

```txt
controller UI
  │
  ├── POST /api/players/prompt ───────┐
  ├── POST /api/players/times-up      │
  └── POST /api/players/clear         │
                                      ▼
                              vibe-control
                                      │
                                      │ HTTP command request
                                      ▼
player laptop                  vibe-player-relay
                                      │
                                      │ JSON-lines external event
                                      ▼
                              local Unix socket
                                      │
                                      ▼
                              Deep Agents CLI
```

Player status remains player-to-controller:

```txt
vibe-player-hook ──POST──▶ vibe-control
heartbeat loop ───POST──▶ vibe-control
```

Command dispatch becomes controller-to-player:

```txt
vibe-control ──POST──▶ vibe-player-relay ──Unix socket──▶ Deep Agents CLI
```

## Player relay

The relay entry point is:

```toml
vibe-player-relay = "control_server.player_relay:main"
```

The relay runs on each player laptop and binds to the player laptop's LAN
interface:

```bash
VIBE_RELAY_HOST=0.0.0.0 VIBE_RELAY_PORT=9771 uv run --project control vibe-player-relay
```

`play.sh` can start it automatically in the player iTerm2 session after it
exports `VIBE_EVENT_SOCKET`, or the event operator can run it as a separate
terminal process during setup.

### Relay env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `VIBE_RELAY_HOST` | `127.0.0.1` | Bind host for the player relay |
| `VIBE_RELAY_PORT` | `9771` | Bind port for the player relay |
| `VIBE_EVENT_SOCKET` | required | Local Deep Agents CLI external-event socket |
| `VIBE_PLAYER_TOKEN` | required | Shared bearer token for controller commands |

### Relay API

`GET /healthz`

Returns whether the relay can see the configured Unix socket.

```json
{
  "ok": true,
  "socket": "/tmp/deepagents-vibe-3001.sock"
}
```

`POST /command`

Headers:

```txt
authorization: Bearer <VIBE_PLAYER_TOKEN>
content-type: application/json
```

Body:

```json
{
  "kind": "command",
  "payload": "/skill:web-vibe Prompt: taco truck"
}
```

or:

```json
{
  "kind": "signal",
  "payload": "times-up"
}
```

The relay forwards the body to the existing Unix socket as:

```json
{
  "kind": "command",
  "payload": "/skill:web-vibe Prompt: taco truck",
  "correlation_id": "vibe-lan-..."
}
```

The HTTP response should mirror the Unix socket ACK:

```json
{
  "ok": true
}
```

If the socket is unavailable, return `503`. If the socket returns a negative
ACK, return `502`. If the token is missing or wrong, return `401`.

## Controller registry

The controller needs to know where each player relay lives. Add a relay URL to
the existing player connection state.

### Option A: explicit env per player

For day-of reliability, start with explicit static mapping:

```bash
export VIBE_PLAYER_3001_RELAY=http://10.20.30.101:9771
export VIBE_PLAYER_3002_RELAY=http://10.20.30.102:9771
export VIBE_PLAYER_TOKEN=<shared-token>
VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

This is simple, deterministic, and pairs well with venue-assigned static IPs.

### Option B: authenticated player self-registration

After the explicit mapping works, allow `play.sh` to report its relay URL only
through an authenticated registration flow:

```json
{
  "port": "3001",
  "relay_url": "http://10.20.30.101:9771"
}
```

Do not accept `relay_url` on the existing unauthenticated
`/api/players/connect` or `/api/players/heartbeat` endpoints. If those endpoints
store a caller-supplied relay URL, any machine that can reach the controller can
spoof a player port and make the controller send `VIBE_PLAYER_TOKEN` to an
attacker-controlled URL.

Self-registration needs a separate authenticated endpoint or authenticated
versions of the existing endpoints. Minimum requirements:

- Require a registration token distinct from `VIBE_PLAYER_TOKEN`.
- Prefer one token per player port, for example `VIBE_PLAYER_3001_REGISTER_TOKEN`.
- Reject registrations for a port whose token does not match.
- Validate `relay_url` against an allowlist, expected host list, or event subnet.
- Store the most recent authenticated relay URL per port and expire it with the
  heartbeat.

This avoids manual env mapping, but it needs a reliable way for the player
laptop to know its LAN IP. Static mapping is safer for the first event run.

## Testing away from the venue

Tailscale is a reasonable way to test the LAN command channel before the event.
Treat Tailscale IPs like the event LAN IPs:

Controller:

```bash
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_PLAYER_3001_RELAY=http://<player-1-tailscale-ip>:9771
export VIBE_PLAYER_3002_RELAY=http://<player-2-tailscale-ip>:9771

cd vibe-coding-olympics/control
VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

Player laptop:

```bash
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_CONTROL_API=http://<controller-tailscale-ip>:8766

cd vibe-coding-olympics
./play.sh 3001

cd vibe-coding-olympics
VIBE_EVENT_SOCKET=/tmp/deepagents-vibe-3001.sock \
  VIBE_RELAY_HOST=0.0.0.0 \
  VIBE_RELAY_PORT=9771 \
  uv run --project control vibe-player-relay
```

Make sure the relay binds to an interface reachable over Tailscale and that
local firewalls allow inbound TCP `9771` from the controller's Tailscale IP.

## Controller dispatch behavior

The `control_server.player_dispatch` module chooses the transport:

1. If a relay URL exists for the target port, POST to the relay.
2. Otherwise, fall back to the existing `iterm_ctrl` Unix socket path.

That keeps local smoke tests working and allows mixed operation during rollout.

Dispatch functions:

```python
async def send_prompt_to_players(ports: list[str] | None, prompt: str) -> list[str]: ...
async def times_up_players(ports: list[str] | None) -> list[str]: ...
async def clear_players(ports: list[str] | None) -> list[str]: ...
async def players_ready(ports: list[str] | None) -> list[str]: ...
async def reset_players(ports: list[str] | None) -> list[str]: ...
```

The FastAPI app calls `player_dispatch` instead of calling `iterm_ctrl`
directly for prompt, ready, clear, times-up, and reset operations.

## Reset command

`reset` is the only command that is not currently a Unix-socket event. It uses
iTerm2 to type `/quit`, waits, then types `deepagents -y`.

For LAN mode, prefer replacing hard reset with `force-clear` for normal
between-round flow. If hard reset is still required, add a relay-only command:

```json
{
  "kind": "admin",
  "payload": "restart-cli"
}
```

The relay can own the local process only if it also starts the CLI. If `play.sh`
continues to start the CLI inside iTerm2, remote hard reset should remain
unsupported for separate laptops.

## Security model

The event VLAN is isolated, but the relay still needs a shared token because it
can inject arbitrary CLI commands.

Minimum event-safe controls:

- Bind relays only on the event LAN.
- Require `Authorization: Bearer <token>` for `POST /command`.
- Do not log prompt bodies at warning/error level.
- Keep `VIBE_PLAYER_TOKEN` out of docs examples except as a placeholder.
- Ask the venue to allow controller-to-player TCP only on the relay port.

## Implementation status

- Done: shared Unix-socket send helper in `control_server.event_socket`.
- Done: `control_server.player_relay` with `/healthz` and `/command`.
- Done: relay tests for success, missing token, bad token, and missing socket.
- Done: controller relay registry from `VIBE_PLAYER_<port>_RELAY` env vars.
- Done: `player_dispatch` transport selection and FastAPI endpoint wiring.
- Done: controller tests proving relay dispatch is preferred and local iTerm2
  fallback still works.
- Not done: automatic relay startup from `play.sh`. Start `vibe-player-relay`
  separately for now.

## Day-of command sketch

Controller:

```bash
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_PLAYER_3001_RELAY=http://10.20.30.101:9771
export VIBE_PLAYER_3002_RELAY=http://10.20.30.102:9771

cd vibe-coding-olympics/control
VIBE_CONTROL_HOST=0.0.0.0 uv run vibe-control
```

Player 1:

```bash
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_CONTROL_API=http://<controller-static-ip>:8766

cd vibe-coding-olympics
./play.sh 3001

cd vibe-coding-olympics
VIBE_EVENT_SOCKET=/tmp/deepagents-vibe-3001.sock \
  VIBE_RELAY_HOST=0.0.0.0 \
  VIBE_RELAY_PORT=9771 \
  uv run --project control vibe-player-relay
```

Player 2:

```bash
export VIBE_PLAYER_TOKEN=<shared-token>
export VIBE_CONTROL_API=http://<controller-static-ip>:8766

cd vibe-coding-olympics
./play.sh 3002

cd vibe-coding-olympics
VIBE_EVENT_SOCKET=/tmp/deepagents-vibe-3002.sock \
  VIBE_RELAY_HOST=0.0.0.0 \
  VIBE_RELAY_PORT=9771 \
  uv run --project control vibe-player-relay
```
