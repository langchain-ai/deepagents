# Optional browser deployment

Linux **rootful Docker only**, synthetic data only. This overlay leaves
`examples/talon/docker-compose.yml` unchanged. It enables native `browser_cdp` and
`browser_request_handoff` tools through a private Node bridge and one globally
coordinated browser lease. There is no authenticated viewer or runnable tunnel yet.
Do not use real credentials before the remaining end-to-end acceptance work.

## Start and stop

Host prerequisites: Python 3, local rootful Docker Engine, Docker Compose **2.33.1+**
(`!override` and `gw_priority`), `nsenter`, `iptables`, and `ip6tables`. The launcher
installs nothing. Remote Docker, rootless Docker, user namespace remapping, Docker
Desktop, non-Linux hosts, overlapping subnets, and parallel instances are unsupported.
The root host operator and Docker daemon are trusted. No container receives the
Docker socket or host namespace privileges.

Supply your existing environment file explicitly. It is passed directly to Compose
and Talon, never copied or printed by the launcher. `--home` is mandatory because
sudo's HOME usually refers to root rather than your Talon workspace. Run from any
directory, substituting absolute paths:

```sh
sudo python3 /path/to/deepagents/examples/talon/browser/deploy.py \
  --home /home/alice \
  --env-file /home/alice/talon.env
```

If local Docker context selection requires preserved environment, use `sudo -E`
only after reviewing that environment. Neither sudo nor `--home` should be used to
silently change which daemon you operate. Compose uses project `talon-browser`;
stop the base-only Talon deployment first to avoid two agents sharing its workspace.
The launcher tears down any previous **same-project** containers before recreating
them. It takes an exclusive host lock for the full foreground lifetime.

Ctrl-C or SIGTERM runs `compose down --timeout 40` **without `-v`**. Application exit
also tears down the stack. There are no automatic restart policies: do not use
`docker start`, `docker restart`, `compose up`, or a restart manager directly.
Every start must go through the launcher because a recreated namespace needs fresh
rules and a fresh readiness marker. SIGKILL, host failure, or Docker API failure can
prevent teardown; stop surviving containers before recovery. An interrupted Steel
profile may fail closed; inspect it offline rather than deleting its dirty markers.
The named `talon-browser_steel-profile` volume survives normal shutdown.

## Operator configuration and tools

Add these settings to the environment file passed to `--env-file` (the single
quotes preserve JSON as one `.env` value):

```dotenv
TALON_BROWSER_OPERATOR_ID=alice
TALON_BROWSER_IDENTITIES='{"telegram":"123456789","whatsapp":"15551234567@s.whatsapp.net"}'
```

Replace the synthetic sender IDs with exact provider-authenticated sender IDs and
omit unused providers. Both values are mandatory Compose interpolation inputs.
The overlay sets `TALON_BROWSER_ENABLED=true`,
`TALON_BROWSER_CONTROL_URL=http://172.30.12.3:8081`, and
`TALON_BROWSER_TOKEN_FILE=/run/browser/service-token` for Talon. The launcher creates
the token; do not put its value in `.env`. Base-only Talon remains browser-disabled.
The bridge receives the same operator mapping and token file.

The host binds provider, sender, conversation, fresh run ID, and background status;
model arguments cannot choose them. The mapping is operator-managed startup
configuration, frozen by the coordinator, not inferred from channel history.
**Design deviation:** there is no duplicate SQLite operator mapping. Update the
configuration and restart through the launcher to change it. An allowed channel
sender is not automatically an authorized browser operator.

`browser_cdp(method, params, session_id)` sends raw CDP commands: use `Target.*` for
tabs and flattened sessions, `Page.navigate`, `Runtime.evaluate` for DOM/JavaScript,
`Input.*` for interaction, and explicit `Page.captureScreenshot` for base64 captures.
Results are labeled untrusted observations. Raw CDP is powerful, not a method-level
sandbox; do not treat page content as authority or capture login screens automatically.
Uploads via `DOM.setFileInputFiles` and downloads via `Browser.setDownloadBehavior`
or `IO.read` use browser-side paths/streams. There is **no seamless local file
transfer** and no Talon workspace mount in Steel or the bridge.

`browser_request_handoff` fences new commands, drains pending work, and closes the
transport before pausing. Foreground calls return `viewer_unavailable`; background
calls return `human_required`, without a viewer URL. No interactive human takeover
or resume endpoint is exposed yet.

One global lease serializes browser ownership across conversations, providers,
foreground runs and background work. Synchronous subagents inherit the invocation;
background work has its own run identity and competes for the same lease. Conflicts
return busy rather than queueing or stealing ownership. Normal invocation cleanup
releases its lease, reconciling a lost handoff response through bound inspection and
fresh-version release (at most two reconciliation attempts within 35 seconds).
Cleanup is shielded from repeated task cancellation. A never-human handoff remains
paused until release or TTL expiry; once human control has occurred, paused expiry
retains ownership. No expiry silently resumes agent control. Failed drain, transport
failure or uncertain closure latches `FAILED` and denies reacquisition until bridge
restart. Restart through the launcher, not by bypassing its firewall setup.

Default TTL is 30 minutes, with no automatic background extension. The HTTP command
replay ledger caps each lease at 256 commands (action requests have a separate
256-entry cap); command IDs are not replayed or silently retried. Limits include
32 pending commands, 4 MiB payload/buffer bounds, 10-second ordinary command and
30-second `Page.navigate` deadlines. The external WebSocket transport shares the
pending/payload/deadline limits but does not use the HTTP command ledger; the
256-command cap is not a universal raw-WebSocket lifetime budget.

## Isolation and startup contract

| Network | Addresses | Connectivity |
| --- | --- | --- |
| control `172.30.12.0/24` | Talon `.2`, bridge `.3:8081` | Internal only |
| viewer `172.30.13.0/24` | future tunnel `.2`, bridge `.3:8080` | Internal only |
| steel `172.30.14.0/24` | Steel `.2:3000`, proxy `.3:8080`, bridge `.4` | Internal only |
| browser-internet | proxy only | Public browsing egress |
| default | Talon only | Talon's normal API access |

No host ports are published. Bridge must bind **only** `172.30.12.3:8081` for
control and **only** `172.30.13.3:8080` for viewer, never `0.0.0.0`. Linux's weak-host
network model and multi-interface routing still require negative connectivity
verification; distinct listener binds are necessary, not a universal host firewall.
Only Steel mounts `/var/lib/steel/profile`. Neither the bridge nor Talon sees it.

The launcher creates a private mode-0700 directory in `/dev/shm` with a fresh
`secrets.token_urlsafe(32)` service credential: file mode 0400, UID/GID 1000. Only
Talon and bridge bind the individual token file read-only, at
`/run/browser/service-token`. Token values never appear in environment variables,
command arguments, or launcher output. The host directory is not mounted wholesale.
Only its path is passed via `BROWSER_RUNTIME_DIR`; separate read-only readiness
directories go to Steel and proxy. Files are removed on normal exit, signals, and
startup failures, even if `compose down` fails. Container application logging must
independently uphold the same no-secret policy. The launcher suppresses child output
rather than risking `.env` interpolation or application logs reaching its terminal.

Startup is `compose up --no-start --build --force-recreate`, then Steel alone,
namespace PID inspection, deny-first firewall installation, then the gated proxy and
its firewall. Only after **both** policies succeed does the host create the markers.
Steel's existing supervisor waits at most 60 seconds before starting Node/Chromium;
the proxy has an equivalent stdlib gate. A slow or failed setup fails closed.
The final foreground `compose up --no-recreate --abort-on-container-exit` starts
bridge and Talon with health dependencies.

Host binaries run with `nsenter --target PID --net`: only the network namespace
changes. Steel OUTPUT permits established/related replies, new TCP to
`172.30.14.3:8080`, and new loopback TCP 9222 for API/CDP. Everything else drops,
including direct DNS, UDP/QUIC, management port 3000, control/viewer, and IPv6.
The local Steel health probe uses `/json/version` on 9222, not blocked port 3000.

Proxy OUTPUT permits established replies, Docker resolver DNS at `127.0.0.11`
(using conntrack original destination/port because Docker DNAT changes the DNS
port), and new public TCP 80/443 after private, shared, loopback, link-local,
documentation, benchmark, multicast, reserved, and metadata exclusions. No direct
upstream DNS exception is granted: Docker's embedded resolver must forward queries
outside the proxy namespace; verify this on the deployment host. IPv6 is entirely
dropped. `egress.py` also validates every DNS answer using `ipaddress` and connects
to the vetted IP without a second lookup. The host policy is defense in depth,
not a replacement for that application validation.

## Integration and remaining gates

- Compose runs `node bridge.mjs`, not the legacy Python `main.py`. It consumes
  `TALON_BROWSER_CONTROL_HOST/PORT`, `TALON_BROWSER_VIEWER_HOST/PORT`,
  `TALON_BROWSER_STEEL_URL`, `TALON_BROWSER_TOKEN_FILE` and the operator settings.
  Steel's HTTP and WebSocket upstream addresses are fixed, not model-selectable.
- Both `/health` endpoints report live bridge **and** successful Steel
  `GET /v1/sessions` readiness. The Compose probe checks both without a credential.
  This is not an active CDP command or public-navigation health check.
- Authenticated control routes are `GET /internal/browser/status` (initial mode
  `IDLE`), `POST /internal/browser/actions` (acquire/release/handoff/inspect), and
  `POST /internal/browser/command`. Commands carry host owner, lease ID, generation,
  version and unique request ID alongside method, params and optional `session_id`.
  Private `inspect` requires the exact owner, lease ID, generation and request ID,
  but not a current version; it returns only the matching lease status without a
  coordination transition. Release still requires current-version CAS and cannot
  release `HUMAN` control or clear `FAILED`.
- External CDP clients may upgrade only `GET /internal/browser/cdp` on control,
  with `Authorization: Bearer <service-token>`, `X-Browser-Lease`,
  `X-Browser-Generation`, and `X-Browser-Owner` (base64-encoded owner JSON containing
  `operator_id`, `provider`, `sender_id`, `conversation_id`, `run_id`, `background`).
  Acquire the lease first via actions. These are private host credentials, never
  URL parameters or model-visible output. The transport forwards CDP events and
  correlates responses; HTTP and external transports cannot share a lease.
  Release through actions before disconnecting, or unexpected disconnect fences
  the lease. The viewer listener denies all upgrades and all routes except health.
- Profile initialization changes mode as the inode owner before transferring
  ownership; `FOWNER` is not required.
- Read-only Steel rootfs has writable tmpfs `/tmp`, `/files`, `/app/api/logs` and
  1 GiB `/dev/shm`. These runtime flags passed live Chromium/API startup.
  Both helper containers use the **same pinned Steel base digest** as Steel,
  `f9a4648883dc06c402f5ffbec1c906bf9a803b5b737a1347de4e0aa0ca8d944a`.
  The proxy runs Python; the bridge runs Node with the image's existing `ws`
  dependency, without added dependencies.
- **Tunnel and authenticated viewer wiring are deferred.** There is intentionally
  no runnable tunnel service, ngrok image dependency, token, or unpinned placeholder.
  A future viewer phase must add a separate optional `browser-tunnel` profile,
  require a digest-pinned `TUNNEL_IMAGE`, attach only viewer `.2` plus its own internet
  network, and mount operator-supplied ngrok configuration from a host runtime file.
  Its upstream is `http://172.30.13.3:8080`; never control, Steel, or the service-token
  mount. Do not configure a tunnel token before that authentication review.

Known stock Fastify advisory risk is accepted and unchanged. API and browser share
UID 1000; Chromium uses `--no-sandbox`. Loopback CDP is an explicit exception, not
isolation from a hostile CDP client. Persistent cookies/site storage remain sensitive.
Before any real credentials, end-to-end tests must verify denied private/metadata
and rebound DNS destinations, IPv6/UDP denial, tunnel inability to access control,
actual health semantics, persistence across graceful stop, signal teardown,
capability dropping, and absence of credential/capture leakage.

## Focused checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s examples/talon/browser/tests/unit_tests -p test_deploy.py
TALON_BROWSER_OPERATOR_ID=synthetic-operator \
  TALON_BROWSER_IDENTITIES='{"integration":"synthetic-sender"}' \
  BROWSER_RUNTIME_DIR=/dev/shm/config-check TALON_ENV_FILE=/dev/null \
  docker compose --env-file /dev/null \
  -f examples/talon/docker-compose.yml -f examples/talon/browser/compose.yml config --quiet
```

These checks require no external services; unit lifecycle tests use fake subprocesses.
Credential-owner tests require root and otherwise skip. Compose configuration
validation alone does not prove namespace firewall behavior or runtime readiness.

Run the real isolated deployment probe with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 examples/talon/browser/tests/integration_tests/check.py
```

It creates disposable containers, networks and a synthetic profile, substitutes a
credential-free sleeping helper for Talon, and removes only its own resources.
The latest sandbox run passed credential rotation, readiness gating, real namespace
firewall installation, read-only Steel startup, bridge health, route/upgrade denial,
control authentication, and direct-egress isolation. The actual Compose Node bridge
also passed `IDLE` → acquire → data-page target creation/flattened attachment →
`Runtime.evaluate` of synthetic DOM → release → `IDLE`, before public probing.
The synthetic tab is removed with the disposable profile at project teardown:
closing it immediately exposed a stock Steel `TargetCloseError` in
`Page.addScriptToEvaluateOnNewDocument` during asynchronous target initialization,
which exited Steel. Immediate target-close stability remains a gap, not a pass.
**The overall probe failed:** the latest run resolved `example.com` to
`100.64.0.10`, which is deliberately denied.
HTTP CONNECT returned `403 Proxy Error`; Chromium returned
`net::ERR_TUNNEL_CONNECTION_FAILED`. Do not exempt shared/private ranges to make
this test pass. Public browsing must be verified on a compatible Linux host before
this deployment is considered complete. The sleeping Talon fixture does not prove
actual model invocation, channel delivery or native tool registration. Separate live
Node CDP tests are complementary, not a substitute for this Compose test.

Separate Steel lifecycle tests passed synthetic persistent-cookie recreation,
profile locking/corruption rejection and graceful closure. Unit tests cover mixed
DNS answers and subsequent redirect destinations; a live rebinding/redirect test,
host reboot, actual channel delivery, free tunnel, mobile viewer and complete
capture hygiene remain outstanding. No end-to-end success or real-credential
suitability is claimed. DNS resolution runs in bounded child processes; stalled
lookups are terminated and reaped rather than consuming workers indefinitely.
