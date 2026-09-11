# Steel profile foundation

Synthetic-only foundation, not yet approved for real credentials or untrusted browsing.
The base digest is pinned to Steel provenance commit
`2b41124d8e2953b0afe355c534e3c9aa71edae26`. The accepted stock Fastify
5.8.5 moderate advisory is unchanged. No dependencies are added.

## Run

```sh
docker build -t talon-steel-foundation examples/talon/browser/steel
docker run --name talon-steel --network none --cap-drop ALL \
  --cap-add CHOWN --cap-add SETUID --cap-add SETGID --cap-add SETPCAP \
  --security-opt no-new-privileges \
  --mount type=volume,source=talon-steel-profile,target=/var/lib/steel/profile \
  talon-steel-foundation
docker stop -t 35 talon-steel
```

Do not publish management ports. Attach only to the caller's isolated browser network.
Only Steel may mount the profile volume; never mount it into the agent container.
Network topology and volume exclusivity across services remain the caller's responsibility.
The wrapper intentionally does not start the upstream nginx/desktop entrypoint;
API is port 3000, Chromium CDP is 9222, and there is no 9223 redirect.

Deployment must set `STEEL_REQUIRE_EGRESS_READY=true` and mount a host-controlled,
read-only `/run/steel-network` directory. The supervisor waits at most 60 seconds
for its `ready` file before launching Node or Chromium; timeout fails closed.
The host must install namespace firewall rules before creating the marker and
remove stale markers before each container start. Direct tests default to false.
This wrapper does not install firewall rules. Loopback TCP 9222 is required for
API-to-CDP communication; inspection found no API startup request to loopback
3000. Prefer health probes from the trusted bridge rather than allowing new
loopback 3000 connections. Firewall health traffic cannot be distinguished from
browser traffic by UID; fixed proxy flags prevent normal page HTTP bypass but
are not protection against hostile CDP clients.

Create sessions with `{}` or configure `sessionId`, `timeout`, `blockAds`,
`dimensions`, `userAgent`, and `timezone`; omit `persist` and `userDataDir`.
Non-foundation options are rejected rather than silently enabling credential,
extension, alternate-profile, Selenium, or log-sink features.

The directory inode is exclusively flocked before validation or permission writes.
A new empty mount is initialized to UID/GID 1000 and mode 0700. Existing profiles
are checked for dirty markers, singleton locks, symlinks, malformed Local State
and Preferences, unclean exit preferences, and Cookies SQLite `quick_check` errors.
Invalid profiles fail with a structured error and are not replaced or repaired.
A marker remains on abnormal termination; investigate offline with the browser
stopped rather than automatically removing it. Validation does not prove every
Chromium database is healthy. Keep backups encrypted and access-controlled.

The Python lease supervisor drops to UID 1000 after spawning Node through
`setpriv` with an empty capability bounding set. Both API and Chromium use UID 1000.
On TERM/INT the wrapper waits for launch, closes Chromium through Puppeteer,
removes the dirty marker only after close succeeds, and exits without relaunch.
The public Steel session-release endpoint also closes without launching an idle
browser. This is NOT bridge lease release: releasing a bridge lease must leave
the Steel session/browser alive and must not call the session-release endpoint.
The scoped integration test exercises only explicit Steel session destruction.
An already UID-1000 supervisor is supported on a UID-1000 profile mount without
root transitions. TERM/INT handlers are installed before preparation/spawn and
signals during spawn are forwarded once the child is assigned. A shutdown timeout
fails dirty. SIGKILL is deliberately not recoverable automatically.

## Capture controls and remaining security gates

The bootstrap imports actual packaged class exports before importing `index.js`;
it does not edit or search/replace upstream JavaScript. It disables target
instrumentation attachment (network, bodies, console, CDP commands, interaction
listeners), discards instrumentation records, strips extension launch arguments,
disables extensions, and disables application Pino logging. Chromium managed
policies disable saved passwords, autofill, browser sign-in and sync. Profile
exports are rejected. These are scoped controls, NOT a guarantee that secrets
never reach disk: cookies and site storage are intentionally persistent, caches,
crash reports and other Chromium state need further review. Live view/screenshot,
action and file endpoints are still upstream functionality and must be isolated.

The proxy is fixed to `http://172.30.14.3:8080`; Chromium's implicit loopback
proxy bypass is disabled and QUIC is disabled. The managed URL blocklist adds
basic local/file restrictions but is not a complete private-IP or DNS-rebinding
filter. Before real credentials, the external proxy MUST validate scheme, ports,
all DNS answers and redirect destinations, reject private/link-local/metadata IPs,
and connect to the validated IP. Firewall all direct browser egress, including
IPv6/UDP, and deny management access from browser processes. These controls are
not implemented or verified by this directory alone.

**UID isolation remains unresolved:** unprivileged Puppeteer cannot launch a
child under a different UID after capabilities are dropped. A root/capability
retaining launcher or a separate browser container with an authenticated broker
would be required. This implementation deliberately does not retain a privileged
launcher. UID-based firewall rules cannot distinguish its API and browser.
Do not treat proxy flags as an isolation boundary against hostile CDP clients.
Chromium currently runs with `--no-sandbox`, matching the container restriction;
container/network isolation is mandatory and Chromium sandbox enablement remains
an additional hardening gate.

## Focused verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s examples/talon/browser/steel/tests/unit_tests
PYTHONDONTWRITEBYTECODE=1 python3 examples/talon/browser/steel/tests/integration_tests/check.py
```

Integration builds the image and uses a unique disposable synthetic profile
with `--network none`, without publishing ports or contacting real websites.
It verifies no Node process starts before the readiness marker, actual Chromium
`/proc` arguments (not just wrapper options), fixed proxy/profile and disabled
extensions/QUIC, localhost HTTP including management ports plus private/metadata
navigation rejection with an absent proxy, ownership/mode, persistent synthetic
cookie across graceful restart into a non-root container, concurrent lease
rejection, configured session creation, explicit Steel release without idle
relaunch, and corrupt JSON rejection. Chromium launches its actual binary rather
than Debian's shell wrapper, which otherwise injects extension arguments.
It removes only its own test containers/volume. It does not verify a production
proxy, private-IP filtering, all capture channels, or different browser/API UIDs.
