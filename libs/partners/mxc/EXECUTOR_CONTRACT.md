# MXC executor contract (observed)

This file records what the MXC command-line executor actually does on this
machine. The parsing in `langchain_mxc/_executor.py` and
`langchain_mxc/sandbox.py` is based on these observations, not on the docs.

| Item | Value |
| --- | --- |
| Date | 2026-11-05 |
| Host | macOS 26.7.2 (build 25G332), Apple Silicon (arm64) |
| Package | `@microsoft/mxc-sdk` 1.0.1 from npm (`npm install @microsoft/mxc-sdk`) |
| Executor | `node_modules/@microsoft/mxc-sdk/bin/arm64/mxc-exec-mac` (Mach-O arm64, 3.9 MB) |
| Backend | Seatbelt (`sandbox_init`), selected automatically by `"containment": "process"` |
| Node | 26.3.0, npm 11.16.0 |

Everything below was produced by running the executor directly. The Linux and
Windows executors (`lxc-exec`, `wxc-exec.exe`) could not be run here; the
notes about them come from `strings` on the bundled binaries and from the
Node SDK source, and are marked as such.

## Differences from the brief

| Brief said | Observed on macOS 1.0.1 |
| --- | --- |
| `--probe` and `--available-backends` are useful flags | `mxc-exec-mac` has neither. `--probe` is a `wxc-exec.exe` (Windows) flag only ("Run the fallback detector and emit JSON, without spawning a sandbox"; it also takes a config). `--available-backends` is an `lxc-exec` (Linux) flag only ("Report host backend availability as JSON and exit"). All three executors accept `--dry-run`, which is what the package uses as the portable validation step. |
| There are JSON schema files in the npm package | None. The package ships `dist/generated/v1_0_0/wire.d.ts`, a TypeScript rendering of the 1.0.0 contract. The request shape below was built from it and then confirmed against the executor's own "unknown field" errors. |
| A policy rejection exits 1 with a `policy_validation` code | A malformed request exits 1 with plain text on stderr starting `Request error`. No JSON and no `policy_validation` string appears on macOS. The string `policy_validation` exists inside all three binaries, so another code path (probably Windows) may emit it. |
| `process.env` is required | It is optional. Omitting it gives a default environment (`HOME=<cwd>`, `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, `TERM`). A supplied array is used verbatim. There is an undocumented `process.inheritDefaultEnv` boolean that layers `env` over the default. |
| The sandboxed program needs explicit read access to `/bin/sh` and friends | On macOS the generated Seatbelt profile has a baseline that already allows reading `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`, `/usr/lib`, `/usr/libexec`, `/usr/share`, `/System`, `/Library`, `/private/etc`, the dyld and timezone caches, and read-write on `/dev/null`, `/dev/zero`, `/dev/random`, `/dev/urandom`, `/dev/tty`. `readonlyPaths: []` still runs `/bin/sh`, system `python3` and `git`. Anything under Homebrew or a user-installed toolchain must be listed. |
| Node SDK calls the executor | The v1 Node SDK runs sandboxes in-process through `libmxc_ffi.dylib`; only its legacy helper spawns `mxc-exec-mac`, and it passes `--config-base64`. The executor is still a supported entry point, but Microsoft should confirm it stays in the npm bundle. |

## Command line

```
Usage: mxc-exec-mac [OPTIONS] [CONFIG_PATH]

Arguments:
  [CONFIG_PATH]  Path to config JSON file (positional)

Options:
      --config <CONFIG>                Path to config JSON file
      --config-base64 <CONFIG_BASE64>  Base64-encoded JSON config
      --debug                          Enable debug/console output
      --experimental                   Enable experimental features
      --dry-run                        Parse and validate config then exit without executing
      --log-file <LOG_FILE>            Path to diagnostic log file (appends, creates if missing)
  -h, --help                           Print help
```

Observed facts about the flags:

- `--config <file>`, a positional path and `--config-base64` all behave identically.
- An unknown flag such as `--probe` exits 2 with a clap usage error on stderr.
- No arguments at all exits 1 with `Error: No config provided. Use a positional path, --config, or --config-base64` on stderr.
- `--debug` writes the generated Seatbelt profile and runner diagnostics to **stdout**, interleaved with workload output. The package never passes it.
- `--dry-run` writes diagnostics and `Dry run completed. Result: validation passed` to stdout and exits 0. It validates the JSON, the schema and backend availability, but not the working directory (a nonexistent `cwd` passes dry-run and fails at spawn).
- A policy the backend cannot enforce fails dry-run with **exit 255** and the reason on **stdout**, for example `Dry run completed. Result: validation failed: macOS Seatbelt cannot enforce network.ingress.hostLoopback='allow' alongside network.ingress.default='deny': ... Set network.ingress.default='allow' as well, or set hostLoopback='deny'. Note that an omitted 'hostLoopback' is 'deny', not an inherit of 'default'.` The same policy without `--dry-run` fails the same way before anything runs.
- A listed path that does not exist produces a stdout warning before the diagnostics, in both dry and real runs: `WARNING: filesystem path '<path>' (in 'readwritePaths') does not exist on the host; the backend may fail at mount time`. Execution continues.
- With `egress.default: allow`, Seatbelt opens the network fully: a loopback `bind` and an HTTPS fetch both work even with `ingress` omitted (which means deny). The package therefore treats `allow_network=True` as egress, ingress and host loopback all `allow`, which every backend accepts.

Flags seen in the other executors (from `strings`, not run):

| Executor | Flags |
| --- | --- |
| `lxc-exec` | `--config`, `--config-base64`, positional, `--debug`, `--dry-run`, `--log-file`, `--available-backends`, `--delete` + `--containername`, `--experimental`, `--setup-hyperlight` |
| `wxc-exec.exe` | `--config`, `--config-base64`, positional, `--debug`, `--dry-run`, `--log-file`, `--probe`, `--audit`, `--audit-verbose`, `--delete` + `--containername`, `--operation`, `--container-id`, `--setup-wslc`, `--telemetry-consent`, `--force-reclaim`, and `-- <command>` to override `process.commandLine` |

`--audit` runs the sandbox in permissive mode and must never be used. The
package has no code path that emits it, `--debug`, `--experimental`, or any
`learningMode` / `captureDenials` field.

## Request shape (contract version 1.0.0)

From `dist/generated/v1_0_0/wire.d.ts` (`OneShotRequest`). The schema is
closed; every unknown field is rejected with the list of accepted fields.

```json
{
  "version": "1.0.0",
  "containment": "process",
  "process": {
    "commandLine": "/bin/sh -c 'echo hello'",
    "cwd": "/abs/path/to/project",
    "env": ["PATH=/usr/bin:/bin", "HOME=/abs/scratch"],
    "timeout": 120000
  },
  "filesystem": {
    "readwritePaths": ["/abs/path/to/project", "/abs/scratch"],
    "readonlyPaths": ["/opt/homebrew"],
    "deniedPaths": ["/Users/me/.ssh"]
  },
  "network": {
    "egress": { "default": "deny" },
    "ingress": { "default": "deny", "hostLoopback": "deny" }
  },
  "ui": { "disable": true, "clipboard": "none", "injection": false }
}
```

Accepted top-level fields: `$schema`, `_comment`, `version`, `containerId`,
`containment`, `lifecycle`, `process`, `filesystem`, `fallback`, `network`,
`ui`, `processContainer`, `lxc`, `seatbelt`, `runtimeConfig`, `telemetry`,
`wslc`. Accepted `process` fields: `commandLine`, `cwd`, `env`,
`inheritDefaultEnv`, `timeout`.

Observed details:

- `version` must be exactly one of the registered versions. `0.8.0-alpha` and `1.1.0` are rejected with `Unsupported contract version. Registered versions are: 0.9.0-alpha, 1.0.0, 1.1.0-alpha`.
- `containment` may be omitted, `"process"` or `"seatbelt"`; all three select Seatbelt here. `"bubblewrap"` and `"processcontainer"` exit 1 with `error: the 'bubblewrap' backend is not available on macOS` (same with `--dry-run`).
- `commandLine` is split like a shell: `/bin/sh -c 'echo a; echo b'` spawns `/bin/sh` with two arguments. The package wraps every command as `/bin/sh -c <shlex-quoted command>` on POSIX.
- `process.timeout` is in milliseconds, must be an integer (`"10"` is rejected: `invalid type: string "10", expected u32`). `0` or omitted means no timeout.
- `cwd` is used as-is. A nonexistent `cwd` fails at spawn with exit 255 and a JSON error envelope (below).
- `env` is used verbatim; the shell adds `PWD`, `SHLVL` and `_`. `env: []` gives an empty environment.
- Relative paths and nonexistent paths in `readwritePaths` are accepted silently and have no effect. The package only sends absolute paths that exist.
- Listing `/dev` in `readonlyPaths` emits `(deny file-write* ... (subpath "/dev"))`, which overrides the baseline and makes `>/dev/null` fail with `Operation not permitted`. The package never lists `/dev`.
- Rules are emitted shallow-to-deep, so a deeper path wins: `deniedPaths: ["~/.ssh"]` with `readonlyPaths: ["~"]` leaves `~/.zshrc` readable and `~/.ssh/config` blocked (verified).
- Read-write paths also get `network-bind` and `network-outbound` for Unix sockets under those paths.
- The workload runs as the same uid as the executor (`id -u` = 501 inside the sandbox). There is no user switching.
- stdin: the workload inherits the executor's stdin. With a terminal attached, `cat` blocks until the timeout. With `</dev/null` it reads EOF immediately. The package always passes `stdin=DEVNULL`.
- Output is not buffered or truncated by the executor: 2,000,000 bytes on stdout and 300,000 bytes on stderr passed through unchanged.

## Exit codes and streams

| Case | Command inside sandbox | Exit | stdout | stderr |
| --- | --- | --- | --- | --- |
| Success | `echo hello; echo "to stderr" >&2; ls` | 0 | `hello\nhello.txt\n` | `to stderr\n` |
| Non-zero exit | `echo before; echo "bad thing" >&2; exit 3` | 3 | `before\n` | `bad thing\n` |
| Exit code > 255 | `exit 300` | 44 | empty | empty |
| Command not found | `nonexistent_cmd_xyz` | 127 | empty | `/bin/sh: nonexistent_cmd_xyz: command not found` |
| Workload killed by signal | `kill -9 $$` or `kill -15 $$` | 255 | output before the kill | empty (no envelope) |
| Timeout (`process.timeout` 2000) | `echo started; sleep 30` | 255 | `started\n` | `{"error":{"code":"backend_error","message":"script timed out after 2000ms"}}` |
| Timeout with a backgrounded child | `echo started; (sleep 60 &); sleep 30` | 255 | `started\n` | same envelope; no `sleep 60` survives, so the executor kills the whole tree |
| Write outside allowed folders | `echo x > /outside/new.txt` | 1 (the shell's) | empty | `/bin/sh: /outside/new.txt: Operation not permitted` |
| Write to a read-only path | `echo x > /usr/share/x.txt` | 1 | empty | `/bin/sh: /usr/share/x.txt: Operation not permitted` |
| `mkdir` outside | `mkdir /outside/newdir` | 1 | empty | `mkdir: /outside/newdir: Operation not permitted` |
| List a directory outside | `ls /outside` | 1 | empty | `ls: /outside: Operation not permitted` |
| Read `~/.ssh` (not listed) | `ls ~/.ssh; cat ~/.ssh/config` | 1 each | empty | `ls: /Users/me/.ssh: Operation not permitted`, `cat: /Users/me/.ssh/config: Operation not permitted` |
| Read `~/.ssh` (`deniedPaths`, `~` read-only) | same | 1 each | empty | identical messages; `~/.zshrc` is readable |
| Read a file that does not exist | `cat ~/.ssh/nope` | 1 | empty | `cat: ...: No such file or directory` (existence is visible, contents are not) |
| Python hitting a denied path | `import` from an unlisted directory | 1 | empty | `PermissionError: [Errno 1] Operation not permitted` |
| Network denied, curl | `curl -sS -m 8 https://example.com` | 6 | empty | `curl: (6) Could not resolve host: example.com` |
| Network denied, Python | `urllib.request.urlopen(...)` | n/a | `URLError <urlopen error [Errno 8] nodename nor servname provided, or not known>` | empty |
| Network denied, DNS tool | `nslookup example.com` | 0 | `bind: Operation not permitted` / `isc_socket_bind: unexpected error` | empty |
| Network allowed | `curl -m 15 https://example.com` | 0 | HTTP 200 | empty |
| Malformed JSON | `{ this is not json` | 1 | empty | `Request error\nConfiguration parse error: Invalid version declaration: key must be a string at line 1 column 3` |
| Missing `version` | | 1 | empty | `Request error\nConfiguration parse error: Invalid version declaration: missing field `version` at line 47 column 1` |
| Old `version` | `"0.8.0-alpha"` | 1 | empty | `Request error\nConfiguration parse error: Unsupported contract version. Registered versions are: 0.9.0-alpha, 1.0.0, 1.1.0-alpha` |
| Unknown top-level field | `"bogusField": 1` | 1 | empty | `Request error\nConfiguration parse error: Invalid one-shot request: Invalid configuration at `bogusField`: unknown field `bogusField`, expected one of ...` |
| Unknown nested field | `process.bogus` | 1 | empty | `... Invalid configuration at `process.bogus`: unknown field `bogus`, expected one of `commandLine`, `cwd`, `env`, `inheritDefaultEnv`, `timeout` ...` |
| Empty `commandLine` | `""` | 1 | empty | `... Invalid configuration at `process.commandLine`: string must not be empty ...` |
| Missing `process` | | 1 | empty | `... Invalid configuration: missing field `process` ...` |
| Bad enum value | `network.egress.default: "maybe"` | 1 | empty | `... unknown variant `maybe`, expected `allow` or `deny` ...` |
| Wrong backend for OS | `"containment": "bubblewrap"` | 1 | empty | `error: the 'bubblewrap' backend is not available on macOS` followed by `Script code length:`, `Working directory:`, `Script timeout:`, `Container name:` lines |
| Nonexistent `cwd` | | 255 | empty | `{"error":{"code":"backend_error","message":"failed to spawn sandboxed process: No such file or directory (os error 2)"}}` |
| Config file missing | `--config /nope.json` | 1 | empty | `Request error\nConfiguration parse error: Configuration file not found: /nope.json` |
| `--dry-run`, valid | | 0 | `Script code length: 49\nWorking directory: ...\nScript timeout: 10000\nContainer name: \nRunner completed in 0ms\nDry run completed. Result: validation passed` | empty |
| `--dry-run`, wrong backend | | 1 | empty | same as the wrong-backend case |

Timing: a trivial command completes in about 30 to 50 ms end to end.

### Three classes of executor output

1. **Workload ran.** Exit code is the workload's (mod 256, or 255 when killed by a signal). stdout and stderr are the workload's, byte for byte. Policy denials show up only as the tool's own error text (`Operation not permitted`, a Python `PermissionError`, `Could not resolve host`). The executor adds nothing.
2. **Request rejected before a sandbox was created.** Exit 1, empty stdout, plain-text stderr. Either `Request error\nConfiguration parse error: <detail>` or `error: the '<backend>' backend is not available on <os>` plus diagnostic lines.
3. **Runtime failure inside the executor.** Exit 255, stdout holds whatever the workload printed, and the **last line** of stderr is a JSON envelope `{"error":{"code":"backend_error","message":"..."}}`. Seen for timeouts and spawn failures. Workload stderr written before the failure precedes the envelope.

The package classifies a result in that order: host-side timeout first, then
class 2 (exit 1 with empty stdout and a recognised prefix), then class 3
(exit 255 with a JSON last line), else class 1.

### Process tree and host-side kills

The executor runs in the caller's process group; it puts the workload shell
in a new process group of its own (`pgid` = shell pid). The executor's own
`process.timeout` kills the whole workload tree. But if the **executor** is
killed from outside with SIGTERM or SIGKILL, the workload shell and its
children keep running as orphans. The package therefore:

- always sets `process.timeout`, so the executor does the primary kill;
- waits `timeout + 5 s` on the host, and only then kills the executor, every
  descendant found through `ps -axo pid=,ppid=`, and their process groups
  (`taskkill /T /F` on Windows).

## Toolchain requirements observed

| Tool | What it needed in `readonlyPaths` |
| --- | --- |
| `/usr/bin/python3` (Apple) | nothing beyond the baseline |
| `/usr/bin/git` (Apple, CommandLineTools shim) | nothing beyond the baseline (`/Library` is in the baseline) |
| `.venv/bin/python` (uv-managed CPython 3.12) | the venv directory and `sys.base_prefix` (`~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none`). The venv `bin` dir alone fails in `dyld` with `Library not loaded: @rpath/libpython3.12.dylib`. |
| `/opt/homebrew/bin/node` | all of `/opt/homebrew`. The Cellar prefix alone fails because node links against `/opt/homebrew/opt/llhttp/lib/libllhttp.9.4.dylib`. |

Hence the package adds, for each of `python`, `python3`, `git`, `node` found on
`PATH`: the directory containing it, the install prefix of the symlink target
(parent of `bin`), the running interpreter's `sys.prefix` and
`sys.base_prefix`, and the whole package-manager root when the tool lives
under `/opt/homebrew`, `/usr/local`, `/home/linuxbrew/.linuxbrew` or
`/nix/store`.

## Not verified here

- **Windows** (`wxc-exec.exe`): `--probe` output format, whether `SYSTEMROOT` and `LOCALAPPDATA` are really required, how `deniedPaths` behave in each AppContainer tier, the exact `Access is denied` text, and whether a policy rejection emits `policy_validation`.
- **Linux** (`lxc-exec` with bubblewrap): the `--available-backends` JSON, the error text for denied reads (bubblewrap simply does not mount the path, so the likely message is `No such file or directory`), and the rule found in the binary that rejects a config when `deniedPaths` is present and another listed path cannot be resolved.
- The Seatbelt network toggle with `ingress.hostLoopback: "allow"` is accepted; whether loopback is actually reachable when egress is allowed was not measured separately.

## UI policy (`ui` block), observed 2026-11-12 on macOS

MXC's documented defaults for the block are deny on all three fields
(`disable: true`, `clipboard: "none"`, `injection: false`); an omitted block is
full lockdown on the backends that enforce it (Windows ProcessContainer, macOS
Seatbelt). Linux bubblewrap does not implement the section. `clipboard` accepts
exactly `none`, `read`, `write`, `all`.

`MXCSandbox(allow_ui=..., clipboard=..., allow_input_injection=...)` maps 1:1
onto the block. Observed with `mxc-exec-mac` 1.0.1 and `--debug`:

| Request `ui` | Generated Seatbelt profile |
| --- | --- |
| defaults | no `ui enabled` / `clipboard enabled` sections; pasteboard lookup not allowed |
| `disable: false` | adds `;; --- ui enabled: allow WindowServer + LaunchServices ---` with `com.apple.windowserver.active` / `.session` |
| `clipboard: "all"` | adds `;; --- clipboard enabled: allow pasteboard ---` `(allow mach-lookup (global-name "com.apple.pasteboard.1"))` |

Dry-run and execution succeed with every combination. However, even with
`disable: false` and `clipboard: "all"`, `pbpaste` prints nothing (exit 0) and
`osascript -e 'the clipboard'` fails with
`Connection Invalid error for service com.apple.hiservices-xpcservice` and
`Can't get clipboard. (-1728)`. The profile opens the pasteboard port but the
other services those tools need stay denied, so on macOS 1.0.1 the clipboard
switch is accepted but not sufficient for the standard tools. The deny side is
enforced. Not verified: whether any macOS tool can read the clipboard under
the opened profile; Windows job-object enforcement of all three fields; the
documented Windows rule that `disable: true` makes the other two fields
ignored.

## Checked against Microsoft's published contract, 2026-11-12

Read from `microsoft/mxc` `docs/containment-policy/1.0.0/policy.md` and
`schemas/stable/mxc-config.schema.1.0.0.json`:

- `"containment": "process"` is valid and "resolves to ProcessContainer on
  Windows, Bubblewrap on Linux, or Seatbelt on macOS". The package keeps
  sending `process`.
- "Windows ProcessContainer refuses a supplied environment missing
  `SYSTEMROOT` or `LOCALAPPDATA`." The package always sends both. This closes
  that open question; behaviour itself is still not run on Windows.
- `deniedPaths`: "Enforcement and interactions with grants are
  backend-specific." Still an open question for Windows and Linux.
- The Windows shell wrap is `cmd.exe /d /s /c "<command>"` (changed from
  `/d /c <command>` on this date, so commands that begin with a quote are not
  mangled). Not yet run on Windows.

## Linux (`lxc-exec`, bubblewrap), observed 2026-11-12

Host: Debian 12 container (`python:3.12-bookworm`) on Docker Desktop, kernel
6.12 aarch64, bubblewrap 0.8.0, Node 24, `@microsoft/mxc-sdk` 1.0.1, tests run
as a non-root user. The npm bundle ships an arm64 `lxc-exec`.

- **Result:** 86 unit and 88 integration tests pass (18 skipped: the 16
  inherited cases, the `~/.ssh` test because the container has none, and the
  Seatbelt-only profile test).
- **Containers:** in a default Docker container every command fails with
  `bwrap: Can't mount proc on /newroot/proc: Operation not permitted` (exit 1,
  empty stdout). `--privileged` is needed. The package classifies a `bwrap: `
  start failure as a backend error, and the probe's liveness step now fails
  closed with that reason instead of passing.
- **Blocked paths read as missing, not refused.** bubblewrap does not mount
  what the policy omits: `ls: cannot access '/p': No such file or directory`,
  `touch: cannot touch '/p': No such file or directory`, and from dash
  `cannot create /p: Directory nonexistent`. On Linux the package adds the
  blocked hint when such a path is outside the policy and it, or its parent
  directory, exists on the host.
- **Network off:** `curl: (6) Could not resolve host`, same as macOS.
- **Network on:** `ingress.default: allow` is rejected at dry-run
  ("Bubblewrap: network.ingress.default='allow' is not supported ... Use
  network.ingress.default='deny'"). `allow_network=True` therefore now sends
  egress allow with ingress and host loopback deny on every OS; macOS accepts
  that shape and reaches the network (verified). Egress allow also requires
  `slirp4netns` and `ip` on `PATH`; without them dry-run fails with
  "network.egress requires 'slirp4netns' on PATH" / "requires 'ip' on PATH". Re-checked 2026-11-17 (Debian 12 container): it also requires `iptables` ("network.egress requires 'iptables', which was not found on PATH or in /usr/local/sbin, ..."); the package fails closed with that message.
  The container had `iproute2`, `iptables` and `nftables` installed when it
  passed; only `ip` was named as missing.
- `--available-backends` prints JSON: `[{"backend": "bubblewrap",
  "capabilities": ["proxyEnforcement"]}]`, with a `warnings` array when a
  helper tool is missing.
- Not verified: a non-container Linux host, x64, and `deniedPaths` against a
  real credential folder.

## Windows, first run: observed 2026-11-17

Host: Windows 11 26H1, build 28000.3501, ARM64. `wxc-exec.exe` 1.0.1 (arm64),
not elevated, `wxc-host-prep.exe` never run.

- A sandbox starts without host preparation. `--probe` reports
  `tier: base-container`, `needsDaclAugmentation: false`,
  `baseContainerSupportsDenyPaths: true`.
- `--dry-run` prints `Dry run completed. Result: validation passed`, as on macOS.
- The request shape built for macOS is accepted unchanged, including the
  `cmd.exe /d /s /c "<command>"` wrap and the env with `SYSTEMROOT` and
  `LOCALAPPDATA`.
- Writes outside the project and to the home folder are blocked. The text is
  `Access is denied.` with no path in it.
- Network off: `curl` reports `Could not resolve host`. Loopback is blocked
  too, so `ping 127.0.0.1` exits at once with code 1.
- **`ui.disable: true` stops some programs from starting.** `git --version`
  fails with `STATUS_DLL_INIT_FAILED (0xC0000142)`. The executor's exit code is
  the NTSTATUS (3221225794), not 255, and the JSON envelope follows the
  plain-text message on the same stderr line with no newline between them.
  `classify_result` handles that shape for exit codes at or above 0xC0000000.
- **`python -m pytest` crashes with `ui.disable: true`**: `Windows fatal
  exception: code 0xc06d007e` (delay-load module not found) inside
  `platform._wmi_query`, exit code 3228369022. `python --version` and plain
  scripts run. Opening the UI (`ui.disable: false`) fixes it (confirmed below).
- Output uses CRLF line endings; the package normalises them to LF.
- `--help` lists `--audit` ("permissive allow mode"). The package never passes it.
- Confirmed with `allow_ui=True` (`ui.disable: false`): `git`, `where`,
  `node`, PowerShell, `whoami`, Python `platform` and `ctypes`, and full
  `pytest` runs all work. With `ui.disable: true` only cmd built-ins, basic
  Python and `curl.exe` run. Package default stays `allow_ui=False` (decided
  2026-11-17: security first, documented as needed on Windows).
- Executor-side timeout works: a 30 s sleep with a 2 s limit returns in 2.7 s.
- `deniedPaths` works: a folder inside a read-only folder stays unreadable
  (`Access is denied.`).
- **Deleting with cmd's `del`/`erase` and PowerShell's `Remove-Item` is denied
  inside `readwritePaths`**, with or without the UI. These work in the same
  folder: Python `os.remove` (including files created outside the sandbox),
  `shutil.rmtree`, Node `fs.unlinkSync`, `git rm`, `git clean`, `rmdir /s /q`
  (including a folder with files), `ren`, `move /y`, overwrite. `attrib` on a
  file in the folder prints `Path not found` for the folder itself. Cause
  unknown; question for Microsoft. The package adds a hint naming working
  alternatives when a delete command gets `Access is denied`.
- File ACLs inside the sandbox show the folder's normal inherited entries
  (temp folders: `OWNER RIGHTS`; project folder: the user). No AppContainer
  entry is visible, so access is granted by the container, not by ACLs.
- Not yet observed on Windows: anything on x64 or on build 26100.
