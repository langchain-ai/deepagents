---
type: backend architecture concept
title: Backends, Virtual Filesystems, and Execution
description: How DeepAgents backend contracts separate virtual file operations, graph state, durable stores, routing, and host shell execution. Covers persisted UTF-8 normalization, LocalShellBackend process cleanup, cross-platform output capture, and why virtual paths are not a security boundary.
tags: [backends, virtual-filesystem, state, durable-storage, routing, sandbox, security, process-lifecycle]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-26T08:03:59.911Z
sources:
  - id: openwiki-source-6a9e782d8dda0eeaf00ce0d3
    resource: repo://libs/deepagents/deepagents/backends/_posix_process.py
  - id: openwiki-source-34e9cac87b7ccad2573a3565
    resource: repo://libs/deepagents/deepagents/backends/_windows_process.py
  - id: openwiki-source-a1549ea98d425efea270be93
    resource: repo://libs/deepagents/deepagents/backends/composite.py
  - id: openwiki-source-f84c83d6fab6028c94be90bc
    resource: repo://libs/deepagents/deepagents/backends/local_shell.py
  - id: openwiki-source-e3efb5f3e4a9e8517eb6d8f5
    resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-07f9eac13e71bcbdb4e6994b
    resource: repo://libs/deepagents/deepagents/backends/state.py
  - id: openwiki-source-21e2b0401425a427d8cea9c1
    resource: repo://libs/deepagents/deepagents/backends/store.py
  - id: openwiki-source-c972622237a22631e36f3625
    resource: repo://libs/deepagents/deepagents/backends/utils.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-ab8bfe29057131f7fc94fb78
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_composite_backend.py
  - id: openwiki-source-ff7e5dfd747aa11f43332e1b
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_file_format.py
  - id: openwiki-source-1f666eed797336e343fc47c8
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_local_shell_backend.py
  - id: openwiki-source-fe312e91906ecaaab36bf427
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_posix_process.py
  - id: openwiki-source-fb022ddbcc554eaabedfa8cd
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_state_backend.py
  - id: openwiki-source-e11958a6a68a079be57552fc
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_store_backend.py
  - id: openwiki-source-d969accd0b5fc0acdd77cd6e
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_windows_process.py
generated: { by: "openwiki/0.4.2", at: "2026-11-26T08:03:59.911Z" }
---

# Backends, Virtual Filesystems, and Execution

A backend is the storage and execution boundary behind agent file tools. `BackendProtocol` makes file operations portable across graph state, stores, host filesystems, and remote sandboxes; it does **not** imply that every backend can execute a process. `SandboxBackendProtocol` is the additional capability required for `execute()`. Consequently, `StateBackend` and `StoreBackend` can read, write, search, glob, and transfer files without offering a shell. `create_deep_agent()` chooses `StateBackend()` when no backend is supplied.

This is an important separation of concerns:

- **Backend routing** selects where a virtual path is stored.
- **Persistence scope** is determined by the selected backend and its configuration.
- **Permissions** are enforced by `FilesystemMiddleware` at the built-in tool layer, not by a direct backend call.
- **Execution isolation** is supplied only by the chosen execution backend. A virtual path prefix is not a sandbox.

See [Filesystem tools](/openwiki/concepts/tools-filesystem.md) for agent-facing tools, [State persistence](/openwiki/concepts/state-persistence.md) for checkpoint and store lifecycle, [permissions and HITL](/openwiki/concepts/permissions-hitl.md) for tool approval, and [security operations](/openwiki/operations/security.md) for deployment guidance.

## The capability contract

`BackendProtocol` uses structured result objects rather than requiring callers to parse command output. The common surface includes `ls`, windowed `read`, literal `grep`, `glob`, `write`, exact-string `edit`, `delete`, and upload/download batches. Results carry either data or an error; transfer responses are per input so a batch can partially succeed. The base protocol supplies asynchronous fallbacks, while implementations may provide native async I/O.

`ReadResult` also records pagination metadata. A valid returned text window has paired `start_line` and `end_line`; when supplied, `next_offset` is the line immediately after that window. Backends must tolerate a negative offset and non-positive limit rather than throwing. In the shared Python slicer, a zero limit returns `no_lines_requested=True`, distinguishing an uninspected window from an empty file.

Search semantics are backend-level rather than shell semantics: `grep` searches a literal substring (not a regular expression) and `max_count` is a total match cap. Glob patterns without `/` match a basename at any depth; slash-containing patterns are relative to the requested root, and a leading `/` anchors to that root. Backends report a refused glob such as one containing `..` as a structured error.

## File data and legacy compatibility

The current `FileData` form is a string plus an encoding and optional timestamps:

```python
{
    "content": str,
    "encoding": "utf-8" | "base64",
    "created_at": "...",  # optional
    "modified_at": "...",  # optional
}
```

Text uses UTF-8. Before persisting a `utf-8` value, the shared file-data constructors make the Python string UTF-8 encodable: a valid UTF-16 surrogate pair represented as two code points is rejoined, while each lone surrogate becomes U+FFFD. This matches the behavior of writing a JavaScript string and prevents later listing, reading, or downloading from failing on invalid Unicode. Base64 content is not changed by this normalization. Binary data is represented by base64 text with `encoding: "base64"`, enabling upload/download to round-trip bytes while text edit rejects binary files. File sizes in the state and store implementations are calculated from UTF-8 bytes of normalized content.

Older state or store records may use `content: list[str]`. The compatibility conversion joins the strings with `"\n"` (and maps an empty list to `""`). Reads, listing size calculation, searches, globs, and downloads can use that normalized view without rewriting the persisted legacy record. Values other than a string or a list consisting solely of strings raise `TypeError`; malformed persisted data is not silently treated as text.

```mermaid
flowchart TD
    Stored["Stored file content"] --> Type{"String or legacy string list"}
    Type -->|"String"| View["Use content"]
    Type -->|"Legacy list"| Join["Join lines for operation"]
    Join --> Observe["Read, list, search, glob, download"]
    Observe --> Keep["Leave stored legacy value unchanged"]
    View --> Change["Write or successful edit"]
    Join --> Change
    Change --> Current["Persist current string representation"]
```

This is a lazy migration path: observation is non-mutating, but a successful content mutation writes the current representation.

A write creates or replaces string content, normalizes it to UTF-8-encodable text, and records `utf-8`. A successful text edit applies the same normalization. An update retains an existing `created_at` when available and refreshes `modified_at`; an edit retains its existing encoding unless the file is rewritten as text. Failed edits do not persist a converted legacy value.

## State and store are different persistence scopes

| Backend | Owner and scope | Lifecycle and operational constraint |
| --- | --- | --- |
| `StateBackend` | The current graph's `files` state channel | Graph/thread-local state; checkpoint configuration controls survival between invocations. |
| `StoreBackend` | A LangGraph `BaseStore` namespace selected by a caller factory | Cross-thread persistence within that namespace; durability is that of the supplied store. |

### `StateBackend`: graph-local files

`StateBackend` may only operate in a LangGraph graph context. It reads `files` through `CONFIG_KEY_READ` with `fresh=True`, which applies pending channel writes before the read. It sends partial updates through `CONFIG_KEY_SEND`; the channel's dictionary merge preserves unrelated keys, and subsequent reads in the same superstep see the queued changes before the node boundary commits them. A delete queues `None` for the exact path and every descendant under `path + "/"`.

There is deliberately no private in-memory fallback outside a graph: missing execution configuration produces an actionable `RuntimeError`. Seed files by passing `files` in invocation state. Use this backend for files whose natural lifetime is the agent thread, not for shared durable records.

### `StoreBackend`: durable namespace-scoped files

`StoreBackend(namespace=..., store=...)` uses an explicitly supplied `BaseStore`, or resolves a graph store at call time. Without either, it raises an actionable `RuntimeError`. The namespace factory receives the active `Runtime`; with an explicit store, a factory that ignores its argument can also run outside graph execution. A runtime-dependent factory outside a graph fails clearly instead of attempting to dereference an absent runtime.

The namespace is a store scope, not a filesystem path. It must be a nonempty tuple of nonempty string components drawn from alphanumerics plus `-`, `_`, `.`, `@`, `+`, `:`, and `~`; wildcard-like and path-like characters are rejected before lookup. This prevents wildcard injection into store lookup, but application identity and authorization must still decide which namespace a caller may use.

Store listing, search, globbing, and recursive deletion page through matching store items and perform virtual-path filtering locally. Recursive deletion removes the exact key plus descendants, treating wildcard-looking keys literally. Its asynchronous operations use native store methods such as `aget`, `aput`, `abatch`, and `asearch`.

## `CompositeBackend`: one virtual tree, multiple owners

`CompositeBackend` presents a default backend plus route-prefix backends as one virtual tree. It chooses the longest matching route. A path equal to a route root such as `/memories` maps to `/` inside that route; a descendant has the prefix stripped; an unmatched path stays with `default`.

```mermaid
flowchart TD
    Request["File operation and virtual path"] --> Match{"Longest matching route"}
    Match -->|"Matched"| Strip["Strip route prefix"]
    Strip --> Target["Call routed backend"]
    Target --> Restore["Restore composite path in result"]
    Match -->|"No match"| Default["Call default backend"]
    Command["Execute command"] --> Exec["Call default backend only"]
```

The diagram shows path routing, not a security or execution boundary.

For `ls`, the composite lists only a matched route; at `/`, it combines the default listing with synthetic directories for each route. Routed listings, grep matches, glob matches, mutation results, and transfer responses are remapped back to composite paths. A routed backend lacking `delete` yields a `DeleteResult` error rather than leaking `NotImplementedError`.

Uploads and downloads are grouped by selected backend, one batch is sent per group, and results are restored to original input order and paths. This preserves per-file partial failures without requiring every backend to receive one transfer at a time.

A root (`/` or `None`) grep searches default and routes. `max_count` is global: the composite forwards only the remaining budget and conservatively marks the result truncated if it skips unvisited routes after the budget is reached. Root glob merges default and routes. A leading `/` anchors a glob to the composite root, so routes are skipped unless the pattern explicitly names their prefix; a named route pattern is rewritten relative to that backend. Merge operations surface the first backend error instead of reporting a misleading partial success.

### Routing never routes a command

`execute()` is intentionally different: it delegates only to `default` when that backend implements `SandboxBackendProtocol`, otherwise it raises `NotImplementedError`. A route mounted from `StoreBackend` is accessible through file operations, not through a shell. A composite is therefore a storage composition mechanism, not access control or containment.

A typical mixed-lifetime layout is:

```python
CompositeBackend(
    default=StateBackend(),
    routes={"/memories/": StoreBackend(namespace=lambda _rt: ("filesystem", "alice"))},
)
```

This makes `/memories/` durable in the selected store namespace while other paths remain graph-local. It does not grant a user access to that namespace.

## Local shell execution is host execution

`LocalShellBackend` combines `FilesystemBackend` with `SandboxBackendProtocol`, but it is explicitly **not** a sandbox. `execute()` starts `subprocess.Popen(..., shell=True)` directly on the host with the process user's permissions and `root_dir` as its working directory. Commands can access accessible host paths, make network connections, modify files and configuration, and spawn processes. Backend operations are not reversible.

> **Danger:** Do not expose `LocalShellBackend` to untrusted users, production services, or multi-tenant workloads. Use a genuinely isolated sandbox backend for service-side execution, and apply tool permissions and human approval as defense in depth. See [sandbox partners](/openwiki/integrations/sandbox-partners.md).

`virtual_mode=True` makes *filesystem-tool* paths relative to `root_dir` and rejects traversal such as `..` and `~`. It does not constrain shell syntax or shell paths: `execute("cat /etc/passwd")` is still governed only by host permissions. Likewise, placing a local shell backend inside a composite changes which backend receives file paths but cannot confine commands beyond the rule that the composite executes on its default.

### Execution lifecycle, timeout, and cancellation

The default command timeout is 120 seconds and must be positive; a positive per-command value overrides it. An empty command produces an error response. The backend starts the shell with stdout and stderr pipes, `stdin=DEVNULL`, text decoding with replacement, the configured environment, and `cwd=root_dir`. By default that environment is empty apart from supplied `env`; `inherit_env=True` copies the parent environment and overlays `env`. On POSIX it also starts a new session, making the shell its process-group leader.

```mermaid
flowchart TD
    Start["execute or aexecute"] --> Spawn["Start host shell with pipes"]
    Spawn --> Capture["Incrementally capture stdout and stderr"]
    Capture --> Done{"Pipes EOF and shell exited"}
    Done -->|"Yes"| Response["Format output and return response"]
    Done -->|"No"| Trigger{"Timeout, interruption, or cancellation"}
    Trigger -->|"Timeout"| Terminate["Terminate shell or POSIX process group"]
    Trigger -->|"Cancellation"| Signal["Signal worker cancellation"]
    Signal --> Terminate
    Trigger -->|"Interruption"| Terminate
    Terminate --> Drain["Drain POSIX pipes within cleanup deadline"]
    Drain --> Reap["Reap direct shell and close pipes"]
    Reap --> Outcome{"Cleanup completed"}
    Outcome -->|"Timeout"| Timed["Return 124 with partial output"]
    Outcome -->|"Cancellation"| Cancelled["Raise cancellation after cleanup grace"]
    Outcome -->|"No"| Warn["Warn process may still run"]
    Warn --> Timed
```

This control flow shows the LocalShellBackend host-process lifecycle. A normal response waits for both pipe EOF and the direct shell to exit; cleanup handles timeout, interruption, and async cancellation.

Output collection is incremental rather than a one-shot `communicate()`. The POSIX reader uses `selectors` to take bounded chunks from both pipes and keeps bytes across cancellation polling attempts, decoding only once complete. The Windows reader polls available named-pipe bytes without reader threads and incrementally decodes each stream. Both require EOF on both pipes *and* direct-shell exit before successful completion, so output from descendants that inherited a pipe is retained rather than silently discarded. This can also mean a shell that has exited does not complete until an inherited writer closes or the command reaches its timeout.

On timeout, the implementation preserves output captured before cleanup, combines it with stderr lines prefixed `[stderr]`, and returns exit code `124`. It applies the configured UTF-8 byte limit only when formatting the combined response, avoiding partial Unicode code points. A nonzero normal exit gets exit-code information appended. If termination or reaping fails, the timeout response warns that the command may still be running and that retrying can create a second copy.

The cleanup boundary is platform-specific. On POSIX, timeout, `KeyboardInterrupt`, and supported async cancellation kill the new process group with `SIGKILL` (falling back to the direct shell if group termination fails), drain output, reap the shell, and close pipes under one cleanup deadline. A descendant that creates its own session or process group escapes that group; `nohup` alone does not. On Windows, cleanup kills and reaps only the direct shell, then closes pipes; descendants can survive and inherited output handles can hold the pipes open. Therefore a timeout response explicitly notes that processes started by the shell may still run even when cleanup of the shell succeeded.

`aexecute()` delegates the built-in implementation to a thread so synchronous process I/O does not block the event loop. It uses a cancellation event and a start handshake so cancellation that wins before execution prevents the worker from launching the command. After cancellation it waits up to a one-second cleanup grace period, including through repeated cancellations. If cleanup exceeds that period, it retains the worker in a background set until it finishes and logs only the failure type, then re-raises cancellation. This bound is for the awaiter's cancellation path, not for interpreter or executor shutdown; overriding `execute()` takes the generic async delegation path and does not receive this built-in cancellation cleanup.

These timeout, output, environment, and cancellation controls are reliability features, not isolation guarantees.

## Focused regression coverage

The backend tests cover the compatibility boundary from both state and store: legacy list reads without mutation, failed malformed values, successful write/edit migration, timestamps, binary transfer round trips, and edit refusal for binary data. State tests also verify that a lone surrogate is persisted and downloaded as U+FFFD, while a split valid surrogate pair becomes its Unicode character before UTF-8 byte-size calculations. State tests additionally cover graph-context behavior; store tests cover namespace validation, paging, recursive literal-key deletion, and native async paths.

Composite tests cover longest-prefix and route-root handling, remapped result paths, root listings, grouped transfers and response order, global grep budgets, anchored glob semantics, merge truncation, error propagation, and default-only execution.

Local-shell tests cover working-directory sharing, environment construction, virtual filesystem paths versus unrestricted shell paths, output byte truncation, undecodable bytes, timeout partial-output responses, and warnings for incomplete cleanup. POSIX integration checks that timeout stops a background descendant in the command group, including after the shell has exited, while cleanup unit tests protect the shared drain-and-reap deadline. Reader tests verify output and newline preservation across polling timeouts, do not copy accumulated POSIX output into transient timeout exceptions, wait for process exit even after pipes close, and retain output from pipe-owning descendants. Windows tests additionally cover its pipe EOF error codes, incremental multibyte decoding, and prompt cleanup when an inherited writer keeps a pipe open. These tests establish observable lifecycle behavior; they do not make host shell execution safe.
