---
type: integration architecture
title: Sandbox and Partner Integrations
description: How dcode discovers optional sandbox providers, acquires or attaches sandboxes, runs setup, and assigns cleanup ownership. Covers the independently distributed partner adapters and Daytona's bounded command execution behavior.
tags: [dcode, sandbox, integrations, providers, lifecycle, partners]
sources:
  - resource: repo://libs/partners/daytona/langchain_daytona/sandbox.py
  - resource: repo://libs/partners/daytona/tests/unit_tests/test_import.py
  - resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-9f207ab48c42b84dcfd05f43
    resource: repo://libs/code/deepagents_code/integrations/sandbox_config.py
  - id: openwiki-source-bcf1f68e7989964d2fcec7aa
    resource: repo://libs/code/deepagents_code/integrations/sandbox_factory.py
  - id: openwiki-source-03e3942e51522a3aa485168d
    resource: repo://libs/code/deepagents_code/integrations/sandbox_provider.py
  - id: openwiki-source-668d65d09330d04370b47300
    resource: repo://libs/code/deepagents_code/integrations/sandbox_registry.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-ea49272aef6bfc33d634a15c
    resource: repo://libs/code/tests/integration_tests/test_sandbox_factory.py
  - id: openwiki-source-ba6aa10dca5a8aea05030887
    resource: repo://libs/code/tests/integration_tests/test_sandbox_operations.py
  - id: openwiki-source-b6c121d977850bad8b8c8068
    resource: repo://libs/code/tests/unit_tests/test_sandbox_factory.py
  - id: openwiki-source-32820da62949c2fdb3f41188
    resource: repo://libs/code/tests/unit_tests/test_sandbox_registry.py
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-da577cbe81ec29338f1388b2
    resource: repo://libs/partners/daytona/pyproject.toml
  - id: openwiki-source-936554ac5f0a201f8696be25
    resource: repo://libs/partners/modal/pyproject.toml
  - id: openwiki-source-8d2c8381956c1c023bcdb565
    resource: repo://libs/partners/runloop/pyproject.toml
  - id: openwiki-source-03a39f44d8ccfde2fd47e57a
    resource: repo://libs/partners/vercel/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-10-24T08:05:08.866Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-10-24T08:05:08.866Z
---

# Sandbox and Partner Integrations

A dcode sandbox provider supplies a `SandboxBackendProtocol`: an execution and filesystem backend identified by `id`. The registry and factory are the dcode-facing integration layer; the provider package remains responsible for its provider SDK, credentials, environment semantics, and backend implementation. Selecting a sandbox routes the execution/filesystem backend; it is not, by itself, a general authorization or isolation policy. See [Backends](/openwiki/concepts/backends.md), [Tools and filesystem](/openwiki/concepts/tools-filesystem.md), and [Security](/openwiki/operations/security.md).

## Provider contract and discovery

A `SandboxProvider` implements synchronous `get_or_create(sandbox_id=..., **kwargs)` and `delete(sandbox_id=..., **kwargs)`. Its `aget_or_create()` and `adelete()` convenience methods run those synchronous operations in worker threads. A provider may advertise `SandboxProviderMetadata`: default working directory, installation hint, whether attachment by ID and named snapshots are supported, and an optional backend module for dependency probing.

`SandboxRegistry` combines three sources under one name:

1. Curated built-ins: `agentcore`, `daytona`, `langsmith`, `modal`, `openshell`, `runloop`, and `vercel`.
2. Installed Python entry points in the `deepagents_code.sandbox_providers` group.
3. Operator-declared providers in `[sandboxes.providers]`.

On a name collision, resolution is **configuration, then entry point, then built-in**. Entry-point enumeration is best-effort, so a discovery error leaves the remaining sources usable. Built-in and configured metadata can be read without constructing a provider. To obtain an entry-point provider's metadata, the registry instantiates it; if that fails, discovery presents a conservative `/workspace` metadata fallback rather than failing the provider list.

```mermaid
flowchart TD
    Request["Provider name"] --> Config{"Configured entry"}
    Config -->|yes| Import["Import class_path"]
    Config -->|no| Entry{"Entry point"}
    Entry -->|yes| Load["Load entry point"]
    Entry -->|no| Builtin["Create built-in provider"]
    Import --> Provider["SandboxProvider"]
    Load --> Provider
    Builtin --> Provider
    Provider --> Backend["Create or attach backend"]
```

*Provider resolution is configuration over entry point over built-in.*

A configuration entry can provide `class_path`, `working_dir`, installation package, capability flags, and provider `params`:

```toml
[sandboxes]
default = "acme"

[sandboxes.providers.acme]
class_path = "my_package:AcmeSandboxProvider"
working_dir = "/workspace"
package = "acme-dcode-sandbox"
supports_snapshot_name = true

[sandboxes.providers.acme.params]
region = "us-east-1"
```

`class_path` is trusted operator configuration: resolving it imports Python. A malformed or unreadable user configuration is recorded as a visible registry configuration error while unrelated startup can continue; malformed provider values that are not TOML tables are ignored. The configured default names the provider selected by a bare `--sandbox`; it never turns on sandbox mode for an invocation that did not opt in. Configuration layering and managed policy are described in [Configuration layering](/openwiki/concepts/config-layering.md).

## Factory lifecycle and ownership

`create_sandbox()` is the common synchronous context-managed lifecycle. It reads provider metadata before acquisition, validates that the provider supports a requested `snapshot_name`, and rejects a snapshot combined with `sandbox_id`: a snapshot applies only when creating a fresh sandbox. It then copies configured provider `params`, overlays explicit `params`, translates `snapshot_name` to the provider's `snapshot` argument, and calls `get_or_create()`.

```mermaid
flowchart TD
    Start["create_sandbox"] --> Validate["Read metadata and validate options"]
    Validate --> Merge["Merge configured then explicit parameters"]
    Merge --> Acquire["get_or_create"]
    Acquire --> Setup{"Setup file supplied"}
    Setup -->|yes| Run["Expand variables and run bash -c"]
    Setup -->|no| Yield["Yield backend"]
    Run --> Yield
    Yield --> Exit["Context exit or failure"]
    Exit --> Owned{"Created by factory"}
    Owned -->|yes| Delete["delete backend id"]
    Owned -->|no| Retain["Leave attached backend"]
```

*The factory deletes a new resource, but never an attached one.*

A setup file is read on the host after acquisition. `${VAR}` expansion uses dcode's active workspace environment—not the server process environment—and the resulting script executes inside the sandbox as `bash -c`. A missing/unreadable setup file, an execution exception, or a nonzero exit prevents the context body from running. Once a fresh backend has been acquired, the `finally` path attempts deletion after setup failure, caller failure, or normal exit. Cleanup errors are reported but do not mask the original exception. Supplying `sandbox_id` changes this ownership rule: the factory attaches and does not delete the supplied resource, including after setup failure.

For the server runtime, dcode enters this context in a worker thread and stores the context for the process lifetime; it registers an `atexit` close after successful startup. If startup is cancelled after a context has entered, the cancellation path closes that context before re-raising cancellation, avoiding a factory-owned sandbox leak. A configured sandbox creation failure is surfaced as startup failure rather than silently replacing the requested backend.

## Daytona: provisioning and command boundaries

The built-in dcode Daytona provider creates a Daytona client from the API key and optional `DAYTONA_API_URL`, then creates a fresh sandbox. It does **not** currently implement `sandbox_id` attachment, even though generic provider metadata can otherwise advertise that capability. Startup polls `echo ready` every two seconds up to its `timeout`; if readiness never succeeds, it best-effort deletes the new sandbox before raising. For snapshots, `DAYTONA_SANDBOX_SNAPSHOT_ID` wins; otherwise an explicit `snapshot` overrides `DAYTONA_SANDBOX_SNAPSHOT_NAME`, and a missing named snapshot may be built from `snapshot_image` or the provider default image. A named snapshot that already exists but is not active is rejected rather than rebuilt.

`DaytonaSandbox` is the partner backend that wraps the created SDK sandbox. It inherits the common `BaseSandbox` filesystem operations and implements command execution with a temporary Daytona process session. `ExecuteResponse` carries combined output, an optional exit code, and a `truncated` marker; callers should use the exit code to distinguish a completed command failure from success.

For a finite `execute()` timeout, one monotonic deadline covers session creation, asynchronous command start, status polling, and final-log retrieval. The adapter avoids making a request once fewer than one second remains, caps each request by the remaining time, and limits each polling sleep to that remainder. `timeout=0` is deliberately different: it means wait indefinitely and sends unbounded command-phase requests. The execute-start argument is adjusted for the Daytona SDK's extra five-second HTTP buffer so that a finite effective request stays within the caller's deadline.

```mermaid
flowchart TD
    Begin["execute command"] --> Deadline{"Finite timeout"}
    Deadline -->|no timeout 0| Create["Create session"]
    Deadline -->|yes| Budget{"At least one second remains"}
    Budget -->|no| TimedOut["Return exit 124"]
    Budget -->|yes| Create
    Create --> Start["Start async command"]
    Start --> Poll["Poll status within remaining deadline"]
    Poll --> Done{"Exit code available"}
    Done -->|yes| Logs["Read final logs within deadline"]
    Done -->|no deadline| TimeoutLogs["Try partial logs within shared grace"]
    Logs --> Result["Return output and exit code"]
    TimedOut --> Cleanup["Bounded session deletion"]
    TimeoutLogs --> Cleanup
    Result --> Cleanup
    Cleanup --> Finish["Return result"]
```

*This is the Daytona adapter's command deadline and cleanup path; it is not a contract for other sandbox adapters.*

A timeout returns exit code `124` and the timeout message, appending partial stdout/stderr if it can be read. Timeout recovery gives log collection and deletion one shared two-second grace period, with at most one second per cleanup request; exhausted grace skips deletion. The adapter always attempts bounded session deletion once session creation was attempted—including when a creation transport timeout leaves remote creation uncertain. Cleanup failures are logged and do not replace a command result. On normal completion, final logs are retrieved before the session is deleted.

Daytona file transfers require absolute paths: invalid relative inputs get the standardized `invalid_path` error without an SDK request. Downloads preserve input order and map missing Daytona content to `file_not_found`; uploads return one response per input. These transfer rules and the command deadline behavior are specific to `langchain-daytona`.

## Installation and optional package boundary

The six curated external adapters—AgentCore, Daytona, Modal, OpenShell, Runloop, and Vercel—are optional `deepagents-code` extras, collectively available through `deepagents-code[all-sandboxes]`. `langsmith[sandbox]` is a regular dcode dependency. The compatibility `deepagents-code[quickjs]` extra intentionally remains empty because QuickJS is already a normal dependency and is middleware rather than a remote sandbox adapter. Missing optional provider imports produce provider-specific installation guidance.

In particular, the `openshell` extra installs `langchain-nvidia-openshell>=0.1.1`. dcode's local development source mappings point at the Daytona, Modal, QuickJS, Runloop, and Vercel partner directories; they do not turn those packages into dcode-owned code or add an OpenShell checkout.

The checked-in Daytona, Modal, Runloop, and Vercel adapters are separate partner distributions. Each depends on Deep Agents and its respective provider SDK and owns its own version, environment, `pyproject.toml`, `Makefile`, and tests. dcode's curated names provide discovery, optional-install packaging, parameter forwarding, and lifecycle policy—not ownership of the adapter implementation. A third-party package can extend the registry by publishing an entry point in `deepagents_code.sandbox_providers`; an operator can instead use a configured `class_path` for an internal or local provider.

## Operating and changing integrations

- Use `--sandbox <provider>` to choose a provider, or bare `--sandbox` only when `[sandboxes].default` is configured. Unknown providers and unsupported snapshot options are rejected before normal startup. ID support is provider-specific: the Daytona provider currently raises `NotImplementedError` if an ID is passed.
- Treat provider metadata as declarations that drive dcode validation and UX, not proof that the provider enforces a security property. Verify provider-specific identity, credential, network, retention, timeout, cancellation, and isolation behavior separately.
- Preserve the factory's ownership boundary when adding a provider: only fresh factory-created resources may be deleted. Ensure setup failures take the same cleanup path.
- Do not generalize Daytona's `124` timeout result, deadline accounting, polling policy, or shared cleanup grace to another adapter without implementing and testing them there.
- The focused dcode unit tests cover registry precedence, entry-point metadata fallback, malformed configuration, snapshot validation, setup failure, attachment ownership, and cleanup error reporting. Daytona unit tests additionally cover fixed and adaptive polling, finite versus unlimited command requests, deadline-bounded requests, partial-output recovery, cleanup failure, and shared cleanup grace. The optional integration suites exercise real creation, command execution, file transfer, and filesystem operations; they are skipped in the release pipeline and some error-path cases remain explicitly skipped pending provider support.
- For repository work on an adapter, follow the partner-package guidance: adding a partner also requires its independent release and CI wiring, including sandbox credential/integration-test surfaces where applicable.

See [Development](/openwiki/operations/development.md) for repository workflows and [Run a dcode session](/openwiki/workflows/run-dcode-session.md) for the end-user session path.
