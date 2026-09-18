---
type: security operations guide
title: Security Boundaries and Secrets
description: Operating guidance for agent authority, filesystem and execution boundaries, MCP configuration and credentials, Talon approvals, and GitHub Actions credentials. It distinguishes mediated controls from containment and documents the repository's scoped CI credential model.
tags: [security, operations, trust-boundaries, secrets, approvals, mcp, github-actions]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-18T16:46:37.183Z
sources:
  - id: openwiki-source-8d4ac162fca0a57f00bb83b7
    resource: repo://.github/SECRETS.md
  - id: openwiki-source-fa750a379507f8fc66395df2
    resource: repo://.github/workflows/_eval.yml
  - id: openwiki-source-7330cb37457ccdb62d7c41c7
    resource: repo://.github/workflows/auto-label-by-package.yml
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-6d4b4e707b8d60b6ccfa3425
    resource: repo://.github/workflows/openwiki-update.yml
  - id: openwiki-source-074ce96a8baea27a6c43328b
    resource: repo://libs/code/deepagents_code/client/launch/server.py
  - id: openwiki-source-216ca680d81dc35eb4d3e76e
    resource: repo://libs/code/deepagents_code/mcp_config.py
  - id: openwiki-source-030d8bd153a9c3ea2a99cb7d
    resource: repo://libs/code/deepagents_code/workspace.py
  - id: openwiki-source-1d73b3e2b56b5f0d27273379
    resource: repo://libs/code/README.md
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-3d157a5857f325aceaade7f1
    resource: repo://libs/talon/deepagents_talon/channels/whatsapp.py
  - id: openwiki-source-31e40ff79779f51cafd03f01
    resource: repo://libs/talon/deepagents_talon/mcp_auth.py
  - id: openwiki-source-111101dcd1462ff54277b1fc
    resource: repo://libs/talon/deepagents_talon/mcp_config.py
  - id: openwiki-source-267468fe937003d4716fe6c2
    resource: repo://libs/talon/deepagents_talon/tool_approvals.py
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
generated: { by: "openwiki/0.4.2", at: "2026-09-18T16:46:37.183Z" }
---

# Security Boundaries and Secrets

## Authority model and deployment boundary

Deep Agents follows a **trust the LLM** model: an agent can perform whatever its exposed tools permit. Prompts, model selection, and a tool's redacted presentation are not enforcement points. Establish authority at tool exposure, approvals, the execution backend, OS identity, network policy, and the deployment environment. The application operator—not this library—must secure the service that invokes an agent, its credentials, persistence, and external integrations.

```mermaid
flowchart TD
    Input["User input or tool result"] --> Model["Model selects a tool call"]
    Model --> Gate{"Approval or policy"}
    Gate -->|deny| Stop["Do not dispatch"]
    Gate -->|allow| Tool["Exposed tool"]
    Tool --> Backend["Selected backend"]
    Backend --> Host["Host process"]
    Backend --> Sandbox["Sandbox or remote environment"]
```

This is the authority path: an approval can stop dispatch, while the selected backend determines where an allowed action executes.

Related: [backends](../concepts/backends.md), [permissions and HITL](../concepts/permissions-hitl.md), [MCP](../integrations/mcp.md), [GitHub Action](../integrations/github-action.md), and [development](development.md).

## Filesystem policy is not execution containment

`FilesystemPermission` is an ordered, first-match policy for filesystem-tool operations. Its `allow`, `deny`, and `interrupt` modes respectively proceed, return a permission-denied error, or delegate an approval decision to `HumanInTheLoopMiddleware`. Permission patterns must be absolute and cannot contain traversal or home-directory shorthand.

This governs the filesystem tools, not arbitrary host reads. In particular, `FilesystemMiddleware` refuses unscoped permissions with an execution-capable backend because it has no equivalent execute-tool permission enforcement. Do not use filesystem rules as a shell, host, or secret-confidentiality boundary. Use a sandbox/VM/container or an OS identity and readable-path set that excludes sensitive material.

## dcode: project, local server, and workspace boundaries

dcode trusts the directory from which it runs; project artifacts are read before a human-approval prompt. Treat an untrusted checkout as untrusted executable influence and use a remote sandbox rather than running it on a trusted workstation.

Its local server chooses an ephemeral port and binds `127.0.0.1`; its subprocess environment sets `LANGGRAPH_AUTH_TYPE=noop`. Consequently, the HTTP interface relies on loopback exposure and host-process isolation rather than server authentication. Do not assume that a shared local host makes loopback a private trust zone.

For server-hosted threads, dcode canonicalizes an existing absolute workspace directory, derives workspace and policy identities, and transactionally binds them to the thread. A subsequent binding that differs is rejected, so a client cannot silently move an existing thread to a different workspace or execution policy. This is an integrity guard for dcode state, not protection against a party that can alter its database or host filesystem.

### MCP environment expansion and OAuth storage

dcode expands `${VAR}` and `${VAR:-default}` in MCP `command`, `url`, `args`, `env`, and `headers`; malformed braced references and an unset required value fail resolution. This is configuration interpolation, not secret mediation: expansion can still pass a credential to a command or remote endpoint.

`FileTokenStorage` writes MCP OAuth token state under the selected profile's state directory. It restricts server-name path components and serializes mutations; token and client information can be persisted together atomically. Token values are sensitive: the OAuth token model's default representation includes them, so logging a token object or an exception that wraps one can disclose credentials. Do not place credential values in configuration, prompts, logs, or diagnostic output.

## Talon: operator authority and mediated administration

Talon is experimental alpha software and explicitly lacks production-grade complete HITL policy, channel administrator controls, sandbox execution isolation, and multi-tenant boundaries. A channel participant with agent access must therefore be treated as holding the operator's effective agent, model, MCP, and local-host authority.

WhatsApp defaults to `self` exposure for the paired account. `allowlist` narrows triggering chats or mentions. `open` requires the explicit `DEEPAGENTS_TALON_WHATSAPP_OPEN_ACK` acknowledgement and permits arbitrary senders; do not use it where that effective operator authority is unacceptable.

### Invocation snapshots and approval policy

`ToolApprovalStore` stores exact tool-name booleans and produces an immutable `ApprovalSnapshot` for an invocation. Only enabled names appear in the resulting approval interrupts; unspecified tools do not prompt. Updates validate the bounded JSON policy, use a lock and revision compare-and-swap, and preserve unrelated entries. A successful change is explicitly available only on the next invocation, so an in-flight turn retains its original snapshot.

Changing the policy is separately operator-authorized: `update_tool_approvals` rejects calls without both a trusted operator marker and an active snapshot. A `false` policy value disables prompting; it does not itself make a tool unavailable or prove authorization. Embedding hosts must populate the operator authorization metadata from a trusted channel boundary, never from model arguments or untrusted inbound metadata.

### MCP configuration redaction is not a confidentiality boundary

`MCPConfigStore` provides `get_mcp_configuration` and `update_mcp_server` against one POSIX regular, non-symlink configuration file. The read tool redacts stored strings except recognized enum values and exact `${ENV_VAR}` references. Updates use a process-local HMAC revision, a sidecar lock, validation, atomic replacement, and schedule reload only after a successful write; stale/busy revisions return conflicts.

These controls mediate the configuration-tool path and prevent lost or stale writes. They **do not** conceal credentials from a Talon agent using the default local shell: a readable absolute path can bypass redaction, revisions, and the update approval. A warning that the config or token directory is inside the workspace is therefore a placement warning, not a filesystem confidentiality boundary. Put secrets where the agent process cannot read them, and use `${ENV_VAR}` references rather than literal values when configuring MCP.

Talon MCP OAuth storage holds cleartext bearer and refresh tokens. It hardens storage with owner-only directories/files, locking, and atomic writes, but the code explicitly describes that as hardening rather than a defense against the default shell backend. Do not log, copy, or expose token material.

## GitHub Actions and CI credentials

`.github/SECRETS.md` is an inventory of intended non-`GITHUB_TOKEN` CI credential scopes and explicitly warns not to record credential values or identifiers. It distinguishes target GitHub configuration from what workflow YAML proves: environment selection does not establish that environment protection, branch policy, or secrets have actually been configured. Verify external GitHub and provider settings separately.

Use GitHub environments and step-level injection to minimize credential reach. For example, the labeling workflow selects the `labeling` environment but injects its provider credential only into the topic-classification step; the ordinary CI workflow defaults `GITHUB_TOKEN` to read-only checks, contents, and pull-request permissions. Reusable eval jobs declare optional provider secrets, run in the `evals` environment, and retain read-only repository contents permission.

The OpenWiki update workflow defaults `GITHUB_TOKEN` to `contents: read`, checks out without persisted credentials, and mints a separate repository-scoped GitHub App token for the update/PR steps with explicit contents and pull-request write permissions. That separation is meaningful only for that App-token path: workflow-level `permissions` restrict `GITHUB_TOKEN`, not a separately minted App installation token. Keep App installations and provider keys least-privileged, scoped to the narrowest environment, and rotate/revoke them after suspected exposure.

## Operational checklist

1. Before opening an untrusted project or channel, choose a real execution boundary: remote sandbox, VM/container, or a dedicated low-privilege identity.
2. Keep filesystem permissions as tool policy only; test that no execution path can read the secrets the agent must not access.
3. Keep dcode's local service away from untrusted local peers, and do not place secrets in project files, prompts, or tool output.
4. For Talon, prefer `self` or a restrictive allowlist; do not equate MCP redaction, a workspace-placement warning, or approval prompts with sandboxing.
5. Review the persisted and active Talon approval revisions before changes. Treat conflicts as a reason to reread and review, not to overwrite policy.
6. Store CI credentials in their narrowest GitHub environment, inject them only into consuming steps, and audit broader repository/organization fallback scopes before deletion or rotation.
7. On suspected compromise, stop the affected runtime; revoke and rotate relevant provider, MCP, channel, and CI credentials; review approval/configuration changes and persisted state; then redeploy only after containment and approval paths are retested.

Focused coverage includes `libs/talon/tests/unit_tests/test_mcp_config.py`, `libs/talon/tests/unit_tests/test_tool_approvals.py`, `libs/talon/tests/unit_tests/test_tool_approval_authorization.py`, and dcode workspace tests.
