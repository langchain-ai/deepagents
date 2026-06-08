# Code Interpreter Baseline Defaults and Swarm Retrofit

## Status

Draft

## Authors

- Deep Agents contributors

## Last Updated

- 2026-06-07

## Summary

This spec defines a default capability contract for Code Interpreter (CI) so skills can rely on stable APIs without custom per-agent wiring. The contract introduces:

1. A built-in subagent dispatch API (`subagent`) aligned to task-tool semantics.
2. A built-in direct model API (`llm`) for one-shot calls with optional schemas.
3. A Node-compatible `fs` subset (`readFile`, `writeFile`, `readdir`) for agent intuition.
4. Two import-free top-level helpers needed for swarm workflows (`glob`, `editFile`).
5. Built-in first-party extensions loaded by default through the existing extension API (no special-case middleware mechanism).

The spec also proposes re-fitting swarm as a thin library on top of this baseline contract, rather than requiring users to manually create and expose `create_swarm_task_tool` through `ptc`.

## Problem Statement

Today, advanced code interpreter patterns require explicit setup by application authors:

1. Swarm usage requires creating `create_swarm_task_tool`.
2. The swarm tool must be exposed through `CodeInterpreterMiddleware(ptc=[...])`.
3. Skills depend on concrete tool names (`task`, `read_file`, `write_file`, ...) instead of a stable CI platform API.

This creates three problems:

1. Onboarding friction: users cannot "just use" swarm or similar orchestration skills.
2. Portability risk: skill code is tightly coupled to current tool names and wiring.
3. Fragmentation: every project reinvents slightly different defaults.

## Goals

1. Define a minimal, stable code interpreter default contract that is available out of the box.
2. Preserve existing behavior for current users during migration.
3. Make swarm an optional layer built on top of baseline code interpreter primitives.
4. Keep public interfaces backwards-compatible until a major version change.

## Non-Goals

1. Replacing Deep Agents `task` tool semantics in this phase.
2. Designing a general cross-runtime RPC protocol for arbitrary interpreter languages.
3. Changing backend security model in this spec.
4. Designing PTC approval / HITL policy in this spec.

## Current Constraints and Observations

1. `SubAgentMiddleware` already supports delegating and returning final message content from subagents.
2. `task` calls currently return final outputs, and parent state excludes subagent internals like `structured_response` unless explicitly surfaced through tool output.
3. `CodeInterpreterMiddleware` PTC is allowlist-based and currently requires explicit tool exposure.
4. Existing swarm tooling (`create_swarm_task_tool`) already supports:
   1. Subagent dispatch.
   2. Direct model invocation.
   3. Optional `response_schema`.
   4. Variant caching for schema-constrained agent paths.
5. `CodeInterpreterMiddleware` already has a production extension lifecycle (`on_setup`, `on_eval`) and first-party swarm extension usage, so baseline defaults can be implemented in the same architecture.
6. Existing backend protocol supports `read`, `write`, `ls`, `glob`, `grep`, `edit`, and upload/download, but does not expose full Node filesystem primitives (`mkdir`, `rename`, `unlink`, `rm`, `stat`).

## Backend Capability Matrix (V1)

This matrix captures expected behavior before any backend protocol expansion.

| Backend | `fs.readFile` (`utf8`) | `fs.readFile` (`base64`) | `fs.writeFile` (`w`) | `fs.writeFile` (`wx`) | `fs.readdir` | `glob` | `editFile` |
|---|---|---|---|---|---|---|---|
| FilesystemBackend | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| StoreBackend | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| StateBackend | Yes | Best-effort | Yes | Yes | Yes | Yes | Yes |
| ContextHubBackend | Yes | No (text-focused) | Yes (text only) | Yes (text only) | Yes | Yes | Yes |
| LangSmithBackend | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| CompositeBackend | Route-dependent | Route-dependent | Route-dependent | Route-dependent | Route-dependent | Route-dependent | Route-dependent |

Notes:

1. `fs.writeFile` overwrite behavior in V1 may route through backend upload primitives when direct `write` is create-only.
2. Capability gaps (for example, no `ls`/`glob` on a backend) are surfaced as normalized unsupported errors rather than silent fallback.
3. Composite behavior depends on the mounted backend for the target path.

## Proposed Design

### 1. CI Baseline Capability Extensions

When CI is enabled, middleware loads built-in first-party extensions that install a stable global namespace:

1. `subagent`
2. `llm`
3. `fs`
4. `glob`
5. `editFile`

Skills should target these baseline APIs, not raw tool names.

### 2. Subagent API

Provide:

```ts
await subagent({
  description: string,
  subagentType: string,
  responseSchema?: JsonSchemaObject,
});
```

#### Semantics

1. API shape mirrors the task-tool contract, but camelCase in JS.
2. `description` is required and maps to subagent instructions.
3. `subagentType` is required.
4. `responseSchema` is optional and requests structured output.
5. Snake_case keys are rejected (no aliases in V1).
6. Return contract:
   1. Without `responseSchema`, return `string`.
   2. With `responseSchema`, return native JS value (`object` / `array` / primitive / `null`).
7. If structured parse/validation fails, throw (no raw-text fallback).

#### Mapping

1. `subagent(...)` maps to existing delegated subagent execution (`task` behavior).
2. Structured output uses existing schema validation and variant-cache behavior from swarm tooling.
3. In deepagents-integrated usage, absence of subagent dispatch runtime is a configuration error and should fail fast.

### 3. Direct Model API

Provide:

```ts
await llm({
  prompt: string,
  responseSchema?: JsonSchemaObject,
});
```

#### Semantics

1. `prompt` is required.
2. `responseSchema` requests structured output.
3. No subagent tools or iterative agent loop are used in this path.
4. No model/provider override fields in V1.
5. Uses the parent agent's effective model/runtime context.
6. Return contract:
   1. Without `responseSchema`, return `string`.
   2. With `responseSchema`, return native JS value (`object` / `array` / primitive / `null`).
7. If structured parse/validation fails, throw (no raw-text fallback).

#### Mapping

1. `llm(...)` maps to the existing direct model invocation path currently exposed by swarm `invoke` mode.
2. This replaces mixed-mode dispatch (`mode="agent" | "invoke"`) with two explicit APIs.

### 4. Filesystem API

Provide a Node-compatible subset:

```ts
await fs.readFile(path, { encoding?: "utf8" | "base64" })
await fs.writeFile(path, data, { encoding?: "utf8" | "base64", flag?: "w" | "wx" })
await fs.readdir(path)
```

#### Semantics

1. `fs` is intentionally limited to an implementable Node-compatible subset in V1.
2. `readFile` / `writeFile` support `encoding: "utf8" | "base64"` only.
3. `writeFile` defaults to overwrite semantics (`flag: "w"`), with create-only via `flag: "wx"`.
4. `readdir` returns `string[]` entry names in V1 (no `withFileTypes` in V1).
5. Path resolution:
   1. Absolute paths are accepted.
   2. Relative paths are resolved against virtual root (`/`).
   3. No mutable `cwd` API in V1.
6. `writeFile` contract:
   1. `flag: "w"` is create-or-overwrite (last writer wins under races).
   2. `flag: "wx"` is create-only and should throw `EEXIST` when the path already exists.
   3. If backend primitives cannot provide atomic create-only, implementation uses best-effort detection and still maps observed collisions to `EEXIST`.
   4. For `encoding: "base64"`, payload is decoded before backend write/upload calls.
7. `readdir` output contract:
   1. Return entry names (not absolute paths).
   2. Sort ascending lexicographically for deterministic output.
   3. Include hidden files when backend includes them.
   4. Exclude `.` and `..`.
   5. Do not append trailing `/` for directories.
8. Unsupported operations or unsupported encoding/backend combinations fail fast with normalized errors.

### 5. Top-Level Swarm Helpers (Import-Free)

Provide two top-level helpers (not under `fs`, no import required):

```ts
await glob(pattern, { cwd?: string })
await editFile({ filePath, oldString, newString, replaceAll?: boolean })
```

#### Semantics

1. These remain outside `fs` to keep `fs` Node-compatible.
2. `glob` maps to backend glob primitives and returns matched paths.
3. `editFile` maps to backend exact-replacement edit semantics.
4. `grep` is not part of V1 baseline.

### 6. Extension API as Implementation Mechanism

Implement baseline CI capabilities using the existing interpreter extension lifecycle.

1. Add first-party extensions loaded by default by `CodeInterpreterMiddleware`:
   1. `SubagentExtension` owns `globalThis.subagent`.
   2. `LlmExtension` owns `globalThis.llm`.
   3. `FilesystemExtension` owns `globalThis.fs`.
   4. `GlobExtension` owns top-level `globalThis.glob`.
   5. `EditFileExtension` owns top-level `globalThis.editFile`.
2. Extension ownership boundaries are strict; no cross-extension re-exports.
3. Each extension uses `on_setup(ctx)` to install its JS surface area and helper modules.
4. Each extension uses `on_eval(ctx, runtime)` for runtime-dependent host bindings that need per-eval `ToolRuntime` and synthesized call ids.
5. Filesystem/glob/edit host functions use `ctx.backend` for backend-backed operations.
6. Keep baseline APIs independent of `tools.*` PTC namespace so defaults work even with `ptc=None`.
7. Ship baseline capability guidance through extension `system_prompt`.

This is the primary V1 architecture. It keeps implementation inside the existing extension contract and avoids adding another middleware-specific mechanism.

#### Extension Sketch

```python
class SubagentExtension(InterpreterExtension):
    @classmethod
    def name(cls) -> str:
        return "subagent"

    async def on_setup(self, ctx: ExtensionContext) -> None:
        # Install JS wrapper for globalThis.subagent
        ...

    async def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime) -> None:
        # Bind per-eval host functions for subagent dispatch
        ...


class LlmExtension(InterpreterExtension):
    @classmethod
    def name(cls) -> str:
        return "llm"

    async def on_setup(self, ctx: ExtensionContext) -> None:
        # Install JS wrapper for globalThis.llm
        ...

    async def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime) -> None:
        # Bind per-eval host functions for direct model invocation
        ...


class FilesystemExtension(InterpreterExtension):
    @classmethod
    def name(cls) -> str:
        return "fs"

    async def on_setup(self, ctx: ExtensionContext) -> None:
        # Install JS wrapper for globalThis.fs
        ...

    async def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime) -> None:
        # Bind per-eval host functions for filesystem operations
        ...


class GlobExtension(InterpreterExtension):
    @classmethod
    def name(cls) -> str:
        return "glob"

    async def on_setup(self, ctx: ExtensionContext) -> None:
        # Install top-level globalThis.glob
        ...


class EditFileExtension(InterpreterExtension):
    @classmethod
    def name(cls) -> str:
        return "editFile"

    async def on_setup(self, ctx: ExtensionContext) -> None:
        # Install top-level globalThis.editFile
        ...
```

#### Extension Composition and Ordering

1. `SubagentExtension`, `LlmExtension`, `FilesystemExtension`, `GlobExtension`, and `EditFileExtension` are always included by default.
2. User-provided `extensions=[...]` remain supported.
3. Baseline names (`subagent`, `llm`, `fs`, `glob`, `editFile`) are reserved.
4. Middleware validates extension collisions for reserved names at construction time and fails fast with a clear error.
5. Baseline extension registration order is fixed:
   1. `SubagentExtension()`
   2. `LlmExtension()`
   3. `FilesystemExtension()`
   4. `GlobExtension()`
   5. `EditFileExtension()`
   6. `*user_extensions`
6. Reserved names cannot be overridden by later setup/eval hooks.
7. Concurrency limits are per capability in V1 (no shared cross-extension limiter):
   1. `subagent` has its own in-flight cap (default 10).
   2. `llm` has its own in-flight cap (default 10).

### 7. Default Capability Loading

Baseline capabilities are installed by default when `CodeInterpreterMiddleware` is used.

Behavior:

1. `subagent`, `llm`, `fs`, `glob`, and `editFile` are available in CI without requiring `ptc`.
2. Users may still pass `ptc=[...]` for additional tool exposure.
3. Users may still pass `extensions=[...]` for custom extension behavior.
4. Baseline exports are reserved names; user tools/extensions cannot shadow them.
5. No new `ci_defaults` keyword argument is introduced.
6. Implementation uses only the extension API path in V1 via built-in baseline extensions.
7. Reserved-name validation happens at middleware construction (fail-fast), not at first eval.

### 8. Swarm Retrofit

Re-implement swarm orchestration as a thin layer over baseline APIs.

1. Swarm dispatch uses `subagent(...)`.
2. Swarm direct one-shot model calls use `llm(...)`.
3. Swarm file workflows use `fs.readFile`, `fs.writeFile`, `fs.readdir`, plus top-level `glob` and `editFile`.
4. `create_swarm_task_tool` remains as a compatibility shim initially.

This turns swarm into a composable policy library, not infrastructure plumbing.

### 9. Configuration Surface (V1)

Add explicit middleware knobs for baseline capability limits:

```python
CodeInterpreterMiddleware(
    # per-capability in-flight caps
    subagent_max_in_flight=10,
    llm_max_in_flight=10,
    # optional per-capability timeout guards (None = rely on existing eval timeout)
    subagent_timeout_s=None,
    llm_timeout_s=None,
)
```

Semantics:

1. Caps are enforced independently per capability, per eval invocation.
2. Timeout, when set, applies to each individual `subagent(...)` / `llm(...)` call.
3. Timeouts and cap breaches map to normalized error codes (see error table).

## Compatibility and Migration

### Phase 0 (Current)

1. Explicit `ptc` and explicit swarm tool creation required.

### Phase 1 (Minor Release, Default-On)

1. Ship baseline CI capability layer enabled by default.
2. Keep all current APIs unchanged.
3. Keep explicit `ptc` configuration supported for extra tools.
4. Publish migration docs for skill authors (`subagent` / `llm` / `fs` / `glob` / `editFile`).

### Phase 2 (Deprecation)

1. Emit deprecation warnings when users wire `create_swarm_task_tool` only to emulate baseline behavior.
2. Keep compatibility shim functional.

### Phase 3 (Major Release Cleanup)

1. Remove obsolete compatibility path if usage is low and migration is complete.

## API Sketches

### Python-facing CI config

```python
# Baseline APIs loaded automatically via built-in extensions
CodeInterpreterMiddleware()

# Optional additional tool exposure / custom extensions
CodeInterpreterMiddleware(
    ptc=[...],  # optional additional tools
    extensions=[...],  # optional custom extensions
)
```

Baseline CI APIs are available by default.

Implementation note: baseline APIs are provided by built-in first-party extensions loaded by default.

### REPL-facing globals

```ts
// always present in CI
subagent(...)
llm(...)
fs.readFile(...)
fs.writeFile(...)
fs.readdir(...)
glob(...)
editFile(...)
```

## Error Handling

### Normalized Error Mapping (V1)

| Surface | Condition | Code | Notes |
|---|---|---|---|
| `subagent` | Unknown `subagentType` | `ERR_SUBAGENT_TYPE_UNKNOWN` | Include available types in message |
| `subagent` | Invalid `responseSchema` | `ERR_SCHEMA_INVALID` | Include size/depth/property limit detail |
| `subagent` | Structured response parse/validation failure | `ERR_STRUCTURED_OUTPUT_INVALID` | Throw; do not fallback to raw text |
| `subagent` | Per-capability cap exceeded | `ERR_CAPACITY_EXCEEDED` | Include configured cap |
| `subagent` | Per-call timeout | `ETIMEDOUT` | Distinct from eval timeout |
| `llm` | Invalid `responseSchema` | `ERR_SCHEMA_INVALID` | Same validator class as `subagent` |
| `llm` | Structured response parse/validation failure | `ERR_STRUCTURED_OUTPUT_INVALID` | Throw; do not fallback to raw text |
| `llm` | Per-capability cap exceeded | `ERR_CAPACITY_EXCEEDED` | Include configured cap |
| `llm` | Per-call timeout | `ETIMEDOUT` | Distinct from eval timeout |
| `fs` | Path not found | `ENOENT` | Node-style |
| `fs` | Create-only write to existing file | `EEXIST` | `flag: "wx"` |
| `fs` | Is directory where file expected | `EISDIR` | Node-style |
| `fs` | Permission denied | `EACCES` | Node-style |
| `fs` | Unsupported operation/backend | `ENOTSUP` / `ERR_NOT_SUPPORTED` | Include backend detail |
| `fs` | Invalid encoding | `ERR_INVALID_ENCODING` | Only `utf8` / `base64` in V1 |
| `glob` / `editFile` | Backend unavailable | `ERR_BACKEND_REQUIRED` | `backend=None` |
| `glob` / `editFile` | Unsupported operation/backend | `ENOTSUP` / `ERR_NOT_SUPPORTED` | Include backend detail |

### subagent

1. Unknown `subagentType`: throw explicit error listing available types.
2. Invalid schema: fail fast with schema limit/depth/shape error.
3. Structured parse/validation failures throw explicit errors.

### llm

1. Invalid schema: fail fast with schema limit/depth/shape error.
2. Provider/model invocation errors surface with normalized error type/message.
3. Structured parse/validation failures throw explicit errors.

### fs namespace

1. Normalize errors to Node-like codes/messages where possible (`ENOENT`, `EEXIST`, `EISDIR`, `EACCES`, `ENOTSUP`/`ERR_NOT_SUPPORTED`).
2. Preserve backend-specific details in message suffixes for debugging.
3. Preserve path security constraints from existing middleware/backend.
4. Invalid encoding or unsupported method/options fail fast.
5. No methods beyond V1 subset are exposed on `fs`.

### glob / editFile

1. Surface backend and validation errors with consistent, explicit messages.
2. `glob` and `editFile` fail fast when backend capability is unavailable.

### `backend=None` Behavior

1. `subagent` and `llm` remain available.
2. `fs`, `glob`, and `editFile` are still defined globals but throw immediately with `ERR_BACKEND_REQUIRED`.
3. This behavior is deterministic and documented in system prompt guidance.

## Security Considerations

1. PTC remains a high-trust mechanism; baseline defaults increase reachable capabilities from REPL code.
2. Existing backend-level path and execution controls remain the primary boundary.
3. Approval/HITL policy for PTC is explicitly out of scope for this spec.

## Performance Considerations

1. Reuse existing variant cache behavior for schema-constrained subagent calls.
2. Reuse schema-constrained direct model invocation path for `llm(...)`.
3. Avoid per-call bridge reinstallation when exported capability set is unchanged.
4. Keep default API wrappers thin and zero-copy where possible.
5. Enforce per-capability concurrency limits for `subagent` and `llm` to bound fan-out.

## Testing Plan

### Unit tests

1. CI baseline exports (`subagent`, `llm`, `fs`, `glob`, `editFile`) are present when CI middleware is enabled.
2. `subagent(...)` enforces camelCase payload shape and rejects snake_case.
3. `subagent(...)` delegates correctly and preserves structured output semantics.
4. `llm(...)` invokes direct model path and supports structured output.
5. Structured outputs return native JS values; invalid structured responses throw.
6. `fs.readFile`, `fs.writeFile`, `fs.readdir` map correctly to backend operations.
7. `fs.writeFile` default overwrite (`flag: "w"`) and create-only (`flag: "wx"`) semantics are respected.
8. `glob` and `editFile` map correctly to backend operations.
9. Per-capability concurrency caps are enforced independently for `subagent` and `llm`.
10. Baseline APIs work when `ptc=None`.
11. Extension-collision validation rejects attempts to shadow reserved baseline names.

### Integration tests

1. Swarm skill works without explicit `create_swarm_task_tool` wiring.
2. Existing explicit wiring still works unchanged.
3. Mixed setup (baseline + extra PTC tools) works.

### Regression tests

1. Existing Deep Agents `task` behavior is unaffected outside CI baseline wrapper.
2. Existing filesystem middleware behavior is unchanged for non-CI callers.

## Rollout Metrics

1. Adoption of baseline defaults vs legacy explicit wiring.
2. Rate of CI/swarm setup errors in user reports.
3. Performance impact on eval latency and token usage.

## Decision

Proceed with baseline CI capability layer and swarm retrofit, with immediate default-on behavior in the next minor release and phased compatibility cleanup.

This gives users zero-config orchestration primitives while preserving current APIs and minimizing migration risk.

## Readiness

1. Open questions: none.
2. Spec readiness: ready for implementation.

## Implementation Checklist

1. Middleware wiring:
   1. Load built-in baseline extensions by default in `CodeInterpreterMiddleware`.
   2. Enforce reserved-name collision checks at middleware construction.
   3. Add V1 configuration knobs for per-capability caps/timeouts.
2. Extensions:
   1. Implement `SubagentExtension` with task-aligned `subagent(...)` contract.
   2. Implement `LlmExtension` with one-shot `llm(...)` contract.
   3. Implement `FilesystemExtension` for `fs.readFile` / `fs.writeFile` / `fs.readdir`.
   4. Implement `GlobExtension` and `EditFileExtension` for top-level helpers.
3. Semantics and normalization:
   1. Implement structured-output native return conversion and throw-on-invalid behavior.
   2. Implement Node-style `fs.writeFile` flag behavior (`w` / `wx`).
   3. Implement deterministic `readdir` output normalization.
   4. Implement normalized error-code mapping and backend-specific detail suffixes.
   5. Implement deterministic `backend=None` throws for `fs` / `glob` / `editFile`.
4. Swarm retrofit:
   1. Migrate swarm internals to baseline APIs (`subagent`, `llm`, `fs`, `glob`, `editFile`).
   2. Keep `create_swarm_task_tool` compatibility shim + deprecation path.
5. Tests:
   1. Add unit coverage for baseline exports, payload validation, return typing, error mapping, and caps.
   2. Add integration coverage for no-wiring swarm workflows and mixed extension/PTC setups.
   3. Add regression coverage for existing `task` and filesystem middleware behavior.
6. Documentation:
   1. Publish migration guide with before/after snippets.
   2. Document backend capability matrix and unsupported-operation behavior.

## Acceptance Criteria

1. A new CI user can run swarm-style parallel subagent workflows without manually creating or exposing `create_swarm_task_tool`.
2. A skill can rely on `subagent`, `llm`, Node-compatible `fs` subset, `glob`, and `editFile` across projects without tool-name-specific rewrites.
3. Existing users with explicit `ptc` and `create_swarm_task_tool` continue to work unchanged during migration.
4. Structured output schemas are supported through both `subagent` and `llm` baseline APIs.
5. Structured output with `responseSchema` is returned as native JS values (not JSON strings), with strict throw-on-invalid behavior.

## Appendix: Migration Examples

### Example 1: Swarm Dispatch

Before:

```ts
const out = await tools.swarmTask({
  description: "Summarize this PR",
  subagent_type: "reviewer",
  response_schema: schema,
  mode: "agent",
});
```

After:

```ts
const out = await subagent({
  description: "Summarize this PR",
  subagentType: "reviewer",
  responseSchema: schema,
});
```

### Example 2: Direct Model Call

Before:

```ts
const out = await tools.swarmTask({
  description: "Extract entities",
  response_schema: schema,
  mode: "invoke",
});
```

After:

```ts
const out = await llm({
  prompt: "Extract entities",
  responseSchema: schema,
});
```

### Example 3: File Workflow

Before:

```ts
const data = await tools.readFile({ file_path: "/notes.txt" });
const matches = await tools.glob({ pattern: "**/*.md" });
await tools.editFile({
  file_path: "/notes.txt",
  old_string: "TODO",
  new_string: "DONE",
});
```

After:

```ts
const data = await fs.readFile("/notes.txt", { encoding: "utf8" });
const matches = await glob("**/*.md");
await editFile({
  filePath: "/notes.txt",
  oldString: "TODO",
  newString: "DONE",
});
```
