# Switchyard in-process library pivot

> Preserved experiment, not the active Harbor topology. After the library path
> exposed structured-content conversion failures, the active plan moved to a
> native Harbor Docker sidecar; see `BRANCH_PLAN.md`.
>
> The runtime-build job was removed from the active reusable workflow after
> GitHub rejected `SocketDev/action` and `PyO3/maturin-action` under LangChain's
> enterprise action allowlist, even when the job was skipped. The source,
> patches, adapter, tests, and resume steps remain here. Resuming CI requires an
> allowlisted internal build action or an official prebuilt Switchyard wheel.

This is the recovery note for the primary Harbor approach: run Switchyard's
`libsy` routing engine inside the Deep Agent process instead of as a Docker
Compose sidecar. Keep `SIDECAR_FALLBACK.md` and its implementation intact until
the library path has completed a real Harbor task.

## Why this path

The Compose topology itself worked, but autonomous-lite includes a roughly
15.26 GiB FeatureBench image. Compose mode starts a generic LangSmith sandbox
and pulls the task image into nested Docker, where extracting the largest layer
exceeded the environment startup window. In-process routing lets Harbor use its
normal task snapshots and removes the sidecar, nested Docker, extra image pull,
and sidecar credential plumbing.

No model or provider calls were made while validating this pivot.

## Pinned upstream sources

- LangChain NVIDIA integration: `langchain-ai/langchain-nvidia` at
  `7d91bd6f706ec5745c69fcea00a39fd7c758e44c`
- Switchyard: `NVIDIA-NeMo/Switchyard` at
  `759658ee7d281b603d923a5f333c6c9a91a6ffe7`
- Local patches:
  - `patches/libsy-python-escalation.patch`
  - `patches/langchain-provider-safe-tool-calls.patch`

The patch is stored with zero context to keep repository whitespace checks
clean. Apply it from the pinned Switchyard checkout with
`git apply --unidiff-zero <path-to-patch>`.

The LangChain integration has merged publicly but is not yet published as a
package. Switchyard's required source reports 0.2.0, while PyPI still only has
0.1.0. The runtime therefore has to be built from the pinned sources for now.

## Missing upstream surface and completed patch

The public LangChain adapter exposes noop, random, task-classifier, and stage
routing. It does not expose escalation or passthrough. Both already exist in
Switchyard's Rust core, so the saved patch adds only PyO3 factories and exports:

```python
algorithms.passthrough(target)
algorithms.llm_escalation(
    judge_target,
    efficient_target,
    capable_target,
    confirmations=2,
    recent_turn_window=28,
    window_message_chars=500,
    max_output_tokens=4096,
)
```

The patch also adds upstream-style tests for passthrough and escalation session
latching. The first draft compiled but was missing the lazy exports in
`switchyard_rust.libsy`; the saved patch now includes those exports and their
type-checking stubs. It was validated against the pinned Switchyard commit with:

- `cargo fmt --all --check`
- `cargo test -p switchyard-py` in `rust:1.96.1`
- a reverse-apply check after application

Temporary upstream checkouts used for that validation were under
`/private/tmp/switchyard-pivot.Xhmgch/`; they are disposable because the
commits and complete patch are recorded here.

## Load-bearing session behavior

Every routed request must carry a stable per-Harbor-task session id:

```python
await algorithm.run(
    request,
    headers={"x-switchyard-session-id": harbor_session_id},
)
```

Without this header, `confirmations=2` never accumulates across turns and an
escalation arm silently behaves like the weak baseline. Use
`HARBOR_SESSION_ID` as the source and fail clearly if it is absent in a routed
run.

## Intended eval implementation

Add `deepagents_harbor/langgraph_project/switchyard_library.py` to:

1. Load the selected existing `routes-<arm>.toml` with `tomllib`.
2. Construct provider-native LangChain models, retaining fields such as Nano's
   `extra_body` thinking configuration.
3. Wrap them with `LangChainLlmClient` and build patched passthrough or
   escalation algorithms.
4. Wrap algorithm calls so the session header is always present.
5. Return the weak/baseline model and `SwitchyardRoutingMiddleware` for
   `create_deep_agent`.

The preserved design used this environment switch:

- `HARBOR_SWITCHYARD_CONFIG` set and no image: in-process library mode
- config and `switchyard_image` both set: Compose sidecar mode
- neither set: ordinary Harbor mode

Library mode used Harbor's normal `--env langsmith --agent langgraph` path. It
does not set a router `base_url` or add Compose flags.

## Runtime packaging plan

Do not compile Rust separately in every Harbor task. A dedicated GitHub job
should build the patched Linux/Python 3.12 Switchyard wheel once, assemble it
with the pinned LangChain adapter and route TOMLs, and upload that runtime as an
artifact. Harbor shard jobs download it into:

```text
deepagents_harbor/langgraph_project/.local_deps/switchyard-runtime
```

The integration conditionally prepends that directory to `sys.path`. Ordinary
`langgraph.json` dependencies remain unchanged. Dependency-fetching CI steps
must use the repository's pinned Socket Firewall action; credentials belong
only in the Harbor run jobs and must never enter the runtime build.

The run workflow needs to forward only variable references for
`HARBOR_SWITCHYARD_CONFIG`, `ANTHROPIC_API_KEY`, `BASETEN_API_KEY`,
`GOOGLE_API_KEY`, and `NVIDIA_API_KEY` into the agent environment. Never write
their values to configs or artifacts.

## First autonomous-lite attempt and fixes

The first 15-task Opus attempt on August 6 was invalid and must not be used as
benchmark data. All 15 trials recorded exceptions. Five agents reached Opus but
failed on their second model turn because the pinned LangChain adapter placed a
normalized `tool_call` block in `AIMessage.content`; Anthropic accepts
provider-native `tool_use` blocks and rejected that neutral type. Other trials
hit LangSmith connection, environment-start, or resource-ratio errors, and the
last trials were canceled once the shared failure was confirmed.

The adapter patch now keeps normalized tool calls only in
`AIMessage.tool_calls`. An offline regression sends a Switchyard assistant/tool
history through the real `langchain-anthropic` serializer and confirms that the
outbound block is `tool_use`, not `tool_call`.

`DeepAgentsLangSmithEnvironment` makes two local Harbor fixes reproducible:

1. Directory archives add every nested path exactly once. Stock Harbor's
   recursive `tar.add` behavior duplicated the staged runtime until the upload
   grew to gigabytes.
2. Requested CPU and memory are raised, never reduced, to satisfy LangSmith's
   2-6 GiB-per-vCPU acceptance band.

For the next canary, use one task and concurrency 1, route through
`DeepAgentsLangSmithEnvironment`, set the environment-build timeout multiplier
to 3, set the storage override to 65,536 MiB, and allow retries only for sandbox
connection/start failures. Unified Evals now exposes both resource controls,
and library-mode workflow retries are restricted to `SandboxConnectionError`
and `EnvironmentStartTimeoutError`. Do not restart the 15-task slice until a
tool-using canary completes with no trial exception.

Offline verification after these fixes:

- 15 pinned-adapter conversion tests passed.
- The provider-format regression produced Anthropic `tool_use` content.
- Opus resolves to `ChatAnthropic`, model `claude-opus-4-8`, a 1M-token input
  profile, a 128K-token output limit, and tool calling enabled.
- 351 eval-package unit tests and 98 workflow/runtime tests passed across
  routing, agent construction, runtime assembly, workflow wiring, resource
  normalization, archive staging, and fail-fast eval configuration.

## Known caveat

The server path explicitly adds an Anthropic prompt-cache breakpoint. The
LangChain adapter does not obviously add equivalent `cache_control` metadata.
The publication uses uncached pricing, so this does not change the chosen
headline methodology, but cache-aware sensitivity and invoice parity may differ
until the behavior is matched or documented.

## Resume checklist

Completed locally:

1. `switchyard_library.py` loads the selected TOML, builds provider-native
   targets, constructs passthrough/escalation, and injects the stable Harbor
   session header.
2. `make_bare_graph` opts into the middleware only when the config is forwarded
   into the agent environment.
3. `_harbor_run.yml` builds the patched manylinux wheel once, combines it with
   the pinned public LangChain adapter, shares it as an artifact, and retains
   the old sidecar when an image is supplied.
4. Library mode retains only the OpenAI and Anthropic provider integrations and
   forwards the fixed provider-key allowlist into the agent process.
5. Provider-safe tool-call conversion, archive staging, and LangSmith resource
   normalization are covered by focused regressions.
6. Hermetic routing, artifact-assembly, dependency-pruning, agent-construction,
   and workflow tests pass locally. These checks make no provider calls.

Still to do:

1. Run one Opus Harbor canary that exercises a tool round-trip.
2. Push the corrective checkpoint and let GitHub rebuild/import-smoke the Linux
   runtime when Actions is reliable.
3. Dispatch autonomous-lite only after the canary has no infrastructure or
   adapter exception.

The sidecar fallback, its diagnostic artifacts, and the possible prewarmed
LangSmith snapshot are documented separately in `SIDECAR_FALLBACK.md`.
