# Switchyard in-process library pivot

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
- Local patch: `patches/libsy-python-escalation.patch`

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
latching. It was validated against the pinned Switchyard commit with:

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

Start with `agent_impl=bare` only. The environment switch should be:

- `HARBOR_SWITCHYARD_CONFIG` set and no image: in-process library mode
- config and `switchyard_image` both set: preserved Compose sidecar mode
- neither set: ordinary Harbor mode

Library mode uses Harbor's normal `--env langsmith --agent langgraph` path. It
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

## Known caveat

The server path explicitly adds an Anthropic prompt-cache breakpoint. The
LangChain adapter does not obviously add equivalent `cache_control` metadata.
The publication uses uncached pricing, so this does not change the chosen
headline methodology, but cache-aware sensitivity and invoice parity may differ
until the behavior is matched or documented.

## Resume checklist

1. Implement `switchyard_library.py` and hermetic unit tests.
2. Wire `make_bare_graph` to opt into it only when
   `HARBOR_SWITCHYARD_CONFIG` is set for library mode.
3. Add the one-time patched runtime build and artifact download jobs.
4. Update `_harbor_run.yml` to select library versus sidecar based on whether
   a Switchyard image was supplied.
5. Run targeted pytest and Ruff checks.
6. Run a no-provider noop/static-client smoke.
7. Dispatch one real Harbor task before autonomous-lite.

The sidecar fallback, its diagnostic artifacts, and the possible prewarmed
LangSmith snapshot are documented separately in `SIDECAR_FALLBACK.md`.
