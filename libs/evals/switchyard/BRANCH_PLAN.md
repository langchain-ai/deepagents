# Plan: run unified evals through Switchyard

Branch: `srimanth/evals/switchyard-harbor`

## Architecture

Each Harbor trial uses its normal task image in Harbor's native Docker
environment on the GitHub runner. Switchyard is a Compose sidecar sharing
`main`'s network namespace, and the agent calls `http://127.0.0.1:4000/v1`.
There is no LangSmith sandbox, nested Docker daemon, public router, or tunnel.

The implementation uses
`deepagents_harbor.switchyard_environment:SwitchyardDockerEnvironment`. The
adapter mounts the selected route TOML, verifies the loopback health endpoint,
captures `/v1/stats`, and stops Switchyard before Harbor starts verification.
Provider variables use sidecar-only aliases in the GitHub step and are mapped
to their standard names only inside the Switchyard service. The stock LangGraph
launcher therefore never sees an upstream provider variable to copy into
`main`; the OpenAI judge key is likewise kept under a verifier-only alias.

Because the sidecar uses `network_mode: service:main`, Harbor's phase network
policy governs both processes. This preserves no-network verifier enforcement
even for tasks that verify in the shared environment. LangSmith remains only as
the tracing/results plugin.

## Implemented on this branch

1. Native Docker overlay at `switchyard/compose/switchyard-docker.yaml`.
2. Custom Docker environment with TOML mounting, health checking, stats
   capture, and pre-verifier sidecar shutdown.
3. Reusable Harbor workflow inputs:
   - `switchyard_config`: `glm`, `opus`, `nano`, `escalation`, `glm-nano`, or
     `opus-nano`
   - `switchyard_image`: public digest-pinned container reference
4. Unified-evals dispatch inputs plumbed into the reusable workflow.
5. Router model kwargs force `/v1/chat/completions` and use the internal sidecar
   URL; the Harbor session header is stable for every turn of a trial.
6. Per-shard aggregation into `switchyard-stats-summary.json`.
7. A one-task GitHub smoke that runs the previously failing LabBench path with
   Opus and rejects any Harbor trial exception.

## Remaining external prerequisite

A public Switchyard container image must exist. NVIDIA's current release
workflow publishes wheels/sdists, not a container. Either:

1. NVIDIA publishes an official image; or
2. LangChain builds NVIDIA's Apache-2.0 Rust-server Dockerfile at an agreed
   Switchyard commit and publishes it under `ghcr.io/langchain-ai/`.

Whichever path is chosen, dispatches require the immutable
`registry/owner/image@sha256:<digest>` form. Never pass provider credentials as
Docker build arguments and verify the image history before publishing.

## First proof

Run one autonomous lite task, one rollout, with the Opus passthrough arm. It
must prove all of the following before increasing scope:

- the GitHub runner pulls the task and Switchyard images;
- `main` reaches Switchyard on loopback and the health check passes;
- the agent completes and the verifier writes a reward;
- the trial artifacts contain `switchyard-stats.json` with `total_requests > 0`;
- Harbor records zero errored or cancelled trials;
- the LangSmith experiment contains the trace and `harbor_reward` feedback.

Then run the three passthrough baselines and the two Nano escalation arms as
separate workflow dispatches. Start with `categories=autonomous`, `profile=lite`,
and `rollouts=1`; raise rollouts after measuring real cost and variance.

## Pricing decision

Publication pricing is **uncached** for parity with NVIDIA's OpenRouter-based
methodology. Raw snapshots retain cache-read/write fields, and the reporter can
still produce cache-aware numbers as a sensitivity analysis. Use:

```bash
python collect_stats.py report \
  --pricing uncached \
  --rate nemotron-3.5-nano=0.05,0.20 \
  switchyard-stats-summary.json
```
