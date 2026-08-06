# Plan: run unified evals through Switchyard

Branch: `srimanth/evals/switchyard-harbor`

## Architecture

Each Harbor trial uses the LangSmith environment in Docker Compose mode.
Switchyard is a private sidecar on that compose network, and the agent calls
`http://switchyard:4000/v1`. There is no public router endpoint or tunnel.

The implementation uses
`deepagents_harbor.switchyard_environment:SwitchyardLangSmithEnvironment`
because Harbor's `--agent-env` reaches the agent process but not compose
services. The adapter:

- copies only an allowlist of provider variables from the runner environment
  into Docker Compose;
- stages the selected rendered route TOML inside the LangSmith sandbox;
- waits for the sidecar health endpoint before Harbor installs/runs the agent;
- captures `/v1/stats` into the trial artifacts before compose teardown.

This also handles separate no-network verifier environments correctly. Harbor
clears extra compose overlays for the verifier, so the verifier remains isolated
and does not receive the Switchyard sidecar or its provider credentials.

## Implemented on this branch

1. Compose overlay at `switchyard/compose/switchyard.yaml`.
2. Custom LangSmith environment with safe secret forwarding, TOML staging,
   health checking, and stats capture.
3. Reusable Harbor workflow inputs:
   - `switchyard_config`: `glm`, `opus`, `nano`, `escalation`, `glm-nano`, or
     `opus-nano`
   - `switchyard_image`: public digest-pinned container reference
4. Unified-evals dispatch inputs plumbed into the reusable workflow.
5. Router model kwargs force `/v1/chat/completions` and use the internal sidecar
   URL; the Harbor session header is stable for every turn of a trial.
6. Per-shard aggregation into `switchyard-stats-summary.json`.
7. A one-task LangSmith smoke runner in `run_harbor_arms.sh`.

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

Run one autonomous lite task, one rollout, with the Nano passthrough arm. It
must prove all of the following before increasing scope:

- the LangSmith sandbox pulls the image;
- `main` resolves `switchyard` and the health check passes;
- the agent completes and the verifier writes a reward;
- the trial artifacts contain `switchyard-stats.json` with `total_requests > 0`;
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
