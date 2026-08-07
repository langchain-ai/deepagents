# Switchyard sidecar fallback

> Historical LangSmith-sandbox topology. The active sidecar plan now uses
> Harbor's native Docker environment; see `BRANCH_PLAN.md`. Keep this document
> for the nested-Docker findings and recovery work already completed.

This preserves the Docker Compose / LangSmith sandbox approach while the
in-process `libsy` middleware path is evaluated. Do not delete the sidecar code;
it remains a viable fallback for server-only Switchyard features.

## Preserved state

- Branch: `srimanth/evals/switchyard-harbor`
- Last pushed sidecar commit: `a63ae3119`
- Remote: `origin/srimanth/evals/switchyard-harbor`
- The working tree also contains unrelated user-owned modified and untracked
  files. Preserve them when switching approaches.

The implementation is additive and lives primarily in:

- `deepagents_harbor/switchyard_environment.py`
- `deepagents_harbor/switchyard_agent.py`
- `switchyard/compose/switchyard.yaml`
- `_harbor_run.yml` and `unified_evals.yml`
- the corresponding Switchyard environment and workflow tests

The topology itself was proven by earlier GPQA canaries:

```text
LangSmith sandbox
└─ Docker Compose
   ├─ main/agent -> Switchyard only
   └─ Switchyard -> model providers
```

Provider credentials are forwarded to Compose through a fixed allowlist and
are never serialized into job configuration. The main task container is
disconnected from the egress network after agent setup. Switchyard statistics
are captured before teardown.

## Current blocker

The autonomous lite suite includes the FeatureBench image
`crystalxyz/add-feature-mlflow-bedrock-autolog:20260624-212604`.

- The image is about 15.26 GiB compressed.
- Its largest layer is about 6.15 GB.
- Pulls in 32 GiB and 64 GiB LangSmith sandboxes passed download but exceeded
  the environment startup window while extracting that layer.
- Both attempts stopped before any model or verifier call, so they spent no LLM
  tokens.
- Diagnostic job output is under
  `libs/evals/harbor-jobs/switchyard-opus-local-featurebench*`.

The failure is nested-Docker image extraction, not Switchyard routing or an
invalid task image. Ordinary Harbor avoids this because it boots the task's
native LangSmith snapshot directly; the Compose path instead starts a generic
sandbox and pulls the task image into its inner Docker daemon.

## Snapshot experiment

A zero-model-token prewarm was started in LangSmith:

- Sandbox: `switchyard-fb-prewarm-4c88234d`
- Intended snapshot: `switchyard-featurebench-prewarm-v1`
- Resources: 8 vCPU, 16 GiB RAM, 64 GiB disk

The public LangSmith documentation confirms that a captured snapshot preserves
the persistent filesystem while excluding processes, memory, sockets, and
`/tmp`. `/var/lib/docker` is on the persistent root filesystem in this sandbox,
so a completed pull should be capturable and testable by booting a fresh
sandbox and running `docker image inspect`.

One Harbor seam must be patched before this can help Compose runs. In the pinned
Harbor implementation, `LangSmithEnvironment.start()` calls
`_start_default_sandbox()` in Compose mode, and `_start_default_sandbox()`
always calls `_start_sandbox(snapshot_name=None)`. Therefore a supplied
`snapshot_name` is ignored for Compose. The custom Switchyard environment needs
to resolve and boot the named snapshot explicitly, with a unit test covering
that path.

## Other operational findings

- Local Codex network isolation cannot currently provide both external network
  access and provider keys to the same child process. Local model runs therefore
  fail before a model call; this is an execution-boundary limitation, not repo
  code.
- GitHub Actions was experiencing a major hosted-runner outage during the last
  CI attempts. Jobs queued or timed out before meaningful execution.
- A direct, non-Switchyard unified autonomous-lite run is 15 tasks and uses the
  normal LangSmith snapshots; it does not hit the nested-Docker problem.

## Resume checklist

1. Confirm the prewarm sandbox or snapshot still exists.
2. Finish the image pull if necessary, stop Docker, sync the filesystem, and
   capture `switchyard-featurebench-prewarm-v1`.
3. Boot a disposable sandbox from that snapshot and prove the image is present
   with `docker image inspect` before touching Harbor.
4. Override Compose startup in `SwitchyardLangSmithEnvironment` so an explicit
   snapshot is honored; add a hermetic unit test.
5. Add an optional workflow snapshot input and run a zero-model Compose boot.
6. Only then run one real GPQA or autonomous task, followed by unified lite.

If the snapshot does not retain Docker's layer store in practice, the remaining
sidecar options are a smaller preloaded base snapshot/image or a hosted
Switchyard server. The in-process middleware path avoids this entire class of
problems and is now the primary experiment.
