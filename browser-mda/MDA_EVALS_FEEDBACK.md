# MDA evals / Harbor — issues log

Running tally from porting a WebVoyager task to Harbor for a Managed Deep Agent
(`browser-mda`, `managed-deepagents` 0.9.0, `harbor[langsmith]` 0.21.0, macOS).

Status: **7 jobs; trial 7 completed with 0 exceptions.** Jobs 1-6 all failed in
trial-environment setup before the agent ran. Once a trial completed, the
LangSmith experiment row carried **native `total_tokens` and `total_cost`** --
which is the thing that is not obtainable when evaluating a deployment over HTTP
(see related issue 5). Harbor is the right tool for this; the friction is all in
getting a trial to run.

The completed trial scored `reward: 0`, and correctly so: the agent had no shell,
so it could not browse (issue 9).

## What works well

- `mda evals init` scaffolds cleanly and is idempotent; it preserves a
  user-edited `harbor-job.json`.
- `--plugin langsmith` records jobs as LangSmith experiments with no extra code.
- `--env langsmith` provisions per-trial sandboxes **and builds the task image on
  LangSmith infrastructure**, so a local Docker daemon is not required. The docs
  list Docker as a prerequisite; it was not needed.

## Issues

### 1. Generated `datasets` entry does not resolve the task — blocking

`mda evals init` writes `"datasets": [{"path": "evals"}]`. With a task at
`evals/<task>/`, Harbor fails:

```
ValueError: Either datasets or tasks must be provided.
```

Replacing it with `"tasks": [{"path": "evals/<task>"}]` works. Possibly because
`evals/` also contains `harbor-job.json`, but the error names neither cause nor
the directory it searched.

**Ask:** make the generated dataset entry resolve tasks, or have the error report
the path searched and what it required.

### 2. `runtime.json` records a venv-specific path that the documented command breaks

`mda evals init` records an absolute interpreter path:

```
.venv/lib/python3.13/site-packages/managed_deepagents/_binaries/mda
```

The documented run command is `uv run --python 3.12 ... harbor run`, which
re-resolves the project venv to 3.12 and invalidates that path:

```
RuntimeError: The MDA executable recorded for evals is missing or unusable;
rerun `mda evals init` for this project.
```

So the documented command can invalidate the state the preceding command wrote.
The error text is good and the fix works, but the ordering is a trap.

**Ask:** resolve the executable at run time, or record it independently of the
venv's Python version.

### 3. No starter `environment/Dockerfile`, and no documented contract — the big one

`mda evals init` generates `harbor-job.json` and the adapter but leaves
`environment/` to the user. The docs show `environment/Dockerfile` in the task
tree and say nothing about what belongs in it. Four of five failures came from
this.

The actual requirements, learned from `mda_harbor/agent.py` rather than docs:

```python
setup():
    upload runners -> upload artifact to /app
    _install_runtime()    # python -c ..., python -m pip install ./__runtime__, pip install -e .
    _upload_identity_fixture()
    _run_setup_script()   # executes /app/sandbox/setup.sh
```

Which means the trial image must provide:

| requirement | failure when missing |
| --- | --- |
| everything the agent's own `sandbox/setup.sh` needs | `npm: command not found` |
| `python` on `PATH` (not just `python3`) | `python: command not found` |
| a pip that may install into system Python | PEP 668 `externally-managed-environment` |

**Ask:** generate a starter `environment/Dockerfile` matching the MDA sandbox base
(Ubuntu + Node + python, PEP 668 handled), and document the contract.

### 4. `sandbox/setup.sh` runs inside the trial image — undocumented

The agent's sandbox provisioning script executes in the Harbor trial container,
so the trial image effectively *is* the agent's sandbox. Nothing in the MDA evals
docs mentions this, and it is the single most surprising behaviour here: a
`setup.sh` written against the managed sandbox base (Ubuntu 26.04, root, Node 24
preinstalled) silently assumes a base the trial image may not match.

**Ask:** document it, and state which base the script can assume.

### 5. Adapter calls `python`, not `python3`

`_install_runtime` shells out to `python -c ...` and `python -m pip`. Images that
ship only `python3` (most slim Debian bases, including `node:*-slim`) fail. A
symlink fixes it, but the requirement is invisible until it fails.

### 6. `pip install` into system Python trips PEP 668

The runtime install is a plain `python -m pip install`, which Debian-family images
refuse:

```
error: externally-managed-environment
```

Worked around with `PIP_BREAK_SYSTEM_PACKAGES=1` and removing the
`EXTERNALLY-MANAGED` marker. A venv, `--break-system-packages`, or `uv` in the
adapter would avoid pushing this onto every task author.

### 7. Failed trials leave empty experiments with no signal in LangSmith

Harbor creates the LangSmith experiment at job start and writes rows only on
trial completion. All five failed during setup, so LangSmith shows five
experiments with `run_count: None` and no indication anything went wrong. From
the UI it is indistinguishable from "still running".

Diagnosis required local files: `.mda/evals/jobs/<job>/<trial>/exception.txt`.

**Ask:** record setup failures against the experiment, or mark it failed, so the
LangSmith view is not silently empty.

### 8. Docs list `Task.md`; Harbor's own docs do not

The MDA evals task tree includes `Task.md`; Harbor's task docs list
`instruction.md`, `task.toml`, `environment/`, `solution/`, `tests/` and state
there is no `Task.md`. Unclear whether it is required, read by anything, or
MDA-specific convention.

### 9. Harbor eval trials give the agent no shell — blocking for any agent that uses one

`managed_deepagents/_eval_filesystem.py` builds a **disk-only** backend for
Harbor trials:

```python
def create_eval_trial_filesystem_backend(...):
    """Build a disk backend for Harbor eval trials under ``app_dir``."""
```

so `execute` is absent and `deepagents` returns:

```
Error: Execution not available. This agent's backend ...
```

The agent reported it plainly and declined to answer from memory:

> "The `execute` tool, which runs the `browse` browser commands, returned
> 'Execution not available' in this environment. I never opened apple.com."

This is the most consequential issue here. **Any agent whose capability depends
on the sandbox shell -- browsing, code execution, anything driving a CLI -- cannot
be evaluated with Harbor today**, even though the project declares `sandbox/` and
that same script is run during trial setup. The setup script is executed but the
runtime it provisions is then unreachable from the agent.

Note the asymmetry: `sandbox/setup.sh` *does* run in the trial (issue 4), so the
trial pays the cost of installing Chrome and the `browse` CLI, and then the agent
cannot invoke either.

**Ask:** provision a sandbox backend with execution for trials, or make the
limitation explicit at `mda evals init` so authors do not build a task the agent
cannot perform.

### 10. `usage` is null in the agent summary

`agent/summary.json` reports `"usage": null`, so per-trial token accounting is
unavailable from the artifact. LangSmith did compute `total_tokens` and
`total_cost` on the experiment row independently, so this is a reporting gap in
the artifact rather than missing data.

## Related MDA platform issues (not evals-specific)

Hit while building the agent under test; relevant because they shaped the eval
design.

1. **Deployment secrets do not reach the sandbox.** Verified: `env | grep
   ANTHROPIC` inside the sandbox is empty. Reasonable isolation, but it means any
   in-sandbox SDK needing a provider key has no supported way to get one.
2. **`proxy_config` header injection did not inject.** With a rule matching
   `api.anthropic.com` and `{"type": "workspace_secret"}`, a bare `curl` from the
   sandbox with *no* auth header still reached Anthropic unmodified and returned
   `invalid x-api-key`. Documented syntax, no effect observed.
3. **Removing a variable from `.env` does not unset the deployment secret.** A
   stale `ANTHROPIC_BASE_URL` kept routing traffic after being deleted and
   redeployed; it had to be overridden with an explicit value.
4. **A workspace secret appears to shadow the deployment env.** A deployment with
   a correct `ANTHROPIC_API_KEY` in `.env` still presented a different key; using
   a uniquely-named variable was the only reliable fix.
5. **No way to get native Cost/Tokens when evaluating a deployment over HTTP.**
   `aevaluate` against a deployment leaves the experiment run with no LLM spans,
   so the native columns stay empty. Neither `RunTree.to_headers()` nor
   `runs.create(langsmith_tracing={"project_name", "example_id"})` linked the
   deployment run — `reference_example_id` stayed `None`. Harbor is the answer
   here, which is a good reason for issue 3 to be fixed.
6. **LangSmith feedback scores cap at ±99,999.9999.** Token counts exceed this, and
   an over-range score 422s the *entire* multipart batch, silently dropping other
   metrics in it. See `benchmarks/webvoyager/COST_METRICS.md`.

---

# Addendum — first completed trial (second agent)

Appended, not edited, so concurrent writers are not clobbered. One correction to
the header tally: **there has now been 1 completed trial.** It got past every
issue above and failed for a new reason, below.

Same versions (`managed-deepagents` 0.9.0, `harbor[langsmith]` 0.21.0, macOS).

### 9. Eval trials give the agent no execution backend — blocking, and silent

**This is the one that matters.** Everything in issues 1-8 is setup friction you
can work around. This one means a sandbox-dependent agent still cannot be
evaluated after you fix them all.

`managed_deepagents/runtime.py::_prepare_eval_definition` unconditionally
replaces the backend for every eval trial:

```python
# Authoring forbids `backend` on definitions; inject via the compiled config.
merged["backend"] = create_eval_trial_filesystem_backend(FilesystemBackend, ...)
```

Checked against the installed runtime:

```
eval-trial backend type : FilesystemBackend
supports_execution(eval): False
supports_execution(LocalShellBackend): True
```

`FilesystemBackend` does not implement `SandboxBackendProtocol`, so the `execute`
tool returns, from `deepagents/middleware/filesystem.py`:

```
Error: Execution not available. This agent's backend does not support
command execution (SandboxBackendProtocol).
```

The contradiction with issue 4 is the striking part: the adapter **does** run
`sandbox/setup.sh` in the trial container and install Chrome — then hands the
agent a backend that cannot reach it. The sandbox is provisioned and unreachable.

What makes it dangerous is the failure mode. The agent behaved correctly, and
said so:

> "I couldn't look this up on Apple's website... The `execute` tool, which runs
> the `browse` browser commands, returned "Execution not available" in this
> environment. I never opened apple.com. I'm not going to answer from memory,
> because that wouldn't be verified against the site."

The verifier scored that `reward: 0`. Nothing in the job result, the reward file,
or the LangSmith experiment distinguishes it from a genuine capability failure —
only the answer prose does. A browsing suite run this way reports ~0% and looks
like a real measurement. We nearly published that number for two agents.

**Ask:** honor a project-supplied execution backend in eval mode, or default to
`LocalShellBackend` rooted at the trial dir. The Harbor container is already the
isolation boundary — the same argument `sandbox/setup.sh` makes when it runs
Chrome with `--no-sandbox`. Failing that, make `define_sandbox()` in a project a
hard error at eval-compile time rather than a silent capability loss at run time.

### 10. A SIGTERM'd job cancels every in-flight trial, and reads as failure

Harbor's CLI converts SIGTERM into `KeyboardInterrupt`
(`harbor/cli/jobs.py::_handle_sigterm`) and exits 144, cancelling all in-flight
trials. Browsing evals run long — remote image build, per-trial Chrome install,
then the agent — so any run tied to a short-lived shell gets killed mid-flight.

The trial record then shows `CancelledError` and `n_cancelled_trials`, plus a
misleading follow-on from cleanup against an already-terminating sandbox:

```
RuntimeError: Sandbox dataplane URL is not available. Did start() complete?
```

That second error names the sandbox and suggests a provisioning fault, which is
not what happened. Two jobs were lost chasing it.

Launch detached instead — and note **`setsid(1)` does not exist on macOS**, so the
obvious `setsid nohup harbor run ...` fails with `command not found` and the job
never starts. Python's `subprocess.Popen(..., start_new_session=True)` works;
`evals/online-mind2web/run_detached.py` in this repo does it.

**Ask:** mention the long-run/detach requirement in the evals docs, and suppress
or reclassify the dataplane error when the cause is cancellation.

### Re: issue 8 — `Task.md` is required by the skill, not by Harbor

Resolved. `mda evals init` prints a handoff to the `eval-engineering` skill
(`npx skills add langchain-ai/langchain-skills --skill eval-engineering`). That
skill owns `Task.md`: it is the human-reviewed control-plane spec, it must sit
beside `task.toml`, and it must **never** be copied or mounted into the agent's
image or workspace. Harbor itself neither requires nor reads it.

So both docs are right and neither says so. **Ask:** state in the MDA evals task
tree that `Task.md` is an eval-engineering artifact, hidden from the agent.

### Re: issue 3 — a starter Dockerfile that clears issues 3, 5 and 6

Offered as the concrete starter that was asked for. Builds on LangSmith and
handles the `python` and PEP 668 traps:

```dockerfile
FROM node:24-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*
# adapter calls `python`, not `python3` (issue 5)
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
# adapter pip-installs into system Python (issue 6)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN rm -f /usr/lib/python3*/EXTERNALLY-MANAGED
WORKDIR /workspace
```

Caveat on the evidence: this image **builds** on LangSmith, confirmed in the job
log, but has not been driven through a full agent phase — the run carrying it was
lost to issue 10, and the one completed trial used the default image. Treat the
`python`/PEP 668 lines as following issues 5 and 6 rather than as independently
re-verified.

### Confirming issue 1 from a second project

Hit identically in a second project (`browser-mda-stagehand`). `tasks` works,
`datasets` does not. Also note that `evals/harbor-job.json` and a second config
(`evals/<suite>/harbor-job.json`) coexist fine — `harbor run --config <path>`
selects one, which is useful for running several suites against one agent.

### One thing that works better than documented

`--env langsmith` builds the task image on LangSmith infrastructure. The docs
list Docker as a prerequisite; with `environment.type = "langsmith"` in
`harbor-job.json`, **no local Docker daemon is needed at all** — Docker Desktop
was never started on this machine. Worth promoting from a footnote: it removes
the heaviest local prerequisite.
