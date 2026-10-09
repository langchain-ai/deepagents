# MDA evals / Harbor — issues log

Running tally from porting a WebVoyager task to Harbor for a Managed Deep Agent
(`browser-mda`, `managed-deepagents` 0.9.0, `harbor[langsmith]` 0.21.0, macOS).

Status: **5 jobs, 0 completed trials.** Every failure was in trial-environment
setup, before the agent ran. The LangSmith integration itself worked throughout.

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
