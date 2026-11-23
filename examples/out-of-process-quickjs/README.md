# Out-of-process QuickJS worker

`CodeInterpreterMiddleware` runs model-written JavaScript inside the agent's own process, sandboxed by wasm. For multi-tenant deployments that want a second wall, this example runs the same QuickJS engine (`quickjs-rs`) in a separate worker process that holds no credentials. `tools.*` calls are sent back to the host, so tools still run and authorize there with the host's credentials. The credentials never reach the worker.

The worker:

- starts with an empty environment, in an empty temp directory, as `nobody` when the host is root;
- has rlimits on CPU time, open files and core dumps;
- on Linux, installs a seccomp filter after QuickJS has loaded. The filter blocks file opens, sockets, exec and new processes;
- talks to the host only through length-prefixed JSON frames on stdin/stdout;
- is thrown away after a timeout, out-of-memory, a crash, the host's wall-clock limit, or `max_evals` evals. A small pool keeps replacements started.

## Run

```bash
uv run python demo.py
```

Run it as root (e.g. in a container) to also see the switch to `nobody`.

```text
1. tools.* calls run in the host:
   {"result": {"assets": [...], "refused": "ToolException: not authorized to delete asset 1"}, "console": "2 high-severity assets", ...}
   worker: {'op': 'ready', 'pid': 1432, 'uid': 65534, 'seccomp': True}
2. Python running inside a worker, as if it had escaped wasm:
   environment                                   ALLOWED: {'LC_CTYPE': 'C.UTF-8'}
   read /tmp/tmp1cj8bitk-token                   denied: Operation not permitted
   read /proc/1/environ                          denied: Operation not permitted
   open a socket                                 denied: Operation not permitted
   run /bin/sh                                   denied: Operation not permitted
3. Runaway scripts recycle the worker:
   Timeout: worker 1432 replaced by 1439, next eval = 2
   OutOfMemory: worker 1439 replaced by 1442, next eval = 2
```

`LC_CTYPE` is set by Python's own locale coercion at startup; nothing is inherited from the host. Step 2 uses a demo-only `probe` frame (`allow_probe=True`) that stands in for code that has broken out of wasm.

## Use it in an agent

`Interpreter.as_tool()` returns an `eval` tool that works with any LangChain agent:

```python
pool = WorkerPool(WorkerConfig(), size=4)
await pool.start()
interpreter = Interpreter(pool, [lookup_asset, delete_asset])
agent = create_deep_agent(model=..., tools=[interpreter.as_tool()])
```

Use one `Interpreter` per conversation. It keeps its worker, and so its JavaScript state, between evals until the worker is recycled.

## Containers and gVisor

The worker only needs stdin and stdout, so stronger isolation is just a command prefix:

```python
WorkerConfig(launcher=["unshare", "--user", "--net", "--"])        # empty network namespace
WorkerConfig(launcher=["runsc", "--rootless", "--network=none", "do"])  # gVisor
```

The same frames also work over a Unix socket to a sidecar container that has no service-account token mounted (`automountServiceAccountToken: false`). Such a sidecar can run under a gVisor `runtimeClassName` with its own memory limit.

## Before production

- **Memory:** QuickJS's heap is capped inside wasm (`memory_limit`), but wasmtime reserves about 4 GB of address space per instance, so `RLIMIT_AS` can't cap the process. Use a container or cgroup memory limit.
- **seccomp:** it's Linux-only. On macOS the worker still gets the process boundary, empty env, rlimits and recycling. The filter here is a denylist; build an allowlist from a trace of a real worker and use `SCMP_ACT_KILL_PROCESS`.
- **Startup:** a worker takes about 1 s to start, almost all of it compiling the engine's wasm. The pool hides that, and wasmtime can load a precompiled module (`Module.deserialize_file`) in about 10 ms if the host writes it read-only for workers.
- **Values:** values cross as JSON, so tools should return JSON-serializable data.
