# Out-of-process QuickJS worker

A deep agent whose model-written JavaScript runs in a separate worker process that holds no credentials. The agent is composed as usual:

```python
pool = WorkerPool(WorkerConfig(), size=2)
await pool.start()
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[WorkerInterpreterMiddleware(pool, [list_tickets, get_asset])],
    checkpointer=InMemorySaver(),
)
```

`WorkerInterpreterMiddleware` gives the agent an `eval` tool. The JavaScript it runs executes in a worker and reaches the host tools only as `tools.list_tickets(...)` / `tools.get_asset(...)`. Those calls are sent back to the agent's process, so the tools run there, with the host's `TICKETS_API_TOKEN` and the run's config (`get_asset` checks the calling user's team). The worker never sees either.

The worker:

- starts with an empty environment, in an empty temp directory, as `nobody` when the host is root;
- has rlimits on CPU time, open files and core dumps;
- on Linux, installs a seccomp filter after QuickJS has loaded. The filter blocks file opens, sockets, exec and new processes;
- talks to the host only through length-prefixed JSON frames on stdin/stdout;
- is thrown away after a timeout, out-of-memory, a crash, the host's wall-clock limit, or `max_evals` evals. A pool keeps replacements started.

`CodeInterpreterMiddleware` runs the same engine (`quickjs-rs`) inside the agent's process. A runtime option on it (an out-of-process slot behind its `_Slot` interface) would make this setup first-class; until then, this example ships its own small middleware.

## Run

```bash
uv run python agent.py
```

The script uses `anthropic:claude-sonnet-4-6` when `ANTHROPIC_API_KEY` is set (in the environment or a `.env`). Without a key, a scripted stand-in model makes one `eval` call, and the worker, the tools and the permission check still run for real:

```text
[eval]
const tickets = await tools.list_tickets({ status: "open" });
const assets = await Promise.allSettled(tickets.map((t) => tools.get_asset({ asset_id: t.asset_id })));
...

[eval result]
{"result": [{"ticket": "T-101 TLS cert expires in 5 days", "asset": {"name": "payments-api", "severity": "critical", ...}},
            {"ticket": "T-102 Outdated OpenSSH", "asset": "ToolException: alice is not allowed to read asset a-2"}, ...], ...}

the code ran in worker {'op': 'ready', 'pid': 1791, 'uid': 65534, 'seccomp': True}
```

To see what a worker can reach if code breaks out of wasm, and how runaway scripts are recycled:

```bash
uv run python escape_check.py
```

```text
Python running inside a worker, as if it had escaped wasm:
   environment                                   ALLOWED: {'LC_CTYPE': 'C.UTF-8'}
   read /tmp/tmp1qz88dud-token                   denied: Operation not permitted
   read /proc/1/environ                          denied: Operation not permitted
   open a socket                                 denied: Operation not permitted
   run /bin/sh                                   denied: Operation not permitted
Runaway scripts recycle the worker:
   Timeout: worker 1754 replaced by 1764, next eval = 2
   OutOfMemory: worker 1764 replaced by 1766, next eval = 2
```

`LC_CTYPE` is set by Python's own locale coercion; nothing is inherited from the host.

## Containers and gVisor

The worker only needs stdin and stdout, so stronger isolation is just a command prefix:

```python
WorkerConfig(launcher=["unshare", "--user", "--net", "--"])            # empty network namespace
WorkerConfig(launcher=["runsc", "--rootless", "--network=none", "do"])  # gVisor
```

The same frames also work over a Unix socket to a sidecar container that has no service-account token mounted (`automountServiceAccountToken: false`). Such a sidecar can run under a gVisor `runtimeClassName` with its own memory limit.

## Before production

- **Memory:** QuickJS's heap is capped inside wasm (`memory_limit`), but wasmtime reserves about 4 GB of address space per instance, so `RLIMIT_AS` can't cap the process. Use a container or cgroup memory limit.
- **seccomp:** it's Linux-only. On macOS the worker still gets the process boundary, empty env, rlimits and recycling. The filter here is a denylist; build an allowlist from a trace of a real worker and use `SCMP_ACT_KILL_PROCESS`.
- **Startup:** a worker takes about 1 s to start, almost all of it compiling the engine's wasm. The pool hides that, and wasmtime can load a precompiled module (`Module.deserialize_file`) in about 10 ms if the host writes it read-only for workers.
- **Values:** values cross as JSON, so tools should return JSON-serializable data.
