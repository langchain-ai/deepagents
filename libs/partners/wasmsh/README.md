# langchain-wasmsh

Wasmsh sandbox integration for [Deep Agents](https://github.com/langchain-ai/deepagents).
Runs bash and Python 3 inside an in-process Pyodide/WASM sandbox locally — no
container or cloud service required.

## Getting started

### Requirements

- Python 3.11+
- Deno 2+ (preferred) or Node.js 20+ — Deno provides sandboxed permissions for defense-in-depth

### Install

```bash
pip install langchain-wasmsh
```

### Create an agent with a wasmsh sandbox

```python
from deepagents import create_deep_agent
from langchain_wasmsh import WasmshSandbox

backend = WasmshSandbox()
try:
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a coding assistant with bash and Python access.",
        backend=backend,
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Write a script that computes fibonacci(10) and run it."}]
    })

    print(result["messages"][-1].content)
finally:
    backend.close()
```

The agent automatically gets `execute`, `read_file`, `write_file`, `edit_file`,
`ls`, `glob`, and `grep` tools — all routed through the sandbox.

## How-to guides

### Seed files before the agent runs

Pass `initial_files` to pre-populate `/workspace`:

```python
backend = WasmshSandbox(
    initial_files={
        "/workspace/data.csv": b"name,score\nalice,95\nbob,87\n",
        "/workspace/config.json": '{"threshold": 90}',
    },
)
```

Both `bytes` and `str` values are accepted (strings are UTF-8 encoded).

### Retrieve files after execution

Use `download_files` to pull artifacts out of the sandbox:

```python
results = backend.download_files(["/workspace/report.txt"])
if results[0].error is None:
    print(results[0].content.decode())
```

### Upload files at runtime

```python
backend.upload_files([("/workspace/input.txt", b"new data")])
```

### Run bash and Python in the same session

Bash and Python share the same `/workspace` filesystem. Write a file in one
language, read it in the other:

```python
# Bash writes a JSON file
backend.execute('echo \'{"status": "ok"}\' > /workspace/status.json')

# Python reads and validates it
result = backend.execute(
    "python3 -c \""
    "import json; "
    "d = json.load(open('/workspace/status.json')); "
    "print(d['status'])\""
)
print(result.output)  # ok
```

### Use a custom working directory

By default, all commands run relative to `/workspace`. Override this:

```python
backend = WasmshSandbox(working_directory="/home/user")
```

### Limit execution budget

Set `step_budget` to cap the number of VM steps per command:

```python
backend = WasmshSandbox(step_budget=100_000)
```

A budget of `0` (the default) means unlimited.

## Reference

### `WasmshSandbox(*, runtime, dist_dir, working_directory, step_budget, initial_files, allowed_hosts)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runtime` | `str \| None` | auto-detect | Runtime path — prefers Deno, falls back to Node.js |
| `dist_dir` | `str \| Path \| None` | auto-resolved | Path to Pyodide distribution assets |
| `working_directory` | `str` | `"/workspace"` | Working directory for `execute()` |
| `step_budget` | `int` | `0` (unlimited) | VM step budget per command |
| `initial_files` | `dict[str, str \| bytes] \| None` | `None` | Files to seed at creation |
| `allowed_hosts` | `list[str] \| None` | `None` (deny all) | Hostnames allowed for network access |

Raises `FileNotFoundError` if neither Deno nor Node.js is found on `PATH`.
When using Node.js, `allowed_hosts` is still enforced at the wasmsh level but
lacks Deno's OS-level permission isolation.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique sandbox identifier (e.g., `wasmsh-python-<uuid>`) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(command, *, timeout=None)` | `ExecuteResponse` | Run a shell command (prepends `cd /workspace &&`). `timeout` is accepted for protocol compatibility but not enforced; use `step_budget` instead. |
| `upload_files(files)` | `list[FileUploadResponse]` | Write files into the sandbox |
| `download_files(paths)` | `list[FileDownloadResponse]` | Read files from the sandbox |
| `close()` | `None` | Shut down the host subprocess |
| `stop()` | `None` | Alias for `close()` |

### Inherited from `BaseSandbox`

These methods delegate to `execute()` and/or `upload_files()` — no additional
setup required:

`write`, `ls`, `glob`, `grep`

`read` and `edit` are overridden to use `download_files`/`upload_files`
instead of `execute()` because the Pyodide runtime's I/O layer does not
support the Python scripts that `BaseSandbox` generates for these operations.

### `ExecuteResponse`

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Combined stdout + stderr |
| `exit_code` | `int \| None` | Exit code, or `None` if unavailable |
| `truncated` | `bool` | Always `False` for wasmsh |

### Error mapping

Diagnostic events from the wasmsh runtime are mapped to `FileOperationError`:

| Diagnostic contains | Mapped to |
|---------------------|-----------|
| `"not found"` | `"file_not_found"` |
| `"directory"` | `"is_directory"` |
| `"permission"` | `"permission_denied"` |
| *(other)* | `"invalid_path"` |

## Explanation

### What runs inside the sandbox

The wasmsh runtime provides 88 shell utilities (including `jq`, `awk`, `rg`,
`fd`, `diff`, `tar`, `gzip`) plus `python`/`python3` via an embedded CPython
interpreter. Both share the same Emscripten POSIX filesystem.

This is **not** a Linux container. There is no kernel, no process isolation, no
`apt`, `pip install`, or `docker`. If you need a full OS environment, use a
container-based provider like `langchain-modal` or `langchain-daytona`.

### How it works

The Python provider launches a Deno subprocess that boots the
Pyodide/Emscripten WebAssembly module. Communication between Python and the
Deno host uses JSON-RPC over stdin/stdout. Deno's permission model provides
defense-in-depth: the subprocess is restricted to reading only the asset
directory (`--allow-read=<dir>`) and accessing only explicitly allowed
network hosts (`--allow-net=<hosts>`).

```
Python process          Deno subprocess
┌─────────────┐        ┌──────────────────────┐
│ WasmshSandbox│──JSON──│ node-host.mjs        │
│ (BaseSandbox)│  RPC   │   ├─ Pyodide/WASM    │
│              │◄─────►│   ├─ wasmsh runtime   │
│              │ stdin/ │   └─ CPython 3.13     │
│              │ stdout │                      │
└─────────────┘        └──────────────────────┘
```

### How the agent uses the sandbox

When you pass a `WasmshSandbox` as the `backend` to `create_deep_agent`, the
agent gains filesystem tools and a shell `execute` tool:

- **Filesystem tools** (`read_file`, `write_file`, `edit_file`, `ls`, `glob`,
  `grep`) are implemented by `BaseSandbox` using POSIX shell commands via
  `execute()`.
- **`execute()`** prepends `cd /workspace &&` to every command.
- **`initial_files`** are written during sandbox creation before any agent
  code runs.

### Lifecycle

The Deno host process starts when `WasmshSandbox()` is constructed and stops
when `close()` (or its alias `stop()`) is called. Always use try/finally to
avoid orphaned Deno processes.
