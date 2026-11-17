# langchain-mxc

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-mxc?label=%20)](https://pypi.org/project/langchain-mxc/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-mxc)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-mxc)](https://pypistats.org/packages/langchain-mxc)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-mxc
npm install @microsoft/mxc-sdk   # the MXC executor
```

```python
from deepagents import create_deep_agent
from langchain_mxc import MXCSandbox

backend = MXCSandbox(root_dir="./my-project")
result = backend.execute("echo hello")
print(result.output)

agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", backend=backend)
```

## 🤔 What is this?

A [Deep Agents](https://github.com/langchain-ai/deepagents) backend that runs the agent's shell commands inside [Microsoft Execution Containers (MXC)](https://github.com/microsoft/mxc) on the machine where the agent runs. The operating system limits what each command can touch: writes go only to the project folder and a scratch folder, reads are limited to those plus the toolchain, credential folders are blocked, and the network is off unless you turn it on.

Unlike remote sandbox providers, MXC does not create a separate machine. Each command gets its own process sandbox over your real project folder, removed when the command exits, so files persist between commands.

Only shell commands go through MXC. The file tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) run in the agent's process and are limited to `root_dir` by Deep Agents' path checks, as with `LocalShellBackend`. **Those checks are not enforced by the operating system; `execute` is.**

## Requirements

| OS | Minimum | Enforcement |
| --- | --- | --- |
| Windows 11 | 24H2, build 26100.9278 or newer | Process container (AppContainer) |
| macOS | 15 or newer, Apple silicon | Seatbelt |
| Linux | bubblewrap 0.5+ with unprivileged user namespaces. `allow_network=True` also needs `slirp4netns`, `ip` (iproute2) and `iptables` | bubblewrap |

MXC ships as the `@microsoft/mxc-sdk` npm package, which needs Node.js 24 or newer to install. Node is not used at runtime.

`MXCSandbox` finds the executor (`wxc-exec.exe`, `mxc-exec-mac` or `lxc-exec`) in this order: the `executor_path` argument; the `MXC_EXECUTOR_PATH` or `MXC_BIN_DIR` environment variable; `node_modules/@microsoft/mxc-sdk` in the working directory or a parent; common global npm locations; `PATH`. If none is found, the constructor raises `MXCExecutorNotFoundError` listing the paths searched.

## Configuration

```python
backend = MXCSandbox(
    root_dir="./my-project",
    read_only_paths=["~/datasets"],  # extra folders commands can read
    read_write_paths=[],  # extra folders commands can write
    allow_network=False,  # outbound network; inbound stays blocked
    allow_ui=False,  # let commands use the window system
    clipboard="none",  # "none", "read", "write" or "all"
    allow_input_injection=False,  # synthetic keyboard and mouse input
    env={"PYTHONUNBUFFERED": "1"},  # added to the minimal sandbox environment
    timeout=120,  # seconds per command
    max_output_bytes=100_000,
    executor_path=None,  # found automatically
)
```

`backend.build_request("echo hi")` returns the exact request sent to MXC.

## Default policy

| Area | Default |
| --- | --- |
| Read and write | `root_dir`, a per-backend scratch folder (`HOME`, `TEMP` and `TMP` point at it), `read_write_paths` |
| Read only | The system toolchain, the folders holding `python`, `python3`, `git` and `node`, and `read_only_paths` |
| Blocked | Everything else. Credential folders (`~/.ssh`, `~/.aws`, `~/.azure`, `~/.gnupg`, `~/.kube`, `~/.docker`, `~/.config/gcloud`, `~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`) stay blocked even under a broader grant such as `read_only_paths=["~"]`, unless named directly |
| Network | Off. `allow_network=True` allows outbound connections to any host; MXC has no per-host allow list |
| Window system, clipboard, input injection | Off (MXC's defaults). Enforced on Windows and macOS; Linux ignores them |
| Environment | A minimal set. The host environment, including API keys, is not passed in |
| Timeout | 120 seconds per command, exit code 124 on timeout, matching `LocalShellBackend` |

## Windows

Observed with `wxc-exec.exe` 1.0.1 on Windows 11 build 28000 (ARM64):

- **Most toolchains need `allow_ui=True`.** With the window system locked, `git`, `node`, PowerShell and `pytest` cannot start, because they load window-system libraries at startup. The default stays locked; coding agents on Windows usually want `MXCSandbox(root_dir=..., allow_ui=True)`. Clipboard and input injection stay off unless set separately. When a program fails for this reason, the response says so and names `allow_ui=False`.
- **`del`, `erase` and PowerShell's `Remove-Item` are denied, even inside `root_dir`.** Python's `os.remove`, `git rm`, `git clean` and `rmdir /s /q` work, and the response tells the agent so when `del` is denied.

## What the model sees when something is blocked

The command's own error, followed by one line naming what the policy blocked, so the agent can tell the user instead of retrying:

```
[stderr] cat: /Users/me/.ssh/config: Operation not permitted

Exit code: 1
Blocked by sandbox policy: /Users/me/.ssh/config. Ask the user to grant access if this is needed.
```

## Security model

- **Fails closed.** On the first `execute`, the backend validates the policy with the executor and runs a canary that must fail to read a file outside every allowed path. If MXC is missing, rejects the policy or cannot start a sandbox, `execute` returns an error and nothing runs. Commands never run unsandboxed.
- **No enforcement-off modes.** MXC's learning, permissive and `--audit` modes are never used, and tests check that no request contains them.
- **Not covered:** commands run as your user, can use CPU, memory and disk inside `root_dir`, and with `allow_network=True` can reach any host including your local network.

[`EXECUTOR_CONTRACT.md`](EXECUTOR_CONTRACT.md) records everything observed about the MXC executors on each OS, including where they differ from MXC's documentation.

## Tests and demo

```bash
make test                 # unit tests, no MXC needed
make integration_tests    # needs the MXC executor; skipped with a reason otherwise

uv sync --group test --group examples
uv run python examples/demo_agent.py --scripted   # real sandbox, no model or API key
uv run python examples/show_policy.py ./my-project --exercise
```

The demo agent fixes a failing test in `examples/sample_repo` inside the sandbox and is then blocked from reading a file outside it. Without `--scripted` it uses a real model (`ANTHROPIC_API_KEY`).

## 📕 Releases & Versioning

See the [changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/partners/mxc/CHANGELOG.md) for release history.

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
