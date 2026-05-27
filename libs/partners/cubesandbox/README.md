# langchain-cubesandbox

[CubeSandbox](https://github.com/TencentCloud/CubeSandbox) sandbox integration for Deep Agents.

CubeSandbox is a high-performance, secure sandbox service built on RustVMM and KVM
that delivers true kernel-level isolation for AI-agent generated code. This
package adapts the official `cubesandbox` Python SDK into a
[`SandboxBackendProtocol`](https://reference.langchain.com/python/deepagents/backends/protocol/SandboxBackendProtocol)
backend so Deep Agents can drive CubeSandbox the same way they drive Daytona,
Modal, Runloop, and other supported sandboxes.

## Quick install

```bash
pip install langchain-cubesandbox
```

## Usage

```python
import cubesandbox

from langchain_cubesandbox import CubeSandbox

sandbox = cubesandbox.Sandbox.create()
backend = CubeSandbox(sandbox=sandbox, timeout=300)
try:
    result = backend.execute("echo hello")
    print(result.output)
finally:
    sandbox.kill()
```

The wrapper inherits `ls / read / write / edit / glob / grep` from
[`BaseSandbox`](https://reference.langchain.com/python/deepagents/backends/sandbox/BaseSandbox)
and implements `execute`, `upload_files`, and `download_files` against the
CubeSandbox SDK.

## Configuration

CubeSandbox is configured primarily through environment variables consumed by
the official SDK:

| Variable | Description |
| --- | --- |
| `CUBE_API_URL` | Base URL of the CubeAPI control plane. |
| `CUBE_API_KEY` | API key when the deployment requires authentication. |
| `CUBE_TEMPLATE_ID` | Default template ID used by `Sandbox.create()`. |
| `CUBE_PROXY_NODE_IP` | If set, the SDK bypasses DNS and connects to this IP via `IPOverrideTransport`. |

Refer to the [CubeSandbox docs](https://docs.cubesandbox.ai/) for deployment
details.

## Implementation notes

* Shell commands run via a small Python wrapper that we send through the
  SDK's `run_code` envd channel ourselves, instead of calling
  `cubesandbox.Sandbox.commands.run`. The wrapper captures the command's
  stdout/stderr with `subprocess.run(..., capture_output=True)`, writes
  stdout verbatim, then appends an unambiguous sentinel framing the integer
  exit code; stderr is written separately so envd routes it to
  `Execution.logs.stderr`. Going around `commands.run` is required because
  that helper concatenates `print(returncode)` immediately after captured
  stdout — when the command's stdout has no trailing newline (e.g. `cat` on
  a file without `\n` at EOF), the exit-code digit ends up glued to the
  last line of output and the SDK's splitlines-based parser fails to
  separate them. Detailed write-up lives in
  `cubesandbox-tryout/CUBESANDBOX_SDK_BUG_REPORT.md`.
* File upload/download use short Python helpers shipped over `run_code`, with
  the payload encoded as base64 inside the script source. This avoids the
  shell-escape hazards of `commands.run` and works for arbitrary binary
  content.
* `cubesandbox` 0.1.0 is synchronous; the async protocol methods
  (`aexecute`, `aupload_files`, `adownload_files`) inherit the default
  `asyncio.to_thread` wrappers from `SandboxBackendProtocol`. When the
  upstream SDK gains a native async client, this wrapper can be upgraded
  in-place.

## Limitations

* CubeSandbox's `IPOverrideTransport` currently focuses on HTTP and does not
  perform hostname-aware SNI for HTTPS endpoints. Direct-connect deployments
  should expose HTTP via CubeProxy.
* `cubesandbox.Filesystem.read()` reads files as text; this wrapper does
  **not** use it. Binary-safe downloads are performed via the base64
  `run_code` helper described above.

## Testing

Unit tests run against mocks and do not require a CubeSandbox deployment:

```bash
make test
```

Integration tests require the environment variables above:

```bash
make integration_test
```

## Releases & versioning

See the LangChain [Releases](https://docs.langchain.com/oss/python/release-policy)
and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.
