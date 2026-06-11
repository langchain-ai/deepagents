# Sandbox providers

`deepagents-code` can execute the agent's code in a **remote sandbox** instead
of your local machine. Sandbox providers are **pluggable**: alongside the
curated built-ins, you can use third-party providers published by other
packages, or declare your own in `~/.deepagents/config.toml`.

This guide covers how to select a provider, the built-in providers, how to
declare config providers, and how to publish a provider from your own package.

## Selecting a provider

Sandbox mode is **opt-in** — by default code runs locally (`--sandbox none`).
Pass `--sandbox` with a provider name to enable it:

```bash
dcode --sandbox daytona
```

Related flags:

- `--sandbox-id <id>` — reattach to an existing sandbox (only for providers
  that support it).
- `--sandbox-snapshot-name <name>` — start from a named snapshot (only for
  providers that advertise snapshot support, e.g. `langsmith`, `runloop`).

Passing `--sandbox` **with no value** resolves to `[sandboxes].default` from
your config file:

```bash
dcode --sandbox
```

> Because `--sandbox` takes an optional value, keep the bare form **last** on
> the command line — otherwise a following subcommand (e.g. `dcode --sandbox
> agents`) is consumed as the flag's value. Pass an explicit provider name to
> avoid ambiguity.

If you name a provider that isn't installed or declared, `dcode` prints
install/config guidance listing the available providers.

## Built-in providers

Built-in providers ship as `deepagents-code` extras. `langsmith` is bundled;
the others require installing an extra first (e.g. `/install daytona` in-app or
`dcode --install daytona`).

| Provider | Working directory | Install | Snapshot names | Reattach by id |
| --- | --- | --- | --- | --- |
| `agentcore` | `/tmp` | extra `agentcore` | no | no |
| `daytona` | `/home/daytona` | extra `daytona` | no | yes |
| `langsmith` | `/root` | bundled | yes | yes |
| `modal` | `/workspace` | extra `modal` | no | yes |
| `runloop` | `/home/user` | extra `runloop` | yes | yes |

## How providers are discovered

A `SandboxRegistry` merges three sources. On a name collision, **config wins
over entry points, which win over built-ins** — so you can always override
discovery from your own config file.

1. **Built-ins** — curated in this repo, installed as `deepagents-code` extras.
2. **Entry points** — third-party packages that publish providers under the
   `deepagents_code.sandbox_providers` entry-point group.
3. **Config providers** — declared under `[sandboxes.providers]` in
   `~/.deepagents/config.toml` (the escape hatch for internal/local packages).

## Config providers (`[sandboxes.providers]`)

Declare a provider in `~/.deepagents/config.toml`. This parallels the
`[models]` provider configuration and uses the same `class_path` trust model.

```toml
[sandboxes]
# Used when you run `dcode --sandbox` with no value.
default = "acme"

[sandboxes.providers.acme]
# Required: the provider class to import, in module.path:ClassName format.
class_path = "acme_sandbox.provider:AcmeProvider"
# Default working directory inside the sandbox.
working_dir = "/workspace"
# Suggested when the provider's dependencies are missing.
package = "acme-dcode-sandbox"
# Capability flags (default: supports_sandbox_id = true, snapshot = false).
supports_sandbox_id = true
supports_snapshot_name = false

# Extra keyword arguments forwarded to provider.get_or_create().
[sandboxes.providers.acme.params]
region = "us-east-1"
```

Field reference:

| Key | Required | Purpose |
| --- | --- | --- |
| `class_path` | yes | Provider class as `module.path:ClassName`. |
| `working_dir` | no | Default working directory (defaults to `/workspace`). |
| `package` | no | Package name suggested when dependencies are missing. |
| `supports_sandbox_id` | no | Whether `--sandbox-id` reattach is allowed. |
| `supports_snapshot_name` | no | Whether `--sandbox-snapshot-name` is allowed. |
| `params` | no | Keyword args forwarded to `get_or_create()`. |

A config entry that names the same key as a built-in **overrides** that
built-in while keeping the built-in's dependency pre-flight check.

> **Trust model:** `class_path` causes `dcode` to import and run arbitrary
> Python from the named module — module-level code executes on import. This is
> the same trust model as model `class_path` and `pyproject.toml` build
> scripts: you control your own machine and your own config file.

Malformed entries are skipped with a warning rather than crashing startup. If
the config file itself can't be parsed, sandbox errors include a breadcrumb
noting that providers/defaults it declared were ignored.

## Publishing a provider from your package

To distribute a provider so users can `dcode --sandbox <name>` after installing
your package, publish an entry point under `deepagents_code.sandbox_providers`.

1. Implement a `SandboxProvider` subclass:

   ```python
   from deepagents_code.integrations.sandbox_provider import (
       SandboxInstallHint,
       SandboxProvider,
       SandboxProviderMetadata,
   )


   class AcmeProvider(SandboxProvider):
       @property
       def metadata(self) -> SandboxProviderMetadata:
           return SandboxProviderMetadata(
               name="acme",
               working_dir="/workspace",
               install=SandboxInstallHint(kind="package", name="acme-dcode-sandbox"),
               supports_sandbox_id=True,
               supports_snapshot_name=False,
           )

       def get_or_create(self, *, sandbox_id=None, **kwargs):
           ...  # return a SandboxBackendProtocol

       def delete(self, *, sandbox_id, **kwargs):
           ...
   ```

   Implement `get_or_create` and `delete`; async callers are handled by the
   base class's `aget_or_create` / `adelete` wrappers. Override the `metadata`
   property so the registry can surface your working directory and capability
   flags without instantiating the provider — if you don't, a generic default
   (`/workspace`, no snapshot support) is used.

2. Register the entry point (e.g. in `pyproject.toml`):

   ```toml
   [project.entry-points."deepagents_code.sandbox_providers"]
   acme = "acme_sandbox.provider:AcmeProvider"
   ```

Once the package is installed, the provider appears in
`dcode --sandbox <name>` and in the "available providers" listing without any
changes to `deepagents-code`.
