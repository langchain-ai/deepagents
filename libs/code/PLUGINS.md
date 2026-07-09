# Plugins

Plugin support is experimental. Set `DEEPAGENTS_CODE_EXPERIMENTAL=1` before
starting `dcode`.

## Add and install

Open `/plugins` for the interactive manager, or use the scriptable CLI:

```bash
dcode plugin marketplace add owner/repo
dcode plugin install formatter@company-tools --scope user --trust
dcode plugin list
```

Marketplace sources may be GitHub repositories, Git URLs, marketplace JSON
URLs, local JSON files, or local directories. Installed plugins are copied into
a versioned cache under `~/.deepagents/plugins/cache/`. Use repeatable
`--plugin-dir PATH` arguments for session-only development copies that should
load directly from a working tree.

Enablement precedence is local, then project, then user. Project enablement is
stored in `<project>/.deepagents/plugins.json`; local overrides and user state
remain under `~/.deepagents/.state/`.

Use `/reload-plugins` after installing, updating, enabling, or disabling a
plugin. Reload builds a complete runtime snapshot and retains the previous
snapshot if the replacement cannot be built.

## Components

| Component | Support |
| --- | --- |
| Skills | `skills/`, manifest paths, root `SKILL.md` extension |
| Prompt commands | `commands/`, nested Markdown, inline manifest commands |
| Agents | `agents/` Markdown with namespaced names |
| Hooks | command hooks for the MVP event set described below |
| MCP | `.mcp.json`, wrapped/direct maps, inline manifest servers |

Runtime names use `plugin-name:component`. Storage and provenance use the full
`plugin-name@marketplace` identifier. MCP servers keep a path-safe internal key
while plugin ownership remains visible in plugin details.

Unsupported components are inventoried and reported rather than failing the
plugin. This includes LSP servers, MCP bundles, output styles, themes, monitors,
plugin settings, channels, user configuration, and package-manager sources.

## Trust

Skills, prompt commands, and agents may shape model behavior. MCP servers and
hooks additionally execute processes or make network requests.

Interactive installation records trust for the exact plugin version and
executable-surface fingerprint. CLI installation is fail-closed unless
`--trust` is supplied or `dcode plugin trust <id>` is run afterward. Changes to
MCP or hook configuration invalidate trust and require another approval.

Plugin-agent `permissionMode`, `hooks`, and `mcpServers` frontmatter is ignored.
Plugin pre-tool hooks can tighten execution policy but cannot override a user
denial. Uninstall only deletes paths contained by the managed plugin cache.

## Hooks MVP

Plugin hooks receive the compatibility envelope with `session_id`,
`transcript_path`, `cwd`, and `hook_event_name`. Tool hooks receive mapped tool
names and inputs, such as `Bash`, `Read`, `Write`, and `Edit`.

Supported client events include session start/end, permission and notification
events, user prompt submission, and normal stop notifications. Supported
server events include pre-tool, post-tool success/failure, and subagent
start/stop notifications.

`PreToolUse` honors `allow`, `deny`, and `ask`; exit code 2 denies and sends
stderr back to the model. `PostToolUse` and `PostToolUseFailure` are
observational and may add model context. `UserPromptSubmit` may block a prompt
or add context. Input and tool-output mutation fields are intentionally ignored.

## Diagnostics

Use these commands when a plugin does not load:

```bash
dcode plugin info formatter@company-tools
dcode doctor
dcode plugin marketplace update company-tools
dcode plugin update formatter@company-tools
```

The `/plugins` Errors tab and `/reload-plugins` report isolate malformed
plugins and components without stopping sibling plugins or the agent runtime.
