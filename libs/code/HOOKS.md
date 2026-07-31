# Hooks

Hooks are user-configured shell commands that run at agent lifecycle events. Each matching handler receives a JSON event payload on stdin and may influence the session through its exit code and stdout.

> **Warning:** Hook commands run on your machine with your user privileges. Treat every `hooks.json` entry as code you trust — especially project-scoped hooks checked into a repository.

## Configuration locations and precedence

| Scope | Path | When it loads |
| --- | --- | --- |
| User | `~/.deepagents/hooks.json` | Always (when the file exists) |
| Project | `{project_root}/.deepagents/hooks.json` | Only after workspace trust |

When both scopes load, project matcher groups are applied first, then user groups. A project handler that stops further processing therefore wins over lower-precedence user handlers for the same event.

### Project workspace trust

Project-scoped hooks can execute arbitrary commands from the repository. Before they load:

- Interactive `dcode` prompts for approval when `.deepagents/hooks.json` is present and the workspace is not already trusted.
- Choosing always-allow persists trust for that canonical workspace root in `~/.deepagents/.state/hooks_trust.json`.
- Cancelling the prompt (Esc / Ctrl+D) aborts startup.
- Denying skips project hooks for the session and continues with user hooks only.
- Headless / CI runs do not prompt; pass `--trust-project-hooks` to opt in for that run.

## Events and matchers

Each top-level key under `"hooks"` is an event name. Values are lists of matcher groups. A group may omit `matcher` (or use `"*"`) to match all values for that event's matcher field. Events with no matcher field reject non-wildcard matchers at load time.

Native tools are matched by their wire names (for example `execute` → `Bash`, `write_file` → `Write`).

| Event | Owner | Matcher field | Fires when |
| --- | --- | --- | --- |
| `SessionStart` | client | `cause` | A session starts (`startup`, `resume`, `clear`, `compact`) |
| `UserPromptSubmit` | client | _(none)_ | The user submits a prompt |
| `SessionEnd` | client | `cause` | A session ends |
| `PermissionRequest` | client | `tool_name` | The client is about to ask for tool permission |
| `Notification` | client | `notification_type` | A client lifecycle notification is emitted |
| `PreToolUse` | server | `tool_name` | Before a tool call runs |
| `PostToolUse` | server | `tool_name` | After a tool call completes |
| `PreCompact` | server | `trigger` | Before conversation compaction |
| `Stop` | server | _(none)_ | After an agent stop turn |
| `SubagentStart` | server | `agent_name` | When a subagent starts |
| `SubagentStop` | server | `agent_name` | When a subagent stops |

## Handler shape

Each matcher group has a `hooks` list of command handlers:

```json
{
  "type": "command",
  "command": "your-shell-command",
  "timeout": 60,
  "statusMessage": "Running policy check"
}
```

- `type` must be `"command"`.
- `command` is required and runs through a shell, so pipes, redirects, and `$VAR` expansion work.
- `argv` is optional; when set, the handler is executed directly from that argument list instead of through a shell.
- `timeout` is optional seconds; when omitted, the event default applies (600s for most events, 30s for `UserPromptSubmit`).
- `statusMessage` is optional UI status text while the handler runs.
- `async: true` is rejected; async command hooks are not supported.

## Examples

### Minimal

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "true"
          }
        ]
      }
    ]
  }
}
```

### Deny a destructive shell command

Matchers use wire tool names. `execute` is exposed as `Bash`. Exit code `2` (or JSON `permissionDecision: "deny"`) denies `PreToolUse`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c \"import json,sys; d=json.load(sys.stdin); cmd=d.get('tool_input',{}).get('command',''); blocked='rm -rf /' in cmd; print(json.dumps({'hookSpecificOutput':{'hookEventName':'PreToolUse','permissionDecision':'deny','permissionDecisionReason':'Refusing destructive root delete'}}) if blocked else '{}')\""
          }
        ]
      }
    ]
  }
}
```

## How handler output affects behavior

Handlers communicate through:

- **Exit code `2`**: treated as a synthetic `decision: "block"`. Interpretation depends on the event (for example deny on `PreToolUse` / `PermissionRequest`, block further processing on `UserPromptSubmit` / `PreCompact`, feedback on `PostToolUse`).
- **Other non-zero exits**: recorded as diagnostics; they do not apply a block decision.
- **JSON stdout** (`HookWireOutput`): may set `continue` / `stopReason`, `systemMessage` (user-visible notice), `additionalContext` via `hookSpecificOutput`, and event-specific fields such as `permissionDecision` on `PreToolUse`.
- **Non-JSON stdout**: becomes additional context for events whose plain-output policy is context (`SessionStart`, `UserPromptSubmit`); otherwise it is a diagnostic.
- **Timeouts**: when a handler exceeds its timeout, it is terminated and recorded as a timeout diagnostic; it does not apply a successful decision.

## Legacy configuration

Older list-shaped `hooks.json` documents are still loaded. Semantically equivalent legacy events are migrated into the Hooks v2 shape automatically; unsupported legacy events are left unmapped and surfaced as load diagnostics.
