# Hooks

Talon reuses the `deepagents-code` command-hook engine for `PreToolUse`,
`PostToolUse`, and `PostToolUseFailure` events on the main agent and local
subagents, plus `Stop` on the main agent.

Add `hooks.json` to the Talon assistant directory. For the CLI, use
`$DEEPAGENTS_TALON_HOME/$DEEPAGENTS_TALON_ASSISTANT_ID/hooks.json`;
the base directory defaults to `~/.deepagents` and the assistant ID to `default`
(so the default path is `~/.deepagents/default/hooks.json`). Configuration is
loaded at the start of each turn; active turns retain their snapshot. For example:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command", "command": "/absolute/path/check-command", "timeout": 10}]
    }]
  }
}
```

Commands receive a JSON event on stdin and may return hook decisions as JSON on
stdout. Shell calls use the compatible `Bash` name and `tool_input.command`.
Hooks run on the **host**, including when tools use a remote sandbox. Only install
trusted commands; hooks retain the shared engine's environment filtering,
timeouts, output limits, and cancellation cleanup.

Local subagents inherit the parent turn's hook snapshot, including background
workers and tasks with additional tools. Each invocation has an isolated session
and transcript; hook payloads identify it with `agent_id` and `agent_type`.
Subagents retain their existing approval restrictions: a hook cannot authorize a
protected action. Hooks inside opaque compiled or remote agents, and the
`SubagentStart`/`SubagentStop` lifecycle events, are not supported.

Workspace `.deepagents/hooks.json`, Claude Code settings, plugin hooks, and client
lifecycle events are not loaded by this adapter. Hook `allow`
decisions do not bypass Talon's tool approvals.

## RTK command rewriting

A `PreToolUse` hook may return `hookSpecificOutput.updatedInput` containing
`{"command": "rtk git status"}` for a shell call. Hooks may also echo unchanged
fields from `tool_input`, as the RTK hook does. Talon approves the rewritten
command before execution; post-tool hooks receive that command too. Other native
arguments (such as timeout) are preserved. Other tools and input fields remain
unsupported and produce a diagnostic. If several handlers rewrite a command,
the last valid rewrite in configuration order wins; each sees the original input.
A denial from any handler still blocks execution.

To use RTK, install `rtk` on the host and in the tool execution environment
(including remote sandboxes). Save this as `rtk-hook.py` and set the `Bash` hook
command above to `python3 /absolute/path/rtk-hook.py`:

```python
import json
import subprocess
import sys

payload = json.load(sys.stdin)
result = subprocess.run(
    ["rtk", "rewrite", "--", payload["tool_input"]["command"]],
    capture_output=True,
    text=True,
    check=False,
)
if result.returncode in (0, 3) and result.stdout.strip():
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "updatedInput": {"command": result.stdout.strip()},
    }}))
elif result.returncode == 2:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Blocked by RTK permission rules",
    }}))
```

RTK exit codes 0 and 3 supply a rewrite, 1 preserves the original command, and
2 blocks execution.
RTK rewriting does not grant tool approval.
