# Hooks

Talon reuses the `deepagents-code` command-hook engine for the main agent's
`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, and `Stop` events.

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

Workspace `.deepagents/hooks.json`, Claude Code settings, plugin hooks, client
lifecycle events, and subagent hooks are not loaded by this adapter. Hook `allow`
decisions do not bypass Talon's tool approvals. As in the shared engine,
`updatedInput` is currently ignored, so RTK automatic command rewriting is not
supported yet.
