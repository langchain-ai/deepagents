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
decisions do not bypass Talon's tool approvals. As in the shared engine,
`updatedInput` is currently ignored, so RTK automatic command rewriting is not
supported yet.
