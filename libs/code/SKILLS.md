# Skill-scoped MCP tools

MCP tools are available immediately by default. To keep a server's tool schemas
out of model context until a skill needs them, set `skillOnly` in that server's
MCP configuration:

```json
{
  "mcpServers": {
    "records": {
      "url": "https://example.com/mcp",
      "skillOnly": true
    }
  }
}
```

List the tools in a discovered skill's `SKILL.md` frontmatter:

```markdown
---
name: records
description: Search and read customer records
metadata:
  include_tools: records_search records_read
---

Search for the customer before reading their record.
```

Use the exact exposed names shown by `/tools` or `/mcp`, not unprefixed
server-side names. `include_tools` is a space-separated string, not a YAML list.
This first pass supports individual tool names, not MCP-server group labels.

The tools become available after the agent reads the skill with `read_file`, or
when you explicitly select it with `/skill:records`. The SDK keeps them available
while the activating skill remains in context. Calls before activation fail as
unknown tools. Plugin-namespaced skills work the same way.

`skillOnly` applies to every permitted tool on that server; tools not listed by
any skill remain unavailable. Existing `allowedTools` and `disabledTools`
filters still apply. Servers without `skillOnly: true` retain eager disclosure,
and listing one of their tools in a skill does not hide it.

Skill-only tools retain MCP approval rules, hooks, and timeouts, including when
used by subagents. They are unavailable when skills are disabled. This is
context management, not a permission boundary: reading a skill is not approval
to call its tools, and local shell access remains separate.

Restart the agent server after changing MCP configuration. Skill-only disclosure
does not defer the MCP connection or tool discovery itself, and the tool catalog
still lists the connected server's tools before activation.
