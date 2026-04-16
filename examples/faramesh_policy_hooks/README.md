# Faramesh Policy Hooks Example

This example shows Deep Agents integration with Faramesh policy decisions using the built-in tool-call hook API.

## What it demonstrates

- `before_tool_call_hooks` in `create_deep_agent(...)`
- External policy decision call before tool execution
- Returning a `ToolMessage` to block disallowed tool calls

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install deepagents langchain-openai
export OPENAI_API_KEY=...
export FARAMESH_POLICY_URL=http://localhost:8080/policy/check
python agent.py
```

## Policy endpoint contract used by this example

Request JSON:

```json
{
  "tool": "read_secret",
  "args": {
    "path": "/etc/shadow"
  }
}
```

Response JSON:

```json
{"allow": true}
```

or

```json
{"allow": false, "reason": "sensitive path is blocked"}
```
