# Pi-style harness profile

A reusable [`HarnessProfile`](https://docs.langchain.com/oss/python/deepagents/profiles)
that mirrors the prompt and tool-description style of the [Pi coding
agent](https://pi.dev/) — a minimal terminal coding harness whose system
prompt is intentionally short and whose detail lives on per-tool
descriptions and a small set of guidelines.

Unlike Deep Agents' built-in profiles (which target a specific
`provider:model`), this one is a *callable* profile: register it under
whatever provider or model key fits your workflow.

## Usage

```python
from deepagents import create_deep_agent
from pi_profile import register_pi_harness

# Apply Pi-style prompt + tool descriptions to every Anthropic model.
register_pi_harness("anthropic")

agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")
```

You can also target a single model:

```python
register_pi_harness("openai:gpt-5.3")
```

Or build the profile yourself and layer extra fields on top before
registering:

```python
from deepagents import (
    GeneralPurposeSubagentProfile,
    register_harness_profile,
)
from dataclasses import replace
from pi_profile import pi_harness_profile

profile = replace(
    pi_harness_profile(),
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
)
register_harness_profile("openai:gpt-5.3", profile)
```

## What the profile does

- Replaces the assembled base system prompt with `PI_BASE_SYSTEM_PROMPT`
  (Pi's short role statement + tool-driven guidelines).
- Overrides the description for each default Deep Agents filesystem tool
  (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`, `execute`)
  with a Pi-flavored description adapted from Pi's tool definitions.
- Leaves middleware and subagents untouched — Pi's "no sub-agents / no
  plan mode / no permission popups" stance is a packaging choice, not
  something that maps onto Deep Agents' required scaffolding. Layer
  `GeneralPurposeSubagentProfile(enabled=False)` on top yourself if you
  want to drop the auto-added general-purpose subagent.

The Pi → Deep Agents tool-name mapping:

| Pi name | Deep Agents name |
| ------- | ---------------- |
| `read`  | `read_file`      |
| `write` | `write_file`     |
| `edit`  | `edit_file`      |
| `bash`  | `execute`        |
| `find`  | `glob`           |
| `ls`    | `ls`             |
| `grep`  | `grep`           |

## Source attribution

Prompt and descriptions are adapted from Pi's coding agent:
<https://github.com/earendil-works/pi/tree/main/packages/coding-agent>
(see `src/core/system-prompt.ts` and `src/core/tools/`).

## Run the tests

```bash
uv run --extra dev pytest
```
