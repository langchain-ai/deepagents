# Deep Agents Code GitHub Action

This repository's root `action.yml` runs `dcode` non-interactively in GitHub Actions.

## Example

```yaml
name: dcode
on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0
      - uses: langchain-ai/deepagents@main
        with:
          prompt: "Review this repository and summarize the highest-risk issues."
          model: "openai:gpt-5.5"
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          shell_allow_list: "recommended,git,gh"
          max_turns: "8"
          quiet: "true"
```

For production workflows, pin `langchain-ai/deepagents` to a reviewed commit SHA instead of `main`.

## Common inputs

- `prompt`: Task text passed to `dcode`.
- `model`: Model as `provider:model` (e.g. `openai:gpt-5.5`) or a bare name (`claude-*`, `gpt-*`, `gemini-*`) with the provider auto-detected.
- `*_api_key`: Provider API keys (`openai_api_key`, `anthropic_api_key`, `google_api_key`).
- `shell_allow_list`: Commands allowed for headless shell execution, such as `recommended,git,gh`.
- `max_turns`: Maximum agentic turns before stopping.
- `task_timeout`: `dcode --timeout` in seconds. The action-level `timeout` input is in minutes.
- `quiet`: Clean stdout output for piping.
- `json`: Emit machine-readable `dcode` JSON output.

## Advanced inputs

The action also forwards headless `dcode` options for skills, startup commands, sandboxes, MCP config, interpreter tools, rubric grading, stdin, and model/profile overrides. Interactive-only options such as `--auto-approve` are intentionally not exposed; use `shell_allow_list` to permit shell commands in workflows. See `action.yml` for the full input list and defaults.

## Outputs

- `response`: The agent's full output (stdout and stderr combined). This is raw, unfiltered agent output — do not assume it is free of secrets or safe to echo into other services.
- `exit_code`: The agent's exit code (also `124` on task timeout).
- `cache_hit`: Whether agent memory was restored from cache (empty when `enable_memory` is `false`).

## Memory

Persistent memory is enabled by default through `actions/cache`. Use:

- `enable_memory: "false"` to disable cache restore/save.
- `memory_scope: pr`, `branch`, or `repo` to control cache sharing.
- `agent_name` to separate memory between multiple action identities.
