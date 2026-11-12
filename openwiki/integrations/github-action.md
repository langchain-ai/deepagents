---
type: composite-action integration
title: GitHub Action Integration
description: Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the root action contract, CLI distribution and runtime assumptions, memory lifecycle, security handoff, and direct workflow tests.
tags: [github-actions, dcode, deepagents-code, ci, automation, memory, mcp, sandbox]
sources:
  - id: openwiki-source-b1423dca16677f7643488f74
    resource: repo://.github/scripts/tests/workflows/test_github_action.py
  - id: openwiki-source-4d9cccca7700db7220ec055e
    resource: repo://.github/workflows/_test.yml
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-96d0addee4aedab20d360121
    resource: repo://action.yml
  - id: openwiki-source-6c2e9cfaa20096e021221d47
    resource: repo://libs/code/CHANGELOG.md
  - id: openwiki-source-ecf20e7a2684ba0d2ae7d701
    resource: repo://libs/code/deepagents_code/client/non_interactive.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
verified:
  - by: openwiki/0.4.2
    at: 2026-11-11T08:04:08.950Z
generated: { by: "openwiki/0.4.2", at: "2026-11-11T08:04:08.950Z" }
---

# GitHub Action Integration

The repository-root composite action, `langchain-ai/deepagents`, adapts one headless `dcode` task to a GitHub Actions step. Root `action.yml` is the public package: its declared `with:` inputs and three outputs are the supported action contract, and its implementation executes through GitHub's composite-action mechanism. The wrapper installs the published `deepagents-code` distribution, validates and maps action inputs to CLI flags, and invokes `dcode` in the requested workspace. It is not the programmatic graph-construction API: `create_deep_agent()` returns a compiled graph for an application, whereas this integration launches the CLI.

## Use in a job

Check out the target repository first. `working_directory` defaults to `.`, so it is the dcode workspace unless a sandbox is selected. Pin a production action reference to a reviewed commit SHA and provide only the job permissions and secrets the task needs.

```yaml
name: dcode review
on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0
      - uses: langchain-ai/deepagents@<reviewed-commit-sha>
        with:
          prompt: "Review this repository and summarize the highest-risk issues."
          model: "openai:gpt-6-astra"
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          shell_allow_list: "recommended,git,gh"
          max_turns: "8"
          task_timeout: "600"
          quiet: "true"
```

`github_token` defaults to `${{ github.token }}`. The run step exposes it only as `GITHUB_TOKEN`, and exposes the provider inputs as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY`. The same supplied token authenticates an optional skills clone. The agent therefore has the authority of the job token and any credentials available in its execution environment.

## Lifecycle and failure boundary

```mermaid
flowchart TD
    Setup["Set up uv"] --> Memory{"Memory enabled"}
    Memory -->|"Yes"| Restore["Resolve and restore memory cache"]
    Memory -->|"No"| Install["Install selected dcode"]
    Restore --> Install
    Install --> Skills{"skills_repo provided"}
    Skills -->|"Yes"| Clone["Clone and install skills"]
    Skills -->|"No"| Build["Validate inputs and assemble command"]
    Clone --> Build
    Build --> Execute["Run bounded headless dcode"]
    Execute --> Publish["Capture response and exit code"]
    Publish --> Save{"Memory enabled"}
    Save -->|"Yes always"| Persist["Save memory cache"]
```

*The composite action restores state before dcode and attempts cache saving after its run even when it fails.*

The installation step resolves latest `deepagents-code` with `uvx --from deepagents-code dcode --version`, or the exact `cli_version` with `uvx --from "deepagents-code==${INPUT_CLI_VERSION}" dcode --version`. A selected version below `0.1.0` fails before execution. That floor covers always-used flags only: an old pin can still reject an optional flag introduced later.

The run step requires a nonempty prompt and builds an argument array. By default it calls `dcode ... --non-interactive "$prompt"`; `stdin: "true"` instead calls `dcode ... --stdin` and supplies the prompt on standard input. `stdin` cannot be combined with `skill`, because piped input with a skill would take dcode's interactive path rather than headless execution. `timeout` is the wrapper's outer wall-clock limit in minutes (default `30`), while `task_timeout` becomes dcode's `--timeout` in seconds.

It rejects invalid boolean forms, validates positive numeric budgets (`timeout`, `max_turns`, `task_timeout`, and `rubric_max_iterations`), permits zero `max_retries`, and validates nonempty `model_params` and `profile_override` as JSON objects when `jq` is available. `interpreter` is tri-state: `true` adds `--interpreter`, `false` adds `--no-interpreter`, and empty preserves dcode's default.

## Public inputs and authority

All `with:` values are strings. Empty optional value inputs are normally omitted; boolean `true` inputs add their corresponding flag.

| Concern | Inputs | Effect |
| --- | --- | --- |
| Task and identity | `prompt`, `working_directory`, `agent_name`, `cli_version` | Select the task, workspace, memory identity, and CLI distribution. |
| Model | `model`, `model_params`, `max_retries`, `profile_override` | Map to the corresponding model flags. |
| Task bounds | `shell_allow_list`, `skill`, `startup_cmd`, `stdin`, `max_turns`, `task_timeout`, `timeout` | Set shell policy, startup behavior, prompt transport, and limits. |
| Response | `quiet`, `no_stream`, `json` | Map directly to dcode output flags. |
| Rubric | `rubric`, `rubric_model`, `rubric_max_iterations` | Pass acceptance criteria and grader controls. |
| Integrations | `mcp_config`, `no_mcp`, `trust_project_mcp`, `interpreter`, `interpreter_tools`, `sandbox`, `sandbox_id`, `sandbox_snapshot_name`, `sandbox_setup` | Direct mappings for supported MCP, interpreter, and sandbox controls. |
| Memory | `enable_memory`, `memory_scope` | Control the cache lifecycle. |

`--auto-approve` is intentionally absent from the action contract. Adding a dcode option does not expose it from the action: it must be declared and mapped deliberately in `action.yml`. In dcode, `--non-interactive` runs one task and exits without human tool approvals; it can still change files and access the network.

At the dcode boundary, no shell allow list disables shell access, a restrictive list enables and gates shell commands, and `all` permits unrestricted shell commands and auto-approves tools. Project stdio MCP is untrusted by default. An explicit MCP config path takes precedence over auto-discovered configuration, `no_mcp` disables MCP tool loading, and `trust_project_mcp` is an explicit grant to load project MCP configuration. An empty `sandbox` means local runner execution, not isolation. Treat prompts, checked-out content, skills, MCP configuration, and agent output as untrusted.

## Memory and skills

With `enable_memory: "true"` (the default), the action caches the named agent memory directory, the global SQLite session database plus WAL/SHM files, and `<working_directory>/.deepagents/AGENTS.md`. Cache namespace begins with `agent_name` (default `agent`) and uses `memory_scope`: `pr` selects an issue/PR number or ref, `branch` selects the ref, and `repo` is repository-wide and default. Unknown scopes warn and fall back to the PR/ref calculation. Restore tries the selected prefix and then the agent-wide prefix; saves append `github.run_id` and run under `always()`. `cache_hit` is the restore result and is empty when memory is disabled.

If `skills_repo` is supplied, the action shallow-clones an `owner/repo`, `owner/repo@ref`, HTTPS, or SSH source into a temporary directory. Every directory containing `SKILL.md` is copied to `.deepagents/skills`; clone failure or no discovered skill fails the action. Pin a reviewed skill revision because this installs instructions into the agent workspace.

## Outputs and failure propagation

| Output | Meaning |
| --- | --- |
| `response` | Combined dcode stdout and stderr; raw, untrusted agent output. |
| `exit_code` | Agent or outer-timeout status; the action exits with this code. |
| `cache_hit` | Memory restore result, or empty when memory is off. |

The action pipes combined output to `tee` but captures `PIPESTATUS[0]`, so a successful `tee` cannot conceal a dcode or timeout failure. It writes `response` with a random `$GITHUB_OUTPUT` heredoc delimiter, preventing agent output from forging output records. The stdin branch uses process substitution rather than a `printf | timeout` pipeline, avoiding a producer SIGPIPE being reported as the task result.

## Distribution contract and CI coverage

The action ref and `cli_version` select independent artifacts: `uses: langchain-ai/deepagents@<ref>` selects the root composite action, while `cli_version` selects either the latest `deepagents-code` distribution or the exact version passed to `uvx`. Pin both when reproducibility matters. The current repository package declares `deepagents-code` version `0.2.7` and Python `>=3.14,<3.15`; the `0.2.0` changelog records the Python 3.14 requirement as a breaking change. A CLI pin is consequently also a Python-runtime compatibility choice.

The focused action test parses `action.yml` and the dcode root parser to catch action-input-to-flag drift and ensure `--auto-approve` stays absent. It executes the actual action run body with stubbed `uvx` and `timeout`, covering validation, stdin/skill exclusion, version command construction, timeout arithmetic, exit propagation, and stdin SIGPIPE behavior.

Run it locally from the repository root:

```bash
python -m pytest .github/scripts/tests/workflows/test_github_action.py -v
```

CI runs that test through `python -m pytest .github/scripts/tests -v` in `Validate Release Options`, after installing `packaging`, `pyyaml`, and `pytest` on Python 3.14 (and Node 24 for other helper tests). The job runs unless change detection identifies a curated release-notes-only apply. It is the direct action coverage: root `action.yml` is not in the Code path selector, which otherwise selects `libs/code`, `libs/deepagents`, and shared CI workflow/action paths for Code lint and Python-3.14 sharded unit tests. CI also always calls the reusable lockfile workflow; it checks every lockfile, while the separate text-hygiene job runs outside push events. These repository gates complement rather than replace behavior tests for the action.
