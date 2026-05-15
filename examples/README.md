<div align="center">
  <a href="https://docs.langchain.com/oss/python/deepagents/overview#deep-agents-overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../.github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="../.github/images/logo-light.svg">
      <img alt="Deep Agents Logo" src="../.github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<h3 align="center">Examples</h3>

<p align="center"><em>Real agents and patterns built on Deep Agents.</em></p>

## Featured

<table>
<tr>
<td width="50%" valign="top">

### Deep Agents Code

A pre-built coding Deep Agent in your terminal — similar to Claude Code or Codex — powered by any LLM. Includes an interactive TUI, web search, remote sandboxes, persistent memory, custom skills, and human-in-the-loop approval.

```bash
curl -LsSf https://langch.in/dcode | bash
```

<sub>[Source](../libs/code/) · [Docs](https://docs.langchain.com/oss/python/deepagents/cli/overview)</sub>

</td>
<td width="50%" valign="top">

### Open SWE

An open-source, async coding agent for your org's internal workflows. Runs each task in an isolated cloud sandbox, integrates with Slack, Linear, and GitHub, and ships PRs end-to-end.

```bash
@open-swe fix this user-reported bug plz!
```

<sub>[Repository](https://github.com/langchain-ai/open-swe) · [Blog post](https://blog.langchain.com/open-swe-an-open-source-framework-for-internal-coding-agents/)</sub>

</td>
</tr>
</table>

## In the wild

Production agents powered by the LangChain stack:

| Project | Description |
|---|---|
| [**LangSmith Fleet**](https://www.langchain.com/langsmith/fleet) | No-code platform for building AI agents from templates; connect your accounts and let the agent handle routine work |
| [**Chat LangChain**](https://chat.langchain.com/) | Documentation assistant that answers questions about LangChain, LangGraph, and LangSmith ([source](https://github.com/langchain-ai/chat-langchain)) |

## All examples

### Research

| Example | Description |
|---|---|
| [**Deep Research**](deep_research/) | Multi-step web research with Tavily, parallel sub-agents, and strategic reflection |
| [**MCP Docs Agent**](deploy-mcp-docs-agent/) | Docs research agent using MCP tools over LangChain documentation |

### Coding

| Example | Description |
|---|---|
| [**Coding Agent**](deploy-coding-agent/) | Autonomous coding agent in a LangSmith sandbox |
| [**Nemotron Research Agent**](nvidia_deep_agent/) | NVIDIA Nemotron Super for research + GPU-accelerated execution via RAPIDS |

### Content

| Example | Description |
|---|---|
| [**Content Builder**](content-builder-agent/) | Blog posts, LinkedIn posts, and tweets with memory (`AGENTS.md`), skills, and subagents |
| [**Text-to-SQL**](text-to-sql-agent/) | Natural language to SQL with planning and skill-based workflows on the Chinook demo database |
| [**LLM Wiki**](llm-wiki/) | Script-first LLM wiki synced via `langsmith hub init/pull/push` |

### Deployable services

| Example | Description |
|---|---|
| [**Content Writer**](deploy-content-writer/) | Content writer with per-user memory and Supabase auth |
| [**GTM Strategist**](deploy-gtm-agent/) | GTM strategy agent coordinating sync and async subagents |
| [**Async Subagent Server**](async-subagent-server/) | Self-hosted Agent Protocol server exposing a researcher as an async subagent |

### Advanced patterns

| Example | Description |
|---|---|
| [**Ralph Loop**](ralph_mode/) | Autonomous looping with fresh context each iteration, using the filesystem for persistence |
| [**RLM Agent**](rlm_agent/) | `create_rlm_agent` helper: recursive REPL + PTC subagent chain for parallel fan-out |
| [**REPL Swarm**](repl_swarm/) | TypeScript `swarm` skill dispatching subagents in parallel from QuickJS |
| [**Agents as Folders**](downloading_agents/) | Download a zip, unzip, and run |
| [**Better Harness**](better-harness/) | Eval-driven outer-loop optimization of a Deep Agents harness |

Each example has its own `README` with setup instructions.

<details>
<summary><h2>Contributing an example</h2></summary>

See the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) for general contribution guidelines.

When adding a new example:

- **Use uv** for dependency management with a `pyproject.toml` and `uv.lock` (commit the lock file)
- **Pin to deepagents version** — use a version range (e.g., `>=0.3.5,<0.4.0`) in dependencies
- **Include a `README`** with clear setup and usage instructions
- **Add tests** for reusable utilities or non-trivial helper logic
- **Keep it focused** — each example should demonstrate one use-case or workflow
- **Follow the structure** of existing examples (see `deep_research/` or `text-to-sql-agent/` as references)

</details>
