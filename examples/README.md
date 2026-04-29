<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="../.github/images/logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="../.github/images/logo-dark.svg">
    <img alt="Deep Agents" src="../.github/images/logo-dark.svg" height="40"/>
  </picture>
</p>

<h3 align="center">Examples</h3>

<p align="center">
  Agents, patterns, and applications you can build with Deep Agents.
</p>

| Example | Description |
|---------|-------------|
| [deep_research](deep_research/) | Multi-step web research agent using Tavily for URL discovery, parallel sub-agents, and strategic reflection |
| [content-builder-agent](content-builder-agent/) | Content writing agent that demonstrates memory (`AGENTS.md`), skills, and subagents for blog posts, LinkedIn posts, and tweets with generated images |
| [text-to-sql-agent](text-to-sql-agent/) | Natural language to SQL agent with planning, skill-based workflows, and the Chinook demo database |
| [deploy-coding-agent](deploy-coding-agent/) | `deepagents deploy` example: autonomous coding agent with a LangSmith sandbox for code execution |
| [deploy-content-writer](deploy-content-writer/) | `deepagents deploy` example: content writing agent with per-user memory and Supabase auth |
| [deploy-mcp-docs-agent](deploy-mcp-docs-agent/) | `deepagents deploy` example: docs research agent that uses MCP tools to search LangChain documentation |
| [deploy-gtm-agent](deploy-gtm-agent/) | `deepagents deploy` example: GTM strategy agent coordinating sync and async subagents |
| [async-subagent-server](async-subagent-server/) | Self-hosted Agent Protocol server exposing a Deep Agents researcher as an async subagent, with a supervisor REPL |
| [nvidia_deep_agent](nvidia_deep_agent/) | Multi-model agent with NVIDIA Nemotron Super for research and GPU-accelerated code execution via RAPIDS |
| [ralph_mode](ralph_mode/) | Autonomous looping pattern that runs with fresh context each iteration, using the filesystem for persistence |
| [rlm_agent](rlm_agent/) | `create_rlm_agent` helper: wraps `create_deep_agent` with a recursive REPL + PTC subagent chain for parallel fan-out across levels |
| [repl_swarm](repl_swarm/) | Skill-module example: a `swarm` skill (TypeScript) dispatches subagents in parallel from inside the QuickJS REPL |
| [downloading_agents](downloading_agents/) | Shows how agents are just folders—download a zip, unzip, and run |
| [better-harness](better-harness/) | Eval-driven outer-loop optimization of a Deep Agents harness using the `better-harness` research artifact |

Each example has its own `README` with setup instructions.

<details>
<summary><h2>Contributing an Example</h2></summary>

See the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview) for general contribution guidelines.

When adding a new example:

- **Use uv** for dependency management with a `pyproject.toml` and `uv.lock` (commit the lock file)
- **Pin to deepagents version** — use a version range (e.g., `>=0.3.5,<0.4.0`) in dependencies
- **Include a `README`** with clear setup and usage instructions
- **Add tests** for reusable utilities or non-trivial helper logic
- **Keep it focused** — each example should demonstrate one use-case or workflow
- **Follow the structure** of existing examples (see `deep_research/` or `text-to-sql-agent/` as references)

</details>
