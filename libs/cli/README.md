# Deep Agents CLI — Deployment Tooling (Deprecated)

[![PyPI - Version](https://img.shields.io/pypi/v/deepagents-cli?label=%20)](https://pypi.org/project/deepagents-cli/#history)
[![PyPI - License](https://img.shields.io/pypi/l/deepagents-cli)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/deepagents-cli)](https://pypistats.org/packages/deepagents-cli)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

> [!WARNING]
> **`deepagents-cli` is deprecated and no longer maintained. This is the final release.** It is superseded by the [`managed-deepagents`](https://pypi.org/project/managed-deepagents/) package and its `mda` CLI, which build, deploy, and run agents from a project directory with no local CLI infrastructure. See the [Managed Deep Agents overview](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview).
>
> **What to do:** install the `mda` CLI and re-scaffold your project:
>
> ```bash
> uv tool install managed-deepagents
> mda init my-agent
> cd my-agent
> mda deploy
> ```
>
> The `mda` CLI covers everything `deepagents init` / `deploy` / `agents` / `mcp-servers` did — project scaffolding, deployment, agent management, and MCP servers — against the Managed Deep Agents platform. See the [quickstart](https://docs.langchain.com/langsmith/python/managed-deep-agents-quickstart) and [CLI reference](https://docs.langchain.com/langsmith/python/managed-deep-agents-cli).
>
> The interactive coding agent (`dcode`) already lives in [`deepagents-code`](https://docs.langchain.com/deepagents-code) and is unaffected by this deprecation.

## Install

```bash
uv tool install deepagents-cli
```

You'll need a LangSmith API key with access to the Managed Deep Agents private
preview ([waitlist](https://www.langchain.com/langsmith-managed-deep-agents-waitlist)).
Export it before running any command, or put it in a repo `.env` or
`~/.deepagents/.env`:

```bash
export LANGSMITH_API_KEY="..."
```

## Usage

```bash
# Scaffold a new project folder
deepagents init my-agent

# Register any MCP servers you intend to use (one-time, per workspace)
deepagents mcp-servers add --url https://tools.langchain.com \
                            --header X-Api-Key=$LANGSMITH_API_KEY \
                            --name Fleet

# init scaffolds an empty tools.json. Add tools that reference a registered
# MCP server's URL (otherwise the agent has no tools), e.g. the Fleet server
# registered above:
#   { "name": "read_url_content", "mcp_server_url": "https://tools.langchain.com",
#     "mcp_server_name": "Fleet", "display_name": "read_url_content" }

# Upsert the project as a managed agent on /v1/deepagents/*
cd my-agent && deepagents deploy
```

`deepagents init` scaffolds an empty `tools.json`, an example skill under
`skills/`, and an example subagent under `subagents/`; edit or remove them to
suit your agent. `tools.json` starts empty because every tool must reference an
MCP server that is already registered in the workspace
(`deepagents mcp-servers add`) — so a freshly scaffolded project deploys
without first registering a server.

New agents default to the `state` backend. To opt into a managed sandbox, set
`agent.json`'s `backend.type` to `sandbox` and configure
`backend.sandbox_config.scope` as `thread` or `agent`; `sandbox_config` can also
include sandbox policy IDs and TTL fields. The CLI does not create or run
sandboxes locally; sandbox lifecycle is handled by the Managed Deep Agents
platform.

```json
{
  "backend": {
    "type": "sandbox",
    "sandbox_config": {
      "scope": "thread",
      "policy_ids": ["policy-id"]
    }
  }
}
```

### Project layout

```text
my-agent/
  agent.json              # name, description, backend, runtime.model, permissions
  AGENTS.md               # system prompt
  tools.json              # tools the agent can call (optional)
  skills/<name>/SKILL.md  # frontmatter-tagged skills (optional)
  subagents/<name>/       # delegated subagent definitions (optional)
```

### Other commands

```bash
deepagents agents list                  # list workspace agents
deepagents agents get <agent_id>        # show one agent
deepagents agents delete <agent_id>     # delete an agent

deepagents mcp-servers list                  # list workspace MCP servers
deepagents mcp-servers add --url URL          # register a server
deepagents mcp-servers get <id|name|url>      # show one server
deepagents mcp-servers tools <id|name|url>    # list a server's tools
deepagents mcp-servers update <id|name|url>   # update server URL or headers
deepagents mcp-servers delete <id|name|url>   # remove a server
deepagents mcp-servers connect <id|name|url>  # start OAuth for a server
```

`get`, `update`, `delete`, and `connect` accept an MCP server's id, exact name,
or URL — a non-id value is resolved against `mcp-servers list` (URLs are matched
ignoring case and trailing slash).

## 📖 Resources

- **[CLI Documentation](https://docs.langchain.com/oss/python/deepagents/cli/overview)**
- **[Changelog](https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md)**
- **[Source code](https://github.com/langchain-ai/deepagents/tree/main/libs/cli)**
- **[Deep Agents SDK](https://github.com/langchain-ai/deepagents)** — underlying agent harness
- **[Deep Agents Code](https://pypi.org/project/deepagents-code/)** — coding agent
- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
