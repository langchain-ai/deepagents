# Migrating `deepagents deploy` to the Managed Deep Agents (`/v1/deepagents`) API

**Status:** Design — pending implementation
**Date:** 2026-05-20
**Branch:** `vic/migrate-da-deploy`
**Author:** Victor Moreira (`victor@langchain.dev`)
**Companion to:** `langchain-ai/docs#4077`, `langchainplus@vic/mda-mcp-servers-and-sandboxes`

## Summary

Replace the current `deepagents deploy` flow (project bundler + `langgraph deploy`
shell-out + Context Hub seed + auto-wired issues board) with a thin REST client
that talks directly to the Managed Deep Agents API at `/v1/deepagents/*`. The
on-disk project shape is redesigned to mirror the API request body 1:1, so the
files a user edits are the same JSON the platform receives.

This is a deliberate breaking change. `deepagents.toml`, `mcp.json`, the bundler,
the codegen templates, the bundled frontend, the auth/memories/sandbox config
sections, the auto-wired issues board, and `deepagents dev` are all removed. The
inspiration is `fleet-dev-kit/fleetctl` — a thin, headless CLI on top of typed
JSON resources, no client-side compilation.

## Motivation

1. **The managed runtime owns everything the bundler used to generate.** Sandbox
   image, hub repo, per-user memories, auth, and the underlying graph all live
   server-side now. The bundler's job has evaporated.
2. **The MDA payload is rich.** `smith-go/fleet/agents/types.go` already accepts
   typed `skills[]`, `subagents[]`, `tools`, `runtime.backend_type`, and a raw
   `files` map. Everything we need fits in a single POST or PATCH.
3. **A 1:1 file-to-payload mapping unlocks round-tripping.** A future
   `deepagents agents pull <id>` can write the same files back out. With TOML
   plus three other formats, that's lossy; with JSON files that mirror the API,
   it's free.

## Non-goals

- Building a typed Python SDK for the API. The docs explicitly note official
  SDKs are coming; we don't duplicate that work.
- Wrapping every `/v1/deepagents/*` endpoint. Only what the deploy workflow
  needs: `agents` CRUD and `mcp-servers` CRUD.
- OAuth-backed MCP server registration. The CLI accepts `--auth-type headers`
  with literal header values; OAuth setup stays in the LangSmith UI for v1.
- Auto-upserting MCP servers from a project file. Secrets stay out of git.
- Backwards compatibility with existing `deepagents.toml` projects. Migration
  is documented but manual.

## Architecture

### CLI surface

```
deepagents init   [name] [--force]
deepagents deploy [--dir PATH] [--dry-run] [--detach] [--reset]
deepagents agents       list
                        get <id> [--include-files]
                        delete <id> [--yes]
deepagents mcp-servers  list
                        add  --url URL [--name N] [--header K=V]... [--auth-type headers]
                        get  <id>
                        delete <id>
```

Removed: `deepagents dev` (no local-iteration path post-migration; deploy + test
in cloud).

### Project layout

```
my-agent/
  agent.json              { name, description?, runtime?, permissions?, extras? }
  AGENTS.md               → system_prompt
  tools.json              → tools (verbatim ToolsConfig; optional)
  skills/
    <skill>/SKILL.md      → skills[].instructions; YAML frontmatter → name + description
    <skill>/<file>        → skills[].files[<filename>]
  subagents/
    <name>/agent.json     { description?, model_id? }  (subagent name = dir name)
    <name>/AGENTS.md      → subagents[].instructions
    <name>/tools.json     → subagents[].tools
  .deepagents/state.json  gitignored; { agent_id, revision, last_deployed_at }
```

Subagent-local skills (`subagents/<name>/skills/`) ship via the raw `files` map
under paths `subagents/<name>/skills/<skill>/SKILL.md`. If the server rejects an
overlap with the typed `subagents[]` entry, v1 logs a warning and skips those
files. This is a known edge to validate during implementation.

### `agent.json` schema

Mirrors `CreateAgentRequest` minus file-derived fields. Every field except
`name` is optional.

```json
{
  "name": "my-agent",
  "description": "Research assistant",
  "runtime": {
    "model": {"model_id": "anthropic:claude-sonnet-4-6"},
    "backend_type": "thread_scoped_sandbox"
  },
  "permissions": {
    "identity": "personal",
    "visibility": "tenant",
    "tenant_access_level": "read"
  },
  "extras": {}
}
```

Subagent `agent.json` is smaller — `name` is the directory name and the typed
`SubagentSpec` only carries `description`, `model_id`, `instructions`, `tools`:

```json
{
  "description": "Searches the web and summarises findings.",
  "model_id": "anthropic:claude-sonnet-4-6"
}
```

### `tools.json` schema

Verbatim from `ToolsConfig` in `smith-go/fleet/agents/types.go`:

```json
{
  "tools": [
    {
      "name": "tavily_web_search",
      "mcp_server_url": "https://tools.langchain.com",
      "mcp_server_name": "Fleet",
      "display_name": "tavily_web_search"
    }
  ],
  "interrupt_config": {
    "https://tools.langchain.com::tavily_web_search::Fleet": true
  }
}
```

### Module layout

```
libs/cli/deepagents_cli/deploy/
    __init__.py        re-exports (existing CLI bootstrap unchanged)
    commands.py        argparse parsers + execute_*_command entrypoints
    project.py         NEW: Project.load(dir) — parses agent.json / AGENTS.md /
                            tools.json / skills/ / subagents/
    payload.py         NEW: build_payload(project, *, mode) → dict
                            (mode = "create" | "patch")
    api_client.py      NEW: thin httpx client (agents + mcp-servers CRUD,
                            X-Api-Key auth, ErrorResponse parsing)
    state.py           NEW: State.load/save against .deepagents/state.json
```

Deleted: `bundler.py`, `templates.py`, `context_hub.py`, `config.py` (the old
TOML parser), `frontend_dist/`. `commands.py` shrinks roughly 4×.

### Deploy orchestration

```python
def _deploy(dir_, *, dry_run, detach, reset):
    project = Project.load(dir_)
    state   = State.load(project.root, reset=reset)
    payload = build_payload(project, mode="patch" if state.agent_id else "create")

    if dry_run:
        print(json.dumps(payload, indent=2))
        return

    client = ApiClient.from_env()
    _check_referenced_servers_exist(client, payload)
    agent  = client.upsert_agent(state.agent_id, payload)   # POST or PATCH;
                                                            # falls back to POST on 404
    state.save(agent_id=agent["id"], revision=agent["revision"])
    _print_result(agent, detach=detach, endpoint=client.endpoint)
```

`Project.load` returns a structured value object holding the parsed contents of
every relevant file; `build_payload` is a pure function over that value object —
no I/O, easy to snapshot-test.

### MCP server management

Servers are workspace-level resources. The deploy flow does **not**
auto-register them. Instead:

1. User runs `deepagents mcp-servers add --url <url> [--header K=V]...` once per
   server, before the first deploy that references it.
2. On `deploy`, we list registered servers (URL match is case-insensitive with
   trailing slashes stripped, matching the server's `interrupt_config` key
   normalisation), build a `{url → id}` map, and validate that every
   `tools[*].mcp_server_url` and `subagents[*].tools.tools[*].mcp_server_url`
   in the payload resolves to a registered server. Unresolved URLs fail fast
   with a hint:
   `MCP server https://… is not registered. Run: deepagents mcp-servers add --url https://… --header X-Api-Key=…`
3. The `{url → id}` map is cached in `state.json` to skip the list-call on
   subsequent deploys; a 404 on a cached id triggers a re-resolve.

### Local state & idempotency

`.deepagents/state.json`:

```json
{
  "schema_version": 1,
  "endpoint": "https://api.smith.langchain.com",
  "agent_id": "01931f7c-3d22-7c4d-9d6e-1e6e5c7b8a9d",
  "revision": "abc123…",
  "last_deployed_at": "2026-05-20T15:42:11Z",
  "mcp_servers": {
    "https://tools.langchain.com": "0193…server-id…"
  }
}
```

Behavior:

- `agent_id` present → PATCH; absent → POST.
- `--reset` deletes the file before running.
- PATCH returning 404 falls through to POST and overwrites state.
- `deepagents agents delete <id>` removes the matching agent_id from state on
  success.
- `endpoint` is stored so a project pinned to a non-default endpoint stays
  consistent across machines.

### Auth and endpoint resolution

Identical to today's deploy command:

- `LANGSMITH_API_KEY` > `LANGCHAIN_API_KEY`, sent as `X-Api-Key`.
- `LANGSMITH_ENDPOINT` > `LANGCHAIN_ENDPOINT` > `https://api.smith.langchain.com`.
- `.env` at the project root is loaded before validation (existing
  `_load_dotenv` helper is reused).

### Error handling

- **Missing auth**: hard fail with `Set LANGSMITH_API_KEY in your .env or environment.`
- **HTTP 4xx**: parse `ErrorResponse` (`type`/`code`/`detail`/`status`), print
  `Detail`, exit non-zero. No retries.
- **HTTP 5xx**: one retry with 1s backoff; on second failure print body and
  exit non-zero.
- **Private preview gate** (403 with a recognisable code): print a tailored
  message pointing at the waitlist URL.
- **Stray legacy `deepagents.toml`**: on `deploy`, detect it and print a
  one-shot migration hint mapping each section to the new file:
    - `[agent]` → `agent.json`
    - `[sandbox].scope` → `agent.json.runtime.backend_type`
    - `[frontend]`, `[auth]`, `[memories]` → "managed runtime; remove these"
  Exit non-zero. Do not auto-migrate.

### Output

On success:

```
Deployed: my-agent
  agent_id: 01931f7c-3d22-7c4d-9d6e-1e6e5c7b8a9d
  revision: abc123de
  https://smith.langchain.com/o/-/agents/01931f7c-3d22-7c4d-9d6e-1e6e5c7b8a9d
```

Without `--detach`, follow up with one `GET /v1/deepagents/agents/<id>/health`
and print a one-line status. With `--detach`, exit immediately after the upsert.

## Mapping table (canonical)

| Local artifact | API payload field |
| --- | --- |
| `agent.json.name` | `name` |
| `agent.json.description` | `description` |
| `agent.json.runtime.model.model_id` | `runtime.model.model_id` |
| `agent.json.runtime.backend_type` | `runtime.backend_type` |
| `agent.json.permissions.*` | `permissions.*` |
| `agent.json.extras` | `extras` |
| `AGENTS.md` | `system_prompt` |
| `tools.json` | `tools` (verbatim) |
| `skills/<name>/SKILL.md` frontmatter | `skills[].name`, `skills[].description` |
| `skills/<name>/SKILL.md` body | `skills[].instructions` |
| `skills/<name>/<file>` (siblings) | `skills[].files[<filename>]` |
| `subagents/<name>/` (dirname) | `subagents[].name` |
| `subagents/<name>/agent.json.description` | `subagents[].description` |
| `subagents/<name>/agent.json.model_id` | `subagents[].model_id` |
| `subagents/<name>/AGENTS.md` | `subagents[].instructions` |
| `subagents/<name>/tools.json` | `subagents[].tools` |
| `subagents/<name>/skills/<skill>/SKILL.md` | `files["subagents/<name>/skills/<skill>/SKILL.md"]` |

## Validation

`Project.load` enforces:

- `agent.json` exists, is valid JSON, has non-empty string `name`.
- `runtime.model.model_id`, if present, is `provider:model_id` form.
- `runtime.backend_type`, if present, is one of `default` / `thread_scoped_sandbox`
  / `agent_scoped_sandbox`.
- `permissions.identity` ∈ `personal|shared`, `visibility` ∈ `tenant|user`,
  `tenant_access_level` ∈ `read|run|write`.
- `AGENTS.md` exists.
- `tools.json`, if present, has `tools` (array) and `interrupt_config` (object)
  keys; each tool has `name` (string) and `mcp_server_url` (string).
- Each `skills/<name>/SKILL.md` has YAML frontmatter with `name` (string),
  `description` (string), and optional `type` (literal `inline`); the body
  after the frontmatter is the skill's `instructions`.
- Each `subagents/<name>/` has `agent.json` and `AGENTS.md`.
- No duplicate skill or subagent names across the tree.
- No `deepagents.toml` or `mcp.json` present (else: migration hint).

`build_payload` then assembles the JSON with no further I/O.

## Testing

Snapshot fixtures under `libs/cli/tests/unit_tests/deploy/fixtures/projects/`:

- `bare/` — `agent.json` + `AGENTS.md` only
- `with_tools/` — adds `tools.json`
- `with_skills/` — adds `skills/` with two skills and supporting files
- `with_subagents/` — adds `subagents/` with their own agent.json/AGENTS.md/tools.json
- `subagent_with_local_skills/` — verifies the raw `files` fallback path
- `with_permissions/` — verifies the permissions block round-trips

Each fixture pairs with an `expected_payload.json` asserted byte-for-byte.

`respx` (httpx mock) covers:

- `test_api_client.py` — agents CRUD, mcp-servers CRUD, error envelope parsing,
  4xx vs 5xx behavior, retry-once-on-5xx, ErrorResponse `detail` surfacing.
- `test_mcp_upsert.py` — list+match, header-flag parsing, cached-id-404
  re-resolve, OAuth refusal.
- `test_commands.py` — end-to-end mocked `deploy` of three fixtures; verifies
  `--dry-run` writes valid payload JSON to stdout; verifies POST-then-PATCH
  flow across two deploys against the same project.

`test_state.py` covers round-trip, missing-file initialisation, `--reset`, and
404-fallback.

Deleted: `test_bundler.py`, `test_frontend_bundle.py`, `test_frontend_config.py`.

## Migration impact

Existing `deepagents init`-scaffolded projects will not work post-migration.
The migration hint printed by `deploy` covers the field-by-field mapping;
running `deepagents init --force` overwrites scaffolded files with the new
shape. We add a one-page `MIGRATION.md` to `libs/cli/` covering the same
mapping for users who want to migrate by hand.

Examples in `examples/deploy-*` need rewriting against the new layout. That's
in scope — examples are dev artifacts, not user data.

The shipped pre-built frontend and its build pipeline (`make build-frontends`,
`libs/cli/frontend/`) are removed entirely.

## Open questions

1. **Subagent-local skills via raw `files` map**: does the server actually
   accept `files["subagents/<name>/skills/..."]` alongside typed `subagents[]`,
   or does it reject overlapping paths? Verify against
   `vic/mda-mcp-servers-and-sandboxes` during implementation; fall back to
   "lift to top-level `skills/` with namespace prefix" if rejected.
2. **Endpoint of the LangSmith trace UI link**: the docs show
   `https://smith.langchain.com/o/-/projects/<name>` for traces but the new
   managed agent has a dedicated agent page; confirm the canonical URL shape
   with the MDA team before locking in the success message.
3. **`extras` map**: today's deploy never used it. Worth keeping in the schema
   for forward-compat, but `init` should not scaffold it.

## Out of scope (deferred)

- `deepagents threads create/run/stream` subcommands. The platform supports
  them; users can curl them. Add when there's a concrete user ask.
- `deepagents agents pull <id>` (writes API state back to disk as the project
  layout). Easy follow-up given the 1:1 mapping; not needed for v1.
- `deepagents agents revisions <id>`. The platform exposes revision history;
  we don't surface it yet.
- OAuth-backed MCP server registration through the CLI.
- A `mcp_servers.json` auto-upsert mode at deploy time. Reasonable convenience
  feature for a v2; keeps secrets out of git for v1.
