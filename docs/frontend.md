# Bundled frontend

`deepagent deploy` can ship a pre-built React chat UI alongside your agent on
the same LangSmith Deployment. Flip one flag in `deepagents.toml`, add a few
env vars, and your users get a full-stack app with authentication and
real-time streaming.

## Enable it

```toml
# deepagents.toml
[agent]
name = "my-agent"

[auth]
provider = "supabase"   # or "clerk"

[frontend]
enabled = true
app_name = "My Agent"   # optional; defaults to [agent].name
```

`[frontend].enabled = true` requires `[auth]` — the shipped UI uses the auth
provider's JWT to talk to the agent.

## Environment variables

### Supabase

```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_PUBLISHABLE_DEFAULT_KEY=ey...
```

The frontend reuses these same keys on the browser side — the anon/publishable
key is safe to expose.

### Clerk

```bash
CLERK_SECRET_KEY=sk_test_...        # backend: JWKS/JWT verification
CLERK_PUBLISHABLE_KEY=pk_test_...   # frontend: <ClerkProvider> init
```

## What you get

- Chat composer with streaming message view
- Auth gate (sign-in, sign-up, sign-out)
- Thread picker for switching between past conversations
- Todos panel, files panel, and subagent activity — all reflecting your deep
  agent's live graph state
- Rich rendering for common tools: file read/write/edit, `write_todos`,
  `ls`/`glob`/`grep`, `think`

The frontend is mounted at `/app` on your deployment; the LangGraph API stays
at the root (`/threads`, `/runs`, `/assistants`).

## Social logins

**Clerk:** turn on any provider in the Clerk dashboard's Social Connections
and it shows up on the sign-in screen automatically — Clerk's `<SignIn/>`
component renders buttons for every enabled provider with no config on our
side.

**Supabase:** v2 ships email/password only. Social login is on the roadmap.
If you need social logins today, pick Clerk.

### Clerk redirect URLs

Clerk's development instance auto-whitelists `localhost`. For production,
add your deployment host under **Domains** in the Clerk dashboard.
