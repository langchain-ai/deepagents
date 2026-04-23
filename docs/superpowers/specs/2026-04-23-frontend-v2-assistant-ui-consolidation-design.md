# Default Frontend v2 — assistant-ui + single codebase

**Date:** 2026-04-23
**Branch:** `vic/deepagent-deploy-frontend-v2`
**Status:** Draft — pending implementation
**Supersedes:** The v1 spec (preserved on `vic/deepagent-deploy-frontend-v1-archive`), which shipped two frontend forks (`frontends/supabase/`, `frontends/clerk/`) on pre-assistant-ui React code.

## 1. Problem

v1 shipped a working bundled chat UI for `deepagent deploy`, but two issues surfaced during smoke testing:

1. **Two forks are a maintenance trap.** `frontends/supabase/` and `frontends/clerk/` were ~95% identical. Every bug fix (token refresh, scroll behavior, static paths, title sync) needed to be applied twice, and bit-rot between the two was inevitable.
2. **The pre-assistant-ui base was hand-rolling things the ecosystem already solves.** Scroll-to-bottom during streaming, incremental message rendering, composer input handling, tool-call rendering — all implemented ad-hoc. Bugs (scroll yanking users back down, token streaming visual hitches) that are solved out of the box by `@assistant-ui/react` + `streamdown`.

v2 replaces both with a single codebase built on assistant-ui, with auth provider selected at runtime via lazy-loaded adapter modules.

## 2. Goals

- **One frontend source tree** (`frontends/ui/`) replacing both v1 forks.
- **One pre-built bundle** shipped as package data. Auth provider (Supabase or Clerk) selected at runtime by reading `window.__DEEPAGENTS_CONFIG__.auth`.
- **Assistant-ui native** — `@assistant-ui/react`, `@assistant-ui/react-langgraph`, `@assistant-ui/react-streamdown`, `streamdown`. No hand-rolled SSE or scroll logic.
- **Pluggable auth** via a thin `AuthAdapter` interface. Adding a new provider is one new adapter file, not a fork.
- **Generic UI copy** — drop "Deep Research" vestiges from default strings. `app_name` (from `[frontend].app_name`) already parameterizes the header; apply the same treatment to empty-state suggestions and panel labels.
- **Fix v1 smoke bugs as a side effect:**
  - Scroll respects user intent (assistant-ui's `ThreadPrimitive.Viewport`).
  - Clerk token refresh lives inside the Clerk adapter.
  - `/app` vs `/app/` routing quirk, `/healthz`, `/favicon.ico`, `/.well-known/` — all still exempted from LangGraph auth middleware (same server-side fix as late-v1).

## 3. Non-goals (v2)

- User-supplied welcome message / suggestions (would need a new `deepagents.toml` field).
- Theme or branding customization beyond `app_name`.
- Ejecting the frontend source into the user's project.
- A plugin API for custom tool-call renderers.
- Server-side rendering / Next.js variant.
- Supabase social logins (deferred from v1; same reasoning).
- Automating OAuth redirect-URL configuration on the provider dashboards.

v2 lands as a single PR that replaces the frontend surface but keeps the v1 `deepagents.toml` config shape intact.

## 4. Design

### 4.1 User-visible surface (unchanged from v1)

```toml
[agent]
name = "my-agent"

[auth]
provider = "supabase"       # or "clerk"

[frontend]
enabled = true
app_name = "My Agent"        # optional; defaults to [agent].name
```

No new fields. Same validation rules (`[frontend].enabled = true` requires `[auth]`).

### 4.2 Env vars (unchanged from late-v1)

Backend (JWT validation, unchanged):

```
# Supabase
SUPABASE_URL
SUPABASE_PUBLISHABLE_DEFAULT_KEY

# Clerk
CLERK_SECRET_KEY
```

Frontend (baked into HTML at deploy time, unchanged from late-v1 cleanup):

```
# Clerk only — publishable key is distinct from the secret key used
# by the backend. Supabase reuses the backend values.
CLERK_PUBLISHABLE_KEY
```

### 4.3 Architecture

```
deepagents monorepo
├── libs/cli/deepagents_cli/
│   └── deploy/
│       ├── frontend_dist/          ← single pre-built bundle (no supabase/ vs clerk/ split)
│       │   ├── index.html
│       │   └── assets/...
│       ├── bundler.py              ← copies the single dist into build_dir
│       ├── config.py               ← unchanged from late-v1 (FrontendConfig + validation)
│       ├── templates.py            ← unchanged from late-v1 (APP_PY, AUTH_BLOCKS w/ path exemption)
│       └── ...
│
└── frontends/
    └── ui/                         ← single source tree
        ├── package.json            ← both auth SDKs as deps; lazy-loaded at runtime
        ├── vite.config.ts          ← base: "/app/"
        ├── index.html              ← placeholder script (same mechanism as v1)
        └── src/
            ├── main.tsx
            ├── App.tsx             ← generic layout
            ├── RuntimeProvider.tsx ← useLangGraphRuntime wiring
            ├── runtimeConfig.ts    ← reads window.__DEEPAGENTS_CONFIG__
            ├── auth/
            │   ├── types.ts        ← AuthAdapter interface
            │   ├── loader.tsx      ← dynamic-import router: loadAuth(provider)
            │   ├── supabase.tsx    ← SupabaseAdapter default export
            │   └── clerk.tsx       ← ClerkAdapter default export (+ 45s refresh loop)
            ├── components/
            │   ├── Thread.tsx
            │   ├── tools.tsx       ← tool-renderer dict
            │   ├── SubagentActivity.tsx
            │   ├── TodosPanel.tsx
            │   ├── FilePanels.tsx
            │   └── ThreadPicker.tsx
            └── lib/
                ├── chatApi.ts
                └── format.ts
```

### 4.4 Auth adapter interface

```typescript
// frontends/ui/src/auth/types.ts
export type SessionState =
  | { status: "loading" }
  | { status: "signed-out" }
  | {
      status: "signed-in";
      accessToken: string;
      userIdentity: string;
      signOut: () => Promise<void>;
    };

export interface AuthAdapter {
  /** Top-level provider component. Initializes the auth SDK. */
  Provider: React.ComponentType<{ children: React.ReactNode }>;
  /** Returns current session state. Hook must be called inside `Provider`. */
  useSession: () => SessionState;
  /** Sign-in / sign-up UI. Rendered when session is signed-out. */
  AuthUI: React.ComponentType;
}
```

Each adapter file exports one `AuthAdapter` as its default.

**Supabase adapter:** initializes `supabase-js` client from `runtimeConfig`, uses `supabase.auth.onAuthStateChange` for auto-refresh (SDK handles refresh tokens natively), renders a custom email/password form as `AuthUI`.

**Clerk adapter:** wraps `<ClerkProvider>`, uses `useAuth()` / `useUser()`, includes a 45-second `setInterval` that calls `getToken({ skipCache: true })` to keep the cached JWT fresh (Clerk's default TTL is ~60s). Renders Clerk's `<SignIn />` / `<SignUp />` as `AuthUI`.

### 4.5 Runtime flow

```
1. index.html loads
2. main.tsx reads window.__DEEPAGENTS_CONFIG__ (injected by the bundler at deploy time)
3. main.tsx calls `loadAuth(config.auth)` — dynamic import of the right adapter
4. `<adapter.Provider>` wraps `<App />`
5. App's root reads `adapter.useSession()`:
     - "loading" → null
     - "signed-out" → `<adapter.AuthUI />`
     - "signed-in" → `<AuthenticatedApp />` with accessToken
6. AuthenticatedApp mounts RuntimeProvider with useLangGraphRuntime; SSE streaming begins when a message is sent
```

### 4.6 Code splitting — why this works

The `loadAuth` function uses dynamic `import()`. Vite splits each adapter (and its transitive SDK) into its own chunk:

- `assets/supabase-<hash>.js` (~50 KB gzipped, contains `@supabase/supabase-js`)
- `assets/clerk-<hash>.js` (~55 KB gzipped, contains `@clerk/clerk-react`)

At runtime, only the chunk matching `config.auth` is fetched. The other chunk sits in the deploy bundle but is never downloaded by the user's browser. Net result: one `npm run build`, one committed `frontend_dist/`, tiny runtime cost for flexibility.

### 4.7 Genericization

Strings and UI elements removed or rewritten:

- `constants.ts` default `APP_DESCRIPTION`: "Your deep agent, deployed." (not "Research chat...").
- `Thread.tsx` empty-state suggestions: generic "Ask a question…" / remove the hardcoded research-specific examples.
- `SubagentActivity` user-facing labels: "Task pipeline" instead of "Research progress" / "Subagent pipeline". Internal component name unchanged.
- Header layout: keep the small logo but neutralize — no "Deep Research" references anywhere in the UI source.

The component surface is already agent-agnostic (panels read from `useGraphValues()`; tool renderers are keyed by tool name). The changes are pure copy.

### 4.8 Tool-call renderers (port from v1)

Keep the same four renderers from `tools.tsx` in the reference repo:

- `FileTool` — `read_file`, `write_file`, `edit_file`
- `SearchTool` — `ls`, `glob`, `grep`
- `ThinkTool` — `think`, `think_tool`
- `TodosTool` — `write_todos` (reads live `useGraphValues().todos`)
- `SubagentTool` — `task` (individual subagent delegation calls)

Other tool calls fall through to assistant-ui's default renderer.

### 4.9 Deep-agent-specific panels (port from v1)

- **`TodosPanel`** — collapsible bottom panel, reads `useGraphValues().todos`.
- **`FilePanels`** — collapsible bottom panel, reads `useGraphValues().files`.
- **`SubagentActivity`** — inline pipeline view between messages. Kept because multi-agent orchestration is the point of deep agents; a user who doesn't use subagents simply sees it stay empty.
- **`ThreadPicker`** — dropdown in the header. Uses `@langchain/langgraph-sdk` `Client.threads.search()`.
- **No `SettingsModal`** — drop. `assistantId` is fixed to the graph key in `langgraph.json`; per-user overrides aren't a v2 concern.

### 4.10 Bundler simplifications vs v1

- `_copy_frontend_dist(config, build_dir)` stops branching on `config.auth.provider`. It copies the single `frontend_dist/` tree unconditionally.
- `frontend_dist/` package-data directory collapses from two subtrees to one.
- `Makefile` `build-frontends` target: one `npm ci && npm run build` pass instead of two.
- `libs/cli/pyproject.toml` `include` glob simplifies.
- The placeholder rewrite (`window.__DEEPAGENTS_CONFIG__ = {...};`) is unchanged — still injects the `auth` field so the frontend knows which adapter to load.

### 4.11 Static-path auth exemption (port from late-v1)

The generated `auth.py` still exempts `/app`, `/app/*`, `/healthz`, `/favicon.ico`, and `/.well-known/*` from Bearer-token validation. Without this, the browser can't even load the HTML to show the sign-in UI. No change from the late-v1 fix.

### 4.12 `deepagents dev`

Unchanged — still re-uses the bundler codepath. `langgraph dev` mounts the same generated `app.py` locally, so `deepagent dev` serves the full v2 frontend at `http://localhost:2024/app/`.

### 4.13 Error handling

| Scenario | Behavior |
|---|---|
| `[frontend].enabled = true`, no `[auth]` | Deploy errors before bundling (unchanged). |
| Required backend or frontend env var missing | Deploy errors with the specific missing var names (unchanged). |
| Auth adapter fails to load (network error on lazy import) | Full-page error: "Could not load authentication. Refresh to retry." |
| `window.__DEEPAGENTS_CONFIG__.auth` is an unknown value | Dev-mode console error + blank page. Prevented at bundle time by existing validation. |
| JWT validation fails on backend | 401 from LangGraph; frontend catches, `useSession()` flips to `signed-out`. |

### 4.14 Testing

- **Unit:** existing `config.py` tests stay. `bundler.py` tests updated to assert a single `frontend_dist/` copy (no per-provider branch).
- **Unit:** new test confirming `_copy_frontend_dist` works for both `auth.provider = "supabase"` and `"clerk"` with the same shipped bundle.
- **Integration:** `deepagents deploy --dry-run` snapshot test, one fixture per auth provider, same shipped bundle consumed by both.
- **Manual pre-release:** `deepagents dev` against a real Supabase project and a real Clerk application, one browser pass per provider. Automated e2e (Playwright) is out of scope for v2.

## 5. Migration from main

Because v2 builds on a fresh branch from `origin/main` (the v1 work lives on `vic/deepagent-deploy-frontend-v1-archive` but never landed), the PR diff against main is exactly the v2 state: no teardown commits, no replaced files.

Backend-side artifacts that v1 landed AND v2 keeps are re-implemented on this branch from the archive as reference. Specifically: `FrontendConfig` dataclass, cross-section validation, `_FRONTEND_EXTRA_ENV` map, `APP_PY_TEMPLATE` (Starlette), static-path auth exemption, `/app` → `/app/` redirect, `http.app` wiring in `langgraph.json`, `Makefile` `build-frontends` target, `docs/frontend.md` documentation, `.gitignore` rules. The code patterns are known; this is targeted re-application, not redesign.

## 6. File changes summary

**New files:**

- `frontends/ui/` — entire source tree (~20 files).
- `libs/cli/deepagents_cli/deploy/frontend_dist/` — single pre-built bundle (committed to git; rebuilt via `make build-frontends` on release).
- `docs/superpowers/specs/2026-04-23-frontend-v2-assistant-ui-consolidation-design.md` — this doc.
- `docs/superpowers/plans/2026-04-23-frontend-v2-<filename>.md` — forthcoming implementation plan.
- `docs/frontend.md` — user-facing docs.
- `Makefile` — `build-frontends` target.

**Modified files (in `libs/cli/`):**

- `deepagents_cli/deploy/config.py` — `FrontendConfig`, validation, starter templates.
- `deepagents_cli/deploy/bundler.py` — frontend copy, placeholder rewrite, `app.py` emit, `http.app` in `langgraph.json`.
- `deepagents_cli/deploy/templates.py` — `APP_PY_TEMPLATE` (Starlette), auth blocks with `_is_public_path` exemption.
- `pyproject.toml` — wheel `include` glob for `frontend_dist/**`.

**Modified files (repo root):**

- `.gitignore` — `frontends/*/node_modules/`, `frontends/*/dist/`, unignore `frontends/**/src/lib/`.

## 7. Open implementation questions

Deferred to the plan phase:

- Whether `frontend_dist/` is committed to git (simpler CI) or generated in the release workflow only.
- How the Makefile target is invoked across OSes (uv-based script vs plain `make`).
- Whether to add a smoke-test script that curls `/app/`, `/healthz`, and `/threads` post-deploy.

## 8. Success criteria

v2 is done when:

- A single `make build-frontends` run produces `libs/cli/deepagents_cli/deploy/frontend_dist/{index.html, assets/...}`.
- `deepagents deploy --dry-run` against both Supabase and Clerk fixtures produces identical `frontend_dist/` contents in the build dir; only `index.html`'s `window.__DEEPAGENTS_CONFIG__` differs.
- Manual smoke: Clerk and Supabase deployments each render the full chat UI, stream responses, populate the todos/files/subagent panels, and survive an idle period longer than the Clerk token TTL without failing.
- `libs/cli/pyproject.toml` has one `include` entry for frontend_dist; no `supabase/` or `clerk/` subpaths anywhere in Python code.
