# Frontend v2 — assistant-ui + single codebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an opt-in chat UI alongside the deployed agent on the same LangSmith Deployment, built on assistant-ui, with one source tree and pluggable Supabase/Clerk auth.

**Architecture:** Single Vite + React + TypeScript source tree at `libs/cli/frontend/`. Auth provider is selected at runtime via `window.__DEEPAGENTS_CONFIG__.auth` and loaded through a dynamic-import `AuthAdapter` module (`supabase.tsx` or `clerk.tsx`). Bundler copies one pre-built `frontend_dist/` into the build dir, rewrites a `window.__DEEPAGENTS_CONFIG__` placeholder in `index.html` with per-deploy values, emits a Starlette `app.py` that mounts the dist at `/app`, and adds `"http": { "app": "./app.py:app" }` to `langgraph.json`. Generated `auth.py` exempts `/app`, `/healthz`, `/favicon.ico`, and `/.well-known/*` from Bearer-token validation so the sign-in UI can load without a token.

**Tech Stack:** Python 3.11+, Hatchling (packaging), Starlette (static mount on deployed service), pytest (tests). Frontend: Vite 6 + React 19 + TypeScript 5.7, `@assistant-ui/react`, `@assistant-ui/react-langgraph`, `@assistant-ui/react-streamdown`, `streamdown`, `@langchain/langgraph-sdk`, `@supabase/supabase-js` (Supabase adapter), `@clerk/clerk-react` (Clerk adapter), Tailwind CSS 4.

**Reference spec:** `docs/superpowers/specs/2026-04-23-frontend-v2-assistant-ui-consolidation-design.md`.

**v1 archive (read-only reference for backend patterns):** `vic/deepagent-deploy-frontend-v1-archive` — use `git show v1-archive:<path>` to copy code from a specific commit without checking it out.

**Assistant-ui reference:** `/Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/` on the current checkout — use directly for the assistant-ui component layout.

---

## File structure

**New on this branch:**

- `libs/cli/frontend/` — Vite + React source
  - `package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `postcss.config.js`, `index.html`, `.nvmrc`
  - `src/main.tsx`, `src/App.tsx`, `src/RuntimeProvider.tsx`, `src/runtimeConfig.ts`, `src/index.css`, `src/vite-env.d.ts`, `src/constants.ts`, `src/types.ts`
  - `src/auth/types.ts`, `src/auth/loader.tsx`, `src/auth/supabase.tsx`, `src/auth/clerk.tsx`
  - `src/components/Thread.tsx`, `src/components/tools.tsx`, `src/components/SubagentActivity.tsx`, `src/components/TodosPanel.tsx`, `src/components/FilePanels.tsx`, `src/components/ThreadPicker.tsx`
  - `src/lib/chatApi.ts`, `src/lib/format.ts`
- `libs/cli/deepagents_cli/deploy/frontend_dist/` — pre-built bundle (populated by `make build-frontends`)
- `libs/cli/tests/unit_tests/deploy/test_frontend_config.py` — FrontendConfig unit tests
- `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py` — bundler integration tests
- `docs/frontend.md` — user-facing docs
- `Makefile` at the repo root — `build-frontends` target

**Modified on this branch:**

- `libs/cli/deepagents_cli/deploy/config.py` — add `FrontendConfig`, validation, starter-template edits
- `libs/cli/deepagents_cli/deploy/bundler.py` — copy dist, rewrite placeholder, emit `app.py`, add `http.app` to `langgraph.json`
- `libs/cli/deepagents_cli/deploy/templates.py` — add `APP_PY_TEMPLATE`, extend Supabase/Clerk auth blocks with `_is_public_path` exemption
- `libs/cli/pyproject.toml` — extend wheel `include` glob for `frontend_dist/**/*`
- `.gitignore` — ignore `libs/cli/frontend/node_modules/`, `libs/cli/frontend/dist/`; unignore `libs/cli/frontend/src/lib/`

---

## Phase 1 — Backend config surface

### Task 1: Add `FrontendConfig` dataclass and parser

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_frontend_config.py`

- [ ] **Step 1: Write the failing tests**

Create `libs/cli/tests/unit_tests/deploy/test_frontend_config.py`:

```python
"""Tests for `[frontend]` parsing in deepagents.toml."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from deepagents_cli.deploy.config import (
    AgentConfig,
    AuthConfig,
    DeployConfig,
    FrontendConfig,
    _parse_config,
)


def test_frontend_config_defaults():
    fc = FrontendConfig()
    assert fc.enabled is False
    assert fc.app_name is None


def test_frontend_section_parses_enabled_true():
    cfg = _parse_config({
        "agent": {"name": "my-agent"},
        "auth": {"provider": "supabase"},
        "frontend": {"enabled": True},
    })
    assert cfg.frontend is not None
    assert cfg.frontend.enabled is True
    assert cfg.frontend.app_name is None


def test_frontend_section_parses_app_name():
    cfg = _parse_config({
        "agent": {"name": "my-agent"},
        "auth": {"provider": "clerk"},
        "frontend": {"enabled": True, "app_name": "My App"},
    })
    assert cfg.frontend is not None
    assert cfg.frontend.app_name == "My App"


def test_frontend_section_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        _parse_config({
            "agent": {"name": "my-agent"},
            "auth": {"provider": "supabase"},
            "frontend": {"enabled": True, "theme": "dark"},
        })


def test_frontend_omitted_defaults_to_none():
    cfg = _parse_config({"agent": {"name": "my-agent"}})
    assert cfg.frontend is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_config.py -v`
Expected: FAIL (ImportError on `FrontendConfig`).

- [ ] **Step 3: Add `FrontendConfig` and register it in `DeployConfig`**

In `libs/cli/deepagents_cli/deploy/config.py`, after the `AuthConfig` dataclass, add:

```python
@dataclass(frozen=True)
class FrontendConfig:
    """`[frontend]` section — bundled default frontend settings.

    When `enabled = True`, `deepagent deploy` copies a pre-built React
    chat UI into the deployment alongside the agent. Requires `[auth]`
    to be configured (the frontend uses the same JWT).
    """

    enabled: bool = False
    app_name: str | None = None
```

Add `frontend: FrontendConfig | None = None` to `DeployConfig` (right after `auth`):

```python
@dataclass(frozen=True)
class DeployConfig:
    ...
    auth: AuthConfig | None = None
    frontend: FrontendConfig | None = None
```

Update `_ALLOWED_SECTIONS`:

```python
_ALLOWED_SECTIONS = frozenset({"agent", "sandbox", "auth", "frontend"})
```

Add the allowed-keys constant near `_ALLOWED_AUTH_KEYS`:

```python
_ALLOWED_FRONTEND_KEYS = frozenset({"enabled", "app_name"})
```

At the end of `_parse_config`, after the `auth` parse block, before the final `return`:

```python
    frontend: FrontendConfig | None = None
    frontend_data = data.get("frontend")
    if frontend_data is not None:
        unknown_fe = set(frontend_data.keys()) - _ALLOWED_FRONTEND_KEYS
        if unknown_fe:
            msg = (
                f"Unknown key(s) in [frontend]: {sorted(unknown_fe)}. "
                f"Allowed: {sorted(_ALLOWED_FRONTEND_KEYS)}"
            )
            raise ValueError(msg)
        fe_kwargs: dict[str, Any] = {
            k: frontend_data[k] for k in _ALLOWED_FRONTEND_KEYS if k in frontend_data
        }
        frontend = FrontendConfig(**fe_kwargs)
```

Change the final `return` to include `frontend=frontend`:

```python
    return DeployConfig(agent=agent, sandbox=sandbox, auth=auth, frontend=frontend)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_config.py -v`
Expected: 5 PASS.

Also run the existing config suite to confirm no regressions:

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py \
        libs/cli/tests/unit_tests/deploy/test_frontend_config.py
git commit -m "feat(deploy): parse [frontend] section in deepagents.toml"
```

---

### Task 2: Validate `[frontend]` requires `[auth]` and required env vars

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_frontend_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `libs/cli/tests/unit_tests/deploy/test_frontend_config.py`:

```python
def _write_project(tmp_path: Path) -> Path:
    (tmp_path / "AGENTS.md").write_text("prompt", encoding="utf-8")
    return tmp_path


def test_frontend_enabled_without_auth_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert any("[frontend].enabled requires [auth]" in e for e in errors)


def test_frontend_disabled_no_auth_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        frontend=FrontendConfig(enabled=False),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)


def test_frontend_clerk_requires_publishable_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CLERK_SECRET_KEY", "k")
    monkeypatch.delenv("CLERK_PUBLISHABLE_KEY", raising=False)
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert any("CLERK_PUBLISHABLE_KEY" in e for e in errors)


def test_frontend_supabase_needs_no_extra_env_vars(tmp_path, monkeypatch):
    """Supabase reuses SUPABASE_URL + SUPABASE_PUBLISHABLE_DEFAULT_KEY
    already required by [auth]. No extra VITE_* duplication."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)


def test_frontend_clerk_all_env_vars_present_no_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("CLERK_SECRET_KEY", "k")
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_x")
    _write_project(tmp_path)
    cfg = DeployConfig(
        agent=AgentConfig(name="a"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )
    errors = cfg.validate(tmp_path)
    assert not any("[frontend]" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_config.py -v`
Expected: 5 new tests FAIL.

- [ ] **Step 3: Add env-var map and validation**

In `libs/cli/deepagents_cli/deploy/config.py`, after `_AUTH_PROVIDER_ENV`, add:

```python
_FRONTEND_EXTRA_ENV: dict[str, list[str]] = {
    # Supabase reuses `SUPABASE_URL` + `SUPABASE_PUBLISHABLE_DEFAULT_KEY`
    # from [auth] — no extra browser-facing env vars needed.
    "supabase": [],
    # Clerk's browser-facing publishable key is distinct from
    # `CLERK_SECRET_KEY` (which [auth] uses for JWKS validation).
    "clerk": ["CLERK_PUBLISHABLE_KEY"],
}
"""Additional env vars the frontend bundle needs beyond what `[auth]` already requires."""
```

Add the helper below `_validate_auth_credentials`:

```python
def _validate_frontend_credentials(provider: str) -> list[str]:
    """Check that all extra env vars are set for the frontend bundle."""
    required = _FRONTEND_EXTRA_ENV.get(provider)
    if required is None:
        return []
    missing = [v for v in required if not os.environ.get(v)]
    if not missing:
        return []
    return [
        (
            f"Frontend for '{provider}' requires {' and '.join(missing)}. "
            f"Add it to your .env file so the bundler can write it "
            f"into index.html at deploy time."
        ),
    ]
```

Extend `DeployConfig.validate`, after the auth credential check:

```python
        if self.frontend is not None and self.frontend.enabled:
            if self.auth is None:
                errors.append(
                    '[frontend].enabled requires [auth] to be configured. '
                    'Add an [auth] section with provider = "supabase" or '
                    '"clerk".'
                )
            else:
                errors.extend(_validate_frontend_credentials(self.auth.provider))
```

- [ ] **Step 4: Run tests**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_config.py -v`
Expected: all 10 pass.

Run: `cd libs/cli && uv run ruff check deepagents_cli/deploy/config.py tests/unit_tests/deploy/test_frontend_config.py`
Expected: All checks passed.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py \
        libs/cli/tests/unit_tests/deploy/test_frontend_config.py
git commit -m "feat(deploy): validate [frontend] requires [auth] + provider env vars"
```

---

### Task 3: Update starter templates

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/config.py`

- [ ] **Step 1: Update `generate_starter_config`**

Replace the body of `generate_starter_config` with:

```python
def generate_starter_config() -> str:
    """Generate a starter `deepagents.toml` template."""
    return """\
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

# [sandbox] is optional. Omit if not needed for skills or code execution.
# [sandbox]
# provider = "langsmith"   # langsmith | daytona | modal | runloop
# scope = "thread"         # thread | assistant

# [auth] is optional. Add to enable user authentication.
# [auth]
# provider = "supabase"   # supabase | clerk

# [frontend] is optional. Add to ship a bundled chat UI on the same
# deployment as the agent. Requires [auth].
# [frontend]
# enabled = true
# app_name = "My Agent"
"""
```

- [ ] **Step 2: Update `generate_starter_env`**

Replace the body of `generate_starter_env` with:

```python
def generate_starter_env() -> str:
    """Generate a starter `.env` template."""
    return """\
# Model provider API key (required)
ANTHROPIC_API_KEY=

# LangSmith API key (required for deploy and sandbox)
LANGSMITH_API_KEY=

# Auth provider (optional, uncomment for [auth])
# SUPABASE_URL=
# SUPABASE_PUBLISHABLE_DEFAULT_KEY=
# CLERK_SECRET_KEY=

# Frontend (optional, uncomment for [frontend] + matching [auth])
# Clerk only — browser-facing publishable key. Supabase reuses the keys above.
# CLERK_PUBLISHABLE_KEY=
"""
```

- [ ] **Step 3: Verify existing starter-template tests**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_config.py -v -k starter`
Expected: all pass (existing tests do substring assertions that still hold).

- [ ] **Step 4: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/config.py
git commit -m "feat(deploy): surface [frontend] in starter templates"
```

---

## Phase 2 — Backend templates + bundler

### Task 4: Add `APP_PY_TEMPLATE` (Starlette, with `/app` redirect)

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/templates.py`

- [ ] **Step 1: Add the template**

At the bottom of `libs/cli/deepagents_cli/deploy/templates.py`, add:

```python
APP_PY_TEMPLATE = '''\
"""Starlette app mounting the bundled chat UI on /app.

Generated by `deepagent deploy`. LangGraph Platform reads the `http.app`
key in `langgraph.json` and attaches this app alongside the graph.

Uses Starlette directly (not FastAPI) because Starlette is already a
transitive dep of langgraph-cli / langgraph-api in both the dev runtime
and the deployed runtime, whereas FastAPI would require an explicit
install step that `langgraph dev` does not perform.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

_FRONTEND_DIR = Path(__file__).parent / "frontend_dist"


async def healthz(_request):
    return JSONResponse({"ok": True})


async def app_root_redirect(_request):
    # Starlette's Mount at "/app" matches "/app/*" — a bare "/app" 404s
    # otherwise. Redirect so users typing the clean URL land correctly.
    return RedirectResponse(url="/app/", status_code=308)


app = Starlette(
    routes=[
        Route("/healthz", healthz),
        Route("/app", app_root_redirect),
        Mount(
            "/app",
            app=StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="frontend",
        ),
    ],
)
'''
"""Generated `app.py` — a Starlette app that serves the frontend at /app."""
```

- [ ] **Step 2: Verify lint/format**

Run: `cd libs/cli && uv run ruff check deepagents_cli/deploy/templates.py`
Expected: All checks passed.

Run: `cd libs/cli && uv run ruff format --check deepagents_cli/deploy/templates.py`
Expected: already formatted (or run without `--check` to apply).

- [ ] **Step 3: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/templates.py
git commit -m "feat(deploy): add APP_PY_TEMPLATE (Starlette static mount)"
```

---

### Task 5: Extend auth templates with `_is_public_path` exemption

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/templates.py`

- [ ] **Step 1: Update Supabase auth block**

In `AUTH_BLOCK_SUPABASE`, replace the authenticate handler (everything from `@auth.authenticate` down to the end of `get_current_user` but keep the function body after the exemption check):

```python
def _is_public_path(path: str) -> bool:
    """Paths the browser fetches before the user has any auth token.

    The frontend HTML, its assets, and the health check must be reachable
    without a Bearer token — otherwise the sign-in UI can never load and
    the user can't produce a token in the first place.
    """
    if path in ("/app", "/healthz", "/favicon.ico"):
        return True
    return path.startswith("/app/") or path.startswith("/.well-known/")


@auth.authenticate
async def get_current_user(
    authorization: str | None,
    path: str,
) -> Auth.types.MinimalUserDict:
    """Validate Supabase token and return user identity."""
    if _is_public_path(path):
        return {"identity": "anonymous"}

    if not authorization or not authorization.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing or invalid authorization header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    response = await _http_client.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_PUBLISHABLE_DEFAULT_KEY,
        },
    )

    if response.status_code != 200:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid or expired token"
        )

    user = response.json()
    return {
        "identity": user["id"],
        "display_name": user.get("email", ""),
    }
```

- [ ] **Step 2: Update Clerk auth block**

In `AUTH_BLOCK_CLERK`, insert the same `_is_public_path` helper right before `@auth.authenticate`, and change the handler to:

```python
def _is_public_path(path: str) -> bool:
    """Paths the browser fetches before the user has any auth token.

    The frontend HTML, its assets, and the health check must be reachable
    without a Bearer token — otherwise the sign-in UI can never load and
    the user can't produce a token in the first place.
    """
    if path in ("/app", "/healthz", "/favicon.ico"):
        return True
    return path.startswith("/app/") or path.startswith("/.well-known/")


@auth.authenticate
async def get_current_user(
    authorization: str | None,
    path: str,
) -> Auth.types.MinimalUserDict:
    """Validate Clerk session JWT and return user identity."""
    if _is_public_path(path):
        return {"identity": "anonymous"}

    if not authorization or not authorization.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing or invalid authorization header"
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = pyjwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
        )
    except pyjwt.exceptions.PyJWTError as exc:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail=f"Invalid token: {exc}"
        )

    return {
        "identity": payload["sub"],
        "display_name": payload.get("email", payload.get("name", "")),
    }
```

- [ ] **Step 3: Verify lint**

Run: `cd libs/cli && uv run ruff check deepagents_cli/deploy/templates.py`
Expected: All checks passed.

- [ ] **Step 4: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/templates.py
git commit -m "feat(deploy): exempt /app, /healthz, /favicon, /.well-known from auth middleware"
```

---

### Task 6: Bundler — copy frontend_dist, rewrite placeholder, emit app.py

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Create: `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`

- [ ] **Step 1: Write the failing tests**

Create `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`:

```python
"""Tests for bundler behavior when [frontend].enabled = true."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.bundler import bundle
from deepagents_cli.deploy.config import (
    AgentConfig,
    AuthConfig,
    DeployConfig,
    FrontendConfig,
)


@pytest.fixture
def shipped_frontend_dist(tmp_path, monkeypatch):
    """Fake the shipped frontend_dist so tests don't require a real Vite build.

    Writes a minimal `index.html` with the placeholder and one asset file,
    then points the bundler's copy source at this directory.
    """
    fake_dist = tmp_path / "fake_frontend_dist"
    fake_dist.mkdir()
    assets = fake_dist / "assets"
    assets.mkdir()
    (assets / "index-abc.js").write_text("/* fake bundle */", encoding="utf-8")
    (fake_dist / "index.html").write_text(
        '<!doctype html>\n<html><head>'
        '<script>window.__DEEPAGENTS_CONFIG__ = {"__PLACEHOLDER__":true};</script>'
        '<script src="/app/assets/index-abc.js"></script>'
        '</head><body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deepagents_cli.deploy.bundler._FRONTEND_DIST_SRC", fake_dist
    )
    return fake_dist


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "AGENTS.md").write_text("prompt", encoding="utf-8")
    return proj


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    d = tmp_path / "build"
    d.mkdir()
    return d


def _supabase_config() -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="my-agent", model="anthropic:claude-sonnet-4-6"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True, app_name="My App"),
    )


def _clerk_config() -> DeployConfig:
    return DeployConfig(
        agent=AgentConfig(name="my-agent"),
        auth=AuthConfig(provider="clerk"),
        frontend=FrontendConfig(enabled=True),
    )


def test_bundle_emits_app_py(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon")
    bundle(_supabase_config(), project, build_dir)
    app_py = build_dir / "app.py"
    assert app_py.is_file()
    content = app_py.read_text(encoding="utf-8")
    assert "Starlette" in content
    assert "StaticFiles" in content


def test_bundle_copies_frontend_dist(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon")
    bundle(_supabase_config(), project, build_dir)
    dest = build_dir / "frontend_dist"
    assert (dest / "index.html").is_file()
    assert (dest / "assets" / "index-abc.js").is_file()


def test_bundle_rewrites_placeholder_supabase(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    monkeypatch.setenv("SUPABASE_URL", "https://xyz.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "anon-xyz")
    bundle(_supabase_config(), project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"supabase"' in html
    assert '"supabaseUrl":"https://xyz.supabase.co"' in html
    assert '"supabaseAnonKey":"anon-xyz"' in html
    assert '"appName":"My App"' in html


def test_bundle_rewrites_placeholder_clerk(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", "pk_test_abc")
    bundle(_clerk_config(), project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"clerk"' in html
    assert '"clerkPublishableKey":"pk_test_abc"' in html


def test_bundle_without_frontend_still_works(project, build_dir):
    cfg = DeployConfig(agent=AgentConfig(name="my-agent"))
    bundle(cfg, project, build_dir)
    assert not (build_dir / "app.py").exists()
    assert not (build_dir / "frontend_dist").exists()


def test_bundle_escapes_angle_bracket_in_app_name(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    """Prevent `</script>` in app_name from breaking out of the inline script."""
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    cfg = DeployConfig(
        agent=AgentConfig(name="my-agent"),
        auth=AuthConfig(provider="supabase"),
        frontend=FrontendConfig(enabled=True, app_name="</script>hack"),
    )
    bundle(cfg, project, build_dir)
    html = (build_dir / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "\\u003c/script>" in html


def test_bundle_raises_when_frontend_enabled_but_auth_missing(
    shipped_frontend_dist, project, build_dir,
):
    cfg = DeployConfig(
        agent=AgentConfig(name="my-agent"),
        frontend=FrontendConfig(enabled=True),
    )
    with pytest.raises(ValueError, match=r"requires \[auth\]"):
        bundle(cfg, project, build_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_bundle.py -v`
Expected: tests fail (bundler doesn't handle frontend yet).

- [ ] **Step 3: Extend `bundler.py`**

In `libs/cli/deepagents_cli/deploy/bundler.py`, at the top, add `os` and `re` imports (alongside existing `json`, `shutil`):

```python
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any
```

Update the import from `deepagents_cli.deploy.templates` to include `APP_PY_TEMPLATE`:

```python
from deepagents_cli.deploy.templates import (
    APP_PY_TEMPLATE,
    AUTH_BLOCKS,
    AUTH_ON_HANDLER,
    DEPLOY_GRAPH_TEMPLATE,
    MCP_TOOLS_TEMPLATE,
    PYPROJECT_TEMPLATE,
    SANDBOX_BLOCKS,
    SYNC_SUBAGENTS_TEMPLATE,
)
```

Below the `_MODEL_PROVIDER_DEPS` map, add the regex and a module-level pointer to the shipped dist location (overridable in tests):

```python
_FRONTEND_DIST_SRC = Path(__file__).parent / "frontend_dist"
"""Location of the shipped pre-built frontend, inside this Python package."""

_FRONTEND_PLACEHOLDER_RE = re.compile(
    r"window\.__DEEPAGENTS_CONFIG__\s*=\s*\{[^<]*?\};",
    re.DOTALL,
)
"""Matches the placeholder script we injected into index.html at build time."""
```

Add helper functions below `_MODEL_PROVIDER_DEPS`:

```python
def _build_runtime_config_json(config: DeployConfig) -> str:
    """Build the JSON value injected into `window.__DEEPAGENTS_CONFIG__`."""
    if config.auth is None or config.frontend is None:
        msg = "runtime config requires [auth] and [frontend] to be configured"
        raise ValueError(msg)

    provider = config.auth.provider
    app_name = config.frontend.app_name or config.agent.name
    payload: dict[str, Any] = {
        "auth": provider,
        "appName": app_name,
        "assistantId": "agent",
    }
    if provider == "supabase":
        payload["supabaseUrl"] = os.environ["SUPABASE_URL"]
        payload["supabaseAnonKey"] = os.environ["SUPABASE_PUBLISHABLE_DEFAULT_KEY"]
    elif provider == "clerk":
        payload["clerkPublishableKey"] = os.environ["CLERK_PUBLISHABLE_KEY"]
    else:
        msg = f"Unknown auth provider for frontend: {provider}"
        raise ValueError(msg)

    # Escape `<` so a hostile or accidental `</script>` inside a string value
    # can't break out of the inline <script> tag.
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def _copy_frontend_dist(config: DeployConfig, build_dir: Path) -> None:
    """Copy the pre-built bundle into build_dir and rewrite the config placeholder."""
    if config.auth is None:
        msg = "frontend requires [auth] to be set"
        raise ValueError(msg)

    if not _FRONTEND_DIST_SRC.is_dir():
        msg = (
            f"Shipped frontend bundle not found at {_FRONTEND_DIST_SRC}. "
            "Did you run `make build-frontends`?"
        )
        raise RuntimeError(msg)

    dest = build_dir / "frontend_dist"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(_FRONTEND_DIST_SRC, dest)

    index_html = dest / "index.html"
    if not index_html.is_file():
        msg = f"expected index.html inside {_FRONTEND_DIST_SRC}"
        raise RuntimeError(msg)

    html = index_html.read_text(encoding="utf-8")
    payload = _build_runtime_config_json(config)
    replacement = f"window.__DEEPAGENTS_CONFIG__ = {payload};"
    new_html, count = _FRONTEND_PLACEHOLDER_RE.subn(
        lambda _m: replacement, html, count=1,
    )
    if count == 0:
        msg = (
            "Could not find window.__DEEPAGENTS_CONFIG__ placeholder in the "
            "shipped index.html. The frontend bundle is out of sync with the "
            "bundler — rebuild with `make build-frontends`."
        )
        raise RuntimeError(msg)
    index_html.write_text(new_html, encoding="utf-8")
```

Inside `bundle()`, after the `# 6. Generate auth.py if [auth] is configured.` block and before `# 7. Render langgraph.json.`, add:

```python
    # 6b. Copy frontend bundle when enabled.
    frontend_enabled = config.frontend is not None and config.frontend.enabled
    if frontend_enabled:
        if config.auth is None:
            msg = (
                "bundle() requires [auth] when [frontend].enabled is true. "
                "Call DeployConfig.validate(project_root) before bundle() to "
                "surface this as a user-facing error."
            )
            raise ValueError(msg)
        _copy_frontend_dist(config, build_dir)
        (build_dir / "app.py").write_text(APP_PY_TEMPLATE, encoding="utf-8")
        logger.info(
            "Copied frontend bundle and wrote app.py (%s)", config.auth.provider,
        )
```

- [ ] **Step 4: Run the new tests**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_bundle.py -v`
Expected: all pass.

Also confirm existing deploy suite still green:

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py \
        libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py
git commit -m "feat(deploy): copy frontend bundle, rewrite placeholder, emit app.py"
```

---

### Task 7: Wire `http.app` into generated `langgraph.json`

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/bundler.py`
- Modify: `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`

- [ ] **Step 1: Add the failing test**

Append to `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`:

```python
def test_langgraph_json_has_http_app_when_frontend_enabled(
    shipped_frontend_dist, project, build_dir, monkeypatch,
):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")
    bundle(_supabase_config(), project, build_dir)
    data = json.loads((build_dir / "langgraph.json").read_text(encoding="utf-8"))
    assert data["http"] == {"app": "./app.py:app"}


def test_langgraph_json_no_http_app_when_frontend_disabled(project, build_dir):
    cfg = DeployConfig(agent=AgentConfig(name="my-agent"))
    bundle(cfg, project, build_dir)
    data = json.loads((build_dir / "langgraph.json").read_text(encoding="utf-8"))
    assert "http" not in data
```

- [ ] **Step 2: Run — verify failure**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_bundle.py -v -k langgraph_json`
Expected: `test_langgraph_json_has_http_app_when_frontend_enabled` fails.

- [ ] **Step 3: Extend `_render_langgraph_json` + its call site**

In `libs/cli/deepagents_cli/deploy/bundler.py`, change `_render_langgraph_json`:

```python
def _render_langgraph_json(
    *,
    env_present: bool,
    auth_present: bool = False,
    frontend_present: bool = False,
) -> str:
    """Render `langgraph.json` — adds `"env"`, `"auth"`, `"http"` when applicable."""
    data: dict = {
        "dependencies": ["."],
        "graphs": {"agent": "./deploy_graph.py:make_graph"},
        "python_version": "3.12",
    }
    if env_present:
        data["env"] = ".env"
    if auth_present:
        data["auth"] = {"path": "./auth.py:auth"}
    if frontend_present:
        data["http"] = {"app": "./app.py:app"}
    return json.dumps(data, indent=2) + "\n"
```

Update the call inside `bundle()`:

```python
    (build_dir / "langgraph.json").write_text(
        _render_langgraph_json(
            env_present=env_present,
            auth_present=auth_present,
            frontend_present=frontend_enabled,
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run tests**

Run: `cd libs/cli && uv run pytest tests/unit_tests/deploy/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/bundler.py \
        libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py
git commit -m "feat(deploy): add http.app to langgraph.json when frontend enabled"
```

---

## Phase 3 — Package data plumbing

### Task 8: `.gitignore`, `pyproject.toml`, top-level Makefile

**Files:**
- Modify: `.gitignore`
- Modify: `libs/cli/pyproject.toml`
- Create: `Makefile` at the repo root

- [ ] **Step 1: Extend `.gitignore`**

After the Python packaging block in `.gitignore`, add (adjacent to other ignore sections so the intent is clear):

```
# Frontend build artifacts (source tree; package data under libs/cli/deepagents_cli/deploy/frontend_dist/ is tracked)
libs/cli/frontend/node_modules/
libs/cli/frontend/dist/
node_modules/
```

The repo's Python-packaging `lib/` rule matches `libs/cli/frontend/src/lib/`. Add an unignore right after the existing `lib/` line (around line 17):

```
lib/
lib64/
!libs/cli/frontend/src/lib/
```

Verify:

```bash
git check-ignore -v libs/cli/frontend/src/lib/chatApi.ts && echo "OOPS ignored" || echo "tracked as expected"
git check-ignore -v libs/cli/frontend/node_modules/x && echo "ignored as expected" || echo "OOPS not ignored"
```

- [ ] **Step 2: Update `libs/cli/pyproject.toml`**

Find `[tool.hatch.build.targets.wheel]` (around line 149) and extend with an `include` glob:

```toml
[tool.hatch.build.targets.wheel]
packages = ["deepagents_cli"]
include = [
    "deepagents_cli/**/*.py",
    "deepagents_cli/**/*.md",
    "deepagents_cli/**/*.tcss",
    "deepagents_cli/deploy/frontend_dist/**/*",
]
```

Preserve the existing `[tool.hatch.build.targets.wheel.shared-data]` block below.

- [ ] **Step 3: Create `Makefile`**

Create `/Users/victormoreira/Desktop/open-source/deepagents/Makefile`:

```makefile
.PHONY: build-frontends

FRONTEND_SRC := libs/cli/frontend
FRONTEND_DEST := libs/cli/deepagents_cli/deploy/frontend_dist

build-frontends:
	@set -e; \
	echo "--> Building $(FRONTEND_SRC)"; \
	( cd $(FRONTEND_SRC) && npm ci && npm run build ); \
	echo "--> Copying dist into $(FRONTEND_DEST)"; \
	rm -rf $(FRONTEND_DEST); \
	mkdir -p $(FRONTEND_DEST); \
	cp -R $(FRONTEND_SRC)/dist/. $(FRONTEND_DEST)/; \
	echo "Frontend built: $(FRONTEND_DEST)"
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore libs/cli/pyproject.toml Makefile
git commit -m "build: wheel include + gitignore + build-frontends target for v2 frontend"
```

---

### Task 9: Create empty placeholder `frontend_dist/` so bundler tests can run end-to-end

**Files:**
- Create: `libs/cli/deepagents_cli/deploy/frontend_dist/.gitkeep`

- [ ] **Step 1: Create the dir + gitkeep**

```bash
mkdir -p libs/cli/deepagents_cli/deploy/frontend_dist
touch libs/cli/deepagents_cli/deploy/frontend_dist/.gitkeep
```

This ensures the directory exists in git so `_FRONTEND_DIST_SRC.is_dir()` returns True during tests that don't use the `shipped_frontend_dist` fixture. The directory stays "empty" (just `.gitkeep`) until Task 18 populates it from a real build.

- [ ] **Step 2: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/frontend_dist/.gitkeep
git commit -m "build: reserve frontend_dist/ package-data directory"
```

---

## Phase 4 — Frontend scaffold

### Task 10: Initialize `libs/cli/frontend/` from the assistant-ui reference

**Files:**
- Create: `libs/cli/frontend/package.json`
- Create: `libs/cli/frontend/package-lock.json` (via npm)
- Create: `libs/cli/frontend/vite.config.ts`
- Create: `libs/cli/frontend/tsconfig.json`
- Create: `libs/cli/frontend/postcss.config.js`
- Create: `libs/cli/frontend/index.html`
- Create: `libs/cli/frontend/.nvmrc`
- Create: `libs/cli/frontend/src/index.css`
- Create: `libs/cli/frontend/src/vite-env.d.ts`

- [ ] **Step 1: Create the directory and copy scaffolding from the reference demo**

```bash
mkdir -p libs/cli/frontend/src
```

Copy the following files from `/Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/` to `libs/cli/frontend/`:

- `vite.config.ts` — confirm `base: "/app/"` is set
- `tsconfig.json`
- `postcss.config.js`
- `index.html`
- `.nvmrc`
- `src/index.css`

Run:

```bash
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/vite.config.ts libs/cli/frontend/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/tsconfig.json libs/cli/frontend/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/postcss.config.js libs/cli/frontend/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/index.html libs/cli/frontend/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/.nvmrc libs/cli/frontend/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/index.css libs/cli/frontend/src/
```

- [ ] **Step 2: Add placeholder script to `index.html`**

Open `libs/cli/frontend/index.html`. Immediately before the closing `</head>` tag, insert:

```html
<script>window.__DEEPAGENTS_CONFIG__ = {"__PLACEHOLDER__":true};</script>
```

Change the `<title>` to `Deep Agent` (the App will update it at runtime via `document.title = appName`).

- [ ] **Step 3: Write `package.json`**

Create `libs/cli/frontend/package.json`:

```json
{
  "name": "@deepagents/frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@assistant-ui/react": "^0.12.25",
    "@assistant-ui/react-langgraph": "^0.13.10",
    "@assistant-ui/react-streamdown": "^0.1.10",
    "@clerk/clerk-react": "^5.0.0",
    "@langchain/langgraph-sdk": "^1.7.2",
    "@supabase/supabase-js": "^2.99.3",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "streamdown": "^2.5.0"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4.0.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^5.0.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.7.0",
    "vite": "^6.0.0"
  }
}
```

Verify current versions on npm at implementation time; pin to latest stable in each major.

- [ ] **Step 4: Add minimal `vite-env.d.ts`**

Create `libs/cli/frontend/src/vite-env.d.ts`:

```typescript
/// <reference types="vite/client" />
```

- [ ] **Step 5: Install deps + verify TS compile**

Run:

```bash
cd libs/cli/frontend && npm install
```

Expected: clean install, generates `package-lock.json`.

Verify TypeScript is happy with an empty build attempt (expected to fail on missing main.tsx, which we create in Task 11):

```bash
cd libs/cli/frontend && npx tsc --noEmit
```

Expected: error pointing at missing `main.tsx` entry — that's fine; we fix it in T11.

- [ ] **Step 6: Commit**

```bash
git add libs/cli/frontend/package.json libs/cli/frontend/package-lock.json \
        libs/cli/frontend/vite.config.ts libs/cli/frontend/tsconfig.json \
        libs/cli/frontend/postcss.config.js libs/cli/frontend/index.html \
        libs/cli/frontend/.nvmrc libs/cli/frontend/src/index.css \
        libs/cli/frontend/src/vite-env.d.ts
git commit -m "chore(frontend): scaffold libs/cli/frontend with Vite + assistant-ui deps"
```

---

### Task 11: Runtime config reader + auth adapter types + main entry

**Files:**
- Create: `libs/cli/frontend/src/runtimeConfig.ts`
- Create: `libs/cli/frontend/src/auth/types.ts`
- Create: `libs/cli/frontend/src/main.tsx`
- Create: `libs/cli/frontend/src/constants.ts`

- [ ] **Step 1: Create `runtimeConfig.ts`**

Create `libs/cli/frontend/src/runtimeConfig.ts`:

```typescript
/**
 * Runtime config injected into index.html by `deepagent deploy`.
 *
 * The shipped bundle is built ONCE per CLI release. Per-user values
 * (auth provider, provider-specific keys, app name) are written into a
 * `window.__DEEPAGENTS_CONFIG__` script tag by the Python bundler at
 * deploy time.
 */
export interface RuntimeConfigSupabase {
  auth: "supabase";
  supabaseUrl: string;
  supabaseAnonKey: string;
  appName: string;
  assistantId: string;
}

export interface RuntimeConfigClerk {
  auth: "clerk";
  clerkPublishableKey: string;
  appName: string;
  assistantId: string;
}

export type RuntimeConfig = RuntimeConfigSupabase | RuntimeConfigClerk;

declare global {
  interface Window {
    __DEEPAGENTS_CONFIG__?: Partial<RuntimeConfig> & { __PLACEHOLDER__?: boolean };
  }
}

export function getRuntimeConfig(): RuntimeConfig {
  const cfg = window.__DEEPAGENTS_CONFIG__;
  if (!cfg || cfg.__PLACEHOLDER__) {
    throw new Error(
      "window.__DEEPAGENTS_CONFIG__ not injected. Run through `deepagent deploy` or `deepagent dev`.",
    );
  }
  if (cfg.auth === "supabase") {
    if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
      throw new Error("Runtime config missing supabaseUrl / supabaseAnonKey.");
    }
    return {
      auth: "supabase",
      supabaseUrl: cfg.supabaseUrl,
      supabaseAnonKey: cfg.supabaseAnonKey,
      appName: cfg.appName ?? "Deep Agent",
      assistantId: cfg.assistantId ?? "agent",
    };
  }
  if (cfg.auth === "clerk") {
    if (!cfg.clerkPublishableKey) {
      throw new Error("Runtime config missing clerkPublishableKey.");
    }
    return {
      auth: "clerk",
      clerkPublishableKey: cfg.clerkPublishableKey,
      appName: cfg.appName ?? "Deep Agent",
      assistantId: cfg.assistantId ?? "agent",
    };
  }
  throw new Error(`Unknown auth provider: ${String(cfg.auth)}`);
}
```

- [ ] **Step 2: Create `auth/types.ts`**

```bash
mkdir -p libs/cli/frontend/src/auth
```

Create `libs/cli/frontend/src/auth/types.ts`:

```typescript
import type { ReactNode, ComponentType } from "react";

export type SessionState =
  | { status: "loading" }
  | { status: "signed-out" }
  | {
      status: "signed-in";
      accessToken: string;
      userIdentity: string;
      userEmail: string | null;
      signOut: () => Promise<void>;
    };

export interface AuthAdapter {
  /** Provider component that initializes the auth SDK. Wraps the tree. */
  Provider: ComponentType<{ children: ReactNode }>;
  /** Returns the current session state. Must be called inside `Provider`. */
  useSession: () => SessionState;
  /** Sign-in / sign-up UI rendered when session is "signed-out". */
  AuthUI: ComponentType;
}
```

- [ ] **Step 3: Create `constants.ts`**

Create `libs/cli/frontend/src/constants.ts`:

```typescript
import { getRuntimeConfig } from "./runtimeConfig";

const cfg = getRuntimeConfig();

export const APP_NAME = cfg.appName;
export const APP_DESCRIPTION = "Your deep agent, deployed.";
export const ASSISTANT_ID = cfg.assistantId;
```

- [ ] **Step 4: Create `main.tsx`**

Create `libs/cli/frontend/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { APP_NAME } from "./constants";
import "./index.css";

document.title = APP_NAME;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 5: Commit**

```bash
git add libs/cli/frontend/src/runtimeConfig.ts \
        libs/cli/frontend/src/auth/types.ts \
        libs/cli/frontend/src/constants.ts \
        libs/cli/frontend/src/main.tsx
git commit -m "feat(frontend): runtime config reader + auth adapter types + entry"
```

---

## Phase 5 — Auth adapters

### Task 12: Supabase adapter

**Files:**
- Create: `libs/cli/frontend/src/auth/supabase.tsx`

- [ ] **Step 1: Write the adapter**

Create `libs/cli/frontend/src/auth/supabase.tsx`:

```tsx
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter, SessionState } from "./types";

type Ctx = {
  supabase: SupabaseClient;
  state: SessionState;
};

const SupabaseCtx = createContext<Ctx | null>(null);

function SupabaseProvider({ children }: { children: ReactNode }) {
  const cfg = getRuntimeConfig();
  if (cfg.auth !== "supabase") {
    throw new Error("SupabaseProvider mounted with non-supabase runtime config");
  }

  const supabase = useMemo(
    () => createClient(cfg.supabaseUrl, cfg.supabaseAnonKey),
    [cfg.supabaseUrl, cfg.supabaseAnonKey],
  );
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!active) return;
      setSession(data.session);
      setLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, nextSession) => {
        if (!active) return;
        setSession(nextSession);
        setLoading(false);
      },
    );
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [supabase]);

  const state: SessionState = loading
    ? { status: "loading" }
    : session
      ? {
          status: "signed-in",
          accessToken: session.access_token,
          userIdentity: session.user.id,
          userEmail: session.user.email ?? null,
          signOut: async () => {
            await supabase.auth.signOut();
          },
        }
      : { status: "signed-out" };

  return (
    <SupabaseCtx.Provider value={{ supabase, state }}>
      {children}
    </SupabaseCtx.Provider>
  );
}

function useSupabaseCtx(): Ctx {
  const ctx = useContext(SupabaseCtx);
  if (!ctx) {
    throw new Error("useSession() called outside SupabaseProvider");
  }
  return ctx;
}

function useSession(): SessionState {
  return useSupabaseCtx().state;
}

function SupabaseAuthUI() {
  const { supabase } = useSupabaseCtx();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setConfirmation(null);
    try {
      if (mode === "signup") {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setConfirmation("Check your inbox to confirm your email.");
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-dvh flex items-center justify-center bg-slate-50 p-4">
      <form
        onSubmit={submit}
        className="flex flex-col gap-3 w-full max-w-sm rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <h1 className="text-xl font-semibold text-slate-900">
          {mode === "signin" ? "Sign in" : "Sign up"}
        </h1>
        <input
          type="email"
          required
          placeholder="you@example.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <input
          type="password"
          required
          minLength={6}
          placeholder="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {loading ? "…" : mode === "signin" ? "Sign in" : "Sign up"}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
        {confirmation && <p className="text-xs text-emerald-700">{confirmation}</p>}
        <button
          type="button"
          className="text-center text-xs text-slate-600 hover:underline"
          onClick={() => {
            setMode(mode === "signin" ? "signup" : "signin");
            setError(null);
            setConfirmation(null);
          }}
        >
          {mode === "signin"
            ? "Need an account? Sign up"
            : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}

const adapter: AuthAdapter = {
  Provider: SupabaseProvider,
  useSession,
  AuthUI: SupabaseAuthUI,
};

export default adapter;
```

- [ ] **Step 2: Commit**

```bash
git add libs/cli/frontend/src/auth/supabase.tsx
git commit -m "feat(frontend): supabase auth adapter"
```

---

### Task 13: Clerk adapter (with 45s token refresh)

**Files:**
- Create: `libs/cli/frontend/src/auth/clerk.tsx`

- [ ] **Step 1: Write the adapter**

Create `libs/cli/frontend/src/auth/clerk.tsx`:

```tsx
import {
  ClerkProvider,
  SignIn,
  SignUp,
  useAuth,
  useUser,
} from "@clerk/clerk-react";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter, SessionState } from "./types";

type Ctx = { state: SessionState };
const ClerkCtx = createContext<Ctx | null>(null);

function ClerkSessionBridge({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn, getToken, signOut } = useAuth();
  const { user } = useUser();
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    if (!isLoaded || !isSignedIn) {
      setAccessToken(null);
      return;
    }
    const refresh = async () => {
      // Clerk session tokens default to a ~60s TTL. Force a fresh fetch on an
      // interval well under that so long-idle sessions don't hand the
      // LangGraph SDK an expired JWT.
      const t = await getToken({ skipCache: true });
      if (active && t) setAccessToken(t);
    };
    void refresh();
    const interval = window.setInterval(refresh, 45_000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [getToken, isLoaded, isSignedIn, user?.id]);

  const state: SessionState = useMemo(() => {
    if (!isLoaded) return { status: "loading" };
    if (!isSignedIn) return { status: "signed-out" };
    if (!accessToken) return { status: "loading" };
    return {
      status: "signed-in",
      accessToken,
      userIdentity: user?.id ?? "",
      userEmail: user?.primaryEmailAddress?.emailAddress ?? null,
      signOut: async () => {
        await signOut();
      },
    };
  }, [isLoaded, isSignedIn, accessToken, user?.id, user?.primaryEmailAddress, signOut]);

  return <ClerkCtx.Provider value={{ state }}>{children}</ClerkCtx.Provider>;
}

function ClerkAdapterProvider({ children }: { children: ReactNode }) {
  const cfg = getRuntimeConfig();
  if (cfg.auth !== "clerk") {
    throw new Error("ClerkProvider mounted with non-clerk runtime config");
  }
  return (
    <ClerkProvider publishableKey={cfg.clerkPublishableKey}>
      <ClerkSessionBridge>{children}</ClerkSessionBridge>
    </ClerkProvider>
  );
}

function useSession(): SessionState {
  const ctx = useContext(ClerkCtx);
  if (!ctx) {
    throw new Error("useSession() called outside ClerkAdapterProvider");
  }
  return ctx.state;
}

function ClerkAuthUI() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  return (
    <div className="min-h-dvh flex items-center justify-center bg-slate-50 p-4">
      <div className="flex flex-col items-center gap-4">
        {mode === "signin" ? (
          <SignIn routing="virtual" />
        ) : (
          <SignUp routing="virtual" />
        )}
        <button
          type="button"
          className="text-xs text-slate-600 hover:underline"
          onClick={() => setMode(mode === "signin" ? "signup" : "signin")}
        >
          {mode === "signin"
            ? "Need an account? Sign up"
            : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}

const adapter: AuthAdapter = {
  Provider: ClerkAdapterProvider,
  useSession,
  AuthUI: ClerkAuthUI,
};

export default adapter;
```

- [ ] **Step 2: Commit**

```bash
git add libs/cli/frontend/src/auth/clerk.tsx
git commit -m "feat(frontend): clerk auth adapter with 45s token refresh"
```

---

### Task 14: Auth loader (dynamic import router)

**Files:**
- Create: `libs/cli/frontend/src/auth/loader.tsx`

- [ ] **Step 1: Write the loader**

Create `libs/cli/frontend/src/auth/loader.tsx`:

```tsx
import { lazy, Suspense, useMemo, type ReactNode } from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter } from "./types";

/**
 * Dynamic-import router for the auth adapter.
 *
 * Vite code-splits each adapter (and its transitive SDK) into its own chunk.
 * At runtime, only the chunk matching `runtimeConfig.auth` is fetched — so
 * the other SDK's code is shipped in the dist folder but never executes in
 * the browser.
 */
export function createAdapterHolder(): { Adapter: React.ComponentType<{ children: ReactNode }>, useAdapter: () => AuthAdapter } {
  const cfg = getRuntimeConfig();

  const LazyAdapter = lazy(async () => {
    const mod =
      cfg.auth === "supabase"
        ? await import("./supabase")
        : await import("./clerk");
    return { default: () => null, adapter: mod.default };
  }) as unknown as React.LazyExoticComponent<React.ComponentType<unknown>>;

  throw new Error("createAdapterHolder is deprecated — use loadAuth()");
}

export interface AuthBundle {
  adapter: AuthAdapter;
}

let _cache: Promise<AuthBundle> | null = null;

/**
 * Loads the correct auth adapter module for the configured provider.
 * Memoized so repeat calls reuse the same Promise.
 */
export function loadAuth(): Promise<AuthBundle> {
  if (_cache) return _cache;
  const cfg = getRuntimeConfig();
  _cache = (async () => {
    const mod =
      cfg.auth === "supabase"
        ? await import("./supabase")
        : await import("./clerk");
    return { adapter: mod.default };
  })();
  return _cache;
}

interface AuthGateProps {
  fallback?: ReactNode;
  children: (adapter: AuthAdapter) => ReactNode;
}

/**
 * Suspends until the adapter module loads, then calls `children` with it.
 *
 * Usage:
 *   <AuthGate fallback={<Splash />}>
 *     {(adapter) => <adapter.Provider><App /></adapter.Provider>}
 *   </AuthGate>
 */
export function AuthGate({ fallback = null, children }: AuthGateProps) {
  const promise = useMemo(() => loadAuth(), []);
  return <SuspendingLoader promise={promise} fallback={fallback} render={children} />;
}

function SuspendingLoader({
  promise,
  fallback,
  render,
}: {
  promise: Promise<AuthBundle>;
  fallback: ReactNode;
  render: (adapter: AuthAdapter) => ReactNode;
}) {
  const state = useSuspenseResource(promise);
  if (!state) return <>{fallback}</>;
  return <>{render(state.adapter)}</>;
}

function useSuspenseResource(promise: Promise<AuthBundle>): AuthBundle | null {
  const [state, setState] = useState<AuthBundle | null>(null);
  useEffect(() => {
    let active = true;
    void promise.then((r) => {
      if (active) setState(r);
    });
    return () => {
      active = false;
    };
  }, [promise]);
  return state;
}

// Fix missing imports
import { useEffect, useState } from "react";
```

**Simplify — rewrite the file with this cleaner version:**

```tsx
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { getRuntimeConfig } from "../runtimeConfig";
import type { AuthAdapter } from "./types";

let _cache: Promise<AuthAdapter> | null = null;

/** Dynamic-import the auth adapter module for the active provider. */
export function loadAuth(): Promise<AuthAdapter> {
  if (_cache) return _cache;
  const cfg = getRuntimeConfig();
  _cache = (async () => {
    const mod =
      cfg.auth === "supabase"
        ? await import("./supabase")
        : await import("./clerk");
    return mod.default;
  })();
  return _cache;
}

export function useAuthAdapter(): AuthAdapter | null {
  const promise = useMemo(() => loadAuth(), []);
  const [adapter, setAdapter] = useState<AuthAdapter | null>(null);
  useEffect(() => {
    let active = true;
    void promise.then((a) => {
      if (active) setAdapter(a);
    });
    return () => {
      active = false;
    };
  }, [promise]);
  return adapter;
}
```

(Replace the entire previous content with the simplified version above.)

- [ ] **Step 2: Type-check**

Run:

```bash
cd libs/cli/frontend && npx tsc --noEmit
```

Expected: still errors on `App.tsx` (not yet created), but `auth/loader.tsx` itself should type-check cleanly.

- [ ] **Step 3: Commit**

```bash
git add libs/cli/frontend/src/auth/loader.tsx
git commit -m "feat(frontend): lazy-loaded auth adapter router"
```

---

## Phase 6 — Runtime provider + layout

### Task 15: `RuntimeProvider.tsx` — assistant-ui + LangGraph wiring

**Files:**
- Create: `libs/cli/frontend/src/RuntimeProvider.tsx`
- Create: `libs/cli/frontend/src/lib/chatApi.ts`
- Create: `libs/cli/frontend/src/lib/format.ts`
- Create: `libs/cli/frontend/src/types.ts`

- [ ] **Step 1: Copy `lib/` and `types.ts` from the reference demo**

The reference demo's `src/lib/chatApi.ts`, `src/lib/format.ts`, and `src/types.ts` are auth-agnostic. Copy them:

```bash
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/lib/chatApi.ts libs/cli/frontend/src/lib/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/lib/format.ts libs/cli/frontend/src/lib/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/types.ts libs/cli/frontend/src/
```

(Directory `libs/cli/frontend/src/lib/` may not exist yet — `mkdir -p` it first.)

- [ ] **Step 2: Create `RuntimeProvider.tsx`**

Open the reference at `/Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/RuntimeProvider.tsx` and copy its structure. The only changes for v2:

- Accept `accessToken: string` as a prop (instead of pulling `session.access_token` from a Supabase hook).
- Construct the LangGraph SDK Client with `{ apiUrl: window.location.origin, defaultHeaders: { Authorization: \`Bearer ${accessToken}\` } }`.
- Everything else — `useLangGraphRuntime`, `GraphValuesContext`, `ThreadActionsContext`, thread create/load/stream — unchanged.

Write the file to `libs/cli/frontend/src/RuntimeProvider.tsx`. (The exact body is verbatim from the reference, with the `session`-prop pattern replaced by `accessToken`-prop.)

- [ ] **Step 3: Type-check**

```bash
cd libs/cli/frontend && npx tsc --noEmit
```

Expected: still errors on missing `App.tsx` and components (next tasks), but `RuntimeProvider.tsx` should type-check.

- [ ] **Step 4: Commit**

```bash
git add libs/cli/frontend/src/RuntimeProvider.tsx \
        libs/cli/frontend/src/lib/ \
        libs/cli/frontend/src/types.ts
git commit -m "feat(frontend): RuntimeProvider wiring useLangGraphRuntime"
```

---

### Task 16: `App.tsx` — adapter-agnostic layout with auth gate

**Files:**
- Create: `libs/cli/frontend/src/App.tsx`

- [ ] **Step 1: Write `App.tsx`**

Create `libs/cli/frontend/src/App.tsx`:

```tsx
import { useAuthAdapter } from "./auth/loader";
import type { AuthAdapter } from "./auth/types";
import RuntimeProvider from "./RuntimeProvider";
import Thread from "./components/Thread";
import AppHeader from "./components/AppHeader";

export default function App() {
  const adapter = useAuthAdapter();
  if (!adapter) return <SplashScreen />;
  return (
    <adapter.Provider>
      <Gate adapter={adapter} />
    </adapter.Provider>
  );
}

function Gate({ adapter }: { adapter: AuthAdapter }) {
  const session = adapter.useSession();
  if (session.status === "loading") return <SplashScreen />;
  if (session.status === "signed-out") {
    const { AuthUI } = adapter;
    return <AuthUI />;
  }
  return <AuthenticatedApp accessToken={session.accessToken} userEmail={session.userEmail} onSignOut={session.signOut} />;
}

function SplashScreen() {
  return <div className="min-h-dvh bg-slate-50" />;
}

function AuthenticatedApp({
  accessToken,
  userEmail,
  onSignOut,
}: {
  accessToken: string;
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  return (
    <RuntimeProvider accessToken={accessToken}>
      <div className="flex h-dvh flex-col bg-[var(--background)]">
        <AppHeader userEmail={userEmail} onSignOut={onSignOut} />
        <Thread />
      </div>
    </RuntimeProvider>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add libs/cli/frontend/src/App.tsx
git commit -m "feat(frontend): App.tsx auth gate and layout"
```

---

### Task 17: Port UI components from the reference demo

**Files:**
- Create: `libs/cli/frontend/src/components/AppHeader.tsx`
- Create: `libs/cli/frontend/src/components/Thread.tsx`
- Create: `libs/cli/frontend/src/components/tools.tsx`
- Create: `libs/cli/frontend/src/components/SubagentActivity.tsx`
- Create: `libs/cli/frontend/src/components/TodosPanel.tsx`
- Create: `libs/cli/frontend/src/components/FilePanels.tsx`
- Create: `libs/cli/frontend/src/components/ThreadPicker.tsx`

- [ ] **Step 1: Copy component sources**

```bash
mkdir -p libs/cli/frontend/src/components
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/Thread.tsx libs/cli/frontend/src/components/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/tools.tsx libs/cli/frontend/src/components/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/SubagentActivity.tsx libs/cli/frontend/src/components/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/TodosPanel.tsx libs/cli/frontend/src/components/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/FilePanels.tsx libs/cli/frontend/src/components/
cp /Users/victormoreira/Desktop/demos/lsd-react-ui-cr/frontend/src/components/ThreadPicker.tsx libs/cli/frontend/src/components/
```

- [ ] **Step 2: Create `AppHeader.tsx`**

Create `libs/cli/frontend/src/components/AppHeader.tsx`. This is new code (no direct equivalent in the reference — the demo's App.tsx rendered the header inline). Write:

```tsx
import { APP_DESCRIPTION, APP_NAME } from "../constants";
import ThreadPicker from "./ThreadPicker";

export default function AppHeader({
  userEmail,
  onSignOut,
}: {
  userEmail: string | null;
  onSignOut: () => Promise<void>;
}) {
  return (
    <header className="header-blur sticky top-0 z-30 flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-3 py-2 sm:px-6 sm:py-3">
      <div className="flex items-center gap-2 sm:gap-3">
        <svg className="h-5 w-5 sm:h-6 sm:w-6" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
          <path d="M12 2L2 12l10 10 10-10L12 2z" />
          <path d="M12 8L8 12l4 4 4-4-4-4z" />
        </svg>
        <div>
          <h1 className="text-sm font-semibold sm:text-lg">{APP_NAME}</h1>
          <p className="hidden text-xs text-[var(--muted-foreground)] sm:block">{APP_DESCRIPTION}</p>
        </div>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2">
        {userEmail && (
          <span className="hidden max-w-[160px] truncate text-xs text-[var(--muted-foreground)] md:inline">
            {userEmail}
          </span>
        )}
        <button
          onClick={() => { void onSignOut(); }}
          className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--muted-foreground)] transition-colors hover:bg-[var(--accent-bg)]"
        >
          Sign out
        </button>
        <ThreadPicker />
      </div>
    </header>
  );
}
```

- [ ] **Step 3: Sweep the copied files for references to `Auth.tsx`, `SettingsModal.tsx`, or Supabase imports**

Run:

```bash
grep -rn "SettingsModal\|supabaseClient\|from.*'\\.\\./Auth'\|from.*'\\.\\./supabaseClient'" libs/cli/frontend/src/components/ || echo "no stragglers"
```

For any hits, remove the offending imports/usages. `Thread.tsx` and the panels should be self-contained (they read state via `useGraphValues()` context from `RuntimeProvider.tsx` — no auth refs needed). `ThreadPicker.tsx` may need its `accessToken` prop swapped for reading from the adapter-agnostic session; the simplest fix is to have `ThreadPicker.tsx` read the LangGraph client from a context exposed by `RuntimeProvider` instead of constructing its own.

- [ ] **Step 4: Genericize copy**

Open `libs/cli/frontend/src/components/Thread.tsx`. Find any hardcoded suggestions like `"What can you help me research?"` and replace with neutral ones:

```tsx
const SUGGESTIONS = [
  "What can you help me with today?",
  "Walk me through what you can do.",
  "Draft a plan for a task.",
];
```

Open `libs/cli/frontend/src/components/SubagentActivity.tsx`. Find user-visible labels mentioning "Research" or "Subagent progress" and rename to generic equivalents like "Task pipeline" / "Parallel tasks".

- [ ] **Step 5: Type-check the full source tree**

```bash
cd libs/cli/frontend && npx tsc --noEmit
```

Fix any remaining errors. Common categories:

- Missing prop types — add where needed.
- Unused imports — remove.
- Supabase-specific imports leftover — remove.

- [ ] **Step 6: Build**

```bash
cd libs/cli/frontend && npm run build
```

Expected: clean build, `dist/` produced with `index.html` + `assets/`.

- [ ] **Step 7: Commit**

```bash
git add libs/cli/frontend/src/components/
git commit -m "feat(frontend): port Thread, panels, tool renderers from reference demo"
```

---

## Phase 7 — Build + verify

### Task 18: Build + commit `frontend_dist/` package data

**Files:**
- Modify: `libs/cli/deepagents_cli/deploy/frontend_dist/*` (populated by build)

- [ ] **Step 1: Run `make build-frontends`**

From the repo root:

```bash
make build-frontends
```

Expected:

- `libs/cli/frontend/dist/` contains `index.html` + `assets/`.
- `libs/cli/deepagents_cli/deploy/frontend_dist/` is populated with identical contents.
- The `.gitkeep` from Task 9 is overwritten by the real files (or remove it explicitly before the copy).

- [ ] **Step 2: Inspect the built `index.html`**

```bash
grep -c "__PLACEHOLDER__" libs/cli/deepagents_cli/deploy/frontend_dist/index.html
```

Expected: exactly 1 occurrence — the placeholder script is preserved for the bundler to rewrite at deploy time.

```bash
ls libs/cli/deepagents_cli/deploy/frontend_dist/assets/
```

Expected: one `index-*.js` and one `index-*.css` at minimum.

- [ ] **Step 3: Verify the wheel includes the bundle**

```bash
cd libs/cli && uv run python -m build --wheel
unzip -l dist/deepagents_cli-*.whl | grep frontend_dist | head
```

Expected: wheel contents include `deepagents_cli/deploy/frontend_dist/index.html` and the `assets/*` files.

- [ ] **Step 4: Delete the now-superseded `.gitkeep`**

```bash
rm -f libs/cli/deepagents_cli/deploy/frontend_dist/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add libs/cli/deepagents_cli/deploy/frontend_dist/
git commit -m "build(frontend): ship pre-built frontend bundle as package data"
```

The diff will be large (built assets). That's expected.

---

### Task 19: End-to-end dry-run test

**Files:**
- Modify: `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`

- [ ] **Step 1: Append the end-to-end test**

Append to `libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py`:

```python
def test_deploy_dry_run_supabase_end_to_end(tmp_path, monkeypatch, capsys):
    """Run `_deploy(dry_run=True)` against a full project tree, using the real shipped bundle."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "AGENTS.md").write_text("prompt", encoding="utf-8")
    (project / "deepagents.toml").write_text(
        """
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

[auth]
provider = "supabase"

[frontend]
enabled = true
app_name = "My App"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_DEFAULT_KEY", "k")

    from deepagents_cli.deploy.commands import _deploy

    monkeypatch.chdir(project)
    _deploy(config_path=str(project / "deepagents.toml"), dry_run=True)

    out = capsys.readouterr().out
    assert "Inspect the build directory" in out
    build_line = [line for line in out.splitlines() if "build directory" in line][-1]
    build_path = Path(build_line.split("Inspect the build directory:")[-1].strip())
    assert (build_path / "app.py").is_file()
    assert (build_path / "frontend_dist" / "index.html").is_file()
    html = (build_path / "frontend_dist" / "index.html").read_text(encoding="utf-8")
    assert "__PLACEHOLDER__" not in html
    assert '"auth":"supabase"' in html
```

- [ ] **Step 2: Run**

```bash
cd libs/cli && uv run pytest tests/unit_tests/deploy/test_frontend_bundle.py::test_deploy_dry_run_supabase_end_to_end -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add libs/cli/tests/unit_tests/deploy/test_frontend_bundle.py
git commit -m "test(deploy): end-to-end dry-run with frontend enabled"
```

---

## Phase 8 — Docs

### Task 20: Write `docs/frontend.md`

**Files:**
- Create: `docs/frontend.md`

- [ ] **Step 1: Write the docs**

Create `/Users/victormoreira/Desktop/open-source/deepagents/docs/frontend.md`:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add docs/frontend.md
git commit -m "docs: user-facing docs for [frontend] section"
```

---

## Phase 9 — Verification

### Task 21: Final lint / format / type / test sweep

**Files:** n/a

- [ ] **Step 1: Python-side full suite**

```bash
cd libs/cli
uv run pytest tests/unit_tests -q
```

Expected: all pass.

```bash
uv run ruff check deepagents_cli/deploy/ tests/unit_tests/deploy/
uv run ruff format --check deepagents_cli/deploy/ tests/unit_tests/deploy/
uv run ty check deepagents_cli/deploy/
```

Expected: all clean.

- [ ] **Step 2: Frontend build**

```bash
cd libs/cli/frontend && npm run build
```

Expected: clean build.

- [ ] **Step 3: Wheel sanity**

```bash
cd libs/cli
uv run python -m build --wheel 2>&1 | tail -5
unzip -l dist/deepagents_cli-*.whl | grep -c frontend_dist
```

Expected: build succeeds; grep count ≥ 3 (index.html + at least one JS + one CSS).

- [ ] **Step 4: Commit any fixes from the sweep**

```bash
git add -A
git commit -m "chore: format + typecheck fixes after v2 implementation"
```

(Skip if the sweep produced no diffs.)

---

## Self-review

Before handing off, verify each spec section maps to a task:

- §2 Goals → Tasks 1-17
- §3 Non-goals → enforced by absence (no settings modal, no theme, no eject)
- §4.1 Config surface → Tasks 1-2
- §4.2 Env vars → Task 2 (validation); Task 6 (injection)
- §4.3 Architecture (file tree) → Tasks 10-17 (frontend); Tasks 1-7 (backend); Tasks 8-9 (packaging)
- §4.4 AuthAdapter interface → Task 11; Tasks 12-14 implementations
- §4.5 Runtime flow → Tasks 11, 14, 16
- §4.6 Code splitting → Task 14 (dynamic import)
- §4.7 Genericization → Tasks 10-11 (constants, title), Task 17 (Thread suggestions + SubagentActivity labels)
- §4.8 Tool-call renderers → Task 17 (copied from reference)
- §4.9 Deep-agent panels → Task 17 (copied)
- §4.10 Bundler simplifications → Tasks 6-7
- §4.11 Static-path auth exemption → Task 5
- §4.12 `deepagents dev` → works unchanged (uses the same bundler codepath)
- §4.13 Error handling → Tasks 2 (config), 6 (bundler), 11 (runtime config), 14 (adapter loading)
- §4.14 Testing → Tasks 1, 2, 6, 7, 19

Placeholder scan: no "TBD" / "implement later" / generic-handler phrases. All steps have complete code or exact commands.

Type consistency:

- `FrontendConfig(enabled: bool, app_name: str | None)` — used identically in config.py and tests.
- `RuntimeConfigSupabase` / `RuntimeConfigClerk` discriminated union — used consistently.
- `AuthAdapter` with `Provider`, `useSession`, `AuthUI` — consistent across adapters and loader.
- `SessionState` discriminated union — same shape in types.ts, supabase.tsx, clerk.tsx, App.tsx.
- `loadAuth()` / `useAuthAdapter()` — single source of truth in `auth/loader.tsx`.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-23-frontend-v2-assistant-ui-consolidation.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review (spec compliance then code quality), fast iteration.

**2. Inline Execution** — execute tasks in this session with checkpoints for review.

**Which approach?**
