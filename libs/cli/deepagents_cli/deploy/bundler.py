"""Bundle a deepagents project for deployment.

Reads the canonical project layout:

```txt
<project>/
    deepagents.toml  # required — agent + sandbox config
    AGENTS.md        # required — system prompt + seeded memory
    .env             # optional — environment variables
    mcp.json         # optional — HTTP/SSE MCP servers
    skills/          # optional — auto-seeded into skills namespace
    user/            # optional — per-user writable memory
        AGENTS.md    # optional — seeded as empty if not provided
```

...and writes everything `langgraph deploy` needs to a build directory.

AGENTS.md and skills are read-only at runtime.  When a `user/`
directory is present, a per-user `AGENTS.md` is seeded (from
`user/AGENTS.md` if provided, otherwise empty) and is writable
at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    MCP_FILENAME,
    SKILLS_DIRNAME,
    USER_DIRNAME,
    DeployConfig,
    SubAgentProject,
    load_subagents,
)
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

logger = logging.getLogger(__name__)

_MODEL_PROVIDER_DEPS: dict[str, str] = {
    "anthropic": "langchain-anthropic",
    "azure_openai": "langchain-openai",
    "baseten": "langchain-baseten",
    "cohere": "langchain-cohere",
    "deepseek": "langchain-deepseek",
    "fireworks": "langchain-fireworks",
    "google_genai": "langchain-google-genai",
    "google_vertexai": "langchain-google-vertexai",
    "groq": "langchain-groq",
    "mistralai": "langchain-mistralai",
    "nvidia": "langchain-nvidia-ai-endpoints",
    "openai": "langchain-openai",
    "openrouter": "langchain-openrouter",
    "perplexity": "langchain-perplexity",
    "xai": "langchain-xai",
}
"""Dependencies inferred from a provider: prefix on the model string."""

_FRONTEND_DIST_SRC = Path(__file__).parent / "frontend_dist"
"""Location of the shipped pre-built frontend, inside this Python package."""

_FRONTEND_PLACEHOLDER_RE = re.compile(
    r"window\.__DEEPAGENTS_CONFIG__\s*=\s*\{[^<]*?\};",
    re.DOTALL,
)
"""Matches the placeholder script we injected into index.html at build time."""


def _build_runtime_config_json(config: DeployConfig) -> str:
    """Build the JSON value injected into `window.__DEEPAGENTS_CONFIG__`.

    Only reached when `[frontend].enabled` and `[auth]` is set —
    validation guarantees both. The `is None` guards below exist so
    the optional fields narrow for type-checkers.
    """
    if config.frontend is None:
        msg = "runtime config requires [frontend] to be configured"
        raise ValueError(msg)
    if config.auth is None:
        msg = "runtime config requires [auth] to be configured"
        raise ValueError(msg)

    app_name = config.frontend.app_name or config.agent.name
    payload: dict[str, Any] = {
        "appName": app_name,
        "assistantId": "agent",
    }
    # Optional UI-customization fields — only injected when the user
    # set them, so the default-bundle case stays small.
    if config.frontend.subtitle is not None:
        payload["subtitle"] = config.frontend.subtitle
    if config.frontend.prompts is not None:
        payload["prompts"] = list(config.frontend.prompts)

    provider = config.auth.provider
    payload["auth"] = provider
    if provider == "supabase":
        payload["supabaseUrl"] = os.environ["SUPABASE_URL"]
        payload["supabaseAnonKey"] = os.environ["SUPABASE_PUBLISHABLE_DEFAULT_KEY"]
    elif provider == "clerk":
        payload["clerkPublishableKey"] = os.environ["CLERK_PUBLISHABLE_KEY"]
    elif provider == "anonymous":
        # No env vars; payload["auth"] = "anonymous" is enough.
        pass
    else:
        msg = f"Unknown auth provider for frontend: {provider}"
        raise ValueError(msg)

    # Escape `<` so a hostile or accidental `</script>` inside a string value
    # can't break out of the inline <script> tag.
    return json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")


def _copy_frontend_dist(config: DeployConfig, build_dir: Path) -> None:
    """Copy the pre-built bundle into build_dir and rewrite the config placeholder."""
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
        lambda _m: replacement,
        html,
        count=1,
    )
    if count == 0:
        msg = (
            "Could not find window.__DEEPAGENTS_CONFIG__ placeholder in the "
            "shipped index.html. The frontend bundle is out of sync with the "
            "bundler — rebuild with `make build-frontends`."
        )
        raise RuntimeError(msg)
    index_html.write_text(new_html, encoding="utf-8")


def bundle(
    config: DeployConfig,
    project_root: Path,
    build_dir: Path,
) -> Path:
    """Create the full deployment bundle in *build_dir*."""
    build_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read AGENTS.md — the system prompt AND (optionally) seeded memory.
    agents_md_path = project_root / AGENTS_MD_FILENAME
    system_prompt = agents_md_path.read_text(encoding="utf-8")

    # 2. Build and write the seed payload: memory (AGENTS.md) + skills/.
    seed = _build_seed(project_root, system_prompt)
    (build_dir / "_seed.json").write_text(
        json.dumps(seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote _seed.json (memories: %d, skills: %d, user_memories: %d)",
        len(seed["memories"]),
        len(seed["skills"]),
        len(seed.get("user_memories", {})),
    )

    # 3. Copy mcp.json if present.
    mcp_present = (project_root / MCP_FILENAME).is_file()
    if mcp_present:
        shutil.copy2(project_root / MCP_FILENAME, build_dir / "_mcp.json")
        logger.info("Copied %s → _mcp.json", MCP_FILENAME)

    # 3b. Copy .env from the project root if present (alongside
    # deepagents.toml). The bundler skips .env when
    # building the seed payload so secrets never land in _seed.json.
    env_src = project_root / ".env"
    env_present = env_src.is_file()
    if env_present:
        shutil.copy2(env_src, build_dir / ".env")
        logger.info("Copied %s → .env", env_src)

    # 4. Load subagents (needed for both deploy_graph.py and pyproject.toml).
    sync_subagents = load_subagents(project_root)

    # 5. Render deploy_graph.py.
    has_user_memories = (project_root / USER_DIRNAME).is_dir()
    has_sync_subagents = bool(sync_subagents)
    (build_dir / "deploy_graph.py").write_text(
        _render_deploy_graph(
            config,
            mcp_present=mcp_present,
            has_user_memories=has_user_memories,
            has_sync_subagents=has_sync_subagents,
        ),
        encoding="utf-8",
    )
    logger.info("Generated deploy_graph.py")

    # 6. Generate auth.py from the [auth] provider, if any. Skipped
    # entirely when [auth] is omitted — in that case LangSmith Cloud's
    # default x-api-key auth applies. Validation guarantees [auth] is
    # set whenever [frontend].enabled, so auth.py is always present
    # for frontend deploys (including the "anonymous" provider, whose
    # permissive handler lets the bundled UI reach /threads).
    frontend_enabled = config.frontend is not None and config.frontend.enabled
    auth_provider: str | None = (
        config.auth.provider if config.auth is not None else None
    )

    auth_present = auth_provider is not None
    if auth_provider is not None:
        (build_dir / "auth.py").write_text(
            _render_auth_py(auth_provider),
            encoding="utf-8",
        )
        logger.info("Generated auth.py (%s)", auth_provider)

    # 6b. Copy frontend bundle when enabled.
    if frontend_enabled:
        _copy_frontend_dist(config, build_dir)
        (build_dir / "app.py").write_text(APP_PY_TEMPLATE, encoding="utf-8")
        logger.info("Copied frontend bundle and wrote app.py (%s)", auth_provider)

    # 7. Render langgraph.json.
    (build_dir / "langgraph.json").write_text(
        _render_langgraph_json(
            env_present=env_present,
            auth_present=auth_present,
            frontend_present=frontend_enabled,
        ),
        encoding="utf-8",
    )

    # 7. Render pyproject.toml.
    subagent_model_providers: list[str] = []
    has_subagent_mcp = False
    for sa in sync_subagents.values():
        model = sa.config.agent.model
        if ":" in model:
            subagent_model_providers.append(model.split(":", 1)[0])
        if (sa.root / MCP_FILENAME).is_file():
            has_subagent_mcp = True

    (build_dir / "pyproject.toml").write_text(
        _render_pyproject(
            config,
            mcp_present=mcp_present,
            subagent_model_providers=subagent_model_providers,
            has_subagent_mcp=has_subagent_mcp,
        ),
        encoding="utf-8",
    )

    return build_dir


def _build_subagent_seed(subagent: SubAgentProject) -> dict:
    """Build the seed entry for a single sync subagent."""
    sa_root = subagent.root
    agent = subagent.config.agent

    memories: dict[str, str] = {
        f"/{AGENTS_MD_FILENAME}": (sa_root / AGENTS_MD_FILENAME).read_text(
            encoding="utf-8"
        ),
    }

    skills: dict[str, str] = {}
    skills_dir = sa_root / SKILLS_DIRNAME
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skills_dir).as_posix()
                skills[f"/{rel}"] = f.read_text(encoding="utf-8")

    mcp_path = sa_root / MCP_FILENAME
    mcp = None
    if mcp_path.is_file():
        mcp = json.loads(mcp_path.read_text(encoding="utf-8"))

    return {
        "config": {
            "name": agent.name,
            "description": agent.description,
            "model": agent.model,
        },
        "memories": memories,
        "skills": skills,
        "mcp": mcp,
    }


def _build_seed(
    project_root: Path,
    system_prompt: str,
) -> dict:
    """Build the `_seed.json` payload.

    Layout::

        {
            "memories":       { "/AGENTS.md": "..." },
            "skills":         { "/<skill>/SKILL.md": "...", ... },
            "user_memories":  { "/AGENTS.md": "..." }
        }

    `memories` and `skills` are read-only at runtime.
    `user_memories` contains a single writable `AGENTS.md` mounted at
    `/memories/user/`, namespaced per user_id.  If the project has a
    `user/` directory (even if empty), an `AGENTS.md` is always seeded.
    """
    memories: dict[str, str] = {f"/{AGENTS_MD_FILENAME}": system_prompt}
    skills: dict[str, str] = {}
    user_memories: dict[str, str] = {}

    skills_dir = project_root / SKILLS_DIRNAME
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skills_dir).as_posix()
                skills[f"/{rel}"] = f.read_text(encoding="utf-8")

    user_dir = project_root / USER_DIRNAME
    if user_dir.is_dir():
        user_agents_md = user_dir / AGENTS_MD_FILENAME
        content = (
            user_agents_md.read_text(encoding="utf-8")
            if user_agents_md.is_file()
            else ""
        )
        user_memories[f"/{AGENTS_MD_FILENAME}"] = content

    seed: dict = {
        "memories": memories,
        "skills": skills,
        "user_memories": user_memories,
    }

    # Sync subagents.
    sync_subagents = load_subagents(project_root)
    if sync_subagents:
        seed["subagents"] = {
            name: _build_subagent_seed(sa) for name, sa in sync_subagents.items()
        }

    return seed


def _render_deploy_graph(
    config: DeployConfig,
    *,
    mcp_present: bool,
    has_user_memories: bool = False,
    has_sync_subagents: bool = False,
) -> str:
    """Render the generated `deploy_graph.py`."""
    provider = config.sandbox.provider
    if provider not in SANDBOX_BLOCKS:
        msg = f"Unknown sandbox provider {provider!r}. Valid: {sorted(SANDBOX_BLOCKS)}"
        raise ValueError(msg)
    sandbox_block, _ = SANDBOX_BLOCKS[provider]

    if mcp_present:
        mcp_tools_block = MCP_TOOLS_TEMPLATE
        mcp_tools_load_call = "tools.extend(await _load_mcp_tools())"
    else:
        mcp_tools_block = ""
        mcp_tools_load_call = "pass  # no MCP servers configured"

    if has_sync_subagents:
        sync_subagents_block = SYNC_SUBAGENTS_TEMPLATE
        sync_subagents_load_call = (
            "all_subagents.extend("
            "await _build_sync_subagents(seed, store, assistant_id))"
        )
    else:
        sync_subagents_block = ""
        sync_subagents_load_call = "pass  # no sync subagents"

    return DEPLOY_GRAPH_TEMPLATE.format(
        model=config.agent.model,
        sandbox_snapshot=config.sandbox.template,
        sandbox_image=config.sandbox.image,
        sandbox_scope=config.sandbox.scope,
        sandbox_block=sandbox_block,
        mcp_tools_block=mcp_tools_block,
        mcp_tools_load_call=mcp_tools_load_call,
        default_assistant_id=config.agent.name,
        has_user_memories=has_user_memories,
        sync_subagents_block=sync_subagents_block,
        sync_subagents_load_call=sync_subagents_load_call,
    )


def _render_auth_py(provider: str) -> str:
    """Render the generated `auth.py` for the given auth provider."""
    if provider not in AUTH_BLOCKS:
        msg = f"Unknown auth provider {provider!r}. Valid: {sorted(AUTH_BLOCKS)}"
        raise ValueError(msg)
    auth_block, _ = AUTH_BLOCKS[provider]
    return auth_block + AUTH_ON_HANDLER


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


def _render_pyproject(
    config: DeployConfig,
    *,
    mcp_present: bool,
    subagent_model_providers: list[str] | None = None,
    has_subagent_mcp: bool = False,
) -> str:
    """Render the deployment package's `pyproject.toml`.

    Deps are inferred — the user never writes them. We add:

    - the LangChain partner package matching the model provider prefix
    - `langchain-mcp-adapters` if `mcp.json` is present
    - the sandbox partner package (daytona/modal/runloop)
    """
    deps: list[str] = []

    provider_prefix = (
        config.agent.model.split(":", 1)[0] if ":" in config.agent.model else ""
    )
    if provider_prefix and provider_prefix in _MODEL_PROVIDER_DEPS:
        deps.append(_MODEL_PROVIDER_DEPS[provider_prefix])

    # Add deps for subagent model providers.
    for sp in subagent_model_providers or []:
        dep = _MODEL_PROVIDER_DEPS.get(sp)
        if dep and dep not in deps:
            deps.append(dep)

    if mcp_present or has_subagent_mcp:
        deps.append("langchain-mcp-adapters")

    _, partner_pkg = SANDBOX_BLOCKS.get(config.sandbox.provider, (None, None))
    if partner_pkg:
        deps.append(partner_pkg)

    if config.auth is not None:
        _, auth_pkg = AUTH_BLOCKS.get(config.auth.provider, (None, None))
        if auth_pkg:
            deps.append(auth_pkg)

    extra_deps_lines = "".join(f'    "{dep}",\n' for dep in deps)

    return PYPROJECT_TEMPLATE.format(
        agent_name=config.agent.name,
        extra_deps=extra_deps_lines,
    )


def print_bundle_summary(config: DeployConfig, build_dir: Path) -> None:
    """Print a human-readable summary of what was bundled."""
    seed_path = build_dir / "_seed.json"
    seed: dict[str, Any] = {"memories": {}, "skills": {}}
    if seed_path.exists():
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to parse %s; summary may be incomplete: %s",
                seed_path,
                exc,
            )

    print(f"\n  Agent: {config.agent.name}")
    print(f"  Model: {config.agent.model}")
    if config.auth is not None:
        if config.auth.provider == "anonymous":
            print("  Auth: anonymous (API open to anyone)")
        else:
            print(f"  Auth: {config.auth.provider}")
    else:
        print("  Auth: none (LangSmith API key required to call the API)")

    memory_files = sorted(seed.get("memories", {}).keys())
    if memory_files:
        print(f"\n  Memory seed ({len(memory_files)} file(s)):")
        for f in memory_files:
            print(f"    {f}")

    user_memory_files = sorted(seed.get("user_memories", {}).keys())
    if user_memory_files:
        print(f"\n  User memory seed ({len(user_memory_files)} file(s)):")
        for f in user_memory_files:
            print(f"    {f}")

    skills_files = sorted(seed.get("skills", {}).keys())
    if skills_files:
        print(f"\n  Skills seed ({len(skills_files)} file(s)):")
        for f in skills_files:
            print(f"    {f}")

    if (build_dir / "_mcp.json").exists():
        print("\n  MCP config: _mcp.json")

    # Subagent summary.
    sync_subagents = seed.get("subagents", {})
    if sync_subagents:
        print(f"\n  Subagents ({len(sync_subagents)}):")
        for name, sa_data in sync_subagents.items():
            desc = sa_data.get("config", {}).get("description", "")
            print(f"    {name} \u2014 {desc}")

    print(f"\n  Sandbox: {config.sandbox.provider}")
    print(f"\n  Build directory: {build_dir}")
    generated = sorted(f.name for f in build_dir.iterdir() if f.is_file())
    print(f"  Generated files: {', '.join(generated)}")
    print()
