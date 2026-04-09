"""Bundle a deepagents project for deployment.

Reads the canonical project layout::

    src/
      agents.md       # required — becomes the system prompt and (optionally) seeded memory
      skills/         # optional — auto-seeded into the skills namespace
      mcp.json        # optional — HTTP/SSE MCP servers
      deepagents.toml

…and writes everything ``langgraph deploy`` needs to a build directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    MCP_FILENAME,
    SKILLS_DIRNAME,
    DeployConfig,
)
import json as _json

from deepagents_cli.deploy.templates import (
    DEPLOY_GRAPH_TEMPLATE,
    MCP_TOOLS_TEMPLATE,
    PYPROJECT_TEMPLATE,
    SANDBOX_BLOCKS,
)

logger = logging.getLogger(__name__)

# Dependencies inferred from a ``provider:`` prefix on the model string.
_MODEL_PROVIDER_DEPS = {
    "anthropic": "langchain-anthropic",
    "openai": "langchain-openai",
    "google_genai": "langchain-google-genai",
    "google_vertexai": "langchain-google-vertexai",
    "groq": "langchain-groq",
    "mistralai": "langchain-mistralai",
}


def bundle(
    config: DeployConfig,
    project_root: Path,
    build_dir: Path,
) -> Path:
    """Create the full deployment bundle in *build_dir*."""
    build_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read agents.md — the system prompt AND (optionally) seeded memory.
    agents_md_path = project_root / AGENTS_MD_FILENAME
    system_prompt = agents_md_path.read_text(encoding="utf-8")

    # 2. Build and write the seed payload: memory (agents.md) + skills/.
    seed = _build_seed(config, project_root, system_prompt)
    (build_dir / "_seed.json").write_text(
        json.dumps(seed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote _seed.json (memories: %d, skills: %d)",
        len(seed["memories"]),
        len(seed["skills"]),
    )

    # 3. Copy mcp.json if present.
    mcp_present = (project_root / MCP_FILENAME).is_file()
    if mcp_present:
        shutil.copy2(project_root / MCP_FILENAME, build_dir / "_mcp.json")
        logger.info("Copied %s → _mcp.json", MCP_FILENAME)

    # 3b. Copy .env from the project root's *parent* if present. This is
    # where users keep their deployment secrets (e.g.
    # ``examples/deploy-coding-agent/.env``) — outside ``src/`` so they
    # never land in a seed file.
    env_src = project_root.parent / ".env"
    env_present = env_src.is_file()
    if env_present:
        shutil.copy2(env_src, build_dir / ".env")
        logger.info("Copied %s → .env", env_src)

    # 4. Render deploy_graph.py.
    (build_dir / "deploy_graph.py").write_text(
        _render_deploy_graph(config, system_prompt, mcp_present=mcp_present),
        encoding="utf-8",
    )
    logger.info("Generated deploy_graph.py")

    # 5. Render langgraph.json.
    (build_dir / "langgraph.json").write_text(
        _render_langgraph_json(env_present=env_present),
        encoding="utf-8",
    )

    # 6. Render pyproject.toml.
    (build_dir / "pyproject.toml").write_text(
        _render_pyproject(config, mcp_present=mcp_present),
        encoding="utf-8",
    )

    return build_dir


def _build_seed(
    config: DeployConfig,  # noqa: ARG001
    project_root: Path,
    system_prompt: str,
) -> dict[str, dict[str, str]]:
    """Build the ``_seed.json`` payload.

    Layout::

        {
            "memories": { "AGENTS.md": "..." },
            "skills":   { "<skill>/SKILL.md": "...", ... }
        }

    ``memories`` always contains ``AGENTS.md`` — the agent reads it at
    runtime via ``/memories/AGENTS.md``. Writes and edits to that path
    are blocked by ``ReadOnlyStoreBackend`` in the generated graph.

    ``skills`` walks ``src/skills/`` if present. Keys are paths relative
    to the skills dir; the runtime namespace handles the scoping.
    """
    # Keys must match what CompositeBackend passes to the mounted
    # StoreBackend after stripping the route prefix: for a read of
    # ``/memories/AGENTS.md`` it calls ``store.read("/AGENTS.md")``.
    # Seed with the same leading-slash convention.
    memories: dict[str, str] = {f"/{AGENTS_MD_FILENAME}": system_prompt}
    skills: dict[str, str] = {}

    skills_dir = project_root / SKILLS_DIRNAME
    if skills_dir.is_dir():
        for f in sorted(skills_dir.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(skills_dir).as_posix()
                skills[f"/{rel}"] = f.read_text(encoding="utf-8")

    return {"memories": memories, "skills": skills}


def _render_deploy_graph(
    config: DeployConfig,
    system_prompt: str,
    *,
    mcp_present: bool,
) -> str:
    """Render the generated ``deploy_graph.py``."""
    provider = config.sandbox.provider
    if provider not in SANDBOX_BLOCKS:
        msg = (
            f"Unknown sandbox provider {provider!r}. "
            f"Valid: {sorted(SANDBOX_BLOCKS)}"
        )
        raise ValueError(msg)
    sandbox_block, _ = SANDBOX_BLOCKS[provider]

    if mcp_present:
        mcp_tools_block = MCP_TOOLS_TEMPLATE
        mcp_tools_load_call = "tools.extend(await _load_mcp_tools())"
    else:
        mcp_tools_block = ""
        mcp_tools_load_call = "pass  # no MCP servers configured"

    return DEPLOY_GRAPH_TEMPLATE.format(
        model=config.agent.model,
        system_prompt=system_prompt,
        sandbox_template=config.sandbox.template,
        sandbox_image=config.sandbox.image,
        sandbox_scope=config.sandbox.scope,
        sandbox_block=sandbox_block,
        mcp_tools_block=mcp_tools_block,
        mcp_tools_load_call=mcp_tools_load_call,
        default_assistant_id=config.agent.name,
    )


def _render_langgraph_json(*, env_present: bool) -> str:
    """Render ``langgraph.json`` — adds ``"env": ".env"`` when a .env was copied."""
    data: dict = {
        "dependencies": ["."],
        "graphs": {"agent": "./deploy_graph.py:make_graph"},
        "python_version": "3.12",
    }
    if env_present:
        data["env"] = ".env"
    return _json.dumps(data, indent=2) + "\n"


def _render_pyproject(config: DeployConfig, *, mcp_present: bool) -> str:
    """Render the deployment package's ``pyproject.toml``.

    Deps are inferred — the user never writes them. We add:

    - the LangChain partner package matching the model provider prefix
    - ``langchain-mcp-adapters`` if ``mcp.json`` is present
    - the sandbox partner package (daytona/modal/runloop)
    """
    deps: list[str] = []

    provider_prefix = config.agent.model.split(":", 1)[0] if ":" in config.agent.model else ""
    if provider_prefix and provider_prefix in _MODEL_PROVIDER_DEPS:
        deps.append(_MODEL_PROVIDER_DEPS[provider_prefix])

    if mcp_present:
        deps.append("langchain-mcp-adapters")

    _, partner_pkg = SANDBOX_BLOCKS.get(config.sandbox.provider, (None, None))
    if partner_pkg:
        deps.append(partner_pkg)

    extra_deps_lines = "".join(f'    "{dep}",\n' for dep in deps)

    return PYPROJECT_TEMPLATE.format(
        agent_name=config.agent.name,
        extra_deps=extra_deps_lines,
    )


def print_bundle_summary(config: DeployConfig, build_dir: Path) -> None:
    """Print a human-readable summary of what was bundled."""
    seed_path = build_dir / "_seed.json"
    seed: dict[str, dict[str, str]] = {"memories": {}, "skills": {}}
    if seed_path.exists():
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    print(f"\n  Agent: {config.agent.name}")
    print(f"  Model: {config.agent.model}")

    memory_files = sorted(seed.get("memories", {}).keys())
    if memory_files:
        print(f"\n  Memory seed ({len(memory_files)} file(s)):")
        for f in memory_files:
            print(f"    {f}")

    skills_files = sorted(seed.get("skills", {}).keys())
    if skills_files:
        print(f"\n  Skills seed ({len(skills_files)} file(s)):")
        for f in skills_files:
            print(f"    {f}")

    if (build_dir / "_mcp.json").exists():
        print("\n  MCP config: _mcp.json")

    print(f"\n  Sandbox: {config.sandbox.provider}")
    print(f"\n  Build directory: {build_dir}")
    generated = sorted(f.name for f in build_dir.iterdir() if f.is_file())
    print(f"  Generated files: {', '.join(generated)}")
    print()
