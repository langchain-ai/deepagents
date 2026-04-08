"""Bundle a deepagents project for deployment.

Reads local AGENTS.md, skills, tools, and MCP config, then generates
all artifacts needed for ``langgraph deploy`` in a build directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from deepagents_cli.deploy.config import DeployConfig
from deepagents_cli.deploy.templates import (
    BEFORE_AGENT_SANDBOX_TEMPLATE,
    DEPLOY_GRAPH_HUB_TEMPLATE,
    DEPLOY_GRAPH_TEMPLATE,
    LANGGRAPH_JSON_TEMPLATE,
    MCP_TOOLS_TEMPLATE,
    PYPROJECT_TEMPLATE,
    SANDBOX_BLOCKS,
    TOOLS_IMPORT_AUTODISCOVER_TEMPLATE,
    TOOLS_IMPORT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _is_hub_backed(config: DeployConfig) -> bool:
    """Hub-backed deploys are the new path; the legacy bundle path is kept
    for content-writer-style examples that haven't migrated yet.

    For now we treat any project as hub-backed regardless of sandbox
    provider — the sandbox dispatch in ``SANDBOX_BLOCKS`` handles
    ``provider = "none"`` by falling back to ``StateBackend``.
    """
    return True


def bundle(
    config: DeployConfig,
    project_root: Path,
    build_dir: Path,
) -> Path:
    """Create the full deployment bundle in *build_dir*.

    Args:
        config: Parsed deploy configuration.
        project_root: Directory containing ``deepagents.toml``.
        build_dir: Target directory for generated artifacts.

    Returns:
        Path to the build directory (same as *build_dir*).
    """
    build_dir.mkdir(parents=True, exist_ok=True)

    # Always copy ``hub.py`` into the build dir as a standalone module —
    # the generated ``deploy_graph.py`` imports ``HubBackend`` from it
    # regardless of whether this deploy uses ``backend = "hub"``, because
    # ``deepagents.backends.hub`` is not yet published to PyPI.
    _bundle_hub_module(build_dir)

    # Bundle the agent_memories sources + skills as a single
    # ``_agent_memories_seed.json`` artifact. The generated graph reads
    # this file on first invocation and uploads each entry to the
    # configured backend via ``BackendProtocol.upload_files`` — same code
    # path for hub-backed and store-backed agent memories.
    seed = _build_agent_memories_seed(config, project_root)
    seed_path = build_dir / "_agent_memories_seed.json"
    seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False))
    logger.info("Created _agent_memories_seed.json with %d entries", len(seed))

    # 2. Copy tools.py if configured
    tools_module_name: str | None = None
    if config.tools.python_file:
        src = project_root / config.tools.python_file
        dst = build_dir / src.name
        shutil.copy2(src, dst)
        tools_module_name = src.stem
        logger.info("Copied tools file: %s", src.name)

    # 3. Copy MCP config if configured
    if config.mcp.config:
        src = project_root / config.mcp.config
        dst = build_dir / "_mcp.json"
        shutil.copy2(src, dst)
        logger.info("Copied MCP config: %s → _mcp.json", config.mcp.config)

    # 4. Copy env file if configured
    if config.deploy.env_file:
        src = project_root / config.deploy.env_file
        dst = build_dir / ".env"
        shutil.copy2(src, dst)
        logger.info("Copied env file: %s", config.deploy.env_file)

    # 5. Generate deploy_graph.py
    deploy_graph = _render_deploy_graph_hub(
        config, project_root, tools_module_name
    )
    (build_dir / "deploy_graph.py").write_text(deploy_graph)
    logger.info("Generated deploy_graph.py")

    # 6. Generate langgraph.json
    langgraph_json = _render_langgraph_json(config)
    (build_dir / "langgraph.json").write_text(langgraph_json)
    logger.info("Generated langgraph.json")

    # 7. Generate pyproject.toml
    pyproject = _render_pyproject(config)
    (build_dir / "pyproject.toml").write_text(pyproject)
    logger.info("Generated pyproject.toml")

    # 8. Write a unique build marker to bust the cloud builder's Docker
    # layer cache. Without this, the cloud build can serve a stale image
    # when only file content (not file presence) has changed.
    import secrets
    import time as _time

    build_id = f"{int(_time.time())}-{secrets.token_hex(4)}"
    (build_dir / ".deepagents-build-id").write_text(build_id + "\n")
    logger.info("Bundle build id: %s", build_id)

    return build_dir


def _bundle_hub_module(build_dir: Path) -> Path:
    """Copy ``deepagents.backends.hub`` into the build dir as a standalone module.

    The generated ``deploy_graph.py`` imports ``HubBackend`` from this
    bundled file (``_deepagents_hub.py``) rather than from
    ``deepagents.backends.hub``, which doesn't exist in the published
    PyPI release of deepagents yet.

    Returns the destination path.
    """
    import deepagents.backends.hub as hub_module  # noqa: PLC0415

    src = Path(hub_module.__file__)
    dst = build_dir / "_deepagents_hub.py"
    shutil.copy2(src, dst)
    logger.info("Bundled hub backend module into %s", dst)
    return dst


def _collect_hub_files(
    config: DeployConfig, project_root: Path
) -> list[tuple[str, bytes]]:
    """Walk the project for files that should land in the hub repo.

    Returns ``(path, content)`` tuples in the shape `BackendProtocol.upload_files`
    expects. Memory sources land at the project root layout, skills under
    ``/skills/``, and the MCP config at ``/.mcp.json``.
    """
    files: list[tuple[str, bytes]] = []

    # Memory sources → / (top-level, e.g. /AGENTS.md). Single-file sources
    # land at /<filename>; directory sources land under /<filename>/.
    for src in config.agent_memories.sources:
        src_path = project_root / src
        if src_path.is_file():
            files.append((f"/{src_path.name}", src_path.read_bytes()))
        elif src_path.is_dir():
            for f in sorted(src_path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(project_root).as_posix()
                    files.append((f"/{rel}", f.read_bytes()))

    # Skills sources → /skills/<skill-name>/SKILL.md
    for src in config.skills.sources:
        src_path = project_root / src
        if not src_path.is_dir():
            continue
        for f in sorted(src_path.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(src_path).as_posix()
                files.append((f"/skills/{rel}", f.read_bytes()))

    # MCP config → /.mcp.json
    if config.mcp.config:
        mcp_path = project_root / config.mcp.config
        if mcp_path.is_file():
            files.append(("/.mcp.json", mcp_path.read_bytes()))

    return files


def _push_to_hub(config: DeployConfig, project_root: Path) -> None:
    """Push agent files (memory, skills, mcp config) to a LangSmith hub repo.

    The hub repo handle is derived from ``config.agent.name``. The files
    are uploaded via `HubBackend.upload_files`, which batches them into a
    single atomic commit.

    The runtime ``deploy_graph.py`` reads from this same repo via
    `HubBackend`, so this function and `_render_deploy_graph_hub` must
    agree on the path layout.
    """
    from deepagents.backends.hub import HubBackend  # noqa: PLC0415

    files = _collect_hub_files(config, project_root)
    if not files:
        logger.warning("No files to push to hub repo %s.", config.agent.name)
        return

    hub = HubBackend.from_env(config.agent.name)
    responses = hub.upload_files(files)
    failed = [r for r in responses if r.error]
    if failed:
        details = "; ".join(f"{r.path}: {r.error}" for r in failed)
        msg = f"Failed to push to hub repo {config.agent.name}: {details}"
        raise RuntimeError(msg)
    logger.info("Pushed %d files to hub repo %s.", len(files), config.agent.name)


def _build_bundle(config: DeployConfig, project_root: Path) -> dict[str, str]:
    """Read memory and skills files into a path→content mapping.

    Memory files are mapped under ``/memory/`` and skills under ``/skills/``
    in the store namespace.

    Args:
        config: Parsed deploy configuration.
        project_root: Project root directory.

    Returns:
        Dict mapping store paths to file content strings.
    """
    bundle: dict[str, str] = {}

    # Memory sources → /memory/<filename>
    for src in config.agent_memories.sources:
        src_path = project_root / src
        if src_path.is_file():
            store_key = f"/memory/{src_path.name}"
            bundle[store_key] = src_path.read_text(encoding="utf-8")
        elif src_path.is_dir():
            for f in sorted(src_path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(src_path)
                    store_key = f"/memory/{rel}"
                    bundle[store_key] = f.read_text(encoding="utf-8")

    # Skills sources → /skills/<skill-name>/SKILL.md
    for src in config.skills.sources:
        src_path = project_root / src
        if not src_path.is_dir():
            continue
        for f in sorted(src_path.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(src_path)
                store_key = f"/skills/{rel}"
                bundle[store_key] = f.read_text(encoding="utf-8")

    return bundle


def _build_agent_memories_seed(
    config: DeployConfig, project_root: Path
) -> dict[str, str]:
    """Build the seed payload for store-backed ``/agent_memories/``.

    Returns a ``{key: content}`` mapping where each key matches the path
    the runtime ``MemoryMiddleware`` / ``SkillsMiddleware`` will look up
    once the file is in the store. Walks the same sources as
    `_collect_hub_files` (memory + skills) and emits keys *without* the
    ``/agent_memories/`` mount prefix because the composite strips the
    prefix before delegating to the store backend.
    """
    seed: dict[str, str] = {}

    # Memory sources land at the project-root layout (e.g. ``/AGENTS.md``).
    for src in config.agent_memories.sources:
        src_path = project_root / src
        if src_path.is_file():
            seed[f"/{src_path.name}"] = src_path.read_text(encoding="utf-8")
        elif src_path.is_dir():
            for f in sorted(src_path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(project_root).as_posix()
                    seed[f"/{rel}"] = f.read_text(encoding="utf-8")

    # Skills sources land under ``/skills/``.
    for src in config.skills.sources:
        src_path = project_root / src
        if not src_path.is_dir():
            continue
        for f in sorted(src_path.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(src_path).as_posix()
                seed[f"/skills/{rel}"] = f.read_text(encoding="utf-8")

    return seed


def _enumerate_memory_keys(
    sources: list[str],
    project_root: Path,
    *,
    file_prefix: str,
    dir_root: str,
) -> list[str]:
    """Enumerate the runtime store keys for a memory source list.

    `MemoryMiddleware` calls ``backend.download_files(sources)`` and treats
    each entry as a single file path, so the rendered ``memory=[...]`` list
    must contain real file keys (not directory placeholders). This walks
    ``sources`` the same way the file collectors do and returns one key
    per file.

    Args:
        sources: Local paths to enumerate (typically
            ``config.agent_memories.sources`` or
            ``config.user_memories.sources``).
        project_root: Project root directory used to resolve relative paths.
        file_prefix: Prefix prepended to every key (e.g.
            ``"/agent_memories/"`` or ``"/user_memories/"``).
        dir_root: For directory sources, controls how the relative path
            inside the key is computed — ``"source"`` makes it relative
            to the source directory, ``"project"`` makes it relative to
            the project root.
    """
    keys: list[str] = []
    for src in sources:
        src_path = project_root / src
        if src_path.is_file():
            keys.append(f"{file_prefix}{src_path.name}")
        elif src_path.is_dir():
            base = src_path if dir_root == "source" else project_root
            for f in sorted(src_path.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(base).as_posix()
                    keys.append(f"{file_prefix}{rel}")
    return keys


def _render_deploy_graph_hub(
    config: DeployConfig,
    project_root: Path,
    tools_module_name: str | None,
) -> str:
    """Render the hub-backed ``deploy_graph.py``.

    Composes:

    - A per-provider sandbox creation block from
      :data:`~deepagents_cli.deploy.templates.SANDBOX_BLOCKS` (langsmith,
      daytona, modal, runloop, or none → ``StateBackend``).
    - The shared tools loader (only emitted when ``[tools].python_file``
      is configured).
    - The MCP loader (only emitted when ``[mcp].config`` is configured).
    - ``/agent_memories/`` mounted to a hub or store backend depending
      on ``[agent_memories].backend``, holding skills and shared memory.
    - ``/user_memories/`` mounted to a user-namespaced StoreBackend.

    The graph is exported as an async ``make_graph`` factory so MCP
    loading can ``await`` cleanly.
    """
    provider = config.sandbox.provider
    if provider not in SANDBOX_BLOCKS:
        msg = (
            f"Unknown sandbox provider {provider!r}. "
            f"Valid: {sorted(SANDBOX_BLOCKS)}"
        )
        raise ValueError(msg)
    sandbox_block, _partner_pkg = SANDBOX_BLOCKS[provider]

    # Tools loader (same logic as the non-hub template)
    tools_import_block = ""
    tools_load_call = "pass  # noqa  (no static tools configured)"
    if tools_module_name:
        if config.tools.functions:
            func_imports = ", ".join(config.tools.functions)
            func_list = ", ".join(config.tools.functions)
            tools_import_block = TOOLS_IMPORT_TEMPLATE.format(
                module_name=tools_module_name,
                function_imports=func_imports,
                function_list=func_list,
            )
        else:
            tools_import_block = TOOLS_IMPORT_AUTODISCOVER_TEMPLATE.format(
                module_name=tools_module_name,
            )
        tools_load_call = "tools.extend(_load_tools())"

    # MCP loader (same template as non-hub)
    mcp_tools_block = ""
    mcp_tools_load_call = "pass  # noqa  (no MCP servers configured)"
    if config.mcp.config:
        mcp_tools_block = MCP_TOOLS_TEMPLATE
        mcp_tools_load_call = "tools.extend(await _load_mcp_tools())"

    agent_memory_keys = _enumerate_memory_keys(
        config.agent_memories.sources,
        project_root,
        file_prefix="/agent_memories/",
        dir_root="project",
    )
    user_memory_keys = _enumerate_memory_keys(
        config.user_memories.sources,
        project_root,
        file_prefix="/user_memories/",
        dir_root="project",
    )
    memory_sources = agent_memory_keys + user_memory_keys

    return DEPLOY_GRAPH_HUB_TEMPLATE.format(
        agent_name=config.agent.name,
        hub_repo=config.agent.name,
        model=config.agent.model,
        system_prompt=config.agent.system_prompt,
        sandbox_template=config.sandbox.template,
        sandbox_image=config.sandbox.image,
        memory_sources=memory_sources,
        agent_memories_backend=config.agent_memories.backend,
        sandbox_block=sandbox_block,
        tools_import_block=tools_import_block,
        tools_load_call=tools_load_call,
        mcp_tools_block=mcp_tools_block,
        mcp_tools_load_call=mcp_tools_load_call,
    )


def _render_deploy_graph(
    config: DeployConfig,
    project_root: Path,
    tools_module_name: str | None,
) -> str:
    """Render the deploy_graph.py server entry point.

    Args:
        config: Parsed deploy configuration.
        tools_module_name: Python module name for tools (e.g. ``"tools"``),
            or ``None`` if no tools file.

    Returns:
        Rendered Python source string.
    """
    # Build tools import block
    tools_import_block = ""
    tools_load_call = ""
    if tools_module_name:
        if config.tools.functions:
            # Explicit function list
            func_imports = ", ".join(config.tools.functions)
            func_list = ", ".join(config.tools.functions)
            tools_import_block = TOOLS_IMPORT_TEMPLATE.format(
                module_name=tools_module_name,
                function_imports=func_imports,
                function_list=func_list,
            )
        else:
            # Auto-discover @tool functions
            tools_import_block = TOOLS_IMPORT_AUTODISCOVER_TEMPLATE.format(
                module_name=tools_module_name,
            )
        tools_load_call = "tools.extend(_load_tools())"

    # Build MCP tools block
    mcp_tools_block = ""
    mcp_tools_load_call = ""
    if config.mcp.config:
        mcp_tools_block = MCP_TOOLS_TEMPLATE
        mcp_tools_load_call = "tools.extend(await _load_mcp_tools())"

    # Build sandbox block (for sandbox-enabled agents)
    before_agent_block = ""
    sandbox_backend_block = "# No sandbox configured"
    if config.sandbox.provider != "none":
        before_agent_block = BEFORE_AGENT_SANDBOX_TEMPLATE.format(
            sandbox_template=config.sandbox.template,
            sandbox_image=config.sandbox.image,
        )
        # In execution context, use sandbox as backend instead of store
        sandbox_backend_block = (
            "# Use sandbox as backend for execution (thread-scoped)\n"
            "    ert = runtime.execution_runtime\n"
            "    if ert is not None:\n"
            "        from langgraph.config import get_config\n"
            "        thread_id = get_config().get('configurable', {}).get('thread_id', 'default')\n"
            "        sandbox_backend = _get_or_create_sandbox(thread_id)\n"
            "        if sandbox_backend is not None:\n"
            "            backend = sandbox_backend"
        )

    # Compute store paths for memory and skills.
    #
    # Memory keys must be enumerated per file: ``MemoryMiddleware`` calls
    # ``backend.download_files(sources)`` and treats each entry as a single
    # file path, so a directory placeholder like ``/memory/`` would silently
    # resolve to ``file_not_found``. Skills, by contrast, are listed via
    # ``ls``/``glob`` by ``SkillsMiddleware``, so a directory root works.
    memory_sources = _enumerate_memory_keys(
        config.agent_memories.sources,
        project_root,
        file_prefix="/memory/",
        dir_root="source",
    )
    skills_sources = ["/skills/"]

    return DEPLOY_GRAPH_TEMPLATE.format(
        agent_name=config.agent.name,
        memory_scope="agent",
        model=config.agent.model,
        system_prompt=config.agent.system_prompt,
        memory_sources=memory_sources,
        skills_sources=skills_sources,
        tools_import_block=tools_import_block,
        tools_load_call=tools_load_call,
        mcp_tools_block=mcp_tools_block,
        mcp_tools_load_call=mcp_tools_load_call,
        before_agent_block=before_agent_block,
        sandbox_backend_block=sandbox_backend_block,
    )


def _render_langgraph_json(config: DeployConfig) -> str:
    """Render the ``langgraph.json`` configuration.

    Hub-backed deploys export an async ``make_graph`` factory so that MCP
    loading can ``await`` cleanly. Non-hub deploys keep the legacy
    ``graph`` symbol path.
    """
    graph_symbol = "make_graph" if _is_hub_backed(config) else "graph"
    lg_config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {
            "agent": f"./deploy_graph.py:{graph_symbol}",
        },
        "python_version": config.deploy.python_version,
    }

    if config.deploy.env_file:
        lg_config["env"] = ".env"

    return json.dumps(lg_config, indent=2)


def _render_pyproject(config: DeployConfig) -> str:
    """Render the ``pyproject.toml`` for the deployment package.

    Adds:

    - ``langchain-mcp-adapters`` when ``[mcp].config`` is set (both
      hub and non-hub deploy graphs use it).
    - The matching partner package for non-LangSmith sandbox providers
      (``langchain-daytona`` / ``langchain-modal`` / ``langchain-runloop``).
      LangSmith and the no-sandbox path don't need extra deps.
    """
    all_deps = list(config.deploy.dependencies)
    if config.mcp.config:
        all_deps.append("langchain-mcp-adapters")

    _, partner_pkg = SANDBOX_BLOCKS.get(config.sandbox.provider, (None, None))
    if partner_pkg:
        all_deps.append(partner_pkg)

    extra_deps_lines = ""
    for dep in all_deps:
        extra_deps_lines += f'    "{dep}",\n'

    return PYPROJECT_TEMPLATE.format(
        agent_name=config.agent.name,
        python_version=config.deploy.python_version,
        extra_deps=extra_deps_lines,
    )


def print_bundle_summary(
    config: DeployConfig,
    build_dir: Path,
    bundle_data: dict[str, str] | None = None,
) -> None:
    """Print a human-readable summary of what was bundled.

    Args:
        config: Parsed deploy configuration.
        build_dir: Build directory path.
        bundle_data: Optional pre-loaded bundle data. If ``None``, reads from
            the ``_bundle.json`` file in *build_dir*.
    """
    if bundle_data is None:
        bundle_path = build_dir / "_bundle.json"
        if bundle_path.exists():
            bundle_data = json.loads(bundle_path.read_text())
        else:
            bundle_data = {}

    print(f"\n  Agent: {config.agent.name}")
    print(f"  Model: {config.agent.model}")
    if config.agent.system_prompt:
        prompt_preview = config.agent.system_prompt[:80]
        if len(config.agent.system_prompt) > 80:
            prompt_preview += "..."
        print(f"  Prompt: {prompt_preview}")

    # Memory files
    memory_files = [k for k in bundle_data if k.startswith("/memory/")]
    if memory_files:
        print(f"\n  Memory ({len(memory_files)} file(s)):")
        for f in memory_files:
            print(f"    {f}")

    # Skills files
    skills_files = [k for k in bundle_data if k.startswith("/skills/")]
    if skills_files:
        print(f"\n  Skills ({len(skills_files)} file(s)):")
        for f in skills_files:
            print(f"    {f}")

    # Tools
    if config.tools.python_file:
        print(f"\n  Tools: {config.tools.python_file}")
        if config.tools.functions:
            print(f"    Functions: {', '.join(config.tools.functions)}")
        else:
            print("    Functions: auto-discover")

    # MCP
    if config.mcp.config:
        print(f"\n  MCP config: {config.mcp.config}")

    # Sandbox
    print(f"\n  Sandbox: {config.sandbox.provider}")

    # Build dir
    print(f"\n  Build directory: {build_dir}")

    # List generated files
    generated = sorted(f.name for f in build_dir.iterdir() if f.is_file())
    print(f"  Generated files: {', '.join(generated)}")
    print()
