"""Artifact bundler for deepagents deploy.

Assembles a deployment directory containing all files needed to deploy
the agent via ``langgraph deploy``:

- ``langgraph.json`` — LangGraph server configuration
- ``deploy_graph.py`` — server-side graph entry point
- ``deploy_config.json`` — serialized DeployConfig for runtime
- ``pyproject.toml`` — Python dependencies
- ``agents/`` — AGENTS.md files
- ``skills/`` — skill directories
- ``.env`` — environment variables (if configured)
- Custom tool modules (if configured)
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from deepagents_cli.deploy.config import DeployConfig

logger = logging.getLogger(__name__)


def bundle_deploy_artifacts(
    config: DeployConfig,
    *,
    project_root: Path | None = None,
) -> Path:
    """Bundle all deployment artifacts into a temporary directory.

    Args:
        config: Parsed deploy configuration.
        project_root: Project root directory for resolving relative paths.
            Defaults to the current working directory.

    Returns:
        Path to the temporary deploy directory.
    """
    if project_root is None:
        project_root = Path.cwd()

    deploy_dir = Path(tempfile.mkdtemp(prefix="deepagents_deploy_"))
    logger.info("Bundling deploy artifacts in %s", deploy_dir)

    # 1. Copy the deploy graph entry point
    _write_deploy_graph(deploy_dir)

    # 2. Serialize the config for runtime consumption
    _write_deploy_config(deploy_dir, config)

    # 3. Bundle AGENTS.md files
    _bundle_memory(deploy_dir, config, project_root)

    # 4. Bundle skills
    _bundle_skills(deploy_dir, config, project_root)

    # 5. Bundle custom tools module
    _bundle_custom_tools(deploy_dir, config, project_root)

    # 6. Bundle MCP config
    _bundle_mcp_config(deploy_dir, config, project_root)

    # 7. Bundle .env file
    _bundle_env_file(deploy_dir, config, project_root)

    # 8. Write pyproject.toml
    _write_pyproject(deploy_dir, config)

    # 9. Generate langgraph.json
    _write_langgraph_json(deploy_dir, config)

    return deploy_dir


def _write_deploy_graph(deploy_dir: Path) -> None:
    """Copy the deploy graph entry point into the bundle."""
    src = Path(__file__).parent / "deploy_graph.py"
    dst = deploy_dir / "deploy_graph.py"
    shutil.copy2(src, dst)


def _write_deploy_config(deploy_dir: Path, config: DeployConfig) -> None:
    """Serialize the deploy config as JSON for runtime loading."""
    # Serialize the full config so the deploy graph can reconstruct it
    data: dict[str, Any] = {
        "agent": config.agent,
        "description": config.description,
        "model": config.model,
        "model_params": config.model_params,
        "prompt": config.prompt,
        "memory": {
            "scope": config.memory.scope,
            "sources": config.memory.sources,
        },
        "skills": {
            "sources": config.skills.sources,
        },
        "tools": {
            "shell": config.tools.shell,
            "shell_allow_list": config.tools.shell_allow_list,
            "web_search": config.tools.web_search,
            "fetch_url": config.tools.fetch_url,
            "http_request": config.tools.http_request,
            "mcp": str(config.tools.mcp) if config.tools.mcp else False,
            "custom": config.tools.custom,
        },
        "backend": {
            "type": config.backend.type,
            "namespace": {
                "scope": config.backend.namespace.scope,
                "prefix": config.backend.namespace.prefix,
            },
            "path": config.backend.path,
        },
    }

    if config.sandbox is not None:
        data["sandbox"] = {
            "provider": config.sandbox.provider,
            "scope": config.sandbox.scope,
            "template": config.sandbox.template,
            "image": config.sandbox.image,
            "setup_script": config.sandbox.setup_script,
        }
    else:
        data["sandbox"] = None

    (deploy_dir / "deploy_config.json").write_text(json.dumps(data, indent=2))


def _resolve_source_path(source: str, project_root: Path) -> Path | None:
    """Resolve a source path, expanding ~ and making relative paths absolute.

    Returns None if the resolved path does not exist.
    """
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.exists():
        logger.warning("Source path does not exist: %s (resolved from %s)", path, source)
        return None
    return path


def _bundle_memory(deploy_dir: Path, config: DeployConfig, project_root: Path) -> None:
    """Copy AGENTS.md files into the bundle."""
    agents_dir = deploy_dir / "agents"
    agents_dir.mkdir(exist_ok=True)

    bundled_sources: list[str] = []
    for i, source in enumerate(config.memory.sources):
        resolved = _resolve_source_path(source, project_root)
        if resolved is None:
            continue
        # Use index prefix to avoid name collisions
        dest_name = f"{i}_{resolved.name}"
        shutil.copy2(resolved, agents_dir / dest_name)
        bundled_sources.append(f"agents/{dest_name}")
        logger.info("Bundled memory source: %s -> %s", source, dest_name)

    # Update the config's memory sources to point to bundled paths
    # (written separately in deploy_config.json)
    if bundled_sources:
        config_path = deploy_dir / "deploy_config.json"
        data = json.loads(config_path.read_text())
        data["memory"]["_bundled_sources"] = bundled_sources
        config_path.write_text(json.dumps(data, indent=2))


def _bundle_skills(deploy_dir: Path, config: DeployConfig, project_root: Path) -> None:
    """Copy skill directories into the bundle."""
    skills_dir = deploy_dir / "skills"
    skills_dir.mkdir(exist_ok=True)

    bundled_sources: list[str] = []
    for source in config.skills.sources:
        resolved = _resolve_source_path(source, project_root)
        if resolved is None:
            continue
        if not resolved.is_dir():
            logger.warning("Skills source is not a directory: %s", resolved)
            continue
        # Copy each skill subdirectory
        for skill_subdir in resolved.iterdir():
            if skill_subdir.is_dir() and (skill_subdir / "SKILL.md").exists():
                dest = skills_dir / skill_subdir.name
                if dest.exists():
                    logger.warning("Skill name collision, skipping: %s", skill_subdir.name)
                    continue
                shutil.copytree(skill_subdir, dest)
                logger.info("Bundled skill: %s", skill_subdir.name)
        bundled_sources.append("skills")

    if bundled_sources:
        config_path = deploy_dir / "deploy_config.json"
        data = json.loads(config_path.read_text())
        data["skills"]["_bundled_sources"] = list(set(bundled_sources))
        config_path.write_text(json.dumps(data, indent=2))


def _bundle_custom_tools(
    deploy_dir: Path, config: DeployConfig, project_root: Path
) -> None:
    """Copy custom tools module into the bundle."""
    if not config.tools.custom:
        return

    # Parse "module.py:variable" format
    parts = config.tools.custom.split(":")
    module_path_str = parts[0]

    resolved = _resolve_source_path(module_path_str, project_root)
    if resolved is None:
        msg = f"Custom tools module not found: {module_path_str}"
        raise FileNotFoundError(msg)

    dest = deploy_dir / resolved.name
    shutil.copy2(resolved, dest)

    # Update config to point to bundled path
    config_path = deploy_dir / "deploy_config.json"
    data = json.loads(config_path.read_text())
    variable = parts[1] if len(parts) > 1 else "tools"
    data["tools"]["_bundled_custom"] = f"./{resolved.name}:{variable}"
    config_path.write_text(json.dumps(data, indent=2))
    logger.info("Bundled custom tools: %s", config.tools.custom)


def _bundle_mcp_config(
    deploy_dir: Path, config: DeployConfig, project_root: Path
) -> None:
    """Copy MCP configuration file into the bundle."""
    if not config.tools.mcp or config.tools.mcp is True:
        return

    mcp_path = str(config.tools.mcp)
    resolved = _resolve_source_path(mcp_path, project_root)
    if resolved is None:
        logger.warning("MCP config not found: %s", mcp_path)
        return

    shutil.copy2(resolved, deploy_dir / ".mcp.json")
    logger.info("Bundled MCP config: %s", mcp_path)


def _bundle_env_file(
    deploy_dir: Path, config: DeployConfig, project_root: Path
) -> None:
    """Copy .env file into the bundle."""
    resolved = _resolve_source_path(config.env, project_root)
    if resolved is None:
        return

    shutil.copy2(resolved, deploy_dir / ".env")
    logger.info("Bundled env file: %s", config.env)


def _write_pyproject(deploy_dir: Path, config: DeployConfig) -> None:
    """Write a pyproject.toml for the deployment."""
    # Determine extra dependencies based on sandbox provider
    extras: list[str] = []
    if config.sandbox is not None:
        provider = config.sandbox.provider
        if provider == "langsmith":
            extras.append('"langsmith-sandbox"')
        elif provider == "modal":
            extras.append('"deepagents-modal"')
        elif provider == "daytona":
            extras.append('"deepagents-daytona"')
        elif provider == "runloop":
            extras.append('"deepagents-runloop"')

    if config.tools.web_search:
        extras.append('"tavily-python"')

    extra_deps = ""
    if extras:
        extra_deps = "\n" + "\n".join(f"    {e}," for e in extras)

    content = f"""\
[project]
name = "deepagents-deploy"
version = "0.0.1"
requires-python = ">={config.python_version}"
dependencies = [
    "deepagents-cli",{extra_deps}
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (deploy_dir / "pyproject.toml").write_text(content)


def _write_langgraph_json(deploy_dir: Path, config: DeployConfig) -> None:
    """Generate the langgraph.json configuration for deployment."""
    langgraph_config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {
            "agent": "./deploy_graph.py:graph",
        },
        "python_version": config.python_version,
    }

    # Env file
    if (deploy_dir / ".env").exists():
        langgraph_config["env"] = ".env"

    # Store config for semantic search indexing
    if config.backend.type == "store":
        langgraph_config["store"] = {
            "index": {
                "embed": "openai:text-embedding-3-small",
                "dims": 1536,
                "fields": ["$"],
            },
        }

    (deploy_dir / "langgraph.json").write_text(json.dumps(langgraph_config, indent=2))
    logger.info("Generated langgraph.json")
