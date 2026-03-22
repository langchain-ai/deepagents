"""Deploy and dev commands for running deepagents on LangGraph Platform.

Generates deployment artifacts (langgraph.json, deploy_graph.py, pyproject.toml)
and delegates to `langgraph deploy` or `langgraph dev` for the actual
build/push/serve lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEPLOY_GRAPH_FILENAME = "deploy_graph.py"
_LANGGRAPH_JSON_FILENAME = "langgraph.json"
_PYPROJECT_FILENAME = "pyproject.toml"
_ENV_FILENAME = ".env"
_BUNDLED_DIR = "bundled"
_AGENTS_MD_DIR = "agents_md"
_SKILLS_DIR = "skills"


@dataclass(frozen=True)
class DeployConfig:
    """Configuration for a deployment."""

    sandbox_type: str = "langsmith"
    model: str | None = None
    agent_name: str = "agent"
    deployment_name: str | None = None
    env_file: str | None = None
    api_key: str | None = None
    dry_run: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


def _collect_agents_md(project_root: Path | None) -> list[tuple[str, str]]:
    """Collect AGENTS.md files from the project.

    Args:
        project_root: Project root directory.

    Returns:
        List of `(relative_name, content)` tuples.
    """
    files: list[tuple[str, str]] = []
    if project_root is None:
        return files

    candidates = [
        ("AGENTS.md", project_root / "AGENTS.md"),
        (".deepagents/AGENTS.md", project_root / ".deepagents" / "AGENTS.md"),
    ]
    for name, path in candidates:
        try:
            if path.exists() and path.is_file():
                files.append((name, path.read_text(encoding="utf-8")))
        except OSError:
            logger.warning("Could not read %s", path, exc_info=True)
    return files


def _collect_skills(project_root: Path | None) -> list[tuple[str, Path]]:
    """Collect skill directories from the project.

    Args:
        project_root: Project root directory.

    Returns:
        List of `(relative_path, absolute_path)` tuples for skill dirs.
    """
    dirs: list[tuple[str, Path]] = []
    if project_root is None:
        return dirs

    candidates = [
        (".deepagents/skills", project_root / ".deepagents" / "skills"),
        (".agents/skills", project_root / ".agents" / "skills"),
    ]
    for rel, path in candidates:
        try:
            if path.exists() and path.is_dir():
                dirs.append((rel, path))
        except OSError:
            logger.warning("Could not access %s", path, exc_info=True)
    return dirs


def _collect_user_agents_md(agent_name: str) -> str | None:
    """Read the user-level AGENTS.md for the given agent.

    Args:
        agent_name: Agent name (e.g. "agent").

    Returns:
        Content string, or `None` if not found or empty.
    """
    user_dir = Path.home() / ".deepagents" / agent_name
    agent_md = user_dir / "AGENTS.md"
    try:
        if agent_md.exists():
            content = agent_md.read_text(encoding="utf-8").strip()
            if content:
                return content
    except OSError:
        logger.warning("Could not read user AGENTS.md at %s", agent_md, exc_info=True)
    return None


def _bundle_files(
    work_dir: Path,
    *,
    project_root: Path | None,
    agent_name: str,
) -> dict[str, Any]:
    """Bundle AGENTS.md and skills into the work directory.

    Args:
        work_dir: Deployment working directory.
        project_root: Project root directory.
        agent_name: Agent name.

    Returns:
        Metadata dict with bundled file info.
    """
    bundled_dir = work_dir / _BUNDLED_DIR
    bundled_dir.mkdir(exist_ok=True)

    metadata: dict[str, Any] = {"agents_md_files": [], "skills_dirs": []}

    agents_md_dir = bundled_dir / _AGENTS_MD_DIR
    agents_md_dir.mkdir(exist_ok=True)

    project_files = _collect_agents_md(project_root)
    for name, content in project_files:
        safe_name = name.replace("/", "_").replace("\\", "_")
        dest = agents_md_dir / safe_name
        dest.write_text(content, encoding="utf-8")
        metadata["agents_md_files"].append(safe_name)

    user_content = _collect_user_agents_md(agent_name)
    if user_content:
        dest = agents_md_dir / "user_AGENTS.md"
        dest.write_text(user_content, encoding="utf-8")
        metadata["agents_md_files"].append("user_AGENTS.md")

    skills_dirs = _collect_skills(project_root)
    skills_dest = bundled_dir / _SKILLS_DIR
    for rel, src_path in skills_dirs:
        dest = skills_dest / rel.replace("/", "_")
        try:
            shutil.copytree(src_path, dest, dirs_exist_ok=True)
            metadata["skills_dirs"].append(str(dest.relative_to(work_dir)))
        except OSError:
            logger.warning("Could not copy skills from %s", src_path, exc_info=True)

    user_skills = Path.home() / ".deepagents" / agent_name / "skills"
    if user_skills.exists() and user_skills.is_dir():
        dest = skills_dest / "user_skills"
        try:
            shutil.copytree(user_skills, dest, dirs_exist_ok=True)
            metadata["skills_dirs"].append(str(dest.relative_to(work_dir)))
        except OSError:
            logger.warning(
                "Could not copy user skills from %s", user_skills, exc_info=True
            )

    built_in_skills = Path(__file__).parent / "built_in_skills"
    if built_in_skills.exists() and built_in_skills.is_dir():
        dest = skills_dest / "built_in"
        try:
            shutil.copytree(built_in_skills, dest, dirs_exist_ok=True)
            metadata["skills_dirs"].append(str(dest.relative_to(work_dir)))
        except OSError:
            logger.warning(
                "Could not copy built-in skills from %s",
                built_in_skills,
                exc_info=True,
            )

    return metadata


_DEPLOY_GRAPH_TEMPLATE = '''\
"""Server-side graph factory for deployed deepagents.

This module is referenced by langgraph.json and exposes the agent graph
via the `get_agent` async factory function. LangGraph Platform calls this
function per-thread to create an agent with its own sandbox.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langgraph.graph.state import RunnableConfig
from langgraph.pregel import Pregel

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "bundled"
AGENTS_MD_DIR = BUNDLED_DIR / "agents_md"
SKILLS_DIR = BUNDLED_DIR / "skills"

_SANDBOX_BACKENDS: dict[str, Any] = {}


def _get_sandbox_type() -> str:
    return os.getenv("SANDBOX_TYPE", "langsmith")


def _create_sandbox(sandbox_id: str | None = None) -> Any:
    """Create or reconnect to a sandbox."""
    sandbox_type = _get_sandbox_type()

    if sandbox_type == "langsmith":
        from deepagents.backends.langsmith import LangSmithSandbox
        from langsmith.sandbox import Sandbox

        if sandbox_id:
            sb = Sandbox.connect(sandbox_id)
        else:
            sb = Sandbox()
        return LangSmithSandbox(sb)

    if sandbox_type == "daytona":
        from langchain_daytona import DaytonaSandbox

        if sandbox_id:
            return DaytonaSandbox(sandbox_id=sandbox_id)
        return DaytonaSandbox()

    if sandbox_type == "modal":
        from langchain_modal import ModalSandbox

        if sandbox_id:
            return ModalSandbox(sandbox_id=sandbox_id)
        return ModalSandbox()

    if sandbox_type == "runloop":
        from langchain_runloop import RunloopSandbox

        if sandbox_id:
            return RunloopSandbox(sandbox_id=sandbox_id)
        return RunloopSandbox()

    msg = f"Unsupported sandbox type: {sandbox_type}"
    raise ValueError(msg)


def _load_bundled_agents_md() -> str:
    """Load all bundled AGENTS.md files into a combined string."""
    if not AGENTS_MD_DIR.exists():
        return ""
    parts: list[str] = []
    for md_file in sorted(AGENTS_MD_DIR.iterdir()):
        if md_file.is_file() and md_file.suffix == ".md":
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except OSError:
                logger.warning("Could not read %s", md_file, exc_info=True)
    return "\\n\\n".join(parts)


def _get_skills_sources() -> list[str]:
    """Get paths to bundled skill directories."""
    if not SKILLS_DIR.exists():
        return []
    sources: list[str] = []
    for entry in sorted(SKILLS_DIR.iterdir()):
        if entry.is_dir():
            sources.append(str(entry))
    return sources


def _get_model() -> Any:
    """Resolve the model from environment configuration."""
    model_spec = os.getenv("DEEPAGENTS_DEPLOY_MODEL")
    if not model_spec:
        return "anthropic:claude-sonnet-4-6"
    return model_spec


async def get_agent(config: RunnableConfig) -> Pregel:
    """Create an agent with a sandbox for the given thread.

    This is the graph factory function called by LangGraph Platform
    for each thread invocation.

    Args:
        config: LangGraph runtime configuration.

    Returns:
        Compiled agent graph.
    """
    from deepagents import create_deep_agent
    from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
    from deepagents.backends.filesystem import FilesystemBackend

    thread_id = config.get("configurable", {}).get("thread_id")

    sandbox_backend = None
    if thread_id:
        sandbox_backend = _SANDBOX_BACKENDS.get(thread_id)

    if sandbox_backend is None:
        try:
            sandbox_backend = _create_sandbox()
            if thread_id:
                _SANDBOX_BACKENDS[thread_id] = sandbox_backend
        except Exception:
            logger.exception("Failed to create sandbox")
            raise

    model = _get_model()
    agents_md_content = _load_bundled_agents_md()
    skills_sources = _get_skills_sources()

    middleware: list[Any] = []

    if agents_md_content:
        agents_md_dir = AGENTS_MD_DIR
        if agents_md_dir.exists():
            memory_sources = [
                str(p) for p in sorted(agents_md_dir.iterdir()) if p.is_file()
            ]
            middleware.append(
                MemoryMiddleware(
                    backend=FilesystemBackend(),
                    sources=memory_sources,
                )
            )

    if skills_sources:
        middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=skills_sources,
            )
        )

    system_prompt = os.getenv("DEEPAGENTS_DEPLOY_SYSTEM_PROMPT", "")

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=[],
        backend=sandbox_backend,
        middleware=middleware,
    ).with_config(config)
'''


def _write_deploy_graph(work_dir: Path) -> Path:
    """Write the deploy_graph.py server-side entry point.

    Args:
        work_dir: Deployment working directory.

    Returns:
        Path to the written file.
    """
    dest = work_dir / _DEPLOY_GRAPH_FILENAME
    dest.write_text(_DEPLOY_GRAPH_TEMPLATE, encoding="utf-8")
    return dest


def _write_langgraph_json(
    work_dir: Path,
    *,
    env_file: str | None = None,
) -> Path:
    """Generate the langgraph.json configuration.

    Args:
        work_dir: Deployment working directory.
        env_file: Optional path to .env file.

    Returns:
        Path to the written file.
    """
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {
            "agent": f"./{_DEPLOY_GRAPH_FILENAME}:get_agent",
        },
        "python_version": f"3.{sys.version_info.minor}",
    }
    if env_file:
        config["env"] = env_file

    dest = work_dir / _LANGGRAPH_JSON_FILENAME
    dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return dest


def _write_pyproject(work_dir: Path, *, sandbox_type: str = "langsmith") -> Path:
    """Generate a pyproject.toml for the deployment.

    Args:
        work_dir: Deployment working directory.
        sandbox_type: Sandbox provider to include as a dependency.

    Returns:
        Path to the written file.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        sdk_version = pkg_version("deepagents")
    except (PackageNotFoundError, ModuleNotFoundError):
        sdk_version = "0.4.11"

    deps = [
        f'"deepagents>={sdk_version}"',
        '"langchain>=1.2.10"',
        '"langgraph>=1.1.2"',
        '"langsmith[sandbox]>=0.7.7"',
    ]

    sandbox_deps = {
        "daytona": '"langchain-daytona>=0.0.4"',
        "modal": '"langchain-modal>=0.0.2"',
        "runloop": '"langchain-runloop>=0.0.3"',
    }
    if sandbox_type in sandbox_deps:
        deps.append(sandbox_deps[sandbox_type])

    deps_str = ",\n    ".join(deps)

    content = f"""\
[project]
name = "deepagents-deploy"
version = "0.0.1"
description = "Deployed deepagents agent"
requires-python = ">=3.11"
dependencies = [
    {deps_str},
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    dest = work_dir / _PYPROJECT_FILENAME
    dest.write_text(content, encoding="utf-8")
    return dest


def _write_env_file(
    work_dir: Path,
    *,
    config: DeployConfig,
    user_env_file: str | None = None,
) -> Path | None:
    """Generate a .env file for the deployment.

    Args:
        work_dir: Deployment working directory.
        config: Deploy configuration.
        user_env_file: Optional user-provided .env file to merge.

    Returns:
        Path to the written file, or `None` if no env vars needed.
    """
    env_vars: dict[str, str] = {}

    env_vars["SANDBOX_TYPE"] = config.sandbox_type

    if config.model:
        env_vars["DEEPAGENTS_DEPLOY_MODEL"] = config.model

    api_key = (
        config.api_key
        or os.environ.get("LANGSMITH_API_KEY")
        or os.environ.get("LANGCHAIN_API_KEY")
    )
    if api_key:
        env_vars["LANGSMITH_API_KEY"] = api_key

    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "DAYTONA_API_KEY",
        "DAYTONA_SERVER_URL",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "RUNLOOP_API_KEY",
    ):
        val = os.environ.get(key)
        if val:
            env_vars[key] = val

    env_vars.update(config.extra_env)

    if user_env_file:
        user_path = Path(user_env_file).expanduser().resolve()
        if user_path.exists():
            for raw_line in user_path.read_text(encoding="utf-8").splitlines():
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, _, v = stripped.partition("=")
                    env_vars[k.strip()] = v.strip()

    if not env_vars:
        return None

    dest = work_dir / _ENV_FILENAME
    lines = [f"{k}={v}" for k, v in sorted(env_vars.items())]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def scaffold_deployment(
    config: DeployConfig,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate all deployment artifacts in a working directory.

    Args:
        config: Deploy configuration.
        project_root: Project root for discovering AGENTS.md and skills.
        output_dir: Output directory. Created as a temp dir if `None`.

    Returns:
        Path to the deployment working directory.
    """
    if output_dir is not None:
        work_dir = output_dir
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="deepagents_deploy_"))

    _bundle_files(work_dir, project_root=project_root, agent_name=config.agent_name)
    _write_deploy_graph(work_dir)

    env_path = _write_env_file(work_dir, config=config, user_env_file=config.env_file)
    env_ref = f"./{_ENV_FILENAME}" if env_path else None
    _write_langgraph_json(work_dir, env_file=env_ref)
    _write_pyproject(work_dir, sandbox_type=config.sandbox_type)

    return work_dir


def _run_langgraph_command(
    command: str,
    work_dir: Path,
    *,
    extra_args: list[str] | None = None,
    api_key: str | None = None,
) -> int:
    """Run a langgraph CLI command as a subprocess.

    Args:
        command: The langgraph subcommand (e.g. "deploy", "dev").
        work_dir: Working directory containing langgraph.json.
        extra_args: Additional CLI arguments.
        api_key: LangSmith API key for deploy.

    Returns:
        Process exit code.
    """
    cmd = [
        sys.executable,
        "-m",
        "langgraph_cli",
        command,
        "--config",
        str(work_dir / _LANGGRAPH_JSON_FILENAME),
    ]

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if api_key:
        env["LANGSMITH_API_KEY"] = api_key

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(work_dir),
            env=env,
            check=False,
        )
    except FileNotFoundError:
        logger.exception(
            "langgraph CLI not found. Install with: pip install langgraph-cli"
        )
        return 1
    except KeyboardInterrupt:
        return 130
    else:
        return result.returncode


def run_deploy(config: DeployConfig) -> int:
    """Run the full deploy flow: scaffold artifacts then call `langgraph deploy`.

    Args:
        config: Deploy configuration.

    Returns:
        Process exit code.
    """
    from deepagents_cli.config import console
    from deepagents_cli.project_utils import find_project_root

    project_root = find_project_root(Path.cwd())

    console.print("[bold]Preparing deployment artifacts...[/bold]")

    work_dir = scaffold_deployment(config, project_root=project_root)

    console.print(f"  Artifacts generated in: {work_dir}")

    if config.dry_run:
        console.print("\n[bold green]Dry run complete.[/bold green]")
        console.print(f"  Inspect artifacts at: {work_dir}")
        console.print(f"  langgraph.json: {work_dir / _LANGGRAPH_JSON_FILENAME}")
        console.print(f"  deploy_graph.py: {work_dir / _DEPLOY_GRAPH_FILENAME}")
        console.print(f"  pyproject.toml: {work_dir / _PYPROJECT_FILENAME}")
        return 0

    console.print("[bold]Deploying to LangGraph Platform...[/bold]")

    extra_args: list[str] = []
    if config.deployment_name:
        extra_args.extend(["--name", config.deployment_name])

    return _run_langgraph_command(
        "deploy",
        work_dir,
        extra_args=extra_args,
        api_key=config.api_key,
    )


def run_dev(config: DeployConfig, *, port: int = 2024, host: str = "127.0.0.1") -> int:
    """Run the dev flow: scaffold artifacts then call `langgraph dev`.

    Args:
        config: Deploy configuration.
        port: Port for the dev server.
        host: Host for the dev server.

    Returns:
        Process exit code.
    """
    from deepagents_cli.config import console
    from deepagents_cli.project_utils import find_project_root

    project_root = find_project_root(Path.cwd())

    console.print("[bold]Preparing development server artifacts...[/bold]")

    work_dir = scaffold_deployment(config, project_root=project_root)

    console.print(f"  Artifacts generated in: {work_dir}")

    if config.dry_run:
        console.print("\n[bold green]Dry run complete.[/bold green]")
        console.print(f"  Inspect artifacts at: {work_dir}")
        return 0

    console.print(f"[bold]Starting LangGraph dev server on {host}:{port}...[/bold]")

    extra_args = [
        "--host",
        host,
        "--port",
        str(port),
        "--no-browser",
    ]

    return _run_langgraph_command(
        "dev",
        work_dir,
        extra_args=extra_args,
        api_key=config.api_key,
    )
