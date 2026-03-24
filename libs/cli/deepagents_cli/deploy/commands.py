"""CLI commands for ``deepagents deploy``.

Provides the ``deploy`` subcommand that bundles agent artifacts and
deploys them to LangGraph Cloud via ``langgraph deploy``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess  # noqa: S404
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_deploy_parser(
    subparsers: Any,
    *,
    make_help_action: Callable[..., type[argparse.Action]],
    add_output_args: Callable[..., None],
) -> None:
    """Register the ``deploy`` subcommand with the CLI argument parser.

    Args:
        subparsers: The subparsers action from the parent parser.
        make_help_action: Factory for custom help actions.
        add_output_args: Adds ``--output`` argument to a parser.
    """
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy agent to LangGraph Cloud",
        add_help=False,
    )
    deploy_parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        default=False,
        help="Show help for the deploy command",
    )

    deploy_parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        default=None,
        help="Path to deepagents.json config file (default: ./deepagents.json)",
    )

    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate the deploy bundle without deploying. "
        "Prints the bundle directory and langgraph.json.",
    )

    deploy_parser.add_argument(
        "--wait",
        action="store_true",
        default=False,
        help="Wait for the deployment to complete before exiting.",
    )

    # Override config fields via CLI flags
    deploy_parser.add_argument(
        "-a",
        "--agent",
        metavar="NAME",
        default=None,
        help="Agent name (overrides deepagents.json 'agent' field)",
    )
    deploy_parser.add_argument(
        "-M",
        "--model",
        metavar="MODEL",
        default=None,
        help="Model to deploy (overrides deepagents.json 'model' field)",
    )
    deploy_parser.add_argument(
        "--sandbox",
        choices=["langsmith", "modal", "daytona", "runloop", "none"],
        default=None,
        metavar="PROVIDER",
        help="Sandbox provider (default: langsmith)",
    )
    deploy_parser.add_argument(
        "--sandbox-scope",
        choices=["assistant", "user", "thread", "user+thread"],
        default=None,
        help="Sandbox lifecycle scope (default: thread)",
    )
    deploy_parser.add_argument(
        "--backend-scope",
        choices=["assistant", "user", "thread", "user+thread"],
        default=None,
        help="Store backend namespace scope (default: assistant)",
    )
    deploy_parser.add_argument(
        "--memory-scope",
        choices=["assistant", "user", "thread", "user+thread"],
        default=None,
        help="Memory (AGENTS.md) namespace scope (default: assistant)",
    )
    deploy_parser.add_argument(
        "--revision-id",
        metavar="ID",
        default=None,
        help="Deployment revision identifier",
    )

    add_output_args(deploy_parser)


def execute_deploy_command(args: argparse.Namespace) -> None:
    """Execute the ``deploy`` subcommand.

    Args:
        args: Parsed CLI arguments.
    """
    if getattr(args, "help", False):
        _show_deploy_help()
        return

    from deepagents_cli.deploy.bundle import bundle_deploy_artifacts
    from deepagents_cli.deploy.config import DeployConfig

    # Load config
    config_path = Path(args.config).resolve() if args.config else None
    try:
        config = DeployConfig.load(config_path)
    except (ValueError, OSError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Apply CLI overrides
    config = _apply_cli_overrides(config, args)

    # Print config summary
    _print_config_summary(config)

    # Resolve project root: use config file's parent directory if an explicit
    # config path was provided, otherwise use the current working directory.
    project_root = config_path.parent if config_path is not None else Path.cwd()

    # Bundle artifacts
    try:
        deploy_dir = bundle_deploy_artifacts(config, project_root=project_root)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error bundling artifacts: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    if args.dry_run:
        _handle_dry_run(deploy_dir, config)
        return

    # Run langgraph deploy
    _run_langgraph_deploy(
        deploy_dir,
        wait=args.wait,
        revision_id=getattr(args, "revision_id", None),
    )


def _apply_cli_overrides(config: DeployConfig, args: argparse.Namespace) -> DeployConfig:
    """Apply CLI flag overrides to the config.

    Creates a new DeployConfig with overridden fields. Uses object.__setattr__
    on the frozen dataclass since we're constructing the final config.
    """
    from deepagents_cli.deploy.config import (
        BackendConfig,
        DeployConfig,
        MemoryConfig,
        NamespaceConfig,
        SandboxConfig,
    )

    overrides: dict[str, Any] = {}

    if args.agent is not None:
        overrides["agent"] = args.agent
    if args.model is not None:
        overrides["model"] = args.model

    # Sandbox overrides
    if args.sandbox is not None:
        if args.sandbox == "none":
            overrides["sandbox"] = None
        else:
            sandbox_scope = args.sandbox_scope or (config.sandbox.scope if config.sandbox else "thread")
            overrides["sandbox"] = SandboxConfig(
                provider=args.sandbox,
                scope=sandbox_scope,
                template=config.sandbox.template if config.sandbox else None,
                image=config.sandbox.image if config.sandbox else None,
                setup_script=config.sandbox.setup_script if config.sandbox else None,
            )
    elif args.sandbox_scope is not None and config.sandbox is not None:
        overrides["sandbox"] = SandboxConfig(
            provider=config.sandbox.provider,
            scope=args.sandbox_scope,
            template=config.sandbox.template,
            image=config.sandbox.image,
            setup_script=config.sandbox.setup_script,
        )

    # Backend scope override
    if args.backend_scope is not None:
        overrides["backend"] = BackendConfig(
            type=config.backend.type,
            namespace=NamespaceConfig(
                scope=args.backend_scope,
                prefix=config.backend.namespace.prefix,
            ),
            path=config.backend.path,
        )

    # Memory scope override
    if args.memory_scope is not None:
        overrides["memory"] = MemoryConfig(
            scope=args.memory_scope,
            sources=config.memory.sources,
        )

    if not overrides:
        return config

    # Reconstruct with overrides
    from dataclasses import asdict

    config_dict = asdict(config)
    config_dict.update(overrides)

    # Need to reconstruct properly since asdict flattens nested dataclasses
    return DeployConfig(
        agent=overrides.get("agent", config.agent),
        description=config.description,
        model=overrides.get("model", config.model),
        model_params=config.model_params,
        prompt=config.prompt,
        memory=overrides.get("memory", config.memory),
        skills=config.skills,
        tools=config.tools,
        backend=overrides.get("backend", config.backend),
        sandbox=overrides["sandbox"] if "sandbox" in overrides else config.sandbox,
        env=config.env,
        python_version=config.python_version,
    )


def _print_config_summary(config: DeployConfig) -> None:
    """Print a summary of the deployment configuration."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(stderr=True)

        table = Table(title="Deploy Configuration", show_header=False, border_style="dim")
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Agent", config.agent)
        table.add_row("Model", config.model)
        table.add_row("Backend", f"{config.backend.type} (scope: {config.backend.namespace.scope})")
        if config.sandbox:
            table.add_row("Sandbox", f"{config.sandbox.provider} (scope: {config.sandbox.scope})")
        else:
            table.add_row("Sandbox", "disabled")
        table.add_row("Memory", f"scope: {config.memory.scope}, sources: {len(config.memory.sources)}")
        table.add_row("Skills", f"sources: {len(config.skills.sources)}")

        tools_enabled = []
        if config.tools.shell:
            tools_enabled.append("shell")
        if config.tools.web_search:
            tools_enabled.append("web_search")
        if config.tools.fetch_url:
            tools_enabled.append("fetch_url")
        if config.tools.http_request:
            tools_enabled.append("http_request")
        if config.tools.custom:
            tools_enabled.append(f"custom({config.tools.custom})")
        table.add_row("Tools", ", ".join(tools_enabled) if tools_enabled else "none")

        console.print(table)
    except ImportError:
        # Fallback without rich
        print(f"Deploying agent '{config.agent}' with model '{config.model}'")  # noqa: T201
        print(f"  Backend: {config.backend.type} (scope: {config.backend.namespace.scope})")  # noqa: T201
        if config.sandbox:
            print(f"  Sandbox: {config.sandbox.provider} (scope: {config.sandbox.scope})")  # noqa: T201


def _handle_dry_run(deploy_dir: Path, config: DeployConfig) -> None:
    """Handle --dry-run: print bundle contents and exit."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        from rich.tree import Tree

        console = Console(stderr=True)
        console.print(f"\n[bold]Deploy bundle:[/bold] {deploy_dir}\n")

        # Show directory tree
        tree = Tree(f"[bold]{deploy_dir.name}/[/bold]")
        for item in sorted(deploy_dir.rglob("*")):
            if item.is_file():
                rel = item.relative_to(deploy_dir)
                tree.add(str(rel))
        console.print(tree)

        # Show langgraph.json
        lg_json = deploy_dir / "langgraph.json"
        if lg_json.exists():
            console.print("\n[bold]langgraph.json:[/bold]")
            console.print(Syntax(lg_json.read_text(), "json", theme="monokai"))

        # Show deploy_config.json
        dc_json = deploy_dir / "deploy_config.json"
        if dc_json.exists():
            console.print("\n[bold]deploy_config.json:[/bold]")
            console.print(Syntax(dc_json.read_text(), "json", theme="monokai"))

    except ImportError:
        print(f"Deploy bundle: {deploy_dir}")  # noqa: T201
        lg_json = deploy_dir / "langgraph.json"
        if lg_json.exists():
            print(f"\nlanggraph.json:\n{lg_json.read_text()}")  # noqa: T201


def _run_langgraph_deploy(
    deploy_dir: Path,
    *,
    wait: bool = False,
    revision_id: str | None = None,
) -> None:
    """Shell out to ``langgraph deploy`` to push the bundle.

    Args:
        deploy_dir: Path to the deployment bundle directory.
        wait: Whether to wait for deployment to complete.
        revision_id: Optional revision identifier.
    """
    # Check that langgraph CLI is available
    langgraph_bin = shutil.which("langgraph")
    if langgraph_bin is None:
        # Try as a Python module
        try:
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "langgraph_cli", "--help"],
                capture_output=True,
                check=True,
            )
            cmd_prefix = [sys.executable, "-m", "langgraph_cli"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(  # noqa: T201
                "Error: langgraph CLI not found. Install with: pip install langgraph-cli",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        cmd_prefix = [langgraph_bin]

    # Build the deploy command
    cmd = [*cmd_prefix, "deploy", "--config", str(deploy_dir / "langgraph.json")]

    if wait:
        cmd.append("--wait")
    if revision_id:
        cmd.extend(["--revision-id", revision_id])

    try:
        from rich.console import Console

        console = Console(stderr=True)
        console.print(f"\n[bold]Running:[/bold] {' '.join(cmd)}\n")
    except ImportError:
        print(f"Running: {' '.join(cmd)}")  # noqa: T201

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(deploy_dir),
            check=False,
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nDeploy cancelled.", file=sys.stderr)  # noqa: T201
        sys.exit(130)


def _show_deploy_help() -> None:
    """Display help text for the deploy command."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console(stderr=True)
        help_text = """\
# deepagents deploy

Deploy your agent to LangGraph Cloud.

## Usage

```
deepagents deploy [options]
```

## Configuration

Create a `deepagents.json` file in your project root:

```json
{
  "agent": "agent",
  "model": "anthropic:claude-sonnet-4-6",
  "memory": {
    "scope": "user",
    "sources": [".deepagents/AGENTS.md"]
  },
  "skills": {
    "sources": [".deepagents/skills"]
  },
  "tools": {
    "shell": true,
    "web_search": true,
    "custom": "./my_tools.py:tools"
  },
  "backend": {
    "type": "store",
    "namespace": { "scope": "user", "prefix": "filesystem" }
  },
  "sandbox": {
    "provider": "langsmith",
    "scope": "thread"
  }
}
```

If no config file is found, defaults are used.

## Scoping

Each resource (backend, memory, sandbox) can be scoped independently:

- **assistant**: Shared across all users and threads
- **user**: Per-user, persists across threads
- **thread**: Per-conversation, isolated
- **user+thread**: Per-user per-conversation

## Options

- `-c, --config PATH`: Path to config file
- `--dry-run`: Show what would be deployed without deploying
- `--wait`: Wait for deployment to complete
- `-a, --agent NAME`: Override agent name
- `-M, --model MODEL`: Override model
- `--sandbox PROVIDER`: Override sandbox provider
- `--sandbox-scope SCOPE`: Override sandbox lifecycle scope
- `--backend-scope SCOPE`: Override backend namespace scope
- `--memory-scope SCOPE`: Override memory namespace scope
- `--revision-id ID`: Set deployment revision ID
"""
        console.print(Markdown(help_text))
    except ImportError:
        print("deepagents deploy - Deploy your agent to LangGraph Cloud")  # noqa: T201
        print("Usage: deepagents deploy [--config PATH] [--dry-run] [options]")  # noqa: T201
