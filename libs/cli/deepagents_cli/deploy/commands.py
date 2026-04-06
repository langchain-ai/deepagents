"""CLI commands for ``deepagents deploy``.

Registered with the CLI via :func:`setup_deploy_parser` in ``main.py``.

Commands:
- ``deepagents deploy`` — Bundle and deploy to LangGraph Platform
- ``deepagents deploy --dry-run`` — Show what would be generated
- ``deepagents deploy init`` — Generate starter ``deepagents.toml``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def setup_deploy_parser(
    subparsers: Any,  # noqa: ANN401  # argparse subparsers uses dynamic typing
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> argparse.ArgumentParser:
    """Setup the deploy subcommand parser.

    Args:
        subparsers: Parent subparsers object.
        make_help_action: Factory for help actions.

    Returns:
        The deploy subparser.
    """

    def _show_help() -> None:
        deploy_parser.print_help()

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Bundle and deploy agent to LangGraph Platform",
        add_help=False,
    )
    deploy_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(_show_help),
    )
    deploy_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deepagents.toml config file (default: ./deepagents.toml)",
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without deploying",
    )

    # Subcommands
    deploy_sub = deploy_parser.add_subparsers(dest="deploy_command")

    init_parser = deploy_sub.add_parser(
        "init",
        help="Generate a starter deepagents.toml",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing deepagents.toml",
    )

    return deploy_parser


def execute_deploy_command(args: argparse.Namespace) -> None:
    """Execute deploy subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        SystemExit: On validation errors or deployment failures.
    """
    if getattr(args, "deploy_command", None) == "init":
        _init_config(force=args.force)
        return

    # Default: bundle and deploy
    _deploy(
        config_path=args.config,
        dry_run=args.dry_run,
    )


def _init_config(*, force: bool = False) -> None:
    """Generate a starter ``deepagents.toml`` in the current directory.

    Args:
        force: Overwrite existing file if ``True``.
    """
    from deepagents_cli.deploy.config import DEFAULT_CONFIG_FILENAME, generate_starter_config

    target = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if target.exists() and not force:
        print(
            f"Error: {DEFAULT_CONFIG_FILENAME} already exists. "
            "Use --force to overwrite."
        )
        raise SystemExit(1)

    target.write_text(generate_starter_config())
    print(f"Created {DEFAULT_CONFIG_FILENAME}")
    print("Edit the config and run `deepagents deploy` to deploy your agent.")


def _deploy(
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Bundle and deploy the agent.

    Args:
        config_path: Path to config file, or ``None`` for default.
        dry_run: If ``True``, generate artifacts but don't deploy.
    """
    from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
    from deepagents_cli.deploy.config import DEFAULT_CONFIG_FILENAME, load_config

    # Resolve config path
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = Path.cwd() / DEFAULT_CONFIG_FILENAME

    project_root = cfg_path.parent

    # Load and validate config
    try:
        config = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {cfg_path}")
        print(f"Run `deepagents deploy init` to create a starter {DEFAULT_CONFIG_FILENAME}.")
        raise SystemExit(1)
    except ValueError as e:
        print(f"Error: Invalid config: {e}")
        raise SystemExit(1)

    errors = config.validate(project_root)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    # Bundle
    if dry_run:
        build_dir = Path(tempfile.mkdtemp(prefix="deepagents-deploy-"))
    else:
        build_dir = Path(tempfile.mkdtemp(prefix="deepagents-deploy-"))

    try:
        bundle(config, project_root, build_dir)
        print_bundle_summary(config, build_dir)

        if dry_run:
            print("Dry run — artifacts generated but not deployed.")
            print(f"Inspect the build directory: {build_dir}")
            return

        # Deploy via langgraph CLI
        _run_langgraph_deploy(build_dir, name=config.agent.name)

    except Exception:
        if dry_run:
            # Keep build dir for inspection on dry run
            raise
        raise


def _run_langgraph_deploy(build_dir: Path, *, name: str) -> None:
    """Shell out to ``langgraph deploy`` in the build directory.

    Args:
        build_dir: Directory containing generated deployment artifacts.
        name: Deployment name (passed as ``--name`` to avoid interactive prompt).

    Raises:
        SystemExit: If ``langgraph`` CLI is not installed or deployment fails.
    """
    import shutil

    if shutil.which("langgraph") is None:
        print(
            "Error: `langgraph` CLI not found. Install it with:\n"
            "  pip install 'langgraph-cli[inmem]'"
        )
        raise SystemExit(1)

    config_path = str(build_dir / "langgraph.json")
    cmd = ["langgraph", "deploy", "-c", config_path, "--name", name, "--verbose"]

    print("Deploying to LangGraph Platform...")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(build_dir), capture_output=True, text=True)

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"\nDeployment failed (exit code {result.returncode}).")
        raise SystemExit(result.returncode)

    print("\nDeployment complete!")
