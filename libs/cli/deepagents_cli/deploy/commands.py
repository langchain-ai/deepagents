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


def setup_deploy_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the top-level ``init``, ``dev``, and ``deploy`` subparsers.

    The three commands used to live under ``deepagents deploy {init,dev}``
    but are now flat: ``deepagents init``, ``deepagents dev``, and
    ``deepagents deploy``. This function registers all three on the root
    subparsers object.
    """
    # deepagents init
    init_parser = subparsers.add_parser(
        "init",
        help="Generate a starter deepagents.toml in the current directory",
        add_help=False,
    )
    init_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: init_parser.print_help()),
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing deepagents.toml",
    )

    # deepagents dev
    dev_parser = subparsers.add_parser(
        "dev",
        help="Bundle and run a local langgraph dev server",
        add_help=False,
    )
    dev_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: dev_parser.print_help()),
    )
    dev_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deepagents.toml (default: ./deepagents.toml)",
    )
    dev_parser.add_argument(
        "--port",
        type=int,
        default=2024,
        help="Port for the langgraph dev server (default: 2024)",
    )
    dev_parser.add_argument(
        "--allow-blocking",
        action="store_true",
        default=True,
        help="Pass --allow-blocking to langgraph dev (default: enabled)",
    )

    # deepagents deploy
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Bundle and deploy agent to LangGraph Platform",
        add_help=False,
    )
    deploy_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: deploy_parser.print_help()),
    )
    deploy_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deepagents.toml (default: ./deepagents.toml)",
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without deploying",
    )


def execute_init_command(args: argparse.Namespace) -> None:
    _init_config(force=args.force)


def execute_dev_command(args: argparse.Namespace) -> None:
    _dev(
        config_path=args.config,
        port=args.port,
        allow_blocking=args.allow_blocking,
    )


def execute_deploy_command(args: argparse.Namespace) -> None:
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


def _dev(
    *,
    config_path: str | None,
    port: int,
    allow_blocking: bool,
) -> None:
    """Bundle the project and run a local ``langgraph dev`` server.

    The bundle is identical to what ``deepagents deploy`` would ship, just
    served locally instead of pushed to LangGraph Platform. Hot-reloading
    is provided by ``langgraph dev`` itself watching the build directory;
    edits to the source project (``deepagents.toml``, skills, AGENTS.md)
    require re-running ``deepagents deploy dev`` to re-bundle.

    Args:
        config_path: Path to ``deepagents.toml``, or ``None`` for default.
        port: Local port for the dev server.
        allow_blocking: Pass ``--allow-blocking`` to ``langgraph dev`` so
            sync HTTP calls inside the graph (e.g. the LangSmith sandbox
            client) don't trigger blockbuster errors.
    """
    import shutil

    from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
    from deepagents_cli.deploy.config import DEFAULT_CONFIG_FILENAME, load_config

    cfg_path = Path(config_path) if config_path else Path.cwd() / DEFAULT_CONFIG_FILENAME
    project_root = cfg_path.parent

    try:
        config = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {cfg_path}")
        raise SystemExit(1) from None

    errors = config.validate(project_root)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    build_dir = Path(tempfile.mkdtemp(prefix="deepagents-dev-"))
    bundle(config, project_root, build_dir)
    print_bundle_summary(config, build_dir)

    if shutil.which("langgraph") is None:
        print(
            "Error: `langgraph` CLI not found. Install it with:\n"
            "  pip install 'langgraph-cli[inmem]'"
        )
        raise SystemExit(1)

    cmd = [
        "langgraph",
        "dev",
        "--no-browser",
        "--port",
        str(port),
    ]
    if allow_blocking:
        cmd.append("--allow-blocking")

    print(f"\nStarting langgraph dev on http://localhost:{port}")
    print(f"Build directory: {build_dir}")
    print(f"Running: {' '.join(cmd)}\n")

    # Pass through stdout/stderr so the user sees the dev server logs live.
    try:
        subprocess.run(cmd, cwd=str(build_dir), check=False)
    except KeyboardInterrupt:
        print("\nShutting down.")


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
