"""CLI commands for `deepagents init`, `dev`, and `deploy`.

Registered with the CLI via `setup_deploy_parsers` in `main.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents_cli.deploy.config import DeployConfig


def setup_deploy_parsers(
    subparsers: Any,  # noqa: ANN401
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> None:
    """Register the top-level `init`, `dev`, and `deploy` subparsers.

    The three commands used to live under `deepagents deploy {init,dev}`
    but are now flat: `deepagents init`, `deepagents dev`, and
    `deepagents deploy`. This function registers all three on the root
    subparsers object.
    """
    # deepagents init
    init_parser = subparsers.add_parser(
        "init",
        help="(beta) Scaffold a new deploy project folder",
        add_help=False,
    )
    init_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Project folder name (will be created in cwd). Prompted if omitted.",
    )
    init_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: init_parser.print_help()),
        help="show this help message and exit",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    # deepagents dev
    dev_parser = subparsers.add_parser(
        "dev",
        help="(beta) Bundle and run a local langgraph dev server",
        add_help=False,
    )
    dev_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: dev_parser.print_help()),
        help="show this help message and exit",
    )
    dev_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deepagents.toml (default: auto-discovered from cwd)",
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
        help="(beta) Bundle and deploy agent to LangGraph Platform",
        add_help=False,
    )
    deploy_parser.add_argument(
        "-h",
        "--help",
        action=make_help_action(lambda: deploy_parser.print_help()),
        help="show this help message and exit",
    )
    deploy_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to deepagents.toml (default: auto-discovered from cwd)",
    )
    deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without deploying",
    )


_BETA_WARNING = (
    "\033[33mWarning: `deepagents deploy` is in beta. "
    "APIs, configuration format, and behavior may change between releases.\033[0m\n"
)


def execute_init_command(args: argparse.Namespace) -> None:
    """Execute the `deepagents init` command."""
    print(_BETA_WARNING)
    name = args.name
    if name is None:
        try:
            name = input("Project name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1) from None
        if not name:
            print("Error: project name is required.")
            raise SystemExit(1)
    _init_project(name=name, force=args.force)


def execute_dev_command(args: argparse.Namespace) -> None:
    """Execute the `deepagents dev` command."""
    print(_BETA_WARNING)
    _dev(
        config_path=args.config,
        port=args.port,
        allow_blocking=args.allow_blocking,
    )


def execute_deploy_command(args: argparse.Namespace) -> None:
    """Execute the `deepagents deploy` command."""
    print(_BETA_WARNING)
    _deploy(
        config_path=args.config,
        dry_run=args.dry_run,
    )


def _init_project(*, name: str, force: bool = False) -> None:
    """Scaffold a deploy project folder.

    Creates `name/` with the canonical layout:

    ```txt
    <name>/
        deepagents.toml
        AGENTS.md
        .env
        mcp.json
        skills/
    ```

    Args:
        name: Project folder name (created under cwd).
        force: Overwrite existing files if `True`.
    """
    from deepagents_cli.deploy.config import (
        AGENTS_MD_FILENAME,
        DEFAULT_CONFIG_FILENAME,
        MCP_FILENAME,
        SKILLS_DIRNAME,
        STARTER_SKILL_NAME,
        generate_starter_agents_md,
        generate_starter_config,
        generate_starter_env,
        generate_starter_mcp_json,
        generate_starter_skill_md,
    )

    project_dir = Path.cwd() / name

    if project_dir.exists() and not force:
        print(f"Error: {name}/ already exists. Use --force to overwrite.")
        raise SystemExit(1)

    project_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, str]] = [
        (DEFAULT_CONFIG_FILENAME, generate_starter_config()),
        (AGENTS_MD_FILENAME, generate_starter_agents_md()),
        (".env", generate_starter_env()),
        (MCP_FILENAME, generate_starter_mcp_json()),
    ]

    for filename, content in files:
        (project_dir / filename).write_text(content, encoding="utf-8")

    # Create skills/ directory with a starter skill.
    skills_dir = project_dir / SKILLS_DIRNAME
    skills_dir.mkdir(exist_ok=True)
    starter_skill_dir = skills_dir / STARTER_SKILL_NAME
    starter_skill_dir.mkdir(exist_ok=True)
    (starter_skill_dir / "SKILL.md").write_text(
        generate_starter_skill_md(), encoding="utf-8"
    )

    print(f"Created {name}/ with:")
    for filename, _ in files:
        print(f"  {filename}")
    print(f"  {SKILLS_DIRNAME}/")
    print(f"    {STARTER_SKILL_NAME}/SKILL.md")
    print("\nNext steps:")
    print(f"  cd {name}")
    print("  # edit deepagents.toml and AGENTS.md")
    print("  deepagents deploy")


def _deploy(
    config_path: str | None = None,
    dry_run: bool = False,
) -> None:
    """Bundle and deploy the agent.

    Args:
        config_path: Path to config file, or `None` for default.
        dry_run: If `True`, generate artifacts but don't deploy.
    """
    from deepagents_cli.config import _load_dotenv
    from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
    from deepagents_cli.deploy.config import (
        DEFAULT_CONFIG_FILENAME,
        find_config,
        load_config,
    )

    # Resolve config path: explicit flag > auto-discovery > cwd fallback
    if config_path:
        cfg_path = Path(config_path)
    else:
        discovered = find_config()
        cfg_path = discovered or Path.cwd() / DEFAULT_CONFIG_FILENAME

    project_root = cfg_path.parent
    # Ensure the project .env is loaded into os.environ before validation.
    # The main CLI bootstrap loads .env lazily (on first `settings` access),
    # but deploy/dev commands may never touch `settings`, so the project
    # .env would be missing when _validate_model_credentials checks os.environ.
    _load_dotenv(start_path=project_root)

    # Load and validate config
    try:
        config = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {cfg_path}")
        print(f"Run `deepagents init` to create a starter {DEFAULT_CONFIG_FILENAME}.")
        raise SystemExit(1) from None
    except ValueError as e:
        print(f"Error: Invalid config: {e}")
        raise SystemExit(1) from None

    errors = config.validate(project_root)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    # Warn + confirm if the frontend was set to anonymous mode. The
    # generated auth.py for this case is permissive (overrides the
    # LangSmith default x-api-key requirement) so the API is reachable
    # by anyone with the deploy URL.
    is_anonymous = (
        config.frontend is not None
        and config.frontend.enabled
        and config.auth is not None
        and config.auth.provider == "anonymous"
    )
    if is_anonymous:
        # ANSI bold-red header + red bullets so this warning is visually
        # distinct from the yellow beta-warning that prints above it
        # (otherwise the "Continue? [y/N]" prompt looks like it's
        # confirming the beta line rather than the anonymous-auth line).
        print(
            "\033[1;31m⚠ This deploy will use ANONYMOUS auth. "
            "The API is open to anyone with the deploy URL.\033[0m"
        )
        print(
            "\033[31m  • Browser UI shows per-browser threads "
            "(cookie-scoped UX, not security).\033[0m"
        )
        print(
            "\033[31m  • Anyone with the URL can call the API directly "
            "(curl /threads, /runs, etc.) — no auth.\033[0m"
        )
        print(
            "\033[31m  • For real per-user auth, add an [auth] section "
            '(provider = "supabase" or "clerk").\033[0m'
        )
        # Skip the interactive confirm on dry-run (no real push happens).
        if not dry_run:
            try:
                answer = (
                    input("Continue with anonymous deploy? [y/N]: ").strip().lower()
                )
            except (EOFError, KeyboardInterrupt):
                print()
                print("Aborted.")
                raise SystemExit(1) from None
            if answer not in {"y", "yes"}:
                print("Aborted.")
                raise SystemExit(1)

    # Bundle
    build_dir = Path(tempfile.mkdtemp(prefix="deepagents-deploy-"))

    try:
        bundle(config, project_root, build_dir)
        print_bundle_summary(config, build_dir)

        if dry_run:
            print("Dry run — artifacts generated but not deployed.")
            print(f"Inspect the build directory: {build_dir}")
            return

        _seed_hub_repo(config, build_dir)

        # Deploy via langgraph CLI.
        _run_langgraph_deploy(build_dir, name=config.agent.name)
        _auto_wire_issues_board_if_hub(config)
    finally:
        if not dry_run:
            import shutil

            shutil.rmtree(build_dir, ignore_errors=True)


def _dev(
    *,
    config_path: str | None,
    port: int,
    allow_blocking: bool,
) -> None:
    """Bundle the project and run a local `langgraph dev` server.

    The bundle is identical to what `deepagents deploy` would ship, just
    served locally instead of pushed to LangGraph Platform. Hot-reloading
    is provided by `langgraph dev` itself watching the build directory;
    edits to the source project (`deepagents.toml`, skills, `AGENTS.md`)
    require re-running `deepagents dev` to re-bundle.

    Args:
        config_path: Path to `deepagents.toml`, or `None` for default.
        port: Local port for the dev server.
        allow_blocking: Pass `--allow-blocking` to `langgraph dev` so
            sync HTTP calls inside the graph (e.g. the LangSmith sandbox
            client) don't trigger blockbuster errors.
    """
    import shutil

    from deepagents_cli.config import _load_dotenv
    from deepagents_cli.deploy.bundler import bundle, print_bundle_summary
    from deepagents_cli.deploy.config import (
        DEFAULT_CONFIG_FILENAME,
        find_config,
        load_config,
    )

    if config_path:
        cfg_path = Path(config_path)
    else:
        discovered = find_config()
        cfg_path = discovered or Path.cwd() / DEFAULT_CONFIG_FILENAME
    project_root = cfg_path.parent
    # Ensure the project .env is loaded before validation (see _deploy).
    _load_dotenv(start_path=project_root)

    try:
        config = load_config(cfg_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {cfg_path}")
        raise SystemExit(1) from None
    except ValueError as e:
        print(f"Error: Invalid config: {e}")
        raise SystemExit(1) from None

    errors = config.validate(project_root)
    if errors:
        print("Config validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    build_dir = Path(tempfile.mkdtemp(prefix="deepagents-dev-"))
    try:
        bundle(config, project_root, build_dir)
        print_bundle_summary(config, build_dir)

        _seed_hub_repo(config, build_dir)

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

        # Pass through stdout/stderr so the user sees dev server logs live.
        try:
            result = subprocess.run(cmd, cwd=str(build_dir), check=False)
        except KeyboardInterrupt:
            print("\nShutting down.")
            return

        if result.returncode != 0:
            print(f"\nDev server exited with error (exit code {result.returncode}).")
            raise SystemExit(result.returncode)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def _seed_hub_repo(config: DeployConfig, build_dir: Path) -> None:
    """Eagerly create the LangSmith Hub agent repo at bundle time.

    Mirrors the per-(process, assistant_id) seeding that the generated
    `deploy_graph.py` performs on first invocation, but runs it from the
    CLI so the repo exists in LangSmith Hub the moment `deepagents deploy`
    (or `deepagents dev`) returns from bundling. The runtime seed path
    stays in place as a defensive no-op: it short-circuits via
    `has_prior_commits()` once this seed has run.

    Per-user hub repos (`{identifier}-user-{slug}`) are intentionally
    *not* seeded here — user identities aren't known until authenticated
    requests arrive at runtime.

    Args:
        config: Loaded `DeployConfig`. Only invoked for hub-backed deploys;
            no-op otherwise.
        build_dir: Directory containing `_seed.json` written by `bundle()`.

    Raises:
        SystemExit: If the hub commit fails or returns per-file errors —
            fail fast at bundle time rather than at first invocation.
    """
    import json

    from deepagents_cli.deploy.context_hub import ContextHubBackend

    if config.memories.backend != "hub":
        return

    seed_path = build_dir / "_seed.json"
    try:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: failed to read {seed_path.name} for hub seed: {exc}")
        raise SystemExit(1) from None

    # Hub-repo layout matches what the runtime ends up writing: when the
    # generated graph calls `CompositeBackend.aupload_files` with paths
    # like `/memories/AGENTS.md`, the composite strips the `/memories/`
    # route prefix before delegating to `ContextHubBackend`. We call the
    # hub backend directly here, so emit the post-strip paths directly.
    batch: list[tuple[str, bytes]] = []
    for path, content in seed.get("memories", {}).items():
        batch.append((path.lstrip("/"), content.encode("utf-8")))
    for path, content in seed.get("skills", {}).items():
        batch.append((f"skills/{path.lstrip('/')}", content.encode("utf-8")))
    for sa_name, sa_data in seed.get("subagents", {}).items():
        sa_prefix = f"subagents/{sa_name}/"
        for path, content in sa_data.get("memories", {}).items():
            batch.append((f"{sa_prefix}{path.lstrip('/')}", content.encode("utf-8")))
        for path, content in sa_data.get("skills", {}).items():
            batch.append(
                (f"{sa_prefix}skills/{path.lstrip('/')}", content.encode("utf-8"))
            )

    identifier = config.memories.identifier or f"-/{config.agent.name}"
    backend = ContextHubBackend(identifier=identifier)

    try:
        if backend.has_prior_commits():
            print(f"Hub repo {identifier} already exists — skipping seed.")
            return
    except Exception as exc:
        print(f"Error: failed to inspect hub repo {identifier}: {exc}")
        raise SystemExit(1) from None

    if not batch:
        print(f"Hub repo {identifier}: nothing to seed.")
        return

    print(f"Creating hub repo {identifier} with {len(batch)} file(s)...")
    try:
        responses = backend.upload_files(batch)
    except Exception as exc:
        print(f"Error: hub seed failed for {identifier}: {exc}")
        raise SystemExit(1) from None

    failures = [r for r in responses if r.error is not None]
    if failures:
        print(f"Error: hub seed had {len(failures)} failed file(s):")
        for resp in failures:
            print(f"  - {resp.path}: {resp.error}")
        raise SystemExit(1)

    print(f"Hub repo {identifier} created.")


def _run_langgraph_deploy(build_dir: Path, *, name: str) -> None:
    """Shell out to `langgraph deploy` in the build directory.

    Args:
        build_dir: Directory containing generated deployment artifacts.
        name: Deployment name (passed as `--name` to avoid interactive prompt).

    Raises:
        SystemExit: If `langgraph` CLI is not installed or deployment fails.
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
    env = os.environ.copy()
    env["LANGGRAPH_CLI_ANALYTICS_SOURCE"] = "deepagents"

    print("Deploying to LangSmith Deployments...")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(build_dir), check=False, env=env)

    if result.returncode != 0:
        print(f"\nDeployment failed (exit code {result.returncode}).")
        raise SystemExit(result.returncode)

    print("\nDeployment complete!")


def _resolve_langsmith_api_key() -> str | None:
    from deepagents_cli.model_config import resolve_env_var

    return resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var("LANGCHAIN_API_KEY")


def _resolve_langsmith_endpoint() -> str:
    from deepagents_cli.model_config import resolve_env_var

    return (
        resolve_env_var("LANGSMITH_ENDPOINT")
        or resolve_env_var("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    ).rstrip("/")


def _resolve_tracer_session_id_by_project_name(
    *, project_name: str, api_key: str
) -> str | None:
    """Resolve tracing project id (session id) by name."""
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    api_url = _resolve_langsmith_endpoint()
    try:
        client = Client(api_url=api_url, api_key=api_key)
        project = client.read_project(project_name=project_name)
        return str(project.id)
    except LangSmithNotFoundError as exc:
        print(f"Warning: Failed to resolve tracing project '{project_name}': {exc}")
    except Exception as exc:
        print(f"Warning: Failed to resolve tracing project '{project_name}': {exc}")
    return None


def _upsert_issues_board_config(
    *,
    session_id: str,
    api_key: str,
    context_hub_repo_handle: str,
) -> None:
    """Best-effort create or patch board config for a deployed agent."""
    import httpx

    from deepagents_cli.model_config import resolve_env_var

    success_codes = {200, 201}
    http_conflict = 409
    endpoint = _resolve_langsmith_endpoint()
    url = f"{endpoint}/v1/platform/sessions/{session_id}/issues-agent"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    tenant_id = resolve_env_var("LANGSMITH_TENANT_ID")
    if tenant_id:
        headers["x-tenant-id"] = tenant_id

    create_payload = {
        "cron_schedule": "0 */6 * * *",
        "heavy_model": "anthropic:issues-agent-heavy",
        "light_model": "anthropic:issues-agent-light",
        "context_hub_repo_handle": context_hub_repo_handle,
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            create_resp = client.post(url, headers=headers, json=create_payload)
            if create_resp.status_code in success_codes:
                print(
                    f"Issues board auto-wired for tracing project {session_id} "
                    f"({context_hub_repo_handle})."
                )
                return
            if create_resp.status_code == http_conflict:
                patch_resp = client.patch(
                    url,
                    headers=headers,
                    json={"context_hub_repo_handle": context_hub_repo_handle},
                )
                if patch_resp.status_code in success_codes:
                    print(
                        "Issues board already existed; updated context hub id "
                        f"to {context_hub_repo_handle}."
                    )
                    return
                print(
                    "Warning: Failed to patch existing issues board: "
                    f"HTTP {patch_resp.status_code} — {patch_resp.text[:300]}"
                )
                return
            print(
                "Warning: Failed to create issues board config: "
                f"HTTP {create_resp.status_code} — {create_resp.text[:300]}"
            )
    except Exception as exc:
        print(f"Warning: Issues board auto-wire failed: {exc}")


def _resolve_context_hub_identifier(config: DeployConfig) -> str:
    return config.memories.identifier or f"-/{config.agent.name}"


def _resolve_context_hub_repo_handle_for_issues_board(
    config: DeployConfig,
) -> str | None:
    identifier = _resolve_context_hub_identifier(config)
    owner, sep, repo_handle = identifier.partition("/")
    if not sep or not owner or not repo_handle:
        print(
            "Warning: Invalid memories identifier for hub-backed deploy: "
            f"{identifier!r}; skipping issues board auto-wire."
        )
        return None
    return repo_handle


def _auto_wire_issues_board_if_hub(config: DeployConfig) -> None:
    """Best-effort issues-board wiring after deploy for hub-backed memories."""
    if config.memories.backend != "hub":
        return

    context_hub_repo_handle = _resolve_context_hub_repo_handle_for_issues_board(config)
    if context_hub_repo_handle is None:
        return
    api_key = _resolve_langsmith_api_key()
    if api_key is None:
        print(
            "Warning: LANGSMITH/LANGCHAIN API key not found; "
            "skipping issues board auto-wire."
        )
        return

    session_id = _resolve_tracer_session_id_by_project_name(
        project_name=config.agent.name,
        api_key=api_key,
    )
    if session_id is None:
        print(
            "Warning: Could not resolve tracing project after deploy; "
            "skipping issues board auto-wire."
        )
        return

    _upsert_issues_board_config(
        session_id=session_id,
        api_key=api_key,
        context_hub_repo_handle=context_hub_repo_handle,
    )
