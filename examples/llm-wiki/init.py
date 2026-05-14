"""Init-specific workflow for the LLM wiki."""

from __future__ import annotations

import json
from urllib.parse import quote

from models import CliDeps, RunResult, RunnerConfig
import helpers


def resolve_internal_source_flag(deps: CliDeps) -> tuple[str, ...]:
    """Resolve an init flag set that enforces internal repo source."""
    help_text = _hub_init_help_text(deps)
    return _resolve_internal_source_flag_from_help(help_text)


def _resolve_internal_source_flag_from_help(help_text: str) -> tuple[str, ...]:
    """Resolve source=internal flags from `hub init --help` output."""

    if "--repo-source" in help_text and "internal" in help_text:
        return ("--repo-source", "internal")
    if "--source" in help_text and "internal" in help_text:
        return ("--source", "internal")
    if "--internal" in help_text:
        return ("--internal",)

    # Newer Hub CLI builds may default to internal source and omit source flags.
    # We still validate source metadata when it is exposed by `hub get`.
    return ()


def _hub_init_help_text(deps: CliDeps) -> str:
    """Return lowercase `hub init --help` output."""
    help_result = deps.run_langsmith_cli(["hub", "init", "--help"])
    return f"{help_result.stdout or ''}\n{help_result.stderr or ''}".lower()


def _resolve_description_flag(
    help_text: str, description: str | None
) -> tuple[str, ...]:
    """Resolve an init description flag when supported."""
    if description is None:
        return ()
    if "--description" in help_text:
        return ("--description", description)
    if "--desc" in help_text:
        return ("--desc", description)
    return ()


def extract_repo_source(payload: dict[str, object]) -> str | None:
    """Extract repo source metadata from hub get payload."""
    direct_source = payload.get("source")
    if isinstance(direct_source, str):
        return direct_source

    repo_source = payload.get("repo_source")
    if isinstance(repo_source, str):
        return repo_source

    for key in ("repo", "data", "repository"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        nested_source = nested.get("source")
        if isinstance(nested_source, str):
            return nested_source

    return None


def _repo_api_path(owner: str | None, repo: str) -> str:
    """Build the repos API path for owner/repo lookup."""
    owner_segment = owner or "-"
    return f"/api/v1/repos/{quote(owner_segment, safe='')}/{quote(repo, safe='')}"


def _owner_repo_from_hub_identifier(hub_identifier: str) -> tuple[str | None, str]:
    """Parse a hub identifier into owner/repo components."""
    if hub_identifier.startswith("-/"):
        repo = hub_identifier[2:]
        if not repo:
            msg = f"Invalid hub identifier {hub_identifier!r}; missing repo handle."
            raise helpers.WikiError(msg)
        return None, repo

    owner, sep, repo = hub_identifier.partition("/")
    if sep == "" or not owner or not repo:
        msg = f"Invalid hub identifier {hub_identifier!r}; expected OWNER/REPO or -/REPO."
        raise helpers.WikiError(msg)
    return owner, repo


def ensure_internal_repo_default(config: RunnerConfig, deps: CliDeps) -> None:
    """Ensure the hub repo exists with `source=internal` before first push."""
    repo_path = _repo_api_path(config.owner, config.repo)
    try:
        result = deps.run_langsmith_cli(["api", repo_path, "--format", "json"])
    except helpers.WikiError as exc:
        if "404" not in str(exc):
            raise

        create_payload = json.dumps(
            {
                "repo_handle": config.repo,
                "repo_type": "agent",
                "is_public": False,
                "source": "internal",
            }
        )
        try:
            deps.run_langsmith_cli(
                [
                    "api",
                    "/api/v1/repos",
                    "-X",
                    "POST",
                    "--body",
                    create_payload,
                    "--format",
                    "json",
                ]
            )
        except helpers.WikiError as create_exc:
            if "409" in str(create_exc):
                return
            raise
        return

    payload = helpers._parse_cli_json_output(result)
    if payload is None:
        msg = "Unable to verify repo source from repos API response."
        raise helpers.WikiError(msg)

    source = extract_repo_source(payload)
    if source is None:
        msg = (
            "Wiki repo exists but has no source metadata. "
            "Delete and recreate it with `source=internal`."
        )
        raise helpers.WikiError(msg)

    if source.lower() != "internal":
        msg = (
            "Wiki repos must use `source=internal`. "
            f"Found source={source!r} for {helpers._hub_cli_repo_arg(helpers._hub_identifier(config.owner, config.repo))!r}."
        )
        raise helpers.WikiError(msg)


def verify_internal_repo_source(hub_identifier: str, deps: CliDeps) -> None:
    """Verify that the target hub repo source is internal."""
    result = deps.run_langsmith_cli(
        ["hub", "get", helpers._hub_cli_repo_arg(hub_identifier), "--format", "json"]
    )
    payload = helpers._parse_cli_json_output(result)
    if payload is None:
        msg = "Unable to verify repo source from `hub get --format json` output."
        raise helpers.WikiError(msg)

    source = extract_repo_source(payload)
    if source is None:
        owner, repo = _owner_repo_from_hub_identifier(hub_identifier)
        api_result = deps.run_langsmith_cli(
            ["api", _repo_api_path(owner, repo), "--format", "json"]
        )
        api_payload = helpers._parse_cli_json_output(api_result)
        if api_payload is None:
            msg = (
                "Unable to verify repo source from `hub get --format json` output. "
                "Fallback repos API response was not valid JSON."
            )
            raise helpers.WikiError(msg)
        source = extract_repo_source(api_payload)
        if source is None:
            msg = (
                "Unable to verify repo source from `hub get --format json` output. "
                "Fallback repos API response also lacked source metadata."
            )
            raise helpers.WikiError(msg)

    if source.lower() != "internal":
        msg = (
            "Wiki repos must use `source=internal`. "
            f"Found source={source!r} for {helpers._hub_cli_repo_arg(hub_identifier)!r}."
        )
        raise helpers.WikiError(msg)


def run_init(config: RunnerConfig, deps: CliDeps) -> RunResult:
    """Initialize a local topic repo and push its first hub revision."""
    config.topic_dir.mkdir(parents=True, exist_ok=True)
    helpers._ensure_no_symlinks(config.topic_dir)
    helpers._ensure_scaffold(config.topic_dir, config.topic)

    hub_identifier = helpers._hub_identifier(config.owner, config.repo)
    ensure_internal_repo_default(config, deps)
    help_text = _hub_init_help_text(deps)
    source_flags = _resolve_internal_source_flag_from_help(help_text)
    description_flags = _resolve_description_flag(help_text, config.description)

    init_args = [
        "hub",
        "init",
        "--type",
        "agent",
        "--dir",
        str(config.topic_dir),
        "--name",
        config.repo,
        "--force",
    ]
    init_args.extend(source_flags)
    init_args.extend(description_flags)
    deps.run_langsmith_cli(init_args)

    helpers._ensure_scaffold(config.topic_dir, config.topic)
    helpers._validate_text_only_directory(config.topic_dir)

    deps.run_langsmith_cli(
        [
            "hub",
            "push",
            helpers._hub_cli_repo_arg(hub_identifier),
            "--type",
            "agent",
            "--dir",
            str(config.topic_dir),
        ]
    )
    verify_internal_repo_source(hub_identifier, deps)
    hub_url = helpers._resolve_hub_url(config.owner, config.repo)
    return RunResult(answer=None, hub_url=hub_url)
