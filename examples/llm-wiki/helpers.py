"""Helper utilities for the LLM wiki example."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LangSmithSandbox
from deepagents.middleware.filesystem import FilesystemPermission
import index as index_helpers
import log as log_helpers
from models import CliDeps, Mode, RunResult, RunnerConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from deepagents.backends.protocol import SandboxBackendProtocol
    from ingest import IngestResult


_ALLOWED_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}
_DEFAULT_SNAPSHOT_NAME = "deepagents-wiki"
_DEFAULT_DOCKER_IMAGE = "python:3"
_DEFAULT_FS_CAPACITY = 16 * 1024**3
_LANGSMITH_BINARY_CANDIDATES = ("langsmith",)
_HUB_COMPATIBLE_BINARIES: set[str] = set()
_BASE_SYSTEM_PROMPT = """You are an expert research synthesizer building a long-lived topic knowledge base.

Mission:
- Build an accurate, high-signal, source-grounded topic corpus in `/wiki/`.
- Treat `/raw/` as immutable evidence inputs.
- Convert raw notes into canonical, reusable understanding.

Reasoning style:
- Read primary source material before writing.
- Distinguish facts from inferences.
- Prefer compression-by-structure over compression-by-omission.
- Keep uncertainty explicit.
- Resolve contradictions when possible; otherwise record both claims and state what is unresolved.

Writing and organization rules:
- Maintain canonical pages per concept/entity/theme rather than many overlapping fragments.
- Keep pages scannable with clear headings.
- Include concise "What changed" summaries in your responses for runner-managed logging.
- Keep `/wiki/index.md` authoritative for navigation.
- Use recent `/log.md` entries as operational recency context before major synthesis.

Evidence rules:
- Every non-trivial claim should be traceable to the ingested source set.
- Avoid introducing unsupported external facts.
- If evidence is weak or missing, say so directly.

Filesystem policy:
- Never write to `/raw/`.
- Never edit `/log.md`; the runner maintains append-only interaction entries.
- Write only under `/wiki/`.
"""


class WikiError(RuntimeError):
    """Raised when the LLM wiki cannot complete a requested operation."""

def _slugify_topic(topic: str) -> str:
    """Convert a topic label into a stable slug."""
    slug_chars: list[str] = []
    last_dash = False
    for char in topic.strip().lower():
        if char.isalnum():
            slug_chars.append(char)
            last_dash = False
            continue
        if not last_dash:
            slug_chars.append("-")
            last_dash = True
    slug = "".join(slug_chars).strip("-")
    return slug or "topic"


def _topic_dir_for(topic: str, explicit: str | None) -> Path:
    """Resolve the local wiki directory path."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.cwd() / "wikis" / _slugify_topic(topic)).resolve()


def _default_topic_from_repo(repo: str) -> str:
    """Create a display topic from a repo name."""
    return repo.replace("-", " ").replace("_", " ").strip().title() or repo


def _normalize_repo_and_owner(
    parser: argparse.ArgumentParser, repo: str, owner: str | None
) -> tuple[str, str | None]:
    """Normalize repo and owner arguments into canonical pieces."""
    candidate_repo = repo.strip()
    candidate_owner = owner.strip() if owner is not None else None

    if not candidate_repo:
        parser.error("--repo must be non-empty")

    if "/" in candidate_repo:
        parsed_owner, sep, parsed_repo = candidate_repo.partition("/")
        if sep == "" or not parsed_owner or not parsed_repo or "/" in parsed_repo:
            parser.error("--repo must be REPO or OWNER/REPO")
        if candidate_owner and candidate_owner != parsed_owner:
            parser.error("--owner must match owner in --repo when both are provided")
        candidate_owner = parsed_owner
        candidate_repo = parsed_repo

    if "/" in candidate_repo:
        parser.error("--repo must not contain additional '/' segments")

    if candidate_owner == "":
        parser.error("--owner must be non-empty when provided")

    return candidate_repo, candidate_owner


def _hub_identifier(owner: str | None, repo: str) -> str:
    """Build a canonical hub identifier string."""
    if owner:
        return f"{owner}/{repo}"
    return f"-/{repo}"


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM wiki (Deep Agents + LangSmith Hub CLI)"
    )
    parser.add_argument(
        "--mode", required=True, choices=["init", "ingest", "query", "lint"]
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Context Hub repo name or owner/name handle",
    )
    parser.add_argument(
        "--owner",
        default=None,
        help="Optional Context Hub owner when --repo is only a repo name",
    )
    parser.add_argument(
        "--topic-dir", default=None, help="Local wiki directory for init mode"
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file or directory for ingest mode (repeatable)",
    )
    parser.add_argument(
        "--note", default=None, help="Optional note to include in ingest/lint prompt"
    )
    parser.add_argument(
        "--question", default=None, help="Question to answer in query mode"
    )
    parser.add_argument(
        "--model", default=None, help="Optional model override for create_deep_agent"
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional hub repo description to set during init (if supported by CLI)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Opt in to ingest review/confirmation before applying wiki updates",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> RunnerConfig:
    """Parse CLI arguments into a runner config."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    mode = args.mode
    if mode == "ingest" and not args.source:
        parser.error("--source is required in ingest mode")
    if mode == "query" and not args.question:
        parser.error("--question is required in query mode")

    repo, owner = _normalize_repo_and_owner(parser, args.repo, args.owner)
    topic = _default_topic_from_repo(repo)

    return RunnerConfig(
        mode=mode,
        topic=topic,
        repo=repo,
        owner=owner,
        topic_dir=_topic_dir_for(topic, args.topic_dir),
        sources=tuple(Path(source).expanduser().resolve() for source in args.source),
        note=args.note,
        question=args.question,
        model=args.model,
        description=args.description,
        review=bool(args.review),
    )


def _resolve_langsmith_binary() -> str:
    """Find an installed LangSmith CLI binary."""
    for candidate in _LANGSMITH_BINARY_CANDIDATES:
        binary = shutil.which(candidate)
        if binary:
            return binary
    msg = (
        "LangSmith CLI was not found on PATH. Install `langsmith` before running "
        "wiki sync."
    )
    raise WikiError(msg)


def _ensure_hub_command_support(binary: str) -> None:
    """Validate that an installed LangSmith CLI provides `hub` commands."""
    if binary in _HUB_COMPATIBLE_BINARIES:
        return

    check = subprocess.run(  # noqa: S603
        [binary, "hub", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        _HUB_COMPATIBLE_BINARIES.add(binary)
        return

    output = (check.stderr or check.stdout).strip()
    cmd = Path(binary).name
    msg = (
        f"`{cmd}` is installed but does not support `hub` commands required by this example. "
        f"Verify with `{cmd} hub --help` and install a hub-capable LangSmith CLI.\n{output}"
    )
    raise WikiError(msg)


def _ensure_mode_prerequisites(mode: Mode) -> None:
    """Validate mode-specific environment prerequisites."""
    if mode in {"ingest", "query", "lint"} and not os.getenv("LANGSMITH_API_KEY"):
        msg = (
            "LANGSMITH_API_KEY is required for ingest/query/lint modes because they run agent "
            "operations inside `langsmith.sandbox`."
        )
        raise WikiError(msg)


def _run_langsmith_cli(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Execute a langsmith CLI command and raise on failures."""
    binary = _resolve_langsmith_binary()
    _ensure_hub_command_support(binary)
    cmd = Path(binary).name

    result = subprocess.run(
        [binary, *args], capture_output=True, text=True, check=False
    )  # noqa: S603
    if result.returncode == 0:
        return result

    output = (result.stderr or result.stdout).strip()
    if "LANGSMITH_API_KEY" in output or "unauthorized" in output.lower():
        msg = (
            "LangSmith authentication failed. Set LANGSMITH_API_KEY and confirm CLI auth. "
            f"Command: {cmd} {' '.join(args)}\n{output}"
        )
        raise WikiError(msg)

    msg = f"{cmd} {' '.join(args)} failed with exit code {result.returncode}:\n{output}"
    raise WikiError(msg)


def _parse_cli_json_output(
    result: subprocess.CompletedProcess[str],
) -> dict[str, object] | None:
    """Parse JSON stdout from a langsmith CLI response."""
    stdout = (result.stdout or "").strip()
    if not stdout:
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _app_base_url() -> str:
    """Compute the LangSmith app base URL from endpoint environment variables."""
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "https"
    host = parsed.netloc or parsed.path
    if host.startswith("api."):
        host = host[4:]
    return f"{scheme}://{host}"


def _resolve_hub_url(owner: str | None, repo: str) -> str:
    """Resolve a browser URL for the hub repo."""
    base = _app_base_url()
    if owner:
        return f"{base}/hub/{owner}/{repo}"
    return f"{base}/hub/{repo}"


def _hub_cli_repo_arg(hub_identifier: str) -> str:
    """Normalize hub id values for cobra-based CLI parsing."""
    if hub_identifier.startswith("-/"):
        return hub_identifier[2:]
    return hub_identifier


def _iter_tree_paths(root_dir: Path) -> Iterator[Path]:
    """Yield all paths rooted under a workspace directory."""
    yield root_dir
    for current_root, dirnames, filenames in os.walk(
        root_dir, topdown=True, followlinks=False
    ):
        parent = Path(current_root)
        for dirname in dirnames:
            yield parent / dirname
        for filename in filenames:
            yield parent / filename


def _ensure_no_symlinks(root_dir: Path) -> None:
    """Reject workspace trees that contain symlinks."""
    for path in _iter_tree_paths(root_dir):
        if not path.is_symlink():
            continue
        with suppress(ValueError):
            relative = path.relative_to(root_dir)
            msg = (
                "Symlinks are not supported in wiki workspaces for security reasons: "
                f"{relative}"
            )
            raise WikiError(msg)
        msg = f"Symlinks are not supported in wiki workspaces for security reasons: {path}"
        raise WikiError(msg)


def _safe_write_text(path: Path, content: str, *, append: bool = False) -> None:
    """Write UTF-8 text while refusing symlink targets."""
    if path.is_symlink():
        msg = f"Refusing to write to symlink path: {path}"
        raise WikiError(msg)

    flags = os.O_WRONLY | os.O_CREAT
    if append:
        flags |= os.O_APPEND
    else:
        flags |= os.O_TRUNC

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow

    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            msg = f"Refusing to write to symlink path: {path}"
            raise WikiError(msg) from exc
        raise

    mode = "a" if append else "w"
    with os.fdopen(descriptor, mode, encoding="utf-8") as handle:
        handle.write(content)


def _write_if_missing(path: Path, content: str) -> None:
    """Write file content only when the target does not already exist."""
    if path.is_symlink():
        msg = f"Refusing to write to symlink path: {path}"
        raise WikiError(msg)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_text(path, content)


def _agents_md(topic: str) -> str:
    """Build default AGENTS.md guidance content."""
    return (
        f"# {topic} Wiki\n\n"
        "Use this file as the wiki schema/config for agent behavior.\n"
        "Keep it concise and co-evolve it as the wiki and workflow change.\n\n"
        "Rules:\n"
        "- Treat `/raw/` as read-only source material.\n"
        "- Ingest flow should be supervised: review takeaways first, then apply updates.\n"
        "- Ingest updates should prioritize canonical concept/entity/theme pages.\n"
        "- Prefer a flat `/wiki/` layout by default; create subdirectories only when they clearly improve organization.\n"
        "- Use `/log.md` as recency context and keep it append-only.\n"
        "- Do not edit `/log.md` directly; the runner appends structured timeline entries.\n"
        "- Keep `/wiki/index.md` current as a content catalog.\n"
    )

def _ensure_scaffold(
    topic_dir: Path, topic: str, *, overwrite_agents: bool = False
) -> None:
    """Ensure required topic workspace files and directories exist."""
    (topic_dir / "raw").mkdir(parents=True, exist_ok=True)
    (topic_dir / "wiki").mkdir(parents=True, exist_ok=True)
    _write_if_missing(
        topic_dir / "wiki" / "index.md",
        index_helpers.empty_index_text(topic),
    )
    _write_if_missing(topic_dir / "log.md", "# Change Log\n")

    agents_path = topic_dir / "AGENTS.md"
    if overwrite_agents or not agents_path.exists():
        _safe_write_text(agents_path, _agents_md(topic))


def _validate_text_only_directory(root_dir: Path) -> None:
    """Validate that all files in a directory are UTF-8 text with allowed suffixes."""
    _ensure_no_symlinks(root_dir)
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            rel = file_path.relative_to(root_dir)
            msg = (
                f"Unsupported file for v1 text-only hub pushes: {rel}. "
                "Allowed extensions: md, txt, json, yaml, yml, csv."
            )
            raise WikiError(msg)
        try:
            file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            rel = file_path.relative_to(root_dir)
            msg = f"File {rel} is not valid UTF-8 text. Binary uploads are not supported in v1."
            raise WikiError(msg) from exc


def _stage_sources(sources: Sequence[Path], workspace_dir: Path) -> list[Path]:
    """Copy and de-duplicate source files into the workspace raw directory."""
    staged: list[Path] = []
    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        if not source.exists() or not source.is_file():
            msg = f"Source file not found: {source}"
            raise WikiError(msg)
        if source.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            msg = (
                f"Unsupported source file type for {source}. "
                "Use text files with extensions: md, txt, json, yaml, yml, csv."
            )
            raise WikiError(msg)

        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            msg = f"Source file must be UTF-8 text: {source}"
            raise WikiError(msg) from exc

        destination = raw_dir / source.name
        suffix = source.suffix
        stem = source.stem
        counter = 2
        while destination.exists() or destination.is_symlink():
            destination = raw_dir / f"{stem}-{counter}{suffix}"
            counter += 1

        _safe_write_text(destination, text)
        staged.append(destination)

    return staged


def _extract_text(content: object) -> str:
    """Extract textual content from agent message payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _extract_final_ai_message(result: dict[str, object]) -> str:
    """Return the final assistant text message from an agent invoke result."""
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        msg_type = getattr(message, "type", None)
        if msg_type is None and isinstance(message, dict):
            msg_type = message.get("type")
        if msg_type not in {"ai", "assistant"}:
            continue

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = _extract_text(content).strip()
        if text:
            return text
    return ""


def _refresh_index(topic: str, workspace_dir: Path) -> None:
    """Rebuild the wiki index page from current markdown pages."""
    index_helpers.refresh_index(topic, workspace_dir, write_text=_safe_write_text)


def _append_log_entry(
    workspace_dir: Path,
    phase: str,
    outcome: str,
    *,
    metadata: dict[str, object] | None = None,
    summary: str | None = None,
) -> None:
    """Append one structured, parseable interaction entry to the wiki log."""
    log_helpers.append_log_entry(
        workspace_dir,
        phase,
        outcome,
        metadata=metadata,
        summary=summary,
        ensure_file=_write_if_missing,
        append_text=lambda path, content: _safe_write_text(path, content, append=True),
    )


def _permissions() -> list[FilesystemPermission]:
    """Define filesystem write policy for wiki operations."""
    return [
        FilesystemPermission(operations=["write"], paths=["/raw/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/AGENTS.md"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/wiki/**"], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/log.md"], mode="deny"),
    ]


def _review_permissions() -> list[FilesystemPermission]:
    """Define filesystem policy for ingest review (read-only over wiki/raw)."""
    return [
        FilesystemPermission(operations=["write"], paths=["/raw/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/wiki/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/log.md"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/AGENTS.md"], mode="deny"),
    ]


@contextmanager
def _create_langsmith_sandbox_backend() -> Iterator[SandboxBackendProtocol]:
    """Create and clean up a LangSmith sandbox-backed execution backend."""
    env_key = os.getenv("LANGSMITH_API_KEY")
    if not env_key:
        msg = "LANGSMITH_API_KEY is required to create the LangSmith sandbox backend."
        raise WikiError(msg)

    try:
        from langsmith.sandbox import SandboxClient  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = "langsmith.sandbox is unavailable. Install with `pip install 'langsmith[sandbox]'`."
        raise WikiError(msg) from exc

    resolved_snapshot = os.getenv("WIKI_SANDBOX_SNAPSHOT", _DEFAULT_SNAPSHOT_NAME)
    docker_image = os.getenv("WIKI_SANDBOX_IMAGE", _DEFAULT_DOCKER_IMAGE)
    fs_capacity_raw = os.getenv(
        "WIKI_SANDBOX_FS_CAPACITY_BYTES", str(_DEFAULT_FS_CAPACITY)
    )
    try:
        fs_capacity = int(fs_capacity_raw)
    except ValueError as exc:
        msg = "WIKI_SANDBOX_FS_CAPACITY_BYTES must be an integer"
        raise WikiError(msg) from exc

    client = SandboxClient(api_key=env_key)
    snapshots = client.list_snapshots(name_contains=resolved_snapshot)
    has_ready_snapshot = any(
        snap.name == resolved_snapshot and snap.status == "ready" for snap in snapshots
    )
    if not has_ready_snapshot:
        client.create_snapshot(
            name=resolved_snapshot,
            docker_image=docker_image,
            fs_capacity_bytes=fs_capacity,
        )

    sandbox = client.create_sandbox(snapshot_name=resolved_snapshot)
    try:
        yield LangSmithSandbox(sandbox=sandbox)
    finally:
        with suppress(Exception):
            client.delete_sandbox(sandbox.name)


def _run_agent_mode(
    workspace_dir: Path,
    topic: str,
    prompt: str,
    model: str | None,
    *,
    permissions: list[FilesystemPermission],
) -> str:
    """Execute one agent operation against the pulled workspace."""
    with _create_langsmith_sandbox_backend() as sandbox_backend:
        workspace_backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)
        backend = CompositeBackend(
            default=sandbox_backend,
            routes={
                "/raw/": workspace_backend,
                "/wiki/": workspace_backend,
                "/log.md": workspace_backend,
                "/AGENTS.md": workspace_backend,
            },
        )
        agent = create_deep_agent(
            model=model,
            backend=backend,
            permissions=permissions,
            system_prompt=_BASE_SYSTEM_PROMPT,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    text = _extract_final_ai_message(result)
    if text:
        return text
    return f"Completed {topic} wiki operation."


def _run_agent_apply_mode(
    workspace_dir: Path, topic: str, prompt: str, model: str | None
) -> str:
    """Run a mutating agent operation against wiki files."""
    return _run_agent_mode(
        workspace_dir,
        topic,
        prompt,
        model,
        permissions=_permissions(),
    )


def _run_agent_review_mode(
    workspace_dir: Path, topic: str, prompt: str, model: str | None
) -> str:
    """Run a read-only ingest review operation."""
    return _run_agent_mode(
        workspace_dir,
        topic,
        prompt,
        model,
        permissions=_review_permissions(),
    )


def _resolve_internal_source_flag(deps: CliDeps) -> tuple[str, ...]:
    """Resolve an init flag set that enforces internal repo source."""
    from init import resolve_internal_source_flag

    return resolve_internal_source_flag(deps)


def _extract_repo_source(payload: dict[str, object]) -> str | None:
    """Extract repo source metadata from hub get payload."""
    from init import extract_repo_source

    return extract_repo_source(payload)


def _verify_internal_repo_source(hub_identifier: str, deps: CliDeps) -> None:
    """Verify that the target hub repo source is internal."""
    from init import verify_internal_repo_source

    verify_internal_repo_source(hub_identifier, deps)


def _run_init(config: RunnerConfig, deps: CliDeps) -> RunResult:
    """Initialize a local topic repo and push its first hub revision."""
    from init import run_init

    return run_init(config, deps)


def _collect_directory_sources(directory: Path) -> list[Path]:
    """Collect allowed file paths from a source directory recursively."""
    from ingest import collect_directory_sources

    return collect_directory_sources(directory)


def _expand_sources(sources: Sequence[Path]) -> list[Path]:
    """Expand source arguments into a deterministic list of file paths."""
    from ingest import expand_sources

    return expand_sources(sources)


def _build_ingest_review_prompt(
    topic: str, staged_paths: Sequence[Path], note: str | None
) -> str:
    """Build the ingest review prompt for staged source material."""
    from ingest import build_ingest_review_prompt

    return build_ingest_review_prompt(topic, staged_paths, note)


def _build_ingest_apply_prompt(
    topic: str,
    staged_paths: Sequence[Path],
    review_summary: str,
    note: str | None,
) -> str:
    """Build the ingest apply prompt after operator approval."""
    from ingest import build_ingest_apply_prompt

    return build_ingest_apply_prompt(topic, staged_paths, review_summary, note)


def _confirm_ingest_apply(review: str, ask_user: Callable[[str], str]) -> bool:
    """Ask operator to approve ingest apply after the review phase."""
    from ingest import confirm_ingest_apply

    return confirm_ingest_apply(review, ask_user)


def _run_ingest_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> IngestResult:
    """Run ingest mode against a pulled workspace directory."""
    from ingest import run_ingest_workspace

    return run_ingest_workspace(config, workspace_dir, deps)


def _run_pull_mode(config: RunnerConfig, deps: CliDeps) -> RunResult:
    """Pull a hub repo, run the selected mode, and push updates."""
    hub_identifier = _hub_identifier(config.owner, config.repo)

    with deps.tempdir_factory() as temp_dir:
        workspace_dir = Path(temp_dir)

        deps.run_langsmith_cli(
            [
                "hub",
                "pull",
                _hub_cli_repo_arg(hub_identifier),
                "--dir",
                str(workspace_dir),
            ]
        )

        _ensure_no_symlinks(workspace_dir)
        _ensure_scaffold(workspace_dir, config.topic)

        if config.mode == "ingest":
            ingest_result = _run_ingest_workspace(config, workspace_dir, deps)
            answer = ingest_result.answer
            should_push = ingest_result.should_push
        elif config.mode == "query":
            from query import run_query_workspace

            query_result = run_query_workspace(config, workspace_dir, deps)
            answer = query_result.answer
            should_push = query_result.should_push
        else:
            from lint import run_lint_workspace

            answer = run_lint_workspace(config, workspace_dir, deps)
            should_push = True

        if should_push:
            _validate_text_only_directory(workspace_dir)
            deps.run_langsmith_cli(
                [
                    "hub",
                    "push",
                    _hub_cli_repo_arg(hub_identifier),
                    "--type",
                    "agent",
                    "--dir",
                    str(workspace_dir),
                ]
            )

        hub_url = _resolve_hub_url(config.owner, config.repo)
        return RunResult(answer=answer, hub_url=hub_url)


def run(config: RunnerConfig, deps: CliDeps | None = None) -> RunResult:
    """Execute the requested wiki workflow."""
    _ensure_mode_prerequisites(config.mode)
    resolved_deps = deps or CliDeps(
        run_langsmith_cli=_run_langsmith_cli,
        run_agent_mode=_run_agent_apply_mode,
        run_agent_review_mode=_run_agent_review_mode,
        ask_user=input,
        tempdir_factory=tempfile.TemporaryDirectory,
    )

    if config.mode == "init":
        return _run_init(config, resolved_deps)
    return _run_pull_mode(config, resolved_deps)


__all__ = [
    "CliDeps",
    "RunResult",
    "RunnerConfig",
    "WikiError",
    "parse_config",
    "run",
]
