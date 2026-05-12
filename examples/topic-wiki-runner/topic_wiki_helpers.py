"""Helper utilities for the topic wiki runner example."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LangSmithSandbox
from deepagents.middleware.filesystem import FilesystemPermission

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from deepagents.backends.protocol import SandboxBackendProtocol


Mode = Literal["init", "ingest", "query", "lint"]
_ALLOWED_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}
_DEFAULT_SNAPSHOT_NAME = "deepagents-topic-wiki"
_DEFAULT_DOCKER_IMAGE = "python:3"
_DEFAULT_FS_CAPACITY = 16 * 1024**3
_LANGSMITH_BINARY_CANDIDATES = ("langsmith", "langsmith-cli")
_HUB_COMPATIBLE_BINARIES: set[str] = set()
_BASE_SYSTEM_PROMPT = """You are an expert research synthesizer building a long-lived topic knowledge base.

Mission:
- Build an accurate, high-signal, source-grounded topic corpus in `/memories/wiki/`.
- Treat `/memories/raw/` as immutable evidence inputs.
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
- Include concise "What changed" updates in the log.
- Keep `wiki/index.md` authoritative for navigation.

Evidence rules:
- Every non-trivial claim should be traceable to the ingested source set.
- Avoid introducing unsupported external facts.
- If evidence is weak or missing, say so directly.

Filesystem policy:
- Never write to `/memories/raw/`.
- Write only under `/memories/wiki/`.
"""


class TopicWikiError(RuntimeError):
    """Raised when the topic wiki runner cannot complete a requested operation."""


@dataclass(frozen=True)
class RunnerConfig:
    """Parsed runner configuration."""

    mode: Mode
    topic: str
    hub_id: str
    topic_dir: Path
    sources: tuple[Path, ...]
    note: str | None
    question: str | None
    model: str | None


@dataclass(frozen=True)
class CliDeps:
    """Injectable dependencies for tests."""

    run_langsmith_cli: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
    run_agent_mode: Callable[[Path, str, str, str | None], str]
    tempdir_factory: Callable[[], tempfile.TemporaryDirectory[str]]


@dataclass(frozen=True)
class RunResult:
    """Output from a runner invocation."""

    answer: str | None
    hub_url: str | None


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


def _default_hub_id(topic: str) -> str:
    """Build the default hub id for a topic."""
    return f"-/{_slugify_topic(topic)}"


def _repo_name_from_hub_id(hub_id: str) -> str:
    """Extract the repository name component from a hub id."""
    owner, sep, repo = hub_id.partition("/")
    if sep == "" or not owner or not repo:
        msg = f"Invalid --hub-id {hub_id!r}; expected [OWNER/]REPO"
        raise TopicWikiError(msg)
    return repo


def _topic_dir_for(topic: str, explicit: str | None) -> Path:
    """Resolve the local topic directory path."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.cwd() / "topic-wikis" / _slugify_topic(topic)).resolve()


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Topic wiki runner (DeepAgents + LangSmith Hub CLI)"
    )
    parser.add_argument(
        "--mode", required=True, choices=["init", "ingest", "query", "lint"]
    )
    parser.add_argument("--topic", required=True, help="Topic name")
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Hub repo id [OWNER/]REPO (default: -/<topic-slug>)",
    )
    parser.add_argument(
        "--topic-dir", default=None, help="Local topic directory for init mode"
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file for ingest mode (repeatable)",
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
    return parser


def parse_config(argv: Sequence[str] | None = None) -> RunnerConfig:
    """Parse CLI arguments into a runner config."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    hub_id = args.hub_id or _default_hub_id(args.topic)
    mode = args.mode

    if mode == "ingest" and not args.source:
        parser.error("--source is required in ingest mode")
    if mode == "query" and not args.question:
        parser.error("--question is required in query mode")

    return RunnerConfig(
        mode=mode,
        topic=args.topic,
        hub_id=hub_id,
        topic_dir=_topic_dir_for(args.topic, args.topic_dir),
        sources=tuple(Path(source).expanduser().resolve() for source in args.source),
        note=args.note,
        question=args.question,
        model=args.model,
    )


def _resolve_langsmith_binary() -> str:
    """Find an installed LangSmith CLI binary."""
    for candidate in _LANGSMITH_BINARY_CANDIDATES:
        binary = shutil.which(candidate)
        if binary:
            return binary
    msg = (
        "LangSmith CLI was not found on PATH. Install a LangSmith CLI binary (`langsmith` or "
        "`langsmith-cli`) before running topic wiki sync."
    )
    raise TopicWikiError(msg)


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
    raise TopicWikiError(msg)


def _ensure_mode_prerequisites(mode: Mode) -> None:
    """Validate mode-specific environment prerequisites."""
    if mode in {"ingest", "query", "lint"} and not os.getenv("LANGSMITH_API_KEY"):
        msg = (
            "LANGSMITH_API_KEY is required for ingest/query/lint modes because they run agent "
            "operations inside `langsmith.sandbox`."
        )
        raise TopicWikiError(msg)


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
        raise TopicWikiError(msg)

    msg = f"{cmd} {' '.join(args)} failed with exit code {result.returncode}:\n{output}"
    raise TopicWikiError(msg)


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


def _resolve_hub_url(
    hub_id: str, deps: CliDeps, push_result: subprocess.CompletedProcess[str]
) -> str | None:
    """Resolve a browser URL for the hub repo after sync."""
    payload = _parse_cli_json_output(push_result) or {}

    owner: str | None = None
    repo: str | None = None

    owner_value = payload.get("owner")
    if isinstance(owner_value, str) and owner_value and owner_value != "-":
        owner = owner_value

    repo_value = payload.get("repo")
    if isinstance(repo_value, str) and repo_value:
        repo = repo_value

    if repo is None:
        _owner, _sep, repo_part = hub_id.partition("/")
        if repo_part:
            repo = repo_part

    try:
        hub_get = deps.run_langsmith_cli(
            [
                "hub",
                "get",
                _hub_cli_repo_arg(hub_id),
                "--format",
                "json",
            ]
        )
    except TopicWikiError:
        hub_get = None

    if hub_get is not None:
        get_payload = _parse_cli_json_output(hub_get) or {}
        full_name = get_payload.get("full_name")
        if isinstance(full_name, str) and "/" in full_name:
            owner_from_full, repo_from_full = full_name.split("/", 1)
            owner = owner_from_full or owner
            repo = repo_from_full or repo
        else:
            get_owner = get_payload.get("owner")
            if isinstance(get_owner, str) and get_owner:
                owner = get_owner
            get_repo = get_payload.get("repo_handle")
            if isinstance(get_repo, str) and get_repo:
                repo = get_repo

    if not repo:
        return None

    base = _app_base_url()
    if owner:
        return f"{base}/hub/{owner}/{repo}"
    return f"{base}/hub/{repo}"


def _hub_cli_repo_arg(hub_id: str) -> str:
    """Normalize hub id values for cobra-based CLI parsing."""
    # Cobra treats values beginning with `-` as flags; `-/repo` is equivalent
    # to passing just `repo` (owner defaults to current tenant).
    if hub_id.startswith("-/"):
        return hub_id[2:]
    return hub_id


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
                f"Symlinks are not supported in topic wiki workspaces for security reasons: "
                f"{relative}"
            )
            raise TopicWikiError(msg)
        msg = f"Symlinks are not supported in topic wiki workspaces for security reasons: {path}"
        raise TopicWikiError(msg)


def _safe_write_text(path: Path, content: str, *, append: bool = False) -> None:
    """Write UTF-8 text while refusing symlink targets."""
    if path.is_symlink():
        msg = f"Refusing to write to symlink path: {path}"
        raise TopicWikiError(msg)

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
            raise TopicWikiError(msg) from exc
        raise

    mode = "a" if append else "w"
    with os.fdopen(descriptor, mode, encoding="utf-8") as handle:
        handle.write(content)


def _write_if_missing(path: Path, content: str) -> None:
    """Write file content only when the target does not already exist."""
    if path.is_symlink():
        msg = f"Refusing to write to symlink path: {path}"
        raise TopicWikiError(msg)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_text(path, content)


def _agents_md(topic: str) -> str:
    """Build default AGENTS.md guidance content."""
    return (
        f"# {topic} Topic Wiki\n\n"
        "Maintain a concise, source-grounded wiki for this topic.\n\n"
        "Rules:\n"
        "- Treat `/memories/raw/` as read-only source material.\n"
        "- Write curated pages to `/memories/wiki/`.\n"
        "- Keep `/memories/wiki/index.md` current and append changes to `/memories/wiki/log.md`.\n"
    )


def _ensure_scaffold(
    topic_dir: Path, topic: str, *, overwrite_agents: bool = False
) -> None:
    """Ensure required topic workspace files and directories exist."""
    (topic_dir / "raw").mkdir(parents=True, exist_ok=True)
    (topic_dir / "wiki").mkdir(parents=True, exist_ok=True)
    _write_if_missing(
        topic_dir / "wiki" / "index.md",
        f"# {topic} Wiki\n\n## Pages\n\n- _No pages yet._\n",
    )
    _write_if_missing(topic_dir / "wiki" / "log.md", "# Change Log\n")

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
            raise TopicWikiError(msg)
        try:
            file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            rel = file_path.relative_to(root_dir)
            msg = f"File {rel} is not valid UTF-8 text. Binary uploads are not supported in v1."
            raise TopicWikiError(msg) from exc


def _stage_sources(sources: Sequence[Path], workspace_dir: Path) -> list[Path]:
    """Copy and de-duplicate source files into the workspace raw directory."""
    staged: list[Path] = []
    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        if not source.exists() or not source.is_file():
            msg = f"Source file not found: {source}"
            raise TopicWikiError(msg)
        if source.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            msg = (
                f"Unsupported source file type for {source}. "
                "Use text files with extensions: md, txt, json, yaml, yml, csv."
            )
            raise TopicWikiError(msg)

        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            msg = f"Source file must be UTF-8 text: {source}"
            raise TopicWikiError(msg) from exc

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
    wiki_dir = workspace_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    pages = [
        path
        for path in sorted(wiki_dir.glob("*.md"))
        if path.name not in {"index.md", "log.md"}
    ]

    lines = [f"# {topic} Wiki", "", "## Pages", ""]
    if not pages:
        lines.append("- _No pages yet._")
    else:
        for page in pages:
            title = page.stem.replace("-", " ").replace("_", " ").strip().title()
            lines.append(f"- [{title}]({page.name})")

    _safe_write_text((wiki_dir / "index.md"), "\n".join(lines).rstrip() + "\n")


def _append_log_entry(workspace_dir: Path, mode: Mode, detail: str) -> None:
    """Append a timestamped operation entry to the wiki log."""
    log_path = workspace_dir / "wiki" / "log.md"
    _write_if_missing(log_path, "# Change Log\n")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _safe_write_text(log_path, f"- {timestamp} | {mode} | {detail}\n", append=True)
    _normalize_log_chronology(log_path)


def _parse_log_timestamp(entry: str) -> datetime | None:
    """Parse an ISO timestamp from a log bullet entry."""
    if not entry.startswith("- "):
        return None
    timestamp = entry[2:].split("|", 1)[0].strip()
    if not timestamp:
        return None
    normalized = timestamp.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _normalize_log_chronology(log_path: Path) -> None:
    """Sort parseable log entries chronologically while preserving non-entry lines."""
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return

    header_lines: list[str] = []
    entry_lines: list[str] = []
    passthrough_lines: list[str] = []

    for line in lines:
        if line.startswith("- "):
            entry_lines.append(line)
            continue
        if entry_lines:
            passthrough_lines.append(line)
            continue
        header_lines.append(line)

    sortable_entries: list[tuple[datetime, int, str]] = []
    unsortable_entries: list[str] = []
    for index, line in enumerate(entry_lines):
        parsed = _parse_log_timestamp(line)
        if parsed is None:
            unsortable_entries.append(line)
            continue
        sortable_entries.append((parsed, index, line))

    sortable_entries.sort(key=lambda item: (item[0], item[1]))
    sorted_lines = (
        header_lines
        + [line for _, _, line in sortable_entries]
        + unsortable_entries
        + passthrough_lines
    )
    _safe_write_text(log_path, "\n".join(sorted_lines).rstrip() + "\n")


def _build_ingest_prompt(
    topic: str, staged_paths: Sequence[Path], note: str | None
) -> str:
    """Build the ingest prompt for staged source material."""
    memory_paths = [f"/memories/raw/{path.name}" for path in staged_paths]
    source_block = "\n".join(f"- {path}" for path in memory_paths)
    note_block = note or "(none)"
    return (
        f"Ingest the staged sources and update the topic knowledge base for '{topic}'.\n\n"
        "Required workflow:\n"
        "1) Read all staged source files under `/memories/raw/`.\n"
        "2) Create or update canonical pages under `/memories/wiki/`.\n"
        "3) De-duplicate overlapping content and merge into strongest canonical page structure.\n"
        "4) Ensure each important claim in wiki pages is grounded in staged evidence.\n"
        "5) Update `/memories/wiki/index.md` with complete page navigation.\n"
        "6) Append one concise ingest entry to `/memories/wiki/log.md` describing files touched and key changes.\n\n"
        "Content requirements:\n"
        "- Focus on high-signal synthesis over raw paraphrase.\n"
        "- If evidence conflicts, include both claims and mark unresolved points.\n"
        "- If information is missing, add explicit open questions in the most relevant page.\n"
        "- Never write to `/memories/raw/`.\n\n"
        f"Staged sources:\n{source_block}\n\n"
        f"Operator note: {note_block}\n"
    )


def _build_query_prompt(topic: str, question: str) -> str:
    """Build the query prompt for answering from wiki pages."""
    return (
        f"Answer this question about '{topic}': {question}\n\n"
        "Required workflow:\n"
        "1) Read `/memories/wiki/index.md` first.\n"
        "2) Read all relevant `/memories/wiki/*.md` pages before answering.\n"
        "3) Provide a direct answer first, then concise supporting rationale.\n"
        "4) Cite the wiki file paths that support the answer.\n\n"
        "Quality bar:\n"
        "- Prioritize grounded claims over speculation.\n"
        "- If the wiki lacks enough evidence, say what is unknown.\n"
        "- If synthesis would materially improve future answers, update/create `/memories/wiki/synthesis.md`.\n"
        "- Never write to `/memories/raw/`.\n"
    )


def _build_lint_prompt(topic: str, note: str | None) -> str:
    """Build the lint prompt for wiki consistency checks."""
    note_text = note or "(none)"
    return (
        f"Run a full consistency lint pass for the '{topic}' wiki under `/memories/wiki/`.\n\n"
        "Required checks:\n"
        "- Fix broken links, stale references, and structural inconsistencies.\n"
        "- Normalize duplicated concepts into one canonical page where appropriate.\n"
        "- Ensure `/memories/wiki/index.md` references all wiki pages.\n"
        "- Ensure major claims remain evidence-grounded and uncertainty is explicit.\n"
        "- Keep `wiki/log.md` chronological and append one lint entry summarizing fixes.\n"
        "- Never write to `/memories/raw/`.\n\n"
        f"Operator note: {note_text}\n"
    )


def _permissions() -> list[FilesystemPermission]:
    """Define filesystem write policy for topic wiki operations."""
    return [
        FilesystemPermission(
            operations=["write"], paths=["/memories/raw/**"], mode="deny"
        ),
        FilesystemPermission(
            operations=["write"], paths=["/memories/wiki/**"], mode="allow"
        ),
    ]


@contextmanager
def _create_langsmith_sandbox_backend() -> Iterator[SandboxBackendProtocol]:
    """Create and clean up a LangSmith sandbox-backed execution backend."""
    env_key = os.getenv("LANGSMITH_API_KEY")
    if not env_key:
        msg = "LANGSMITH_API_KEY is required to create the LangSmith sandbox backend."
        raise TopicWikiError(msg)

    try:
        from langsmith.sandbox import SandboxClient  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = "langsmith.sandbox is unavailable. Install with `pip install 'langsmith[sandbox]'`."
        raise TopicWikiError(msg) from exc

    resolved_snapshot = os.getenv("TOPIC_WIKI_SANDBOX_SNAPSHOT", _DEFAULT_SNAPSHOT_NAME)
    docker_image = os.getenv("TOPIC_WIKI_SANDBOX_IMAGE", _DEFAULT_DOCKER_IMAGE)
    fs_capacity_raw = os.getenv(
        "TOPIC_WIKI_SANDBOX_FS_CAPACITY_BYTES", str(_DEFAULT_FS_CAPACITY)
    )
    try:
        fs_capacity = int(fs_capacity_raw)
    except ValueError as exc:
        msg = "TOPIC_WIKI_SANDBOX_FS_CAPACITY_BYTES must be an integer"
        raise TopicWikiError(msg) from exc

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
    workspace_dir: Path, topic: str, prompt: str, model: str | None
) -> str:
    """Execute one agent operation against the pulled workspace."""
    with _create_langsmith_sandbox_backend() as sandbox_backend:
        memories_backend = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)
        backend = CompositeBackend(
            default=sandbox_backend, routes={"/memories/": memories_backend}
        )
        agent = create_deep_agent(
            model=model,
            backend=backend,
            permissions=_permissions(),
            system_prompt=_BASE_SYSTEM_PROMPT,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

    text = _extract_final_ai_message(result)
    if text:
        return text
    return f"Completed {topic} wiki operation."


def _run_init(config: RunnerConfig, deps: CliDeps) -> RunResult:
    """Initialize a local topic repo and push its first hub revision."""
    config.topic_dir.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlinks(config.topic_dir)
    _ensure_scaffold(config.topic_dir, config.topic, overwrite_agents=True)

    repo_name = _repo_name_from_hub_id(config.hub_id)
    deps.run_langsmith_cli(
        [
            "hub",
            "init",
            "--type",
            "agent",
            "--dir",
            str(config.topic_dir),
            "--name",
            repo_name,
            "--force",
        ]
    )

    _ensure_scaffold(config.topic_dir, config.topic, overwrite_agents=True)
    _validate_text_only_directory(config.topic_dir)

    push_result = deps.run_langsmith_cli(
        [
            "hub",
            "push",
            _hub_cli_repo_arg(config.hub_id),
            "--type",
            "agent",
            "--dir",
            str(config.topic_dir),
        ]
    )
    hub_url = _resolve_hub_url(config.hub_id, deps, push_result)
    return RunResult(answer=None, hub_url=hub_url)


def _run_ingest_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> None:
    """Run ingest mode against a pulled workspace directory."""
    staged = _stage_sources(config.sources, workspace_dir)
    prompt = _build_ingest_prompt(config.topic, staged, config.note)
    deps.run_agent_mode(workspace_dir, config.topic, prompt, config.model)
    _refresh_index(config.topic, workspace_dir)
    sources_text = ", ".join(path.name for path in staged)
    detail = f"sources=[{sources_text}]"
    if config.note:
        detail += f" note={config.note}"
    _append_log_entry(workspace_dir, "ingest", detail)


def _run_query_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> str:
    """Run query mode and return the model answer."""
    question = config.question or ""
    prompt = _build_query_prompt(config.topic, question)
    answer = deps.run_agent_mode(workspace_dir, config.topic, prompt, config.model)
    _append_log_entry(workspace_dir, "query", f"question={question}")
    return answer


def _run_lint_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> None:
    """Run lint mode to improve wiki consistency and links."""
    prompt = _build_lint_prompt(config.topic, config.note)
    deps.run_agent_mode(workspace_dir, config.topic, prompt, config.model)
    _refresh_index(config.topic, workspace_dir)
    _append_log_entry(workspace_dir, "lint", "lint pass")


def _run_pull_mode(config: RunnerConfig, deps: CliDeps) -> RunResult:
    """Pull a hub repo, run the selected mode, and push updates."""
    with deps.tempdir_factory() as temp_dir:
        workspace_dir = Path(temp_dir)

        deps.run_langsmith_cli(
            [
                "hub",
                "pull",
                _hub_cli_repo_arg(config.hub_id),
                "--dir",
                str(workspace_dir),
                "--yes",
            ]
        )

        _ensure_no_symlinks(workspace_dir)
        _ensure_scaffold(workspace_dir, config.topic)

        if config.mode == "ingest":
            _run_ingest_workspace(config, workspace_dir, deps)
            answer: str | None = None
        elif config.mode == "query":
            answer = _run_query_workspace(config, workspace_dir, deps)
        else:
            _run_lint_workspace(config, workspace_dir, deps)
            answer = None

        _validate_text_only_directory(workspace_dir)
        push_result = deps.run_langsmith_cli(
            [
                "hub",
                "push",
                _hub_cli_repo_arg(config.hub_id),
                "--type",
                "agent",
                "--dir",
                str(workspace_dir),
            ]
        )
        hub_url = _resolve_hub_url(config.hub_id, deps, push_result)
        return RunResult(answer=answer, hub_url=hub_url)


def run(config: RunnerConfig, deps: CliDeps | None = None) -> RunResult:
    """Execute the requested topic wiki workflow."""
    _ensure_mode_prerequisites(config.mode)
    resolved_deps = deps or CliDeps(
        run_langsmith_cli=_run_langsmith_cli,
        run_agent_mode=_run_agent_mode,
        tempdir_factory=tempfile.TemporaryDirectory,
    )

    if config.mode == "init":
        return _run_init(config, resolved_deps)
    return _run_pull_mode(config, resolved_deps)


__all__ = [
    "CliDeps",
    "RunResult",
    "RunnerConfig",
    "TopicWikiError",
    "parse_config",
    "run",
]
