"""Independent spec-compliance verification tool for the dcode agent.

`make_verify_tool` builds a model-invoked tool that re-reads the original task
verbatim from agent state and the produced files from the sandbox, then asks a
fresh-context LLM judge whether every named identifier in the task appears exactly
in the implementation. Sourcing the spec from state (not from a model-supplied
argument) prevents the agent from paraphrasing the contract into the checker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  (needed at runtime for tool injection)
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel


VERIFY_IMPLEMENTATION_DESCRIPTION = """\
Verify your implementation against the original task spec using an independent,
fresh-context checker.

WHEN TO CALL — before concluding the task, call this whenever the task specifies
any exact contract:
- file names or paths (e.g. /app/result.txt)
- field, message, type, function, class, or variable names
- a schema, proto/IDL, or wire format
- exact output text or output format

You do NOT pass the spec. The checker re-reads the original task verbatim and your
files on disk itself, so it cannot inherit any renaming or assumption you made. It
returns PASS/FAIL with the exact mismatches (e.g. `val` vs `value`). Treat any FAIL
as a real defect in your code: fix it and call this again. Do not finish a contract
task until it returns PASS.

Optional `focus`: narrow the check to a file/aspect; empty verifies everything.
"""

_JUDGE_SYSTEM_PROMPT = """\
You are a strict spec-compliance checker. You are given the ORIGINAL TASK (verbatim)
and the FILES an agent produced to satisfy it.

Enumerate EVERY concrete, named item the task requires — file names and paths;
service/message/type/field/enum names; function and class names; ports; wire-format
tokens; and exact output strings or formats — and confirm each appears EXACTLY as
written in the produced files. Matching is literal and case-sensitive: `value` is NOT
`val`, `/app/result.txt` is NOT `/app/results.txt`. A near-match is a MISMATCH.

You cannot run code; judge only by the textual presence and exact spelling of each
item in the files shown. If a file the task requires is genuinely absent from what you
were given, report it as MISMATCH (missing). But if the provided files were truncated
or look incomplete, do NOT fail those items — say you could not verify them.

Report each item as MATCH or MISMATCH with expected-vs-found, then end with exactly
one line: `VERDICT: PASS` (every required item matches) or `VERDICT: FAIL`.
"""


class VerifySchema(BaseModel):
    """Arguments for the `verify_implementation` tool."""

    focus: str | None = Field(
        default=None,
        description=(
            "Optional: narrow the check to a specific file or aspect. "
            "Leave empty to verify the whole implementation against the task spec."
        ),
    )


_SKIP_DIR_PARTS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
    }
)
_SKIP_ROUTE_PREFIXES = ("/large_tool_results/", "/conversation_history/")
_TEXT_EXTS = frozenset(
    {
        ".py",
        ".proto",
        ".go",
        ".js",
        ".ts",
        ".java",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".md",
        ".sql",
        ".sh",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".cs",
        ".html",
        ".css",
        ".cfg",
        ".ini",
        ".xml",
    }
)


def _content_to_text(content: object) -> str:
    """Flatten a message `content` (str or list-of-blocks) to plain text.

    Returns:
        The concatenated text content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _original_task(runtime: ToolRuntime) -> str:
    """Return the original task text verbatim from the first human message.

    Returns:
        The first human message's text, or an empty string if absent.
    """
    state = runtime.state or {}
    messages = state.get("messages") or []
    first = next((m for m in messages if getattr(m, "type", None) == "human"), None)
    if first is None and messages:
        first = messages[0]
    if first is None:
        return ""
    return _content_to_text(getattr(first, "content", "")).strip()


def _is_skippable(path: str) -> bool:
    """Return True for scratch routes and noise directories."""
    if path.startswith(_SKIP_ROUTE_PREFIXES):
        return True
    parts = path.strip("/").split("/")
    return any(part in _SKIP_DIR_PARTS or part.endswith(".egg-info") for part in parts)


def _has_text_ext(path: str) -> bool:
    """Return True if the path has a known text/code extension."""
    lower = path.lower()
    return any(lower.endswith(ext) for ext in _TEXT_EXTS)


def _eligible_paths(
    matches: Sequence[Mapping[str, Any]] | None, focus: str | None
) -> list[str]:
    """Filter glob matches to eligible text files, ordered for inclusion.

    Returns:
        Eligible file paths in inclusion order.
    """
    paths: list[str] = []
    for info in matches or []:
        if info.get("is_dir"):
            continue
        path = info.get("path")
        if not path or _is_skippable(path) or not _has_text_ext(path):
            continue
        paths.append(path)
    if focus:
        paths.sort(key=lambda p: (focus not in p, p.count("/"), p))
    else:
        paths.sort(key=lambda p: (p.count("/"), p))
    return paths


def _file_chunk(
    path: str, file_data: dict[str, Any] | None, max_file_bytes: int
) -> str | None:
    """Render one file as a fenced block.

    Returns:
        The fenced block, or None if the file is binary/empty and should be skipped.
    """
    if not file_data:
        return None
    if file_data.get("encoding") not in {None, "utf-8"}:
        return None  # binary / base64 — skip
    content = file_data.get("content", "")
    if isinstance(content, list):  # legacy v1 line-list format
        content = "\n".join(content)
    elif not isinstance(content, str):
        content = str(content)
    if len(content) > max_file_bytes:
        content = content[:max_file_bytes] + "\n...[truncated]..."
    return f"### FILE: {path}\n```\n{content}\n```"


def _bundle_note(omitted: int) -> str:
    """Return a truncation note for omitted files, or empty string."""
    if not omitted:
        return ""
    return (
        f"\n\n[NOTE: {omitted} additional file(s) were not shown due to size "
        "limits and may be incomplete; do not fail items solely because they "
        "were not shown.]"
    )


def _judge_human(spec: str, bundle: str) -> str:
    """Build the human message for the judge from the spec and file bundle.

    Returns:
        The formatted human-message text.
    """
    return (
        "ORIGINAL TASK (verbatim):\n<task-spec>\n"
        f"{spec}\n</task-spec>\n\n"
        f"FILES PRODUCED:\n{bundle}\n\n"
        "List each required named item as MATCH or MISMATCH (expected vs found), "
        "then end with exactly one line: `VERDICT: PASS` or `VERDICT: FAIL`."
    )


def _preflight(spec: str, n_files: int) -> str | None:
    """Return a fail-soft report when there is nothing to judge.

    Returns:
        A report string when the spec or files are missing, else None.
    """
    if not spec:
        return (
            "VERIFICATION INCOMPLETE: could not read the original task from state; "
            "not verified. Verify manually before concluding."
        )
    if n_files == 0:
        return (
            "VERDICT: FAIL\nNo implementation files were found at the working "
            "directory to verify. Ensure your deliverables exist on disk, then "
            "re-run verification."
        )
    return None


def _format_report(text: str) -> str:
    """Ensure a verdict line is present.

    Returns:
        The report text, defaulting to FAIL if the checker omitted a verdict.
    """
    if "VERDICT:" not in text.upper():
        return (
            f"{text}\n\nVERDICT: FAIL "
            "(no explicit verdict from the checker; treat as unverified)"
        )
    return text


def _incomplete_report(exc: Exception) -> str:
    """Build the fail-soft report for any verification failure.

    Returns:
        A fail-soft report string describing the error. A verifier must never
        raise, so every failure (file access, judge call, anything) becomes this.
    """
    return (
        f"VERIFICATION INCOMPLETE: the checker could not complete "
        f"({type(exc).__name__}); not verified. Re-run verification or verify "
        "manually before concluding."
    )


def make_verify_tool(
    model: str | BaseChatModel,
    backend: BackendProtocol,
    *,
    cwd: str,
    max_files: int = 20,
    max_total_bytes: int = 120_000,
    max_file_bytes: int = 20_000,
) -> BaseTool:
    """Build the `verify_implementation` tool bound to a model, backend, and cwd.

    Returns:
        A `StructuredTool` the agent can call to verify its work against the spec.
    """
    judge_model = model if not isinstance(model, str) else init_chat_model(model)

    # The whole body of each variant is wrapped: a verification tool must never
    # raise, or it would crash the agent it is meant to help. Any failure (file
    # access, judge call, etc.) becomes a fail-soft INCOMPLETE report.
    def _verify(runtime: ToolRuntime, focus: str | None = None) -> str:
        try:
            spec = _original_task(runtime)
            paths = _eligible_paths(backend.glob("**/*", path=cwd).matches, focus)
            chunks: list[str] = []
            total = 0
            omitted = 0
            for path in paths:
                if len(chunks) >= max_files or total >= max_total_bytes:
                    omitted += 1
                    continue
                result = backend.read(path)
                if getattr(result, "error", None):
                    continue
                chunk = _file_chunk(
                    path, getattr(result, "file_data", None), max_file_bytes
                )
                if chunk is None:
                    continue
                chunks.append(chunk)
                total += len(chunk)
            bundle = "\n\n".join(chunks) + _bundle_note(omitted)
            preflight = _preflight(spec, len(chunks))
            if preflight is not None:
                return preflight
            judge_messages = [
                SystemMessage(_JUDGE_SYSTEM_PROMPT),
                HumanMessage(_judge_human(spec, bundle)),
            ]
            response = judge_model.invoke(judge_messages)
            return _format_report(_content_to_text(response.content).strip())
        except Exception as exc:  # noqa: BLE001  (a verifier must never crash the agent)
            return _incomplete_report(exc)

    async def _averify(runtime: ToolRuntime, focus: str | None = None) -> str:
        try:
            spec = _original_task(runtime)
            glob_result = await backend.aglob("**/*", path=cwd)
            paths = _eligible_paths(glob_result.matches, focus)
            chunks: list[str] = []
            total = 0
            omitted = 0
            for path in paths:
                if len(chunks) >= max_files or total >= max_total_bytes:
                    omitted += 1
                    continue
                result = await backend.aread(path)
                if getattr(result, "error", None):
                    continue
                chunk = _file_chunk(
                    path, getattr(result, "file_data", None), max_file_bytes
                )
                if chunk is None:
                    continue
                chunks.append(chunk)
                total += len(chunk)
            bundle = "\n\n".join(chunks) + _bundle_note(omitted)
            preflight = _preflight(spec, len(chunks))
            if preflight is not None:
                return preflight
            judge_messages = [
                SystemMessage(_JUDGE_SYSTEM_PROMPT),
                HumanMessage(_judge_human(spec, bundle)),
            ]
            response = await judge_model.ainvoke(judge_messages)
            return _format_report(_content_to_text(response.content).strip())
        except Exception as exc:  # noqa: BLE001  (a verifier must never crash the agent)
            return _incomplete_report(exc)

    return StructuredTool.from_function(
        name="verify_implementation",
        description=VERIFY_IMPLEMENTATION_DESCRIPTION,
        func=_verify,
        coroutine=_averify,
        args_schema=VerifySchema,
        infer_schema=False,
    )
