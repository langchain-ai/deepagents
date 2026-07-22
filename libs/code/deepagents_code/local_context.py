"""Middleware for injecting local context into system prompt.

Detects git state, project structure, package managers, runtimes, and
directory layout by running a bash script via the backend. Because the
script executes inside the backend (local shell or remote sandbox), the
same detection logic works regardless of where the agent runs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NotRequired,
    Protocol,
    cast,
    runtime_checkable,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime

    from deepagents_code.mcp_tools import MCPServerInfo


_TOOL_NAME_DISPLAY_LIMIT = 10
"""Maximum number of tool names shown per MCP server in the system prompt."""

_DETECT_SCRIPT_TIMEOUT = 30
"""Timeout in seconds for the environment detection script."""

_MCP_ERROR_DETAIL_LIMIT = 200
"""Max characters of an MCP server error surfaced in the system prompt."""

_TRACING_PROJECT_NAME_LIMIT = 200
"""Max characters of a LangSmith project name surfaced in the system prompt."""


def _sanitize_error_detail(error: str | None) -> str:
    """Make an untrusted MCP error string safe to embed in the system prompt.

    The error originates from exception text or MCP config-file contents, so it
    is untrusted input flowing into the system prompt (prompt-injection and
    log-forging risk). Strip hidden/deceptive Unicode, flatten control
    characters and newlines to spaces so the value cannot break out of its
    single bullet line or inject fake instruction lines, collapse runs of
    whitespace, and bound the length.

    Args:
        error: Raw error message, or `None`.

    Returns:
        A single-line, length-bounded, sanitized string. Falls back to
        `"unknown error"` when no usable message remains.
    """
    if not error:
        return "unknown error"
    sanitized = sanitize_control_chars(error, max_length=_MCP_ERROR_DETAIL_LIMIT)
    return sanitized or "unknown error"


def _sanitize_tracing_project_name(project: str) -> str:
    """Make an untrusted LangSmith project name safe for the system prompt.

    Project names can originate from a workspace `.env` file or process
    environment. Flatten hidden/control characters and bound the length before
    embedding them in prompt bullets so a crafted value cannot inject extra
    prompt lines.

    Args:
        project: Raw LangSmith project name.

    Returns:
        A single-line, length-bounded, sanitized project name. Falls back to
        `"unknown project"` when no usable text remains.
    """
    sanitized = sanitize_control_chars(project, max_length=_TRACING_PROJECT_NAME_LIMIT)
    return sanitized or "unknown project"


def _quote_tracing_project_name(project: str) -> str:
    """JSON-quote a sanitized LangSmith project name for prompt insertion.

    Args:
        project: Sanitized LangSmith project name.

    Returns:
        JSON string literal for the project name.
    """
    return json.dumps(project, ensure_ascii=False)


def _build_mcp_context(servers: list[MCPServerInfo]) -> str:
    """Format MCP server/tool inventory for the system prompt.

    Args:
        servers: List of connected MCP server metadata.

    Returns:
        Formatted markdown string, or `""` if no servers.
    """
    if not servers:
        return ""

    total_tools = sum(len(s.tools) for s in servers)
    lines = [f"**MCP Servers** ({len(servers)} servers, {total_tools} tools):"]

    for server in servers:
        if not server.tools:
            # `status`/`error` always exist on the frozen dataclass; the
            # `__post_init__` invariant guarantees a non-`ok` status carries a
            # non-`None` error. The error is untrusted (exception/config text),
            # so it is sanitized and isolated in an `<error>` delimiter before
            # reaching the prompt.
            if server.status == "error":
                detail = _sanitize_error_detail(server.error)
                lines.append(
                    f"- **{server.name}** ({server.transport}): "
                    f"FAILED TO LOAD — <error>{detail}</error>. "
                    "Treat this integration as temporarily unavailable; "
                    "tell the user the server failed to load and suggest "
                    "restarting the MCP server."
                )
            elif server.status == "unauthenticated":
                detail = _sanitize_error_detail(server.error)
                lines.append(
                    f"- **{server.name}** ({server.transport}): "
                    f"NEEDS LOGIN — <error>{detail}</error>. "
                    "This integration requires authentication before its "
                    "tools are available; tell the user and suggest running "
                    "`/mcp` to log in."
                )
            elif server.status == "disabled":
                lines.append(
                    f"- **{server.name}** ({server.transport}): (disabled by user)"
                )
            else:
                # `ok` with no tools (genuinely empty). `awaiting_reconnect` is a
                # transient UI-only status that never reaches this function (the
                # middleware is always built from a fresh preload), but it would
                # also render benignly here.
                lines.append(
                    f"- **{server.name}** ({server.transport}): (no tools registered)"
                )
            continue

        names = [t.name for t in server.tools]
        if len(names) > _TOOL_NAME_DISPLAY_LIMIT:
            shown = ", ".join(names[:_TOOL_NAME_DISPLAY_LIMIT])
            remaining = len(names) - _TOOL_NAME_DISPLAY_LIMIT
            lines.append(
                f"- **{server.name}** ({server.transport}): "
                f"{shown}, and {remaining} more"
            )
        else:
            lines.append(
                f"- **{server.name}** ({server.transport}): {', '.join(names)}"
            )

    return "\n".join(lines)


def _build_tracing_context(
    agent_project: str | None,
    user_project: str | None,
) -> str:
    """Format LangSmith tracing project names for the system prompt.

    Surfaces both projects so the agent can look up the right traces with the
    LangSmith MCP server or CLI: the project its own runs are traced to, and
    the user's original project that shell commands trace to. The
    shell-command line is shown only when the user's project differs from the
    agent's (after sanitizing both), avoiding a redundant duplicate line.

    Args:
        agent_project: Project receiving the agent's own traces, or `None`
            when LangSmith tracing is not enabled.
        user_project: User's original `LANGSMITH_PROJECT`, used by code the
            agent runs in the shell.

    Returns:
        Formatted markdown string, or `""` when tracing is disabled.
    """
    if not agent_project:
        return ""

    safe_agent_project = _sanitize_tracing_project_name(agent_project)
    quoted_agent_project = _quote_tracing_project_name(safe_agent_project)
    lines = [
        "**LangSmith Tracing**:",
        f"- Agent traces: project {quoted_agent_project}",
    ]
    if user_project:
        safe_user_project = _sanitize_tracing_project_name(user_project)
        if safe_user_project != safe_agent_project:
            quoted_user_project = _quote_tracing_project_name(safe_user_project)
            lines.append(f"- Shell-command traces: project {quoted_user_project}")
    return "\n".join(lines)


@runtime_checkable
class _ExecutableBackend(Protocol):
    """Any backend that supports `execute(command) -> ExecuteResponse`."""

    def execute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse: ...


@runtime_checkable
class _AsyncExecutableBackend(Protocol):
    """Any backend that provides an async `aexecute` method."""

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Timeout is forwarded to backend, not used as asyncio timeout
    ) -> ExecuteResponse: ...


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context detection script
#
# Outputs markdown describing the current working environment. Each section
# is guarded so that missing tools or unsupported environments are silently
# skipped -- external tools like git, tree, python3, and node are checked
# with `command -v` before use.
#
# The script is built from section functions so each piece can be tested
# independently. Independent sections run as parallel background subshells;
# see build_detect_script() for the orchestration logic.
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """CWD line and Git metadata used by other sections.

    Returns:
        Bash snippet that prints the header and sets `CWD`, `IN_GIT`, and `ROOT`.
    """
    return r"""CWD="$(pwd)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

# --- Check git and resolve its root once ---
IN_GIT=false
ROOT=""
if command -v git >/dev/null 2>&1; then
  GIT_INFO="$(git rev-parse --is-inside-work-tree --show-toplevel 2>/dev/null)"
  GIT_MODE="${GIT_INFO%%$'\n'*}"
  case "$GIT_MODE" in
    true)
      IN_GIT=true
      ROOT="${GIT_INFO#*$'\n'}"
      ;;
    false) IN_GIT=true ;;  # Bare repository or the Git directory itself.
  esac
fi"""


def _section_project() -> str:
    """Language, monorepo, project-root display, virtual-env detection.

    Returns:
        Bash snippet (requires `CWD` and `ROOT` from header).
    """
    return r"""# --- Project ---
PROJ_LANG=""
[ -f pyproject.toml ] || [ -f setup.py ] && PROJ_LANG="python"
[ -z "$PROJ_LANG" ] && [ -f package.json ] && PROJ_LANG="javascript/typescript"
[ -z "$PROJ_LANG" ] && [ -f Cargo.toml ] && PROJ_LANG="rust"
[ -z "$PROJ_LANG" ] && [ -f go.mod ] && PROJ_LANG="go"
[ -z "$PROJ_LANG" ] && { [ -f pom.xml ] || [ -f build.gradle ]; } && PROJ_LANG="java"

MONOREPO=false
{ [ -f lerna.json ] || [ -f pnpm-workspace.yaml ] \
  || [ -d packages ] || { [ -d libs ] && [ -d apps ]; } \
  || [ -d workspaces ]; } && MONOREPO=true

ENVS=""
{ [ -d .venv ] || [ -d venv ]; } && ENVS=".venv"
[ -d node_modules ] && ENVS="${ENVS:+${ENVS}, }node_modules"

HAS_PROJECT=false
{ [ -n "$PROJ_LANG" ] || { [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ]; } \
  || $MONOREPO || [ -n "$ENVS" ]; } && HAS_PROJECT=true

if $HAS_PROJECT; then
  echo "**Project**:"
  [ -n "$PROJ_LANG" ] && echo "- Language: ${PROJ_LANG}"
  [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && echo "- Project root: \`${ROOT}\`"
  $MONOREPO && echo "- Monorepo: yes"
  [ -n "$ENVS" ] && echo "- Environments: ${ENVS}"
  echo ""
fi"""


def _section_package_managers() -> str:
    """Python and Node package manager detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Package managers ---
PKG=""
if [ -f uv.lock ]; then PKG="Python: uv"
elif [ -f poetry.lock ]; then PKG="Python: poetry"
elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then PKG="Python: pipenv"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.uv\]' pyproject.toml 2>/dev/null; then PKG="Python: uv"
  elif grep -q '\[tool\.poetry\]' pyproject.toml 2>/dev/null; then PKG="Python: poetry"
  else PKG="Python: pip"
  fi
elif [ -f requirements.txt ]; then PKG="Python: pip"
fi

NODE_PKG=""
if [ -f bun.lockb ] || [ -f bun.lock ]; then NODE_PKG="Node: bun"
elif [ -f pnpm-lock.yaml ]; then NODE_PKG="Node: pnpm"
elif [ -f yarn.lock ]; then NODE_PKG="Node: yarn"
elif [ -f package-lock.json ] || [ -f package.json ]; then NODE_PKG="Node: npm"
fi
[ -n "$NODE_PKG" ] && PKG="${PKG:+${PKG}, }${NODE_PKG}"
[ -n "$PKG" ] && echo "**Package Manager**: ${PKG}" && echo ""
"""


def _section_runtimes() -> str:
    """Python and Node runtime version detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Runtimes ---
_RT_TMP="${_DCT:-}"
_RT_CLEANUP=false
if [ -z "$_RT_TMP" ]; then
  _RT_TMP="$(mktemp -d)" || exit 1
  _RT_CLEANUP=true
fi

HAS_PYTHON=false
if command -v python3 >/dev/null 2>&1; then
  python3 --version > "$_RT_TMP/runtime_python" 2>/dev/null &
  HAS_PYTHON=true
fi
HAS_NODE=false
if command -v node >/dev/null 2>&1; then
  node --version > "$_RT_TMP/runtime_node" 2>/dev/null &
  HAS_NODE=true
fi
wait

RT=""
if $HAS_PYTHON && [ -s "$_RT_TMP/runtime_python" ]; then
  IFS= read -r PV < "$_RT_TMP/runtime_python"
  PV="${PV#* }"
  PV="${PV%% *}"
  [ -n "$PV" ] && RT="Python ${PV}"
fi
if $HAS_NODE && [ -s "$_RT_TMP/runtime_node" ]; then
  IFS= read -r NV < "$_RT_TMP/runtime_node"
  NV="${NV#v}"
  [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
fi
$_RT_CLEANUP && rm -rf "$_RT_TMP"
[ -n "$RT" ] && echo "**Detected Runtimes**: ${RT}" && echo ""
"""


def _section_git() -> str:
    """Git branch or detached HEAD commit, main branches, uncommitted changes.

    Returns:
        Bash snippet (requires `IN_GIT` from header).
    """
    return r"""# --- Git ---
if $IN_GIT; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if [ "$BRANCH" = "HEAD" ]; then
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null)"
    GT="**Git**: Detached HEAD at \`${COMMIT}\`"
  else
    GT="**Git**: Current branch \`${BRANCH}\`"
  fi

  MAINS=""
  for b in $(git for-each-ref --format='%(refname:short)' \
      refs/heads/main refs/heads/master 2>/dev/null); do
    case "$b" in
      main) MAINS="${MAINS:+${MAINS}, }\`main\`" ;;
      master) MAINS="${MAINS:+${MAINS}, }\`master\`" ;;
    esac
  done
  [ -n "$MAINS" ] && GT="${GT}, ${MAINS} available"

  DC=$(git status --porcelain 2>/dev/null | awk 'END { print NR }')
  if [ "$DC" -gt 0 ]; then
    if [ "$DC" -eq 1 ]; then GT="${GT}, 1 uncommitted change"
    else GT="${GT}, ${DC} uncommitted changes"
    fi
  fi

  echo "$GT"
  echo ""
fi"""


def _section_gh_cli() -> str:
    """GitHub CLI search JSON-field affordances from the installed `gh`.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- GitHub CLI ---
if command -v gh >/dev/null 2>&1; then
  _gh_json_fields() {
    gh search "$1" --help 2>/dev/null \
      | awk '
        /^JSON FIELDS/ { in_fields = 1; next }
        in_fields && /^$/ { exit }
        in_fields {
          sub(/^[[:space:]]+/, "")
          gsub(/[[:space:]]+/, " ")
          fields = fields (fields ? " " : "") $0
        }
        END {
          sub(/^ /, "", fields)
          sub(/ $/, "", fields)
          if (fields != "") print fields
        }
      '
  }

  _GH_TMP="${_DCT:-}"
  _GH_CLEANUP=false
  if [ -z "$_GH_TMP" ]; then
    _GH_TMP="$(mktemp -d)" || exit 1
    _GH_CLEANUP=true
  fi
  _gh_json_fields prs > "$_GH_TMP/gh_prs_fields" &
  _gh_json_fields issues > "$_GH_TMP/gh_issues_fields" &
  wait

  GH_PRS_FIELDS=""
  GH_ISSUES_FIELDS=""
  [ -s "$_GH_TMP/gh_prs_fields" ] \
    && IFS= read -r GH_PRS_FIELDS < "$_GH_TMP/gh_prs_fields"
  [ -s "$_GH_TMP/gh_issues_fields" ] \
    && IFS= read -r GH_ISSUES_FIELDS < "$_GH_TMP/gh_issues_fields"
  $_GH_CLEANUP && rm -rf "$_GH_TMP"
  if [ -n "$GH_PRS_FIELDS" ] || [ -n "$GH_ISSUES_FIELDS" ]; then
    echo "**GitHub CLI**:"
    [ -n "$GH_PRS_FIELDS" ] \
      && echo "- \`gh search prs --json\` fields: ${GH_PRS_FIELDS}"
    [ -n "$GH_ISSUES_FIELDS" ] \
      && echo "- \`gh search issues --json\` fields: ${GH_ISSUES_FIELDS}"
    case ",$GH_PRS_FIELDS," in
      *mergedAt*) ;;
      *) echo "- \`gh search prs --json\` does not expose \`mergedAt\`;"
         echo "  use \`gh pr view --json mergedAt\` per PR for merge timestamps." ;;
    esac
    echo ""
  fi
fi"""


def _section_test_command() -> str:
    """Test command detection (make test / pytest / npm test).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Test command ---
TC=""
if [ -f Makefile ] && grep -qE '^tests?:' Makefile 2>/dev/null; then TC="make test"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.pytest' pyproject.toml 2>/dev/null \
      || [ -f pytest.ini ] || [ -d tests ] || [ -d test ]; then
    TC="pytest"
  fi
elif [ -f package.json ] \
    && grep -q '"test"' package.json 2>/dev/null; then
  TC="npm test"
fi
[ -n "$TC" ] && echo "**Run Tests**: \`${TC}\`" && echo ""
"""


def _section_files() -> str:
    """Directory listing (filtered, capped at 20).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Files ---
FILE_SUMMARY=$(
  { ls -1 2>/dev/null; [ -e .deepagents ] && echo .deepagents; } |
  sort -u |
  awk '
    BEGIN {
      excluded["node_modules"] = excluded["__pycache__"] = 1
      excluded[".pytest_cache"] = excluded[".mypy_cache"] = 1
      excluded[".ruff_cache"] = excluded[".tox"] = 1
      excluded[".coverage"] = excluded[".eggs"] = 1
      excluded["dist"] = excluded["build"] = 1
    }
    !($0 in excluded) {
      total++
      if (shown < 20) files[++shown] = $0
    }
    END {
      print total + 0
      print shown + 0
      for (i = 1; i <= shown; i++) print files[i]
    }
  '
)
TOTAL="${FILE_SUMMARY%%$'\n'*}"
FILE_DETAILS="${FILE_SUMMARY#*$'\n'}"
SHOWN="${FILE_DETAILS%%$'\n'*}"
SHOWN_FILES="${FILE_DETAILS#*$'\n'}"

if [ "$TOTAL" -gt 0 ]; then
  if [ "$SHOWN" -lt "$TOTAL" ]; then
    echo "**Files** (showing ${SHOWN} of ${TOTAL}):"
  else
    echo "**Files** (${TOTAL}):"
  fi
  while IFS= read -r f; do
    if [ -d "$f" ]; then echo "- ${f}/"
    else echo "- ${f}"
    fi
  done <<< "$SHOWN_FILES"
  echo ""
fi"""


def _section_tree() -> str:
    """`tree -L 3` output.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Tree ---
if command -v tree >/dev/null 2>&1; then
  TREE_EXCL='node_modules|.venv|__pycache__|.pytest_cache'
  TREE_EXCL="${TREE_EXCL}|.git|.mypy_cache|.ruff_cache"
  TREE_EXCL="${TREE_EXCL}|.tox|.coverage|.eggs|dist|build"
  T_PREVIEW=$(tree -L 3 --noreport --dirsfirst \
    -I "$TREE_EXCL" 2>/dev/null | sed -n '1,22p;23{p;q;}')
  if [ -n "$T_PREVIEW" ]; then
    PREVIEW_LINES=$(printf '%s\n' "$T_PREVIEW" | awk 'END { print NR }')
    T="$T_PREVIEW"
    TREE_TRUNCATED=false
    if [ "$PREVIEW_LINES" -gt 22 ]; then
      T=$(printf '%s\n' "$T_PREVIEW" | sed -n '1,22p')
      TREE_TRUNCATED=true
    fi
    echo "**Tree** (3 levels):"
    echo '```text'
    echo "$T"
    $TREE_TRUNCATED && echo "... (more lines truncated)"
    echo '```'
    echo ""
  fi
fi"""


def _section_makefile() -> str:
    """First 20 lines of Makefile (falls back to git root in monorepos).

    Returns:
        Bash snippet (requires `ROOT` and `CWD` from `_section_header`).
    """
    return r"""# --- Makefile ---
MK=""
if [ -f Makefile ]; then
  MK="Makefile"
elif [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && [ -f "${ROOT}/Makefile" ]; then
  MK="${ROOT}/Makefile"
fi
if [ -n "$MK" ]; then
  echo "**Makefile** (\`${MK}\`, first 20 lines):"
  echo '```makefile'
  awk 'NR <= 20 { print; next } { print "... (truncated)"; exit }' "$MK"
  echo '```'
fi"""


def build_detect_script() -> str:
    """Concatenate all section functions into the full detection script.

    Independent sections run as parallel background jobs writing to temp
    files, then results are concatenated in the original display order.
    The header (sets `CWD`, `IN_GIT`, and `ROOT`) and project section run first
    because later sections depend on their variables.

    Returns:
        Complete bash heredoc ready for `backend.execute()`.
    """
    # Header (sets CWD, IN_GIT, ROOT) + project run synchronously for others
    serial_prefix = f"{_section_header()}\n{_section_project()}"

    # These sections are independent — run them in parallel.
    # Subshells inherit parent variables (IN_GIT, ROOT, CWD) via fork.
    # Individual exit codes are not tracked because sections legitimately
    # exit non-zero when they have nothing to report (e.g. no runtimes).
    parallel_sections = [
        ("02_pkgmgr", _section_package_managers()),
        ("03_runtimes", _section_runtimes()),
        ("04_git", _section_git()),
        ("05_gh_cli", _section_gh_cli()),
        ("06_testcmd", _section_test_command()),
        ("07_files", _section_files()),
        ("08_tree", _section_tree()),
        ("09_makefile", _section_makefile()),
    ]

    # Build parallel wrapper: each section runs in a subshell writing to a
    # temp file. Section stderr is discarded to prevent noise leakage.
    parallel_setup = "_DCT=$(mktemp -d) || exit 1\ntrap 'rm -rf \"$_DCT\"' EXIT"
    parallel_block = "\n".join(
        f'(\n{body}\n) > "$_DCT/{name}" 2>/dev/null &'
        for name, body in parallel_sections
    )
    cat_line = "cat " + " ".join(f'"$_DCT/{name}"' for name, _ in parallel_sections)

    body = f"{serial_prefix}\n{parallel_setup}\n{parallel_block}\nwait\n{cat_line}"
    return f"bash <<'__DETECT_CONTEXT_EOF__'\n{body}\n__DETECT_CONTEXT_EOF__\n"


DETECT_CONTEXT_SCRIPT = build_detect_script()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LocalContextState(AgentState):
    """State for local context middleware."""

    _local_context: NotRequired[Annotated[str, PrivateStateAttr]]
    """Private formatted local context cached for prompt injection.

    The context is intentionally stored in private state rather than recomputed
    before every model call: volatile sections such as git status, file lists,
    and directory trees would otherwise churn the system prompt and reduce
    provider prompt-cache hits across a conversation.
    """

    _local_context_refreshed_at_cutoff: NotRequired[Annotated[int, PrivateStateAttr]]
    """Cutoff index of the summarization event we last refreshed for.

    Stored in LangGraph checkpointed state (isolated per thread) and private
    (not exposed to subagents via `PrivateStateAttr`). Used to avoid redundant
    re-runs of the detection script for the same summarization event.
    """


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LocalContextMiddleware(AgentMiddleware):
    """Inject local context (git state, project structure, etc.) into the system prompt.

    Runs a bash detection script via `backend.execute()` on first interaction
    and again after each summarization event, stores the result in state, and
    appends it to the system prompt on every model call.

    Because the script runs inside the backend, it works for both local shells
    and remote sandboxes.
    """

    state_schema = LocalContextState

    def __init__(
        self,
        backend: _ExecutableBackend | _AsyncExecutableBackend,
        *,
        mcp_server_info: list[MCPServerInfo] | None = None,
        tracing_project: str | None = None,
        user_tracing_project: str | None = None,
    ) -> None:
        """Initialize with a backend that supports shell execution.

        Args:
            backend: Backend instance that provides shell command execution.
            mcp_server_info: MCP server metadata to include in the system prompt.
            tracing_project: LangSmith project the agent's own runs trace to, or
                `None` when tracing is disabled (the tracing section is omitted).
            user_tracing_project: User's original `LANGSMITH_PROJECT` used by
                shell commands the agent runs.
        """
        self.backend = backend
        tracing_context = _build_tracing_context(tracing_project, user_tracing_project)
        mcp_context = _build_mcp_context(mcp_server_info or [])
        self._static_context = "\n\n".join(
            context for context in (tracing_context, mcp_context) if context
        )

    @staticmethod
    def _handle_detect_result(result: ExecuteResponse) -> str | None:
        """Validate detection script output and normalize it for state storage.

        Args:
            result: Execution result from the backend.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection script %s; "
                "context will be omitted. Output: %.200s",
                f"exited with code {result.exit_code}"
                if result.exit_code is not None
                else "did not report an exit code",
                output or "(empty)",
            )
            return None
        if not output:
            logger.debug(
                "Local context detection script succeeded but produced no output"
            )
        return output or None

    def _run_detect_script(self) -> str | None:
        """Run the environment detection script.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        backend = self.backend
        if not isinstance(backend, _ExecutableBackend):
            logger.debug(
                "Skipping sync local context detection; backend %s only "
                "supports async execution",
                type(backend).__name__,
            )
            return None
        try:
            result = backend.execute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except NotImplementedError:
            # Expected for async-only backends (e.g. HarborSandbox) that
            # define a stub execute() raising NotImplementedError.
            logger.debug(
                "Backend %s does not support sync execute; "
                "context detection deferred to async path",
                type(backend).__name__,
            )
            return None
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    # override - state parameter is intentionally narrowed from
    # AgentState to LocalContextState for type safety within this middleware.
    def before_agent(  # ty: ignore[invalid-method-override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """Run context detection on first interaction and refresh after summarization.

        On the first invocation, runs the detection script and stores the result.
        After a summarization event (indicated by a new `_summarization_event`
        in state), re-runs the script to capture any environment changes that
        occurred during the session.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with `_local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `_local_context`) to prevent
                retry loops.

                Returns `None` if context is already set and no refresh is
                needed, or if initial detection fails.
        """
        # --- Post-summarization refresh ---
        # _summarization_event is a private field from SummarizationState.
        # At runtime the merged state dict contains all middleware fields;
        # accessed as untyped dict value because LocalContextState does not
        # (and should not) redeclare it.
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = self._run_detect_script()
                if output:
                    return {
                        "_local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                # Script failed — record cutoff to avoid retry loop,
                # keep existing `_local_context`.
                return {"_local_context_refreshed_at_cutoff": cutoff}

        # --- Initial detection (first invocation) ---
        if state.get("_local_context"):
            return None

        output = self._run_detect_script()
        if output:
            return {"_local_context": output}
        return None

    async def _arun_detect_script(self) -> str | None:
        """Run the environment detection script asynchronously.

        Prefers `aexecute` when the backend implements `_AsyncExecutableBackend`.
        Falls back to running the sync detection script in a thread pool
        for sync-only backends.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        backend = self.backend
        if not (
            isinstance(backend, _AsyncExecutableBackend)
            and inspect.iscoroutinefunction(backend.aexecute)
        ):
            try:
                return await asyncio.to_thread(self._run_detect_script)
            except Exception:
                logger.warning(
                    "Local context detection via sync fallback failed "
                    "(backend: %s); context will be omitted from system prompt",
                    type(backend).__name__,
                    exc_info=True,
                )
                return None
        try:
            result = await backend.aexecute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    async def abefore_agent(  # ty: ignore[invalid-method-override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """Async variant of `before_agent` for use in async execution contexts.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with `_local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `_local_context`) to prevent
                retry loops.

                Returns `None` if context is already set and no refresh is
                needed, or if initial detection fails.
        """
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = await self._arun_detect_script()
                if output:
                    return {
                        "_local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                return {"_local_context_refreshed_at_cutoff": cutoff}

        if state.get("_local_context"):
            return None

        output = await self._arun_detect_script()
        if output:
            return {"_local_context": output}
        return None

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """Append local context and MCP info to the system prompt if available.

        Args:
            request: The model request to potentially modify.

        Returns:
            Modified request with context appended, or `None`.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("_local_context", "")
        system_prompt = request.system_prompt or ""

        if local_context:
            if self._static_context:
                prompt_parts = (system_prompt, local_context, self._static_context)
            else:
                prompt_parts = (system_prompt, local_context)
        elif self._static_context:
            prompt_parts = (system_prompt, self._static_context)
        else:
            return None

        return request.override(system_prompt="\n\n".join(prompt_parts))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject local context into system prompt (async).

        Args:
            request: The model request being processed.
            handler: The async handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)


__all__ = ["LocalContextMiddleware"]
