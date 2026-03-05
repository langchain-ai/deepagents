"""Middleware for injecting local context into system prompt.

Detects git state, project structure, package managers, runtimes, and
directory layout by running a bash script via the backend. Because the
script executes inside the backend (local shell or remote sandbox), the
same detection logic works regardless of where the agent runs.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime


@runtime_checkable
class _ExecutableBackend(Protocol):
    """Any backend that supports `execute(command) -> ExecuteResponse`."""

    def execute(self, command: str) -> ExecuteResponse: ...


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
# independently.
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """CWD line and IN_GIT flag (used by other sections).

    Returns:
        Bash snippet that prints the header and sets `CWD` / `IN_GIT`.
    """
    return r"""CWD="$(pwd)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

# --- Check git once ---
IN_GIT=false
if command -v git >/dev/null 2>&1 \
    && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
fi"""


def _section_project() -> str:
    """Language, monorepo, git root, virtual-env detection.

    Returns:
        Bash snippet (requires `CWD` / `IN_GIT` from header).
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

ROOT=""
$IN_GIT && ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

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
RT=""
if command -v python3 >/dev/null 2>&1; then
  PV="$(python3 --version 2>/dev/null | awk '{print $2}')"
  [ -n "$PV" ] && RT="Python ${PV}"
fi
if command -v node >/dev/null 2>&1; then
  NV="$(node --version 2>/dev/null | sed 's/^v//')"
  [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
fi
[ -n "$RT" ] && echo "**Runtimes**: ${RT}" && echo ""
"""


def _section_git() -> str:
    """Git branch, main branches, uncommitted changes.

    Returns:
        Bash snippet (requires `IN_GIT` from header).
    """
    return r"""# --- Git ---
if $IN_GIT; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  GT="**Git**: Current branch \`${BRANCH}\`"

  MAINS=""
  for b in $(git branch 2>/dev/null | sed 's/^[* ]*//'); do
    case "$b" in
      main) MAINS="${MAINS:+${MAINS}, }\`main\`" ;;
      master) MAINS="${MAINS:+${MAINS}, }\`master\`" ;;
    esac
  done
  [ -n "$MAINS" ] && GT="${GT}, main branch available: ${MAINS}"

  DC=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  if [ "$DC" -gt 0 ]; then
    if [ "$DC" -eq 1 ]; then GT="${GT}, 1 uncommitted change"
    else GT="${GT}, ${DC} uncommitted changes"
    fi
  fi

  echo "$GT"
  echo ""
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
EXCL='node_modules|__pycache__|\.pytest_cache'
EXCL="${EXCL}|\.mypy_cache|\.ruff_cache|\.tox"
EXCL="${EXCL}|\.coverage|\.eggs|dist|build"
FILES=$(
  { ls -1 2>/dev/null; [ -e .deepagents ] && echo .deepagents; } |
  grep -vE "^(${EXCL})$" |
  sort -u
)
if [ -n "$FILES" ]; then
  TOTAL=$(echo "$FILES" | wc -l | tr -d ' ')
  SHOWN_FILES=$(echo "$FILES" | head -20)
  SHOWN=$(echo "$SHOWN_FILES" | wc -l | tr -d ' ')
  echo "**Files** (${SHOWN} shown):"
  echo "$SHOWN_FILES" | while IFS= read -r f; do
    if [ -d "$f" ]; then echo "- ${f}/"
    else echo "- ${f}"
    fi
  done
  [ "$SHOWN" -lt "$TOTAL" ] && echo "... ($((TOTAL - SHOWN)) more files)"
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
  T=$(tree -L 3 --noreport --dirsfirst \
    -I "$TREE_EXCL" 2>/dev/null | head -22)
  if [ -n "$T" ]; then
    echo "**Tree** (3 levels):"
    echo '```text'
    echo "$T"
    echo '```'
    echo ""
  fi
fi"""


def _section_makefile() -> str:
    """First 20 lines of Makefile (falls back to git root in monorepos).

    Returns:
        Bash snippet (requires `ROOT` from `_section_project`).
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
  head -20 "$MK"
  TL=$(wc -l < "$MK" | tr -d ' ')
  [ "$TL" -gt 20 ] && echo "... (truncated)"
  echo '```'
fi"""


def build_detect_script() -> str:
    """Concatenate all section functions into the full detection script.

    Returns:
        Complete bash heredoc ready for `backend.execute()`.
    """
    sections = [
        _section_header(),
        _section_project(),
        _section_package_managers(),
        _section_runtimes(),
        _section_git(),
        _section_test_command(),
        _section_files(),
        _section_tree(),
        _section_makefile(),
    ]
    body = "\n".join(sections)
    return f"bash <<'__DETECT_CONTEXT_EOF__'\n{body}\n__DETECT_CONTEXT_EOF__\n"


DETECT_CONTEXT_SCRIPT = build_detect_script()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LocalContextState(AgentState):
    """State for local context middleware."""

    local_context: NotRequired[str]
    """Formatted local context: cwd, project, package managers,
    runtimes, git, test command, files, tree, Makefile.
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

    def __init__(self, backend: _ExecutableBackend) -> None:
        """Initialize with a backend that supports shell execution.

        Args:
            backend: Backend instance that provides shell command execution.
        """
        self.backend = backend

    def _run_detect_script(self) -> str | None:
        """Run the environment detection script.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        try:
            result = self.backend.execute(DETECT_CONTEXT_SCRIPT)
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(self.backend).__name__,
                exc_info=True,
            )
            return None

        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection script %s; context will be omitted. Output: %.200s",
                f"exited with code {result.exit_code}"
                if result.exit_code is not None
                else "did not report an exit code",
                output or "(empty)",
            )
            return None
        if not output:
            logger.debug("Local context detection script succeeded but produced no output")
        return output or None

    # override - state parameter is intentionally narrowed from
    # AgentState to LocalContextState for type safety within this middleware.
    def before_agent(  # type: ignore[override]
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
            State update with `local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `local_context`) to prevent
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
            event: SummarizationEvent = raw_event  # type: ignore[assignment]
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = self._run_detect_script()
                if output:
                    return {
                        "local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                # Script failed — record cutoff to avoid retry loop,
                # keep existing local_context.
                return {"_local_context_refreshed_at_cutoff": cutoff}

        # --- Initial detection (first invocation) ---
        if state.get("local_context"):
            return None

        output = self._run_detect_script()
        if output:
            return {"local_context": output}
        return None

    @staticmethod
    def _get_modified_request(request: ModelRequest) -> ModelRequest | None:
        """Append local context to the system prompt if available.

        Args:
            request: The model request to potentially modify.

        Returns:
            Modified request with context appended, or `None`.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("local_context", "")

        if not local_context:
            return None

        system_prompt = request.system_prompt or ""
        new_prompt = system_prompt + "\n\n" + local_context
        return request.override(system_prompt=new_prompt)

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
