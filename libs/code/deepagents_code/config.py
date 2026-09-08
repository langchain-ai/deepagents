"""Configuration, constants, and model creation."""

from __future__ import annotations

import functools
import importlib
import json
import keyword
import logging
import os
import re
import shlex
import shutil
import sys
import threading
from collections import UserDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import (
    dataclass,
    field as dataclass_field,
    replace as dataclass_replace,
)
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from deepagents_code._constants import FIREWORKS_PROVIDER_ID_PREFIX
from deepagents_code._env_vars import (
    AUTO_CLASSIFIER_MODEL,
    AUTO_CLASSIFIER_TIMEOUT,
    DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
    DISABLED_PROJECT_MCP_SERVERS,
    FORKED_SUBAGENTS,
    HIDE_SPLASH_VERSION,
    READ_PROJECT_DOTENV,
    UI_CHARSET_MODE,
    is_env_truthy,
)
from deepagents_code._git import resolve_git_branch
from deepagents_code._paths import (
    DEEPAGENTS_HOME_ENV,
    DEFAULT_PROFILE_MARKER_ENV,
    PATHS,
)
from deepagents_code._version import __version__

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Mapping,
        MutableMapping,
        Sequence,
    )

    from langchain_core.runnables import RunnableConfig

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import (
        ConfigResolver,
        RankedProviderValue,
    )
    from deepagents_code.configuration.types import ProviderStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy bootstrap: dotenv loading, LANGSMITH_PROJECT override, and start-path
# detection are deferred until first access of `credentials` (via module
# `__getattr__`).  This avoids disk I/O and path traversal during import for
# callers that never touch credentials (e.g. `deepagents --help`).
# ---------------------------------------------------------------------------


@dataclass
class _BootstrapState:
    """Mutable state captured by `_ensure_bootstrap()`."""

    done: bool = False
    """Whether `_ensure_bootstrap()` has executed."""

    start_path: Path | None = None
    """Working directory captured at bootstrap time for dotenv and discovery."""

    original_langsmith_project: str | None = None
    """Caller's `LANGSMITH_PROJECT` before the app overrides it for traces."""

    launch_langsmith_env: dict[str, str | None] = dataclass_field(default_factory=dict)
    """LangSmith values inherited from the user's launch environment."""

    user_langsmith_env: dict[str, str | None] = dataclass_field(default_factory=dict)
    """Launch and project-dotenv LangSmith values intended for user commands."""

    error: BaseException | None = None
    """Why bootstrap did not finish, when it was cut short.

    Bootstrap tolerates its own failures so the app still starts. But a later
    step that needs captured state can only report "not captured", which names
    a symptom rather than a cause. Keeping the exception lets that step chain
    it.
    """


_bootstrap_state = _BootstrapState()
"""State captured and mutated by lazy bootstrap."""

_bootstrap_lock = threading.Lock()
"""Guards `_ensure_bootstrap()` against concurrent access from the main thread
and the prewarm worker thread."""

_singleton_lock = threading.Lock()
"""Guards lazy construction of process-wide config, runtime, and console state."""

_dotenv_loaded_values: dict[str, str] = {}
"""Environment values injected by our dotenv loader and safe to refresh later."""

_reconciled_tracing_values: dict[str, tuple[str | None, str | None]] = {}
"""Original and published tracing values, kept out of later workspace baselines."""

_orphaned_tracing_disabled_notice: str | None = None
"""One-shot TUI notice populated when bootstrap disables orphaned tracing."""

_INHERITED_PYTHONPATH_ENV = "DEEPAGENTS_INHERITED_PYTHONPATH"
"""Carrier var that relays a launch-time `PYTHONPATH` to agent `execute` commands.

`PYTHONPATH` is stripped from the server interpreter's environment (see
`server._SERVER_ENV_DENYLIST`) to keep an untrusted import path off `sys.path`
during startup. The launch-time value is instead carried in this var and
re-applied only to the approval-gated shell backend's `execute` subprocesses by
`agent._apply_inherited_pythonpath`.
"""

_USER_LANGSMITH_ENV_CARRIER = "DEEPAGENTS_USER_LANGSMITH_ENV"
"""Private client-to-server carrier for user-command LangSmith settings.

Holds JSON with two mappings. `launch` is the user's pre-bootstrap environment;
`user` is `launch` overlaid with the project `.env`. `restore_user_langsmith_env`
treats them differently -- `launch` is the higher-precedence layer -- so the
distinction has to survive the trip.
"""

_TRACING_ENABLE_ENV_VARS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)
"""Env vars LangChain/LangSmith read to decide whether tracing is enabled."""

_TRACING_RUNS_ENDPOINTS_ENV_VARS = (
    "LANGSMITH_RUNS_ENDPOINTS",
    "LANGCHAIN_RUNS_ENDPOINTS",
)
"""Env vars the LangSmith SDK parses into replica trace ingestion targets."""

_USER_LANGSMITH_ENV_VARS = (
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_SESSION",
    "LANGCHAIN_SESSION",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
    "LANGSMITH_PROFILE",
    "LANGSMITH_CONFIG_FILE",
    *_TRACING_ENABLE_ENV_VARS,
    *_TRACING_RUNS_ENDPOINTS_ENV_VARS,
)
"""LangSmith settings restored for approval-gated user commands.

The runs-endpoints vars belong here because `_tracing_can_upload_from` treats
them as a live ingest target: their value is a JSON object carrying `api_url`
and `api_key` pairs, so leaving them out both ignored a replica config the user
set for their own commands and left any agent-side value in place.

`LANGSMITH_GATEWAY_API_KEY` is deliberately absent: it authenticates model
calls, not tracing, and nothing here overwrites it.
"""


def _langsmith_selectors_from(env: Mapping[str, str | None]) -> dict[str, str | None]:
    """Snapshot every supported LangSmith selector from `env`.

    Returns:
        One entry per selector, `None` where the var is absent so applying
        the snapshot removes it rather than writing an empty value.
    """
    return {var: env.get(var) for var in _USER_LANGSMITH_ENV_VARS}


_DOTENV_DENIED_ENV_KEYS = frozenset(
    {
        "BASH_ENV",
        "BASHOPTS",
        "CDPATH",
        "COMSPEC",
        DEEPAGENTS_HOME_ENV,
        DEFAULT_PROFILE_MARKER_ENV,
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GIT_ASKPASS",
        "GIT_DIR",
        "GIT_EDITOR",
        "GIT_EXEC_PATH",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PAGER",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
        "GLOBIGNORE",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "PATH",
        "PYTHONEXECUTABLE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SHELLOPTS",
        "SSH_ASKPASS",
        "SYSTEMROOT",
        "WINDIR",
        READ_PROJECT_DOTENV,
        _INHERITED_PYTHONPATH_ENV,
        _USER_LANGSMITH_ENV_CARRIER,
    }
)
"""Environment keys that no `.env` file may inject.

Project dotenv files are untrusted (they travel with cloned repositories), and
even the global dotenv is loaded after the launch profile has been selected.
Neither may replace that profile/trust root. The remaining entries prevent a
dotenv file from turning environment loading into code execution in child
processes. Every entry belongs to one of these categories, so do not remove one
without checking which category it belongs to:

- Profile/trust relocation (`DEEPAGENTS_HOME`): this is captured from the
    inherited environment before dotenv loading. Allowing either dotenv layer
    to change it would make project-controlled configuration capable of moving
    the files treated as user-trusted.

- Dynamic-linker preload/audit (`DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`,
    `LD_AUDIT`, `LD_LIBRARY_PATH`, `LD_PRELOAD`): force a loader to map an
    attacker-supplied shared object into every spawned binary.
- Interpreter startup/path (`NODE_OPTIONS`, `PATH`, `PYTHONEXECUTABLE`,
    `PYTHONHOME`, `PYTHONPATH`, `PYTHONSTARTUP`, `_INHERITED_PYTHONPATH_ENV`):
    hijack which interpreter/binary runs or what it imports at startup.
- Shell startup hooks (`BASH_ENV`, `ENV`, `BASHOPTS`, `SHELLOPTS`, `CDPATH`,
    `GLOBIGNORE`): `BASH_ENV`/`ENV` source a file on every non-interactive shell;
    `SHELLOPTS`/`BASHOPTS` can force `xtrace`/alias expansion; `CDPATH`/
    `GLOBIGNORE` alter path/glob resolution. The agent runs detection and
    `execute` commands through non-interactive shells, so these are live vectors.
- Askpass hijack (`GIT_ASKPASS`, `SSH_ASKPASS`): point credential prompts at an
    attacker-controlled binary.
- Git config/exec injection (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_OBJECT_DIRECTORY`,
    `GIT_EXEC_PATH`, `GIT_EDITOR`, `GIT_PAGER`, `GIT_SSH`, `GIT_SSH_COMMAND`):
    `dcode` runs `git rev-parse`/`git status`/`git for-each-ref` during startup
    local-context detection (`local_context.build_detect_script`), before any
    HITL approval. These keys redirect git's object store/exec path or hook its
    editor/pager/transport helpers into attacker-controlled binaries. The
    numbered `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n` family
    and the inline `GIT_CONFIG_PARAMETERS` blob are worse: they inject arbitrary
    config values such as `core.fsmonitor`/`core.pager`/`core.sshCommand`, which
    git executes. The numbered keys cannot be enumerated in a static set, so
    they are matched by prefix in `_is_dotenv_denied_env_key`.

`_INHERITED_PYTHONPATH_ENV` is denied so a project `.env` cannot smuggle a
`PYTHONPATH` into agent `execute` commands through the carrier var; the carrier
is only meant to relay a value the user set in their launch environment.
`_USER_LANGSMITH_ENV_CARRIER` is denied for the same reason: it decides which
LangSmith flags and API key user commands run under, so only the client's own
capture -- its launch environment plus the project `.env` -- may populate it.

`READ_PROJECT_DOTENV` is denied from *every* `.env` (not just the project one)
because it is a trust decision about the loader itself: if a project `.env`
could set it, first-write-wins would let that file pin it `true` and block the
trusted global `.env` from opting out, and if the global file set it the
project file would already have loaded by the time it was read. The option is
resolved from the trusted global file (read directly, before the project file)
plus the process env and config.toml, never from dotenv injection.

Matching is case-sensitive on POSIX because the protected consumers (the
dynamic linker, bash, CPython, git) read these names only in their canonical
case, so a lowercase `bash_env` injected into the environment is inert there.
On Windows, however, environment variable names are case-insensitive and
`os.environ` normalizes assigned keys to uppercase, so a lowercase
`git_config_key_0` in a `.env` would become an active `GIT_CONFIG_KEY_0` for a
spawned `git`. `_is_dotenv_denied_env_key` therefore compares the uppercased
key, which is a superset on POSIX (denied names are already uppercase, so the
extra denials like `git_config_count` are of otherwise-inert spellings).
"""

_DOTENV_DENIED_ENV_KEY_PREFIXES = (
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_GLOBAL",
)
"""Prefixes of env keys that must not be injected from a `.env` file.

`GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n` are numbered pairs and cannot be listed
exhaustively; the rest are single keys whose `GIT_CONFIG_` prefix makes them
unambiguous config-injection vectors. `GIT_CONFIG_SYSTEM`/`GIT_CONFIG_GLOBAL`
point git at attacker-controlled config files, which can carry the same
executable values (`core.fsmonitor`, `core.pager`, ...).
"""


def _is_dotenv_denied_env_key(key: str) -> bool:
    """Return whether a dotenv key is denied from any `.env` file.

    Combines the exact-match `_DOTENV_DENIED_ENV_KEYS` set with the
    `_DOTENV_DENIED_ENV_KEY_PREFIXES` family so numbered git config keys
    (`GIT_CONFIG_KEY_0`, ...) are denied without enumeration. The key is
    uppercased before both checks so a lowercase or mixed-case spelling cannot
    slip past on Windows, where `os.environ` assignment would normalize it back
    to the active uppercase form.
    """
    normalized = key.upper()
    return normalized in _DOTENV_DENIED_ENV_KEYS or normalized.startswith(
        _DOTENV_DENIED_ENV_KEY_PREFIXES
    )


def _report_denied_env_key(key: str, dotenv_path: Path, *, is_project: bool) -> None:
    """Report a denied dotenv key at a level matching who could have set it.

    A project `.env` is untrusted, so a denied key there is expected and stays
    at debug. The user's own global `.env` is trusted. Silently dropping a key
    the user deliberately wrote leaves them with a setting that never takes
    effect and no way to find out why, so every denied key from that file is
    reported, not `DEEPAGENTS_HOME` alone.

    The report goes to stderr as well as the logger. The package installs a
    buffering handler at import, which stops `logging.lastResort` from writing
    warnings to the terminal, so a `logger.warning` alone would be visible only
    under `--debug`. `_debug` prints and logs for the same reason.

    `DEEPAGENTS_HOME` gets its own sentence: it selects the profile that owns
    this file, so it cannot be read from it.
    """
    if is_project:
        # Log the key only — the value is attacker-controlled.
        logger.debug("Ignoring denied env key %r from %s", key, dotenv_path)
        return
    if key.upper() == DEEPAGENTS_HOME_ENV:
        message = (
            f"Ignoring {DEEPAGENTS_HOME_ENV} in {dotenv_path}: it selects the "
            "profile that owns that file, so it must be set in the launching "
            "shell environment instead (for example 'export "
            f"{DEEPAGENTS_HOME_ENV}=...')."
        )
    else:
        message = (
            f"Ignoring {key!r} in {dotenv_path}: this variable cannot be set "
            "from a .env file. Set it in your shell environment instead."
        )
    print(f"Warning: {message}", file=sys.stderr)  # noqa: T201  # user-facing
    logger.warning("%s", message)


_LANGGRAPH_DEFAULT_RECURSION_LIMIT_ENV = "LANGGRAPH_DEFAULT_RECURSION_LIMIT"
"""Upstream recursion default that only trusted user input may override."""


_PROJECT_DOTENV_DENIED_ENV_KEYS = frozenset(
    {
        DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        DISABLED_PROJECT_MCP_SERVERS,
        AUTO_CLASSIFIER_MODEL,
        AUTO_CLASSIFIER_TIMEOUT,
        FORKED_SUBAGENTS,
        _LANGGRAPH_DEFAULT_RECURSION_LIMIT_ENV,
        "TERM_PROGRAM",
    }
)
"""Env keys a *project* `.env` must not inject, even though they are otherwise
safe process-env inputs.

The first two are the env form of the user-level project-MCP allow/deny lists
(`model_config.load_mcp_server_trust_lists`). Their whole purpose is to be a
*user-level* decision: naming a project MCP server here pre-approves it from an
untrusted `.mcp.json` (stdio → local command execution; remote → SSRF and
`${VAR}` header exfiltration during the discovery preflight). A project `.env`
travels with a cloned repo, so honoring it would let an attacker commit
`.mcp.json` + `.env` and self-approve their own servers — exactly the trust
boundary the feature exists to hold.

`AUTO_CLASSIFIER_MODEL` chooses the model that authorizes gated tool calls in
Auto approval mode. Honoring it from a project `.env` would let a cloned repo
silently point that review at a weaker model — degrading the control, including
its resistance to prompt injection in the untrusted material it reads — which is
a repo-supplied downgrade of a user-level security decision, not a project build
setting. Choosing a classifier stays available through the trusted surfaces:
shell exports, the global `~/.deepagents/.env`, `[models].auto_classifier` in
`~/.deepagents/config.toml`, `--auto-classifier-model`, and `/auto model`.

`FORKED_SUBAGENTS` controls whether dcode's built-in `general-purpose` subagent
uses fork mode. A cloned repository must not alter whether the subagent inherits
the parent conversation and private state, so only the shell or global `.env`
may set it.

`LANGGRAPH_DEFAULT_RECURSION_LIMIT` controls the graph step budget whenever no
Deep Agents override is configured. A project value would bypass the bounded
`runtime.recursion_limit` resolver, so only the shell or global `.env` may set
the upstream default.

Membership is checked against the uppercased key for the same reason as
`_is_dotenv_denied_env_key`: on Windows, `os.environ` normalizes assigned keys
to uppercase, so a lowercase `langgraph_default_recursion_limit` would slip
past an exact-match check yet become the active variable the local API reads.

`AUTO_CLASSIFIER_TIMEOUT` tunes the same control's review deadline, so it is
denied for the same reason: a cloned repo could otherwise stall every gated
batch up to the ceiling, or squeeze the budget until reviews time out and the
session degrades into repeated denials and approval prompts.
`[models].auto_classifier_timeout` in
`~/.deepagents/config.toml` and the trusted env surfaces still set it.

`TERM_PROGRAM` identifies the terminal from which the user launched Deep
Agents Code and is included in trace metadata. A project `.env` must not
replace it: python-dotenv expands environment-variable references, which could
otherwise forward a launch-environment value to the trace backend.

Unlike `_DOTENV_DENIED_ENV_KEYS` (denied from *any* `.env` because they turn
`.env` loading into code execution), these are denied only from the *project*
`.env`: the user's own global `~/.deepagents/.env` and their shell exports are
legitimate, trusted sources and continue to set them. The loader reads plain
`os.environ`, so blocking injection here — before the value ever reaches
`os.environ` — is what keeps that read trustworthy.
"""


def _find_dotenv_from_start_path(start_path: Path) -> Path | None:
    """Find the nearest `.env` file from an explicit start path upward.

    Args:
        start_path: Directory to start searching from.

    Returns:
        Path to the nearest `.env` file, or `None` if not found.
    """
    current = start_path.expanduser().resolve()
    for parent in [current, *list(current.parents)]:
        candidate = parent / ".env"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            logger.warning("Could not inspect .env candidate %s", candidate)
            continue
    return None


# Frozen before either dotenv layer is inspected.
_GLOBAL_DOTENV_PATH = PATHS.profile.dotenv_file


def _dotenv_files_are_same(first: Path | None, second: Path) -> bool:
    """Return whether two dotenv paths identify the same file."""
    if first is None:
        return False
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        # Identity uncertainty must not let a project file become trusted by
        # loading it again through the configured profile path.
        return True


_active_environment: ContextVar[Mapping[str, str] | None] = ContextVar(
    "deepagents_code_active_environment", default=None
)


def active_environment() -> Mapping[str, str]:
    """Return the construction-scoped environment or the live process environment.

    Returns:
        Active immutable snapshot or `os.environ` outside construction.
    """
    environment = _active_environment.get()
    return os.environ if environment is None else environment


def _environment_is_scoped() -> bool:
    """Return whether a construction-scoped environment snapshot is bound.

    Credential readers fork on this: under a workspace snapshot they resolve
    from the snapshot, because the process-global caches describe the launch
    environment instead.

    Returns:
        `True` while a `use_environment` block is active.
    """
    return _active_environment.get() is not None


@contextmanager
def use_environment(environ: Mapping[str, str] | None) -> Iterator[None]:
    """Bind an immutable environment snapshot for the duration of the block.

    The binding is a `ContextVar`, so it is copied into `asyncio.to_thread`
    workers, and it ends when the block exits. Anything that runs later -- a
    lazy model switch, a summary, a compaction rebuild -- is outside it, which
    is why those middlewares carry an `environ` field and re-enter this
    contextmanager at request time instead of relying on the ambient binding.

    `None` binds nothing, and `active_environment` then falls back to
    `os.environ`.
    """
    if environ is None:
        yield
        return
    frozen = (
        environ
        if isinstance(environ, MappingProxyType)
        else MappingProxyType(dict(environ))
    )
    token = _active_environment.set(frozen)
    try:
        yield
    finally:
        _active_environment.reset(token)


def _environment_key(key: str) -> str:
    """Normalize an environment key using the host platform's semantics.

    Returns:
        Uppercase key on Windows, otherwise the original key.
    """
    return key.upper() if sys.platform == "win32" else key


class _InterpolationEnv(UserDict[str, "str | None"]):
    """Interpolation mapping that normalizes reference names like the host.

    Keys are stored normalized, but a `${...}` reference carries the name as
    written. On Windows `${proxy_url}` must find `PROXY_URL`, the way a lookup
    in `os.environ` would.
    """

    def __getitem__(self, key: str) -> str | None:
        return self.data[_environment_key(key)]

    def get(self, key: str, default: str | None = None) -> str | None:
        """Look a reference name up through the host's key normalization.

        `UserDict.get` reads `self.data` directly, so it has to be overridden
        alongside `__getitem__`. `dotenv` resolves every reference with `get`.

        Returns:
            The stored value, or `default` when the name is absent.
        """
        return self.data.get(_environment_key(key), default)


def _dotenv_values_from(
    dotenv_path: Path,
    environ: Mapping[str, str],
) -> dict[str, str | None]:
    """Parse one dotenv file with interpolation against an explicit mapping.

    Interpolation is hand-rolled because `dotenv`'s own resolver layers
    `os.environ` into the mapping it interpolates against (see
    `dotenv.main.resolve_variables`), which would defeat the scoped snapshot in
    `environ`. So `dotenv_values(interpolate=True)` is not a valid
    simplification here, however much it looks like one.

    `environ` stays the top layer, matching how the caller applies the parsed
    values: a key already present there wins, so a `${...}` reference to it must
    resolve to the winning value and not to the shadowed one in this file.

    Returns:
        Parsed values with references resolved against `environ`.
    """
    from dotenv.main import DotEnv
    from dotenv.variables import parse_variables

    parsed = DotEnv(
        dotenv_path=str(dotenv_path),
        interpolate=False,
    ).dict()
    resolved: dict[str, str | None] = {}
    # Built once: `resolved` only grows, and each new key is folded in below, so
    # later values see every earlier one without re-copying the environment.
    interpolation_env = _InterpolationEnv(
        {_environment_key(key): value for key, value in environ.items()}
    )
    for parsed_key, value in parsed.items():
        key = _environment_key(parsed_key)
        resolved[key] = (
            None
            if value is None
            else "".join(
                atom.resolve(interpolation_env) for atom in parse_variables(value)
            )
        )
        # `environ` outranks this file, so a key it already carries keeps its
        # value for every later reference.
        if key not in interpolation_env:
            interpolation_env[key] = resolved[key]
    return resolved


def _dotenv_environment(
    *,
    start_path: Path | None,
    environ: Mapping[str, str],
    include_global: bool = True,
    unreadable: list[Path] | None = None,
    project_layer: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply the project/global dotenv stack to an explicit environment mapping.

    Args:
        start_path: Directory to begin project dotenv discovery from.
        environ: Environment the files are layered under. Its keys win.
        include_global: Whether the global profile `.env` contributes. Pass
            `False` for the user-command path: the global file configures the
            agent, not the user's own commands.
        unreadable: Collects the path of each file that could not be read, for
            a caller that must tell "the file sets nothing" apart from "the file
            could not be read". A read failure is otherwise only logged.
        project_layer: Filled with the result as it stands before the global
            `.env` contributes, i.e. what `include_global=False` would return.
            Lets a caller that needs both layers get them from one pass instead
            of re-walking and re-parsing the whole stack.

    Returns:
        A new effective environment mapping.
    """
    env = {_environment_key(key): value for key, value in environ.items()}
    # The baseline is the caller's environment, before any file contributes to
    # it. Each file interpolates against this rather than against `env`, so one
    # file's values can never expand into another's. Without it a project `.env`
    # reaches keys it is denied (`_PROJECT_DOTENV_DENIED_ENV_KEYS`) by defining
    # a name the trusted global `.env` interpolates.
    baseline = dict(env)

    def apply_dotenv(dotenv_path: Path | None, *, is_project: bool) -> None:
        if dotenv_path is None:
            return
        try:
            values = _dotenv_values_from(dotenv_path, baseline)
        except (OSError, ValueError):
            logger.warning(
                "Could not read dotenv at %s; environment may be incomplete",
                dotenv_path,
                exc_info=True,
            )
            if unreadable is not None:
                unreadable.append(dotenv_path)
            return
        for key, value in values.items():
            if value is None:
                continue
            if _is_dotenv_denied_env_key(key):
                _report_denied_env_key(key, dotenv_path, is_project=is_project)
                continue
            if is_project and key.upper() in _PROJECT_DOTENV_DENIED_ENV_KEYS:
                # A committed project `.env` must not set a user-level trust
                # decision -- MCP trust lists, or the Auto classifier model and
                # deadline that authorize this repo's own tool calls; the
                # global `.env` and the shell may (is_project=False). The key
                # is uppercased so a case variant cannot slip past on Windows,
                # where `os.environ` assignment normalizes it back to the
                # active uppercase form.
                logger.debug(
                    "Ignoring project-denied env key %r from %s", key, dotenv_path
                )
                continue
            if key in env:
                continue
            env[key] = value

    discovery_root = start_path or Path.cwd()
    try:
        project_dotenv = _find_dotenv_from_start_path(discovery_root)
    except OSError:
        logger.warning(
            "Could not inspect project dotenv at %s; environment may be incomplete",
            start_path or "cwd",
            exc_info=True,
        )
        project_dotenv = None
        if unreadable is not None:
            unreadable.append(discovery_root)
    global_is_project = _dotenv_files_are_same(project_dotenv, _GLOBAL_DOTENV_PATH)

    # The project file loads first, so the trusted global `.env` can only fill
    # in vars it left unset. `read_project_dotenv` is denied from *every* `.env`
    # (see `_DOTENV_DENIED_ENV_KEYS`), so neither file can inject it -- but the
    # global file is a legitimate place to opt out, so read just that key from
    # it *before* the project file is touched. Otherwise a hostile project file
    # would load, and could pin the var true, before the trusted opt-out was
    # ever seen. The ordering here is the control; it is not incidental.
    global_toggle: dict[str, str] = {}
    try:
        if not global_is_project and _GLOBAL_DOTENV_PATH.is_file():
            raw = _dotenv_values_from(_GLOBAL_DOTENV_PATH, baseline).get(
                READ_PROJECT_DOTENV
            )
            if raw is not None:
                global_toggle[READ_PROJECT_DOTENV] = raw
    except (OSError, ValueError):
        logger.warning(
            "Could not read global dotenv at %s; global defaults may be incomplete",
            _GLOBAL_DOTENV_PATH,
            exc_info=True,
        )

    from deepagents_code.config_manifest import resolve_read_project_dotenv

    if resolve_read_project_dotenv(global_dotenv=global_toggle):
        apply_dotenv(project_dotenv, is_project=True)
    else:
        logger.debug(
            "Skipping project dotenv at %s: startup.read_project_dotenv is false",
            start_path or "cwd",
        )
    if project_layer is not None:
        project_layer.update(env)
    if include_global and not global_is_project:
        try:
            global_dotenv = (
                _GLOBAL_DOTENV_PATH if _GLOBAL_DOTENV_PATH.is_file() else None
            )
        except OSError:
            logger.warning(
                "Could not inspect global dotenv at %s; global defaults may be "
                "incomplete",
                _GLOBAL_DOTENV_PATH,
                exc_info=True,
            )
            global_dotenv = None
        apply_dotenv(global_dotenv, is_project=False)
    return env


def _preview_dotenv_environ(*, start_path: Path | None = None) -> dict[str, str]:
    """Return the effective dotenv environment without mutating `os.environ`.

    Returns:
        Effective environment for the requested project path.
    """
    env = _environment_before_tracing_reconcile()
    _strip_dotenv_loaded_values(env)
    return _dotenv_environment(start_path=start_path, environ=env)


def _apply_env_values(
    env: MutableMapping[str, str], values: Mapping[str, str | None]
) -> None:
    """Apply a resolved environment snapshot to `env` in place.

    `None` is how every snapshot here spells "the user did not set this", so it
    removes the key rather than writing an empty value -- an empty
    `LANGSMITH_API_KEY` is not the same as an absent one to the SDK.
    """
    for key, value in values.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value


def _strip_dotenv_loaded_values(env: MutableMapping[str, str]) -> None:
    """Remove values this loader injected, keeping any modified since.

    Callers that build a child environment or reload the stack have to start
    from the shell's own values rather than from a previous application.
    """
    for key, value in list(_dotenv_loaded_values.items()):
        if env.get(key) == value:
            env.pop(key)


def _resolve_env_var_from(env: Mapping[str, str], name: str) -> str | None:
    """Resolve an env var from a mapping using app prefix precedence.

    Returns:
        The resolved value, or `None` when absent or empty.
    """
    from deepagents_code.model_config import _ENV_PREFIX

    if not name.startswith(_ENV_PREFIX):
        prefixed = f"{_ENV_PREFIX}{name}"
        if prefixed in env:
            return env[prefixed] or None
    return env.get(name) or None


def _user_langsmith_project_from(env: Mapping[str, str]) -> str | None:
    """Return the caller's project when bootstrap replaced the canonical value."""
    from deepagents_code._env_vars import LANGSMITH_PROJECT

    project = env.get("LANGSMITH_PROJECT")
    agent_project = env.get(LANGSMITH_PROJECT)
    if _bootstrap_state.done and agent_project and project == agent_project:
        return _bootstrap_state.original_langsmith_project
    return project


def _load_dotenv(
    *,
    start_path: Path | None = None,
    refresh_loaded: bool = False,
    capture_user_langsmith: bool = False,
) -> bool:
    """Load the effective project/global dotenv stack into `os.environ`.

    Shell values win over the nearest project `.env`, which wins over the global
    profile `.env`. Previously injected values are removable for serialized
    single-workspace reloads; workspace server runtimes use immutable previews.

    Args:
        start_path: Directory to use for project `.env` discovery.
        refresh_loaded: Remove values previously injected by this loader before
            applying the current project/global dotenv stack. Values modified
            after loading are preserved.
        capture_user_langsmith: Save launch and project-dotenv LangSmith values
            before global defaults are applied, initializing a missing launch
            snapshot from the environment below refreshed dotenv values.

    Returns:
        Whether dotenv loading injected at least one value.
    """
    if refresh_loaded:
        _strip_dotenv_loaded_values(os.environ)
        _dotenv_loaded_values.clear()

    baseline = dict(os.environ)
    # The project layer alone, because the global profile `.env` configures the
    # agent rather than the user's own commands. Collected from the same pass:
    # a second `include_global=False` call would re-walk the tree and re-parse
    # every file, and the two passes would then have to stay identical for the
    # capture to keep meaning what it says.
    project: dict[str, str] | None = {} if capture_user_langsmith else None
    if capture_user_langsmith:
        _initialize_launch_langsmith_env(baseline)
    effective = _dotenv_environment(
        start_path=start_path, environ=baseline, project_layer=project
    )
    for key, value in effective.items():
        if key not in baseline:
            os.environ[key] = value
            _dotenv_loaded_values[key] = value
    if project is not None:
        _bootstrap_state.user_langsmith_env = _langsmith_selectors_from(project)
    return bool(effective.keys() - baseline.keys())


_TRACING_API_KEY_ENV_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
"""Env vars that hold the LangSmith API key used for trace ingestion."""

_TRACING_BRIDGED_ENABLE_ENV_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
"""Tracing flags bootstrap propagates from a `DEEPAGENTS_CODE_` prefix.

`dcode doctor` runs before `_apply_prefixed_langsmith_env` bridges these to
their canonical names, so it must resolve them prefix-aware (via
`resolve_env_var`) to predict the runtime's effective state.

The remaining flags in `_TRACING_ENABLE_ENV_VARS` are not bridged, so only
their canonical form takes effect.
"""

_PREFIXED_LANGSMITH_ENV_VARS = (
    *_TRACING_API_KEY_ENV_VARS,
    *_TRACING_BRIDGED_ENABLE_ENV_VARS,
)
"""LangSmith vars bridged from app-prefixed overrides to SDK names.

Derived from `_TRACING_BRIDGED_ENABLE_ENV_VARS` so `dcode doctor`, which reads
that tuple to predict the bridging, cannot disagree with the runtime.
"""

_PREFIX_RESOLVED_TRACING_ENV_VARS = (
    *_PREFIXED_LANGSMITH_ENV_VARS,
    "LANGSMITH_PROJECT",
)
"""Tracing selectors whose `DEEPAGENTS_CODE_` override wins when publishing.

`LANGSMITH_PROJECT` is here but not in `_PREFIXED_LANGSMITH_ENV_VARS` because
its prefixed form is bridged by `_apply_default_langsmith_project` rather than
by `_apply_prefixed_langsmith_env`.
"""

_TRACING_ENDPOINT_ENV_VARS = ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
"""Env vars that point tracing at a non-default (self-hosted/proxied) endpoint."""

LANGSMITH_US_ENDPOINT = "https://api.smith.langchain.com"
"""Canonical LangSmith SaaS endpoint for the US region (the SDK default)."""

LANGSMITH_EU_ENDPOINT = "https://eu.api.smith.langchain.com"
"""Canonical LangSmith SaaS endpoint for the EU region."""


def normalize_langsmith_endpoint(value: str) -> str:
    """Resolve a LangSmith endpoint shorthand to its canonical URL.

    Maps the case-insensitive region aliases `us`/`eu` to the LangSmith SaaS
    endpoints so the CLI `--base-url` flag and the TUI `/auth` prompt share one
    decode. Any other non-empty value is returned stripped and unchanged (a
    self-hosted or proxied URL); empty input returns an empty string.

    Args:
        value: A region alias, a full endpoint URL, or an empty string.

    Returns:
        The canonical endpoint URL, the stripped literal value, or `""`.
    """
    cleaned = value.strip()
    if not cleaned:
        return ""
    alias = cleaned.lower()
    if alias == "us":
        return LANGSMITH_US_ENDPOINT
    if alias == "eu":
        return LANGSMITH_EU_ENDPOINT
    return cleaned


def _is_langsmith_sdk_default_endpoint(value: str) -> bool:
    """Return whether `value` is the LangSmith SDK's default US SaaS endpoint.

    Profiles and configs often surface `LANGSMITH_US_ENDPOINT` even when the
    user never chose a custom ingest target. That default is not a keyless
    custom endpoint: the SDK treats it the same as leaving `api_url` unset, so
    upload-target checks and client kwargs should ignore it.
    """
    cleaned = value.strip().rstrip("/")
    if not cleaned:
        return False
    default = LANGSMITH_US_ENDPOINT.rstrip("/")
    return cleaned.lower() == default.lower()


def is_http_url(value: str) -> bool:
    """Return whether `value` is a non-empty `http`/`https` URL with a host.

    Guards the LangSmith endpoint so a stored API key is never paired with a
    non-HTTP, malformed, or schemeless value that could route trace ingestion
    (and the key) somewhere unintended.

    Args:
        value: Candidate endpoint URL.

    Returns:
        `True` when `value` parses as an `http`/`https` URL with a network
            location that contains no whitespace.
    """
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    # A real host never contains whitespace, but `urlparse` keeps an internal
    # space in the netloc (e.g. "exa mple.com"). Such a value would be stored,
    # written to `LANGSMITH_ENDPOINT`, and its traces may then be dropped at
    # ingest. Reject it loudly at save time.
    return not any(char.isspace() for char in parsed.netloc)


_TRACING_RECONCILED_ENV_VARS = _USER_LANGSMITH_ENV_VARS
"""Vars the LangSmith SDK reads from `os.environ` to pick a trace destination.

Deliberately the same tuple as `_USER_LANGSMITH_ENV_VARS` rather than a second
list of the same names: the set the agent publishes and the set restored for
the user's own commands have to move together. Listed twice, a new selector
added to only one side fails silently and asymmetrically -- missing here, the
previous workspace's value stays in `os.environ`; missing there, the agent's
value leaks into `execute`.

The profile pair belongs in that set because `langsmith.client._profiles` reads
both straight off `os.environ`: `LANGSMITH_PROFILE` selects the profile and
`LANGSMITH_CONFIG_FILE` the file holding it. That profile supplies the API key
and endpoint when no canonical var does, so leaving the pair out let one
workspace's profile choose where the next workspace's traces went.
"""


def _environment_before_tracing_reconcile() -> dict[str, str]:
    """Undo our tracing publication in a copy, preserving unrelated env edits.

    Returns:
        Environment with unchanged published selectors restored to their inputs.
    """
    env = dict(os.environ)
    _apply_env_values(
        env,
        {
            var: original
            for var, (original, published) in _reconciled_tracing_values.items()
            if env.get(var) == published
        },
    )
    return env


def _tracing_environment_values(environ: Mapping[str, str]) -> dict[str, str | None]:
    """Resolve the canonical selectors published to the LangSmith SDK.

    Returns:
        Selector values, with `None` denoting an unset variable.
    """
    return {
        var: (
            _resolve_env_var_from(environ, var)
            if var in _PREFIX_RESOLVED_TRACING_ENV_VARS
            else environ.get(var) or None
        )
        for var in _TRACING_RECONCILED_ENV_VARS
    }


def reconcile_tracing_environment(environ: Mapping[str, str]) -> None:
    """Publish a workspace's tracing settings to the process environment.

    Redaction and project naming decide from the active workspace snapshot, but
    the LangSmith SDK reads `os.environ` directly. Left to diverge, the snapshot
    can say "not tracing" while the SDK traces on -- uploading unredacted -- or
    the reverse, where the TUI offers `/trace` and nothing is ever ingested.

    Writes the canonical name, resolving `DEEPAGENTS_CODE_`-prefixed overrides
    first, because the SDK only reads canonical names. A var the workspace does
    not set is removed, so the previous workspace's `.env` cannot linger.

    Remember the environment before publication so later workspace previews
    can recover shell values and remove earlier dotenv contributions.

    The SDK caches env reads, so its caches are dropped afterwards.

    Args:
        environ: The active workspace environment snapshot.
    """
    baseline = _environment_before_tracing_reconcile()
    values = _tracing_environment_values(environ)
    _reconciled_tracing_values.update(
        {var: (baseline.get(var), value) for var, value in values.items()}
    )
    _apply_env_values(os.environ, values)
    _clear_langsmith_env_caches()


def _clear_langsmith_env_caches() -> None:
    """Drop the LangSmith SDK's cached environment reads."""
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
        ls_utils.get_tracer_project.cache_clear()
    except Exception:  # cache shape is upstream's, not ours
        # A stale cache means the SDK keeps the previous workspace's answer,
        # which the caller cannot detect. Say so rather than continuing as if
        # the reconcile took effect.
        logger.warning(
            "Could not clear the LangSmith environment caches; tracing may "
            "still use the previous workspace's settings",
            exc_info=True,
        )


class _LangSmithProfileConfig(Protocol):
    """Subset of LangSmith profile client config fields used at bootstrap."""

    api_url: str | None
    """Base URL for a custom self-hosted or proxied LangSmith endpoint."""

    api_key: str | None
    """API key from the active LangSmith profile."""

    oauth_access_token: str | None
    """OAuth access token from the active LangSmith profile."""

    oauth_refresh_token: str | None
    """OAuth refresh token from the active LangSmith profile."""


_QUIET_SDK_LOGGER_NAMES = (
    "deepagents.profiles.harness.harness_profiles",
    "genai-prices",
    "langchain",
    "langsmith",
)

_MCP_STREAMABLE_HTTP_LOGGER_NAME = "mcp.client.streamable_http"
_MCP_SSE_LOGGER_NAME = "mcp.client.sse"

_MCP_SHUTDOWN_RACE_MESSAGES: Mapping[str, frozenset[str]] = {
    _MCP_STREAMABLE_HTTP_LOGGER_NAME: frozenset(
        {"Error parsing JSON response", "Error parsing SSE message"}
    ),
    _MCP_SSE_LOGGER_NAME: frozenset({"Error in sse_reader"}),
}
"""MCP transport log messages whose `try` block writes to the session stream.

Keyed by the logger that emits them. Every listed `except` clause wraps an
`await read_stream_writer.send(session_message)`, so a closed/broken-resource
error reaching one of them means the session's streams are already torn down —
the app is quitting — and the record says nothing actionable.

Both streamable-HTTP messages are needed: a server that answers with a single
JSON body takes `_handle_json_response`, while one that streams its response
takes `_handle_sse_event`, and both send into the read stream from inside the
logging `try`. `Error in post_writer` is deliberately absent — it wraps the
entire write loop, so a closed stream there can also mean the transport was
orphaned while the session was still live, which is worth seeing.
"""


@functools.cache
def _closed_resource_error_types() -> tuple[type[BaseException], ...]:
    """Return anyio's closed/broken-resource exception types.

    Imported lazily and cached rather than at module scope: nothing has pulled
    in `anyio` by the time `_quiet_sdk_logging` runs, so importing it there
    would add work to the startup path of every invocation.

    The import is guarded because this runs on the logging path. `logging`
    swallows exceptions raised by `Handler.emit`, but *not* those raised by
    `Logger.filter` — they propagate into the `except` block that called
    `logger.exception`. Returning an empty tuple degrades to keeping the record.
    In practice the import always succeeds: the transports that emit these
    records import `anyio` themselves.
    """
    try:
        import anyio
    except ImportError:  # pragma: no cover
        return ()
    return (anyio.ClosedResourceError, anyio.BrokenResourceError)


class _McpShutdownRaceFilter(logging.Filter):
    """Drop an MCP client transport's shutdown-race records.

    When the app quits while an MCP response is in flight, the transport's
    reader task sends the parsed message into a session stream that is already
    closed, and the surrounding `except` logs the resulting closed-resource
    error with a full traceback. Nothing downstream can act on that record: the
    `send` that raised *is* the transport's only channel back to the caller, so
    the follow-up `await read_stream_writer.send(exc)` raises the same error
    instead of delivering it. The in-flight request still fails loudly by
    another route — the session's receive loop hands every pending response
    stream an `ErrorData(CONNECTION_CLOSED)` on its way out, so the waiting
    `send_request` raises `McpError("Connection closed")`.

    Matching is deliberately narrow — a per-logger message list, and only anyio
    stream errors — so real transport faults stay on stderr. A network-level
    drop cannot reach this filter: `httpcore` remaps anyio's
    `BrokenResourceError`/`ClosedResourceError` onto `httpx.ReadError` and
    friends before httpx re-raises, and this filter never walks `__cause__`, so
    those records are kept. What remains matches only the teardown of an
    in-process memory stream.
    """

    def __init__(self, messages: frozenset[str]) -> None:
        """Build a filter for one transport logger's shutdown-race messages.

        Args:
            messages: Exact `logger.exception` messages to drop when they carry
                a closed/broken-resource error.
        """
        super().__init__()
        self._messages = messages

    def filter(self, record: logging.LogRecord) -> bool:
        """Return `False` for shutdown-race records, `True` for everything else."""
        if record.msg not in self._messages:
            return True
        if record.exc_info is None or record.exc_info[1] is None:
            # `logger.exception()` outside an active exception yields
            # `(None, None, None)` — there is no error to classify.
            return True
        return not self._is_closed_resource(record.exc_info[1])

    @classmethod
    def _is_closed_resource(cls, exc: BaseException) -> bool:
        """Check for anyio closed/broken-resource errors, including grouped ones.

        Args:
            exc: The exception from the log record's `exc_info`.

        Returns:
            `True` if *exc* is an anyio closed/broken-resource error, or a
            `BaseExceptionGroup` whose every leaf is one, `False` otherwise.
        """
        if isinstance(exc, _closed_resource_error_types()):
            return True
        if isinstance(exc, BaseExceptionGroup):
            # Defensive: no current call site logs from a task-group boundary,
            # but if one ever does, the error arrives wrapped. Every leaf must
            # be a stream teardown — a group that also carries a real fault
            # stays visible rather than being suppressed wholesale.
            return bool(exc.exceptions) and all(
                cls._is_closed_resource(sub) for sub in exc.exceptions
            )
        return False


def _install_mcp_shutdown_race_filter(*, debug_enabled: bool) -> None:
    """Filter the MCP transports' shutdown-race records unless debug is on.

    Each filter must live on the logger that emits the records: logger-level
    filters only run for records logged directly on that logger, so attaching
    one to the `mcp` root would never see the transports' records.

    With `DEEPAGENTS_CODE_DEBUG` set, an already-installed filter is *removed*
    rather than merely not added. A logger-level filter drops the record for
    every handler, including the debug file handler `_quiet_sdk_logging`
    attaches, so leaving one in place would keep the race out of the debug log.

    Args:
        debug_enabled: Whether `DEEPAGENTS_CODE_DEBUG` is truthy.
    """
    for name, messages in _MCP_SHUTDOWN_RACE_MESSAGES.items():
        transport_logger = logging.getLogger(name)
        existing = [
            f for f in transport_logger.filters if isinstance(f, _McpShutdownRaceFilter)
        ]
        if debug_enabled:
            for f in existing:
                transport_logger.removeFilter(f)
            continue
        if not existing:
            transport_logger.addFilter(_McpShutdownRaceFilter(messages))


def _quiet_sdk_logging() -> None:
    """Keep non-actionable SDK diagnostics off the terminal.

    The harness-profile resolver, tracing SDKs, and the `genai-prices` price
    updater emit diagnostics on their own logger hierarchies. With no handler
    attached, warnings reach Python's last-resort stderr handler and can bleed
    into command output or the alternate-screen TUI -- the price updater logs
    at ERROR from a background thread once an hour, so an offline or
    proxied session would otherwise get a stderr line over the TUI on every
    failed refresh. Route them to the debug log when
    `DEEPAGENTS_CODE_DEBUG` is set, otherwise attach a `NullHandler` so they stay
    off the terminal. Other Deep Agents loggers remain untouched so actionable
    runtime warnings are still visible.

    The `mcp` hierarchy is handled differently: it gets no `NullHandler`, so its
    diagnostics stay on stderr, and only the shutdown-race records named in
    `_MCP_SHUTDOWN_RACE_MESSAGES` are filtered out. Under debug those transport
    loggers get the file handler instead, so the filtered records are still
    recorded rather than reaching last-resort stderr over the TUI.
    """
    from deepagents_code._debug import configure_debug_logging
    from deepagents_code._env_vars import DEBUG, is_env_truthy

    debug_enabled = is_env_truthy(DEBUG)
    _install_mcp_shutdown_race_filter(debug_enabled=debug_enabled)
    if debug_enabled:
        for mcp_logger_name in _MCP_SHUTDOWN_RACE_MESSAGES:
            configure_debug_logging(logging.getLogger(mcp_logger_name))
    for name in _QUIET_SDK_LOGGER_NAMES:
        sdk_logger = logging.getLogger(name)
        if debug_enabled:
            configure_debug_logging(sdk_logger)
        if not sdk_logger.handlers:
            sdk_logger.addHandler(logging.NullHandler())


def _load_langsmith_profile_config(
    env: Mapping[str, str] | None = None,
) -> _LangSmithProfileConfig | None:
    """Return the active LangSmith profile client config, if available."""
    try:
        client_module = importlib.import_module("langsmith.client")
    except ImportError:
        return None

    profiles = getattr(client_module, "_profiles", None)
    if profiles is None:
        return None

    if env is None:
        return profiles.load_profile_client_config()

    from unittest.mock import patch

    with patch.dict(os.environ, env, clear=True):
        return profiles.load_profile_client_config()


def _has_langsmith_profile_credentials(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the LangSmith profile config has usable auth material."""
    config = _load_langsmith_profile_config(env)
    if config is None:
        return False

    return bool(
        config.api_key or config.oauth_access_token or config.oauth_refresh_token
    )


def _has_langsmith_profile_custom_endpoint(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the LangSmith profile points at a custom endpoint.

    The SDK default US SaaS URL does not count: profiles often store it
    even when the user never chose a custom ingest target.
    """
    config = _load_langsmith_profile_config(env)
    if config is None:
        return False

    api_url = (config.api_url or "").strip()
    return bool(api_url) and not _is_langsmith_sdk_default_endpoint(api_url)


def _build_orphaned_tracing_disabled_notice() -> str:
    """Return the user-facing notice for disabled orphaned tracing."""
    base = (
        "LangSmith tracing was disabled because tracing is enabled but no "
        "credentials were found."
    )
    if shutil.which("langsmith"):
        return (
            f"{base} Set LANGSMITH_API_KEY or run `langsmith auth login`, "
            "then restart dcode."
        )
    return f"{base} Set LANGSMITH_API_KEY, then restart dcode."


def consume_orphaned_tracing_disabled_notice() -> str | None:
    """Return and clear the pending orphaned-tracing notice, if any."""
    global _orphaned_tracing_disabled_notice  # noqa: PLW0603

    notice = _orphaned_tracing_disabled_notice
    _orphaned_tracing_disabled_notice = None
    return notice


def _disable_set_tracing_flags() -> list[str]:
    """Set every configured tracing-enable flag to `false`.

    Returns:
        Env var names that were disabled.
    """
    disabled = [var for var in _TRACING_ENABLE_ENV_VARS if var in os.environ]
    for var in disabled:
        os.environ[var] = "false"
    return disabled


def _validate_user_langsmith_env(value: object) -> dict[str, str | None] | None:
    """Validate one complete LangSmith selector mapping from the carrier.

    Returns:
        The typed mapping when valid, otherwise `None`.
    """
    if not isinstance(value, dict) or value.keys() != set(_USER_LANGSMITH_ENV_VARS):
        return None
    mapping = cast("dict[str, object]", value)
    result: dict[str, str | None] = {}
    for key in _USER_LANGSMITH_ENV_VARS:
        item = mapping[key]
        if item is not None and not isinstance(item, str):
            return None
        result[key] = item
    return result


def _initialize_launch_langsmith_env(env: Mapping[str, str]) -> None:
    """Capture a missing launch snapshot from an environment baseline."""
    if _validate_user_langsmith_env(_bootstrap_state.launch_langsmith_env) is not None:
        return
    _bootstrap_state.launch_langsmith_env = _langsmith_selectors_from(env)


def _decode_user_langsmith_env(
    encoded: str,
) -> tuple[dict[str, str | None], dict[str, str | None]] | None:
    """Decode launch and user-command LangSmith mappings from the carrier.

    Returns:
        The launch/user pair when valid, otherwise `None`.
    """
    try:
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict) or decoded.keys() != {"launch", "user"}:
        return None
    launch = _validate_user_langsmith_env(decoded["launch"])
    user = _validate_user_langsmith_env(decoded["user"])
    return (launch, user) if launch is not None and user is not None else None


def _encode_user_langsmith_env() -> str:
    """Encode the trusted user-command LangSmith environment for the server.

    Returns:
        Compact JSON containing launch and user-command selector mappings.

    Raises:
        RuntimeError: If bootstrap did not capture every supported selector.
    """
    for values in (
        _bootstrap_state.launch_langsmith_env,
        _bootstrap_state.user_langsmith_env,
    ):
        if _validate_user_langsmith_env(values) is None:
            msg = (
                "Cannot start the server: your LangSmith settings were not "
                "captured at startup, so approval-gated commands could not be "
                "given your own credentials."
            )
            if _bootstrap_state.error is not None:
                # Bootstrap swallowed this to keep the app starting. Chain it,
                # or the report names only the symptom.
                msg = f"{msg} Startup failed earlier: {_bootstrap_state.error}"
            elif not _bootstrap_state.done:
                msg = f"{msg} Startup has not run yet."
            raise RuntimeError(msg) from _bootstrap_state.error
    return json.dumps(
        {
            "launch": _bootstrap_state.launch_langsmith_env,
            "user": _bootstrap_state.user_langsmith_env,
        },
        separators=(",", ":"),
    )


def _strip_user_langsmith_env(
    env: dict[str, str], *, prefixed_only: bool = False
) -> None:
    """Remove LangSmith selectors from `env`.

    Args:
        env: Environment for user commands, modified in place.
        prefixed_only: Drop only the `DEEPAGENTS_CODE_`-prefixed names, for
            callers that go on to write the canonical names themselves.
    """
    from deepagents_code.model_config import _ENV_PREFIX

    for var in _USER_LANGSMITH_ENV_VARS:
        if not prefixed_only:
            env.pop(var, None)
        env.pop(f"{_ENV_PREFIX}{var}", None)


def _report_unusable_langsmith_carrier() -> None:
    """Report that user commands lost their own LangSmith settings.

    Goes to stderr as well as the logger, for the reason given in
    `_report_denied_env_key`: the buffering handler installed at import means a
    `logger.warning` alone is visible only under `--debug`.

    That is not enough here, and the gap is known. The only caller runs during
    the agent build, which happens in the server subprocess, and `launch.server`
    redirects its stdout and stderr to a temporary file surfaced only under
    `--debug` too. So in the one process where a carrier exists, both channels
    are invisible: the user's `execute` commands silently lose their LangSmith
    auth. Closing this needs a server-to-client notice channel -- the
    `consume_orphaned_tracing_disabled_notice` pattern is a module global read
    by `app.py` in the *client*, so it cannot carry this one.
    """
    message = (
        "Could not read your LangSmith settings for approval-gated commands, "
        "so they will run without LangSmith credentials. This usually means "
        "the dcode client and server are running different versions; restart "
        "dcode. Tracing for your own commands is unaffected otherwise."
    )
    print(f"Warning: {message}", file=sys.stderr)  # noqa: T201  # user-facing
    logger.warning("%s", message)


def restore_user_langsmith_env(
    env: dict[str, str], *, start_path: Path | None = None
) -> None:
    """Restore launch and project-dotenv LangSmith settings for user commands.

    Precedence, highest first: the launch shell the user started `dcode` from,
    then the project `.env`. The global profile `.env` is deliberately excluded,
    because it configures the agent rather than the user's own commands.

    Also pops the client-to-server carrier and every `DEEPAGENTS_CODE_`-prefixed
    selector, so `agent.py` can pass the result with `inherit_env=False`.

    Args:
        env: Environment for user commands, modified in place.
        start_path: Project directory whose `.env` supplies the lower layer.
            Falls back to the mapping bootstrap captured for this process.
    """
    encoded = env.pop(_USER_LANGSMITH_ENV_CARRIER, None)
    launch = _bootstrap_state.launch_langsmith_env
    values = _bootstrap_state.user_langsmith_env
    if encoded is not None:
        decoded = _decode_user_langsmith_env(encoded)
        if decoded is None:
            # Falling back to this process's bootstrap state would hand user
            # commands the agent's own key and trace project -- the leak this
            # function exists to prevent. Run them with no LangSmith auth at
            # all instead, which fails visibly rather than silently mislabeling
            # the user's traces.
            _strip_user_langsmith_env(env)
            _report_unusable_langsmith_carrier()
            return
        launch, values = decoded

    _strip_user_langsmith_env(env, prefixed_only=True)
    _apply_env_values(env, launch)
    if start_path is not None:
        unreadable: list[Path] = []
        recomputed = _dotenv_environment(
            start_path=start_path,
            environ=env,
            include_global=False,
            unreadable=unreadable,
        )
        if unreadable:
            # The recompute reports "sets nothing" for a file it could not
            # read, which would drop the project's selectors from every user
            # command. The client already resolved them for this project and
            # shipped them in the carrier, so keep those instead of treating
            # an unreadable file as an empty one.
            logger.warning(
                "Could not read %s; keeping the LangSmith settings captured at "
                "launch for user commands",
                ", ".join(str(path) for path in unreadable),
            )
        else:
            values = recomputed
    _apply_env_values(env, _langsmith_selectors_from(values))


def _disable_orphaned_tracing() -> None:
    """Disable LangSmith tracing when enabled without a usable API key.

    LangChain enables tracing whenever a tracing flag is truthy, regardless of
    credentials. With no env or profile key the background tracer retries
    ingestion and floods `langsmith.client` 401 errors into the TUI (most visibly
    at the atexit flush). When a tracing flag is set but no credentials are
    resolvable, unset the flags so tracing never starts.

    A custom endpoint (`LANGSMITH_ENDPOINT`/`LANGCHAIN_ENDPOINT`, or a profile
    `api_url`) or replica endpoints (`LANGSMITH_RUNS_ENDPOINTS`/
    `LANGCHAIN_RUNS_ENDPOINTS`) signal tracing can upload without a top-level
    API key, so those explicitly configured targets are trusted and left alone.
    The SDK loggers are quieted separately by `_quiet_sdk_logging`, so
    any residual ingest errors stay off the TUI.
    """
    global _orphaned_tracing_disabled_notice  # noqa: PLW0603

    if not _tracing_enabled():
        return

    env = dict(os.environ)
    # Match SDK endpoint precedence: a populated env var wins over the profile,
    # even when it is the default US URL. Only consult the profile when no
    # endpoint env var is set.
    has_env_endpoint = any(
        (env.get(var) or "").strip() for var in _TRACING_ENDPOINT_ENV_VARS
    )
    has_custom_endpoint = any(
        (value := (env.get(var) or "").strip())
        and not _is_langsmith_sdk_default_endpoint(value)
        for var in _TRACING_ENDPOINT_ENV_VARS
    )
    if (
        has_custom_endpoint
        or (not has_env_endpoint and _has_langsmith_profile_custom_endpoint())
        or _has_langsmith_runs_endpoints_from(env)
    ):
        return

    has_key = any(
        (os.environ.get(var) or "").strip() for var in _TRACING_API_KEY_ENV_VARS
    )
    if has_key or _has_langsmith_profile_credentials():
        return

    disabled = _disable_set_tracing_flags()
    _orphaned_tracing_disabled_notice = _build_orphaned_tracing_disabled_notice()
    logger.warning(
        "LangSmith tracing is enabled (%s) but no API key is set; disabling "
        "tracing to avoid repeated authentication failures. Set LANGSMITH_API_KEY "
        "to enable tracing, or unset the tracing flag to silence this warning.",
        ", ".join(disabled),
    )


def _apply_default_langsmith_project() -> None:
    """Route agent traces to the default project when none is configured.

    When tracing is active but neither the prefixed override nor a base
    `LANGSMITH_PROJECT` is set, ingestion would land in the SDK's `default`
    project while `get_langsmith_project_name` advertises `deepagents-code`.
    Set the default explicitly so the displayed/looked-up name matches where
    traces are actually ingested (and `/trace` resolves once a run flushes).
    """
    if os.environ.get("LANGSMITH_PROJECT"):
        return

    if not _tracing_enabled():
        return

    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT_DEFAULT


def apply_stored_langsmith_auth(*, replace_project: bool = False) -> None:
    """Apply a `/auth`-stored LangSmith key, tracing, and redaction now.

    Args:
        replace_project: Whether the stored LangSmith project should replace
            the current process `LANGSMITH_PROJECT`. Startup leaves this false
            so an explicit environment value remains authoritative; the `/auth`
            save path sets it true because the saved project is the newest user
            choice for the already-running session.
    """
    from deepagents_code.model_config import apply_stored_service_credentials

    _apply_prefixed_langsmith_env()
    apply_stored_service_credentials()
    _apply_stored_langsmith_tracing(replace_project=replace_project)
    _disable_orphaned_tracing()
    _apply_default_langsmith_project()
    configure_langsmith_secret_redaction()


def _warn_on_prefixed_langsmith_override(canonical: str, prefixed: str) -> None:
    """Explain why an app-prefixed LangSmith value replaced its canonical peer."""
    from deepagents_code._env_vars import SUPPRESS_ENV_OVERRIDE_WARNING
    from deepagents_code.model_config import _ENV_PREFIX

    logger.warning(
        "%s and %s are both set to different values. Deep Agents Code uses %s "
        "for this session (the %s-prefixed value takes precedence). The %s you "
        "exported in your own shell is unaffected. This is expected. To silence "
        "this warning, unset %s or set %s=1.",
        canonical,
        prefixed,
        prefixed,
        _ENV_PREFIX,
        canonical,
        canonical,
        SUPPRESS_ENV_OVERRIDE_WARNING,
    )


def _apply_prefixed_langsmith_env() -> None:
    """Bridge app-prefixed LangSmith overrides to names read by the SDK."""
    from deepagents_code._env_vars import SUPPRESS_ENV_OVERRIDE_WARNING
    from deepagents_code.model_config import _ENV_PREFIX

    suppress_warning = is_env_truthy(SUPPRESS_ENV_OVERRIDE_WARNING)
    for canonical in _PREFIXED_LANGSMITH_ENV_VARS:
        prefixed = f"{_ENV_PREFIX}{canonical}"
        if prefixed not in os.environ:
            continue
        value = os.environ[prefixed]
        conflict = canonical in os.environ and os.environ[canonical] != value
        # Propagated unconditionally, empty string included: `FOO=""` is how a
        # user explicitly disables a flag, so an `if value:` guard here would
        # silently ignore the override.
        os.environ[canonical] = value
        if conflict and not suppress_warning:
            _warn_on_prefixed_langsmith_override(canonical, prefixed)


def _apply_stored_langsmith_tracing(*, replace_project: bool = False) -> None:
    """Enable tracing (and apply a custom project) for a `/auth`-stored key.

    Storing a LangSmith key via `/auth` is a deliberate opt-in to tracing, but
    a key alone never starts tracing — the SDK only traces when a tracing-enable
    flag is truthy. So when a key is stored, turn tracing on by default.

    The opt-out is intentionally non-destructive and session-scoped: an explicit
    falsy tracing flag (most simply `DEEPAGENTS_CODE_LANGSMITH_TRACING=false`,
    which bootstrap bridges to `LANGSMITH_TRACING`) is honored and tracing stays
    off, so the stored key can be paused without deleting it. A custom stored
    project is applied to `LANGSMITH_PROJECT` when the user has not set one,
    unless `replace_project` is set for the immediate `/auth` save path. A stored
    endpoint (e.g. the EU region) is applied to `LANGSMITH_ENDPOINT` with the
    same precedence via `_apply_stored_langsmith_endpoint`.

    No-op when no LangSmith key is stored, so a key supplied only through the
    environment keeps the prior behavior (tracing stays off unless a flag is
    set).

    A stored key is trusted by *presence*, not validity: this never pings
    LangSmith (a network round-trip at startup would fight the package's
    startup-perf budget). So a stored-but-invalid key (typo'd, revoked, or for
    the wrong workspace) still force-enables tracing, and its traces are then
    silently dropped at ingest with only SDK-internal 401s — which
    `_quiet_sdk_logging` routes away from the TUI. `_disable_orphaned_tracing`
    and `consume_orphaned_tracing_disabled_notice` guard only the *absent*-key
    case, not the invalid-key case. If traces never appear, the key is the first
    thing to re-check via `/auth`.

    The store is read exactly once: a single corrupt-file `RuntimeError` is
    logged and treated as "no stored key" rather than being raised (bootstrap
    must never crash the app) or partially applied.
    """
    from deepagents_code import auth_store
    from deepagents_code._env_vars import classify_env_bool
    from deepagents_code.model_config import LANGSMITH_SERVICE

    try:
        creds = auth_store.load_credentials()
    except RuntimeError:
        logger.warning(
            "Could not read the stored LangSmith credential; the credential file "
            "may be corrupt. Re-add the key via /auth."
        )
        return
    entry = creds.get(LANGSMITH_SERVICE)
    # No-op unless a LangSmith API key was stored via `/auth`. A key supplied
    # only through the environment never lands here, keeping its prior behavior
    # (tracing stays off unless a flag is set).
    if entry is None or entry["type"] != "api_key" or not entry["key"]:
        return
    if _stored_langsmith_key_is_suppressed(entry["key"]):
        return

    # The key was bridged onto LANGSMITH_API_KEY by
    # `apply_stored_service_credentials`. Decide whether to enable tracing.
    flags = [
        classify_env_bool(os.environ[var])
        for var in _TRACING_ENABLE_ENV_VARS
        if var in os.environ
    ]
    if any(flag is False for flag in flags):
        # Explicit, deliberate opt-out — keep the key but make the opt-out
        # authoritative over sibling SDK tracing flags.
        _disable_set_tracing_flags()
        return
    if not any(flag is True for flag in flags):
        os.environ["LANGSMITH_TRACING"] = "true"

    _apply_stored_langsmith_endpoint(
        entry.get("base_url") or None, replace=replace_project
    )

    project = entry.get("project") or None
    if replace_project:
        if project:
            os.environ["LANGSMITH_PROJECT"] = project
        else:
            os.environ.pop("LANGSMITH_PROJECT", None)
        return
    if project and not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = project


def _stored_langsmith_key_is_suppressed(stored_key: str) -> bool:
    """Return whether an env override keeps `stored_key` from taking effect."""
    from deepagents_code.model_config import _ENV_PREFIX

    prefixed_names = [f"{_ENV_PREFIX}{name}" for name in _TRACING_API_KEY_ENV_VARS]
    prefixed_values = [
        os.environ.get(name) or None for name in prefixed_names if name in os.environ
    ]
    if prefixed_values:
        return any(value != stored_key for value in prefixed_values)
    env_key = os.environ.get("LANGSMITH_API_KEY")
    return bool(env_key and env_key != stored_key)


def _apply_stored_langsmith_endpoint(endpoint: str | None, *, replace: bool) -> None:
    """Apply a `/auth`-stored LangSmith endpoint to `LANGSMITH_ENDPOINT`.

    Writes a stored endpoint to the canonical `LANGSMITH_ENDPOINT` and clears the
    `LANGCHAIN_ENDPOINT` alternate so the SDK can't read a stale value through it.
    Precedence mirrors the stored project:

    - `replace` (the immediate `/auth` save): the stored endpoint replaces the
        current value, and a blank endpoint (the US default) clears both names so
        ingestion falls back to the LangSmith SaaS default.
    - Startup (`replace=False`): a non-empty `LANGSMITH_ENDPOINT`/`LANGCHAIN_ENDPOINT`
        already in the environment stays authoritative, so a stored endpoint is
        applied only when neither is set. A stored credential without an endpoint
        never clears an existing env value (self-hosted setups keep working).

    Like a stored key, a stored endpoint is trusted by *presence*, not
    reachability: this never connects to it. A wrong-but-well-formed endpoint (a
    typo'd or dead host) is applied anyway, and its traces may then be dropped at
    ingest. `is_http_url` rejects the obviously malformed cases at save time, but
    if traces never appear the stored endpoint is worth re-checking via `/auth`
    alongside the key.

    Args:
        endpoint: The stored endpoint URL, or `None` when none is stored.
        replace: Whether the stored value should overwrite the current
            environment (the immediate `/auth` save path).
    """
    canonical, alternate = _TRACING_ENDPOINT_ENV_VARS
    if replace:
        if endpoint:
            os.environ[canonical] = endpoint
        else:
            os.environ.pop(canonical, None)
        os.environ.pop(alternate, None)
        return
    if not endpoint:
        return
    if any(os.environ.get(var) for var in _TRACING_ENDPOINT_ENV_VARS):
        return
    os.environ[canonical] = endpoint
    # Past the guard above both endpoint vars are falsy, so this only clears an
    # empty-string `LANGCHAIN_ENDPOINT`; it keeps canonical as the one name the
    # SDK reads and mirrors the `replace` branch's alternate-clearing.
    os.environ.pop(alternate, None)


def _ensure_bootstrap() -> None:
    """Run one-time bootstrap: dotenv loading and `LANGSMITH_PROJECT` override.

    Idempotent and thread-safe — subsequent calls are no-ops. Called
    automatically by `_get_credentials()` when `credentials` is first accessed.

    The flag is set in `finally` so that partial failures (e.g. a
    malformed `.env`) still mark bootstrap as done — preventing infinite retry
    loops. Exceptions are caught and logged at ERROR level; the app proceeds
    with the environment as-is.
    """
    if _bootstrap_state.done:
        return

    with _bootstrap_lock:
        if _bootstrap_state.done:  # double-check after acquiring lock
            return

        try:
            # First, because this needs nothing but `os.environ`. Anything below
            # can raise into the handler, which this contract says to survive;
            # leaving the snapshot empty instead makes `_encode_user_langsmith_env`
            # refuse, and the server never starts at all.
            _bootstrap_state.launch_langsmith_env = _langsmith_selectors_from(
                os.environ
            )
            _bootstrap_state.user_langsmith_env = dict(
                _bootstrap_state.launch_langsmith_env
            )

            from deepagents_code.project_utils import (
                get_server_project_context as _get_server_project_context,
            )

            ctx = _get_server_project_context()
            _bootstrap_state.start_path = ctx.user_cwd if ctx else None
            _load_dotenv(
                start_path=_bootstrap_state.start_path,
                capture_user_langsmith=True,
            )

            # `configure_debug_logging` already ran at import, before the `.env`
            # above was loaded. Re-run it so a `DEEPAGENTS_CODE_DEBUG` set only in
            # `.env` installs the file handler now (idempotent for the same path),
            # ensuring later failures are actually written to the debug log.
            from deepagents_code._debug import configure_debug_logging

            configure_debug_logging(logging.getLogger("deepagents_code"))

            # Keep dependency logging out of command output and the TUI. Route it
            # to the debug log when enabled, otherwise swallow it via NullHandler.
            _quiet_sdk_logging()

            # Capture AFTER dotenv loading so .env-only values are visible,
            # but BEFORE the override below replaces it.
            _bootstrap_state.original_langsmith_project = os.environ.get(
                "LANGSMITH_PROJECT"
            )

            # CRITICAL: Override LANGSMITH_PROJECT to route agent traces to a
            # separate project. LangSmith reads LANGSMITH_PROJECT at invocation
            # time, so we override it here and preserve the user's original
            # value for the project shown in the UI and in stream metadata.
            # Shell commands get their own selectors from
            # `restore_user_langsmith_env`, not from this snapshot.
            from deepagents_code._env_vars import LANGSMITH_PROJECT

            deepagents_project = os.environ.get(LANGSMITH_PROJECT)
            if deepagents_project:
                os.environ["LANGSMITH_PROJECT"] = deepagents_project

            # Bridge prefixed and stored service keys, apply stored LangSmith
            # tracing defaults, disable orphaned tracing, and route active tracing
            # to the displayed project. Keeping this in one helper lets `/auth`
            # save apply the same state inside an already-running TUI session.
            apply_stored_langsmith_auth()
        except Exception as exc:
            _bootstrap_state.error = exc
            logger.exception(
                "Bootstrap failed; .env values and LANGSMITH_PROJECT override "
                "may be missing. The app will proceed with environment as-is.",
            )
        finally:
            _bootstrap_state.done = True


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from rich.console import Console

    from deepagents_code._git import RepositoryMetadata

MODE_PREFIXES: dict[str, str] = {
    "shell_incognito": "!!",
    "shell": "!",
    "command": "/",
}
"""Maps each non-normal mode to its trigger character."""

MODE_DISPLAY_GLYPHS: dict[str, str] = {
    "shell_incognito": "$",
    "shell": "$",
    "command": "/",
}
"""Maps each non-normal mode to its display glyph shown in the prompt/UI."""

if MODE_PREFIXES.keys() != MODE_DISPLAY_GLYPHS.keys():
    _only_prefixes = MODE_PREFIXES.keys() - MODE_DISPLAY_GLYPHS.keys()
    _only_glyphs = MODE_DISPLAY_GLYPHS.keys() - MODE_PREFIXES.keys()
    msg = (
        "MODE_PREFIXES and MODE_DISPLAY_GLYPHS have mismatched keys: "
        f"only in PREFIXES={_only_prefixes}, only in GLYPHS={_only_glyphs}"
    )
    raise ValueError(msg)

_MODE_PREFIXES_BY_LENGTH: tuple[tuple[str, str], ...] = tuple(
    sorted(MODE_PREFIXES.items(), key=lambda item: len(item[1]), reverse=True)
)
"""Mode entries ordered longest-prefix-first.

Pre-sorted at import so `detect_mode_prefix` runs in constant time per
keystroke without re-sorting.
"""


def detect_mode_prefix(text: str) -> tuple[str, str] | None:
    """Return the longest mode prefix and mode for `text`, if any.

    Longer prefixes win so multi-character triggers like `!!` are matched
    before their single-character prefixes (`!`).

    Args:
        text: Input text that may start with a mode trigger.

    Returns:
        Tuple of `(prefix, mode)` for the longest matching trigger, otherwise
        `None`.
    """
    for mode, prefix in _MODE_PREFIXES_BY_LENGTH:
        if text.startswith(prefix):
            return prefix, mode
    return None


class CharsetMode(StrEnum):
    """Character set mode for TUI display."""

    UNICODE = "unicode"
    """Always use Unicode glyphs (e.g. `⏺`, `✓`, `…`)."""

    ASCII = "ascii"
    """Always use ASCII-safe fallbacks (e.g. `(*)`, `[OK]`, `...`)."""

    AUTO = "auto"
    """Detect charset support at runtime and pick Unicode or ASCII."""


@dataclass(frozen=True)
class Glyphs:
    """Character glyphs for TUI display."""

    tool_prefix: str  # ⏺ vs (*)
    ellipsis: str  # … vs ...
    checkmark: str  # ✓ vs [OK]
    error: str  # ✗ vs [X]
    circle_empty: str  # ○ vs [ ]
    circle_filled: str  # ● vs [*]
    square_filled: str  # ■ vs [#]
    checkbox_empty: str  # ☐ vs [ ]
    checkbox_checked: str  # ☑ vs [x]
    output_prefix: str  # ⎿ vs L
    spinner_frames: tuple[str, ...]  # Braille vs ASCII spinner
    pause: str  # ⏸ vs ||
    newline: str  # ⏎ vs \\n
    warning: str  # ⚠ vs [!]
    question: str  # ? vs [?]
    hourglass: str  # ⏳ vs [~]
    retry: str  # ↻ vs [R]
    tool: str  # wrench vs [T]
    file: str  # memo vs [F]
    arrow_up: str  # up arrow vs ^
    arrow_down: str  # down arrow vs v
    arrow_right: str  # right arrow vs ->
    separator: str  # middle dot vs |
    tree_branch: str  # tree branch vs |-
    tree_last: str  # final tree branch vs `-
    bullet: str  # bullet vs -
    cursor: str  # cursor vs >
    disclosure_collapsed: str  # ▸ vs >
    disclosure_expanded: str  # ▾ vs v

    # Box-drawing characters
    box_horizontal: str  # ─ vs -
    box_horizontal_heavy: str  # ━ vs =

    # Diff-specific
    hunk_break: str  # ⋮ vs :
    # Distinct from `ellipsis`, which is identical in Unicode mode but
    # ASCII-expands to "..." — three cells would overflow the diff's
    # line-number column, which is only `max(2, len(str(max_line)))` wide,
    # and break the vertical alignment every row shares.
    line_continuation: str  # … vs .

    # Status bar
    git_branch: str  # "↗" vs "git:"


UNICODE_GLYPHS = Glyphs(
    tool_prefix="⏺",
    ellipsis="…",
    checkmark="✓",
    error="✗",
    circle_empty="○",
    circle_filled="●",
    square_filled="■",
    checkbox_empty="☐",
    checkbox_checked="☑",
    output_prefix="⎿",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    pause="⏸",
    newline="⏎",
    warning="⚠",
    question="?",
    hourglass="⏳",
    retry="↻",
    tool="🔧",
    file="📝",
    arrow_up="↑",
    arrow_down="↓",
    arrow_right="→",
    separator="·",
    tree_branch="├",
    tree_last="└",
    bullet="•",
    cursor="›",  # noqa: RUF001  # Intentional Unicode glyph
    disclosure_collapsed="▸",
    disclosure_expanded="▾",
    # Box-drawing characters
    box_horizontal="─",
    box_horizontal_heavy="━",
    # Diff-specific
    hunk_break="⋮",
    line_continuation="…",
    # Status bar
    git_branch="↗",
)
"""Glyph set for terminals with full Unicode support."""

ASCII_GLYPHS = Glyphs(
    tool_prefix="(*)",
    ellipsis="...",
    checkmark="[OK]",
    error="[X]",
    circle_empty="[ ]",
    circle_filled="[*]",
    square_filled="[#]",
    checkbox_empty="[ ]",
    checkbox_checked="[x]",
    output_prefix="L",
    spinner_frames=("(-)", "(\\)", "(|)", "(/)"),
    pause="||",
    newline="\\n",
    warning="[!]",
    question="[?]",
    hourglass="[~]",
    retry="[R]",
    tool="[T]",
    file="[F]",
    arrow_up="^",
    arrow_down="v",
    arrow_right="->",
    separator="|",
    tree_branch="|-",
    tree_last="`-",
    bullet="-",
    cursor=">",
    disclosure_collapsed=">",
    disclosure_expanded="v",
    # Box-drawing characters
    box_horizontal="-",
    box_horizontal_heavy="=",
    # Diff-specific
    hunk_break=":",
    line_continuation=".",
    # Status bar
    git_branch="git:",
)
"""Glyph set for terminals limited to 7-bit ASCII."""

_glyphs_cache: Glyphs | None = None
"""Module-level cache for detected glyphs."""

_charset_mode_cache: CharsetMode | None = None
"""Module-level cache for the detected charset mode."""

_editable_cache: tuple[bool, str | None] | None = None
"""Module-level cache for editable install info: (is_editable, source_path)."""

_langsmith_url_cache: tuple[str, str] | None = None
"""Module-level cache for successful LangSmith project URL lookups."""

_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS = 2.0
"""Max seconds to wait for LangSmith project URL lookup.

Kept short so tracing metadata can never stall app flows.
"""


def _get_deepagents_version() -> str | None:
    """Resolve the installed Deep Agents SDK version for diagnostics.

    Editable installs can leave package metadata behind the source checkout, so
    this uses the shared resolver that prefers the editable source marker and
    appends an `+editable` suffix. A sibling monorepo workspace whose marker
    trails dcode's exact pin reports the pinned release baseline instead.

    Returns:
        The resolved Deep Agents SDK version, or `None` when unavailable.
    """
    # Imported lazily on purpose: `extras_info` pulls in `packaging`, which we
    # keep off `config`'s module-import path (the startup hot path). Do not
    # hoist this to the top of the module. The import is also guarded so a
    # broken/absent `packaging` can never crash best-effort diagnostic metadata.
    try:
        from deepagents_code.extras_info import resolve_sdk_version

        sdk_version, status = resolve_sdk_version()
    except ImportError:
        logger.warning(
            "Could not import resolve_sdk_version for SDK version metadata",
            exc_info=True,
        )
        return None
    return sdk_version if status == "resolved" else None


def _format_lc_version(base_version: str, *, editable: bool) -> str:
    """Format an `lc_versions` value with editable-install context.

    Editable installs get an `editable` PEP 440 local segment (`1.2.3+editable`).
    The SDK stamps `lc_versions["deepagents"]` the same way and LangChain merges
    the two dicts, so both entries in a single trace share one encoding. It is
    also what `dcode_client_deepagents_version` already reports.

    Args:
        base_version: The base version string.
        editable: Whether the distribution is installed in editable mode.

    Returns:
        The version string, with an `editable` local segment when `editable`.
    """
    if not editable:
        return base_version
    # Imported lazily on purpose, for the same reason as `_get_deepagents_version`
    # above: this pulls in `packaging`, which we keep off `config`'s module-import
    # path (the startup hot path). Do not hoist this to the top of the module.
    # Import from `extras_info` rather than `deepagents._version`: the SDK copy
    # only exists in releases newer than the exact `deepagents` pin, and a
    # PyPI-resolved SDK would raise `ImportError` here on every editable run.
    from deepagents_code.extras_info import _with_editable_local_version

    return _with_editable_local_version(base_version)


def _contract_editable_path(path: str) -> str:
    """Contract an editable path beneath a usable home directory.

    Returns:
        The contracted path, or the original path when no usable home exists.
    """
    try:
        home_path = Path.home()
    except RuntimeError:
        return path
    if not home_path.is_absolute():
        return path
    home = str(home_path)
    return "~" + path[len(home) :] if path.startswith(home) else path


def _resolve_editable_info() -> tuple[bool, str | None]:
    """Parse PEP 610 `direct_url.json` once and cache both results.

    Returns:
        Tuple of (is_editable, contracted_source_path). The path is
        `~`-contracted when it falls under the user's home directory, or
        `None` when the install is non-editable or the path is unavailable.
    """
    global _editable_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _editable_cache is not None:
        return _editable_cache

    editable = False
    path: str | None = None

    try:
        dist = distribution("deepagents-code")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            editable = data.get("dir_info", {}).get("editable", False)
            if editable:
                url = data.get("url", "")
                if url.startswith("file://"):
                    path = _contract_editable_path(url2pathname(urlparse(url).path))
    except (PackageNotFoundError, FileNotFoundError, json.JSONDecodeError, TypeError):
        logger.debug(
            "Failed to read editable install info from PEP 610 metadata",
            exc_info=True,
        )

    _editable_cache = (editable, path)
    return _editable_cache


def _is_editable_install() -> bool:
    """Check if deepagents-code is installed in editable mode.

    Uses PEP 610 `direct_url.json` metadata to detect editable installs.

    Returns:
        `True` if installed in editable mode, `False` otherwise.
    """
    return _resolve_editable_info()[0]


def _get_editable_install_path() -> str | None:
    """Return the `~`-contracted source directory for an editable install.

    Returns `None` for non-editable installs or when the path cannot be
    determined.
    """
    return _resolve_editable_info()[1]


def _detect_charset_mode() -> CharsetMode:
    """Auto-detect terminal charset capabilities (cached for the process).

    Returns:
        The detected CharsetMode based on environment and terminal encoding.
    """
    global _charset_mode_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _charset_mode_cache is not None:
        return _charset_mode_cache
    _charset_mode_cache = _compute_charset_mode()
    return _charset_mode_cache


def _compute_charset_mode() -> CharsetMode:
    """Compute terminal charset capabilities from environment and encoding.

    Returns:
        The detected CharsetMode based on environment and terminal encoding.
    """
    prefixed = os.environ.get(UI_CHARSET_MODE)
    mode = prefixed if prefixed is not None else os.environ.get("UI_CHARSET_MODE")
    mode = (mode or "auto").lower()
    if mode == "unicode":
        return CharsetMode.UNICODE
    if mode == "ascii":
        return CharsetMode.ASCII

    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return CharsetMode.UNICODE
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    return CharsetMode.UNICODE if "utf" in lang.lower() else CharsetMode.ASCII


def get_glyphs() -> Glyphs:
    """Get the glyph set for the current charset mode.

    Returns:
        The appropriate Glyphs instance based on charset mode detection.
    """
    global _glyphs_cache  # noqa: PLW0603  # Module-level cache requires global statement
    if _glyphs_cache is not None:
        return _glyphs_cache

    mode = _detect_charset_mode()
    _glyphs_cache = ASCII_GLYPHS if mode == CharsetMode.ASCII else UNICODE_GLYPHS
    return _glyphs_cache


def reset_glyphs_cache() -> None:
    """Reset the glyphs and charset-mode caches (for testing)."""
    global _glyphs_cache, _charset_mode_cache  # noqa: PLW0603  # Module-level caches require global statement
    _glyphs_cache = None
    _charset_mode_cache = None


def is_ascii_mode() -> bool:
    """Check whether the terminal is in ASCII charset mode.

    Convenience wrapper so widgets can branch on charset without importing
    both `_detect_charset_mode` and `CharsetMode`.

    Returns:
        `True` when the detected charset mode is ASCII.
    """
    return _detect_charset_mode() == CharsetMode.ASCII


def newline_shortcut() -> str:
    """Return the terminal-appropriate label for the newline keyboard shortcut.

    Prefers `Shift+Enter` when the terminal is known to support the kitty
    keyboard protocol, either via conservative terminal-identity heuristics
    or the `DEEPAGENTS_CODE_KITTY_KEYBOARD` override. Falls back to
    `Option+Enter` on macOS and `Ctrl+J` elsewhere — both survive legacy
    terminals that strip the shift modifier from `Enter`.

    Returns:
        A human-readable shortcut string,
            e.g. `'Shift+Enter'`, `'Option+Enter'`, or `'Ctrl+J'`.
    """
    from deepagents_code.terminal_capabilities import supports_kitty_keyboard_protocol

    if supports_kitty_keyboard_protocol():
        return "Shift+Enter"
    return "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"


_UNICODE_BANNER = f"""
██████╗  ███████╗ ███████╗ ██████╗    ▄▓▓▄
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗  ▓•███▙
██║  ██║ █████╗   █████╗   ██████╔╝  ░▀▀████▙▖
██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝      █▓████▙▖
██████╔╝ ███████╗ ███████╗ ██║          ▝█▓█████▙
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝           ░▜█▓████▙
                                          ░█▀█▛▀▀▜▙▄
                                        ░▀░▀▒▛░░  ▝▀▘

 █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
                                                  v{__version__}
"""
_ASCII_BANNER = f"""
 ____  ____  ____  ____
|  _ \\| ___|| ___||  _ \\
| | | | |_  | |_  | |_) |
| |_| |  _| |  _| |  __/
|____/|____||____||_|

    _    ____  ____  _   _  _____  ____
   / \\  / ___|| ___|| \\ | ||_   _|/ ___|
  / _ \\| |  _ | |_  |  \\| |  | |  \\___ \\
 / ___ \\ |_| ||  _| | |\\  |  | |   ___) |
/_/   \\_\\____||____||_| \\_|  |_|  |____/
                                  v{__version__}
"""


def get_banner() -> str:
    """Get the appropriate banner for the current charset mode.

    Returns:
        The text art banner string (Unicode or ASCII based on charset mode).

            Includes "(local)" suffix when installed in editable mode.
    """
    if _detect_charset_mode() == CharsetMode.ASCII:
        banner = _ASCII_BANNER
    else:
        banner = _UNICODE_BANNER

    if is_env_truthy(HIDE_SPLASH_VERSION):
        return banner.replace(f"v{__version__}", "")

    if _is_editable_install():
        banner = banner.replace(f"v{__version__}", f"v{__version__} (local)")

    return banner


MAX_ARG_LENGTH = 150
"""Character limit for tool argument values in the UI.

Longer values are truncated with an ellipsis by `truncate_value`
in `tool_display`.
"""

_git_branch_cache: dict[str, str | None] = {}
"""Per-cwd cache of resolved git branch names.

Avoids repeated git branch resolution within the same session. Keyed by
`str(Path.cwd())`; `None` values indicate the directory is not inside a git
repository or that resolution failed.
"""


def _get_git_branch() -> str | None:
    """Return the current git branch name, or `None` if not in a repo."""
    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.debug("Could not determine cwd for git branch lookup", exc_info=True)
        return None
    if cwd in _git_branch_cache:
        return _git_branch_cache[cwd]

    try:
        branch = resolve_git_branch(cwd) or None
    except OSError:
        logger.debug("Could not determine git branch", exc_info=True)
        branch = None

    _git_branch_cache[cwd] = branch
    return branch


_repo_metadata_cache: dict[str, RepositoryMetadata | None] = {}
"""Per-cwd cache of resolved repository metadata."""


def _get_git_commit_sha() -> str | None:
    """Return the current `HEAD` commit SHA, or `None` if unavailable.

    Resolved fresh on every call (unlike the branch/repo lookups): `HEAD` moves
    whenever the agent or user commits, checks out, or resets within a session,
    and each turn's trace must record the commit that was current for that turn.
    """
    from deepagents_code._git import resolve_git_commit_sha

    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.debug("Could not determine cwd for git commit lookup", exc_info=True)
        return None

    try:
        return resolve_git_commit_sha(cwd) or None
    except OSError:
        logger.debug("Could not determine git commit", exc_info=True)
        return None


def _get_repository_metadata() -> RepositoryMetadata | None:
    """Return parsed `origin` repository metadata, or `None`."""
    from deepagents_code._git import parse_repository_metadata, resolve_git_remote_url

    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.debug("Could not determine cwd for git remote lookup", exc_info=True)
        return None
    if cwd in _repo_metadata_cache:
        return _repo_metadata_cache[cwd]

    repo: RepositoryMetadata | None = None
    try:
        remote_url = resolve_git_remote_url(cwd)
        if remote_url:
            repo = parse_repository_metadata(remote_url)
    except OSError:
        logger.debug("Could not determine git remote", exc_info=True)

    _repo_metadata_cache[cwd] = repo
    return repo


# coding-agent-v1 contract literals. See `build_coding_agent_metadata`.
CODING_AGENT_PURPOSE = "coding"
"""Fixed `ls_agent_purpose` literal identifying the coding-agent trace class."""

CODING_AGENT_INTEGRATION = "deepagents-code"
"""Stable `ls_integration` id for this plugin (unchanged for backward-compat)."""

CODING_AGENT_RUNTIME = "Deep Agents Code"
"""User-facing `ls_agent_runtime` name."""

CODING_AGENT_TRACE_SCHEMA_VERSION = "coding-agent-v1"
"""Version of the coding-agent trace-metadata contract this build emits."""


def build_coding_agent_metadata(
    *,
    thread_id: str,
    turn_id: str | None,
    turn_number: int | None,
    cwd: str,
    git_branch: str | None,
    sandbox_type: str | None,
    user_id: str | None,
) -> dict[str, Any]:
    """Build the shared coding-agent-v1 trace-metadata block.

    Implements the `coding-agent-v1` contract for Deep Agents Code:
    one helper that stamps the identity block, plugin/runtime versions, turn
    markers, and repo/git/cwd attribution. The six identity/version keys and
    `thread_id` are always present; the optional keys whose value is unknown are
    omitted (per the contract), so callers can pass `None` for any of them.

    Because Deep Agents Code is itself the runtime — there is no separate CLI
    package — `ls_integration_version` and `ls_agent_runtime_version` both come
    from the `deepagents-code` package version (`__version__`). The underlying
    `deepagents` SDK version is surfaced separately as
    `dcode_client_deepagents_version` by `build_stream_config`.

    Scope-restricted contract keys are intentionally NOT produced here:
    `approval_policy` (root/interrupted only) and `ls_subagent_id` /
    `ls_subagent_type` (subagent only). This metadata propagates trace-wide
    through the LangGraph stream config (and, for subagents, the per-key config
    merge of langgraph#7926 / deepagents#3634), so any key placed here lands on
    every descendant run. Emitting a run-type-scoped key would therefore leak it
    onto run types outside its contract `appliesTo` set — a hard validator
    failure — and the LangGraph runtime exposes no clean per-run-type metadata
    seam to scope them. See `build_stream_config` for the full rationale.

    Args:
        thread_id: Stable conversation id; also set as top-level `thread_id`.
        turn_id: Per-turn id (uuid4 / message id), or `None`.
        turn_number: 1-based per-thread turn index, or `None`.
        cwd: Current working directory, or empty string when unavailable.
        git_branch: Current branch name, or `None`.
        sandbox_type: Sandbox provider name, or `None`/`"none"` when inactive.
        user_id: Stable pseudonymous user id, or `None`.

    Returns:
        The contract metadata dict with unknown keys omitted.
    """
    metadata: dict[str, Any] = {
        "ls_agent_purpose": CODING_AGENT_PURPOSE,
        "ls_integration": CODING_AGENT_INTEGRATION,
        "ls_agent_runtime": CODING_AGENT_RUNTIME,
        "thread_id": thread_id,
        "ls_trace_schema_version": CODING_AGENT_TRACE_SCHEMA_VERSION,
        "ls_integration_version": __version__,
        "ls_agent_runtime_version": __version__,
    }

    if turn_id:
        metadata["turn_id"] = turn_id
    if turn_number is not None:
        metadata["turn_number"] = turn_number

    repo = _get_repository_metadata()
    if repo is not None:
        repository_url, repository_provider, repository_name = repo
        metadata["repository_url"] = repository_url
        metadata["repository_provider"] = repository_provider
        metadata["repository_name"] = repository_name

    if git_branch:
        metadata["git_branch"] = git_branch
    commit_sha = _get_git_commit_sha()
    if commit_sha:
        metadata["git_commit_sha"] = commit_sha
    if cwd:
        metadata["cwd"] = cwd

    if user_id:
        metadata["user_id"] = user_id
    if sandbox_type and sandbox_type != "none":
        metadata["sandbox_type"] = sandbox_type

    return metadata


def build_stream_config(
    thread_id: str,
    assistant_id: str | None,
    *,
    sandbox_type: str | None = None,
    turn_id: str | None = None,
    turn_number: int | None = None,
    auto_approve: bool = False,
    skill_name: str | None = None,
) -> RunnableConfig:
    """Build the LangGraph stream config dict.

    Stamps the shared `coding-agent-v1` trace-metadata contract via
    `build_coding_agent_metadata` — identity block, plugin/runtime versions,
    turn markers, and repo/git/cwd attribution — onto `metadata`. Metadata set
    here propagates trace-wide to every run in the graph (root, llm, tool, and
    subagent subgraphs), which is exactly what the contract's "always" and
    "where-known" keys require, so the helper output is stamped once here.

    Scope-restricted contract keys are deliberately not emitted. `approval_policy`
    (root/interrupted only) and `ls_subagent_id` / `ls_subagent_type` (subagent
    only) cannot live in this trace-wide metadata: LangGraph propagates each key
    to all descendant runs (per-key config merge, langgraph#7926 /
    deepagents#3634), so they would leak onto run types outside their contract
    `appliesTo` set and fail validation. This runtime exposes no clean
    per-run-type metadata seam to scope them, so they are omitted by design
    rather than leaked. (Subagent runs still inherit the parent/root `thread_id`
    and all required keys, satisfying the contract's grouping rule.)

    Also injects the dcode version into `metadata["lc_versions"]` so LangSmith
    traces can be correlated with specific releases. `create_deep_agent` supplies
    the SDK version through the compiled graph config, and LangChain merges
    nested metadata dictionaries so both versions survive at stream time.

    Also records `dcode_client_deepagents_version` as a dcode-client diagnostic.
    This describes the Deep Agents package installed alongside the TUI, which
    can differ from a remote graph's Deep Agents runtime version. Editable
    installs carry an `+editable` suffix, matching how the SDK stamps
    `lc_versions["deepagents"]`; for sibling monorepo packages that suffix
    identifies workspace HEAD relative to the pinned published SDK baseline.

    Also records `editable` as an always-present boolean so editable dcode installs
    can be filtered without parsing the `lc_versions["deepagents-code"]` suffix.

    Also records `dcode_experimental=True` when `DEEPAGENTS_CODE_EXPERIMENTAL`
    is enabled, so experimental runs are filterable in trace metadata.

    Also records `dcode_auto_approve=True` when auto-approve ("YOLO") mode is
    active, so runs that ran tools without HITL approval are filterable in trace
    metadata. This is a diagnostic key, not the contract-scoped `approval_policy`
    key (see above), so it is safe to stamp trace-wide.

    Also records `dcode_term_program` from `TERM_PROGRAM` when that is non-empty
    after stripping, so traces are groupable by launch environment (e.g.
    "iTerm.app", "vscode", "Apple_Terminal"). Blank values are treated as unset
    to match every other reader of this variable — some shells export
    `TERM_PROGRAM=""` rather than leaving it unset, and terminals that never set
    it (Windows Terminal, ssh, the Linux console) omit the key entirely rather
    than forming a junk grouping bucket. This is a diagnostic key, not part of
    the contract.

    Args:
        thread_id: The app session thread identifier. Set both on
            `configurable.thread_id` and as the top-level `metadata.thread_id`
            used by the contract for grouping turns.
        assistant_id: The dcode agent identifier, if any. When set, it is
            surfaced in trace metadata under `dcode_agent_name` and
            `agent_name`.
        sandbox_type: Sandbox provider name for trace metadata, or `None` if no
            sandbox is active.
        turn_id: Stable per-turn id for the current user prompt, or `None`.
        turn_number: 1-based per-thread turn index, or `None`.
        auto_approve: Whether auto-approve ("YOLO") mode is active for this turn.
            When `True`, `dcode_auto_approve=True` is recorded in trace metadata.
        skill_name: Invoked skill name to record in trace metadata, or `None`.

    Returns:
        Config dict with `configurable` and `metadata` keys, plus
            `recursion_limit` when one is configured.
    """
    from datetime import UTC, datetime

    try:
        cwd = str(Path.cwd())
    except OSError:
        logger.warning("Could not determine working directory", exc_info=True)
        cwd = ""

    from deepagents_code._env_vars import EXPERIMENTAL, USER_ID

    metadata: dict[str, Any] = build_coding_agent_metadata(
        thread_id=thread_id,
        turn_id=turn_id,
        turn_number=turn_number,
        cwd=cwd,
        git_branch=_get_git_branch(),
        sandbox_type=sandbox_type,
        user_id=os.environ.get(USER_ID) or None,
    )

    # Mark experimental runs so they are filterable in trace metadata.
    if is_env_truthy(EXPERIMENTAL):
        metadata["dcode_experimental"] = True

    # Mark auto-approve ("YOLO") runs so they are filterable in trace metadata.
    if auto_approve:
        metadata["dcode_auto_approve"] = True

    if skill_name:
        metadata["ls_skill_name"] = skill_name

    # Record the launch environment so traces are groupable by terminal.
    # Blank is treated as unset, matching the other readers. Not a contract key.
    term_program = os.environ.get("TERM_PROGRAM", "").strip()
    if term_program:
        metadata["dcode_term_program"] = term_program

    # Legacy / diagnostic keys preserved for backward-compatibility during the
    # coding-agent-v1 rollout (not part of the contract).
    editable = _is_editable_install()
    metadata["editable"] = editable
    metadata["lc_versions"] = {
        "deepagents-code": _format_lc_version(__version__, editable=editable)
    }
    deepagents_version = _get_deepagents_version()
    if deepagents_version is not None:
        metadata["dcode_client_deepagents_version"] = deepagents_version
    if assistant_id:
        metadata.update(
            {
                "dcode_agent_name": assistant_id,
                "agent_name": assistant_id,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
    }

    # Send configured limits with each run so the setting takes effect.
    from deepagents_code.config_manifest import resolve_recursion_limit

    resolved_recursion_limit = resolve_recursion_limit()
    if resolved_recursion_limit is not None:
        config["recursion_limit"] = resolved_recursion_limit

    return config


class _ShellAllowAll(list):  # noqa: FURB189  # sentinel type, not a general-purpose list subclass
    """Sentinel subclass for unrestricted shell access.

    Using a dedicated type instead of a plain list lets consumers use
    `isinstance` checks, which survive serialization/copy unlike identity
    checks (`is`).
    """


SHELL_ALLOW_ALL: list[str] = _ShellAllowAll(["__ALL__"])
"""Sentinel value returned by `parse_shell_allow_list` for `--shell-allow-list=all`."""


def parse_shell_allow_list(allow_list_str: str | None) -> list[str] | None:
    """Parse shell allow-list from string.

    Args:
        allow_list_str: Comma-separated list of commands, `'recommended'` for
            safe defaults, or `'all'` to allow any command.

            `'all'` must be the sole value — it is not recognized inside a
            comma-separated list (unlike `'recommended'`).

            Can also include `'recommended'` in the list to merge with custom
            commands.

    Returns:
        List of allowed commands, `SHELL_ALLOW_ALL` if `'all'` was specified,
            or `None` if no allow-list configured.

    Raises:
        ValueError: If `'all'` appears alongside other commands.
    """  # noqa: DOC502 - propagates from `parse_shell_allow_list_items`
    if not allow_list_str:
        return None

    # Special value 'all' allows any shell command
    if allow_list_str.strip().lower() == "all":
        return SHELL_ALLOW_ALL

    # Special value 'recommended' uses our curated safe list
    if allow_list_str.strip().lower() == "recommended":
        return list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    # Split by comma and strip whitespace
    commands = [cmd.strip() for cmd in allow_list_str.split(",") if cmd.strip()]

    return parse_shell_allow_list_items(commands)


def parse_shell_allow_list_items(items: list[str]) -> list[str] | None:
    """Parse an already-split shell allow-list.

    Unlike `parse_shell_allow_list`, this takes separate elements so no comma
    reparsing occurs — a command name containing a comma (valid on POSIX)
    survives intact instead of being split into two entries.

    Args:
        items: Individual allow-list entries, e.g. a TOML array's elements.
            `'all'` must be the sole entry; `'recommended'` merges the curated
            safe list at its position.

    Returns:
        List of allowed commands, `SHELL_ALLOW_ALL` if `'all'` was the sole
        entry, or `None` if every entry was blank.

    Raises:
        ValueError: If `'all'` is combined with other commands.
    """
    commands = [item.strip() for item in items if item.strip()]
    if not commands:
        return None

    if len(commands) == 1 and commands[0].lower() == "all":
        return SHELL_ALLOW_ALL

    # Reject ambiguous input: 'all' mixed with other commands
    if any(cmd.lower() == "all" for cmd in commands):
        msg = (
            "Cannot combine 'all' with other commands in --shell-allow-list. "
            "Use '--shell-allow-list all' alone to allow any command."
        )
        raise ValueError(msg)

    # If "recommended" is in the list, merge with recommended commands
    result = []
    for cmd in commands:
        if cmd.lower() == "recommended":
            result.extend(RECOMMENDED_SAFE_SHELL_COMMANDS)
        else:
            result.append(cmd)

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in result:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique


INTERPRETER_PTC_SAFE_PRESET: frozenset[str] = frozenset({"read_file", "glob", "grep"})
"""Strictly read-only PTC allowlist for `interpreter_ptc="safe"`.

Limited to tools that are **not** in `_add_interrupt_on()` to begin with, so
exposing them through PTC does not introduce a new HITL bypass. Network
tools (`web_search`, `fetch_url`), subagent dispatch (`task`), shell
execution (`execute`), and file writes (`write_file`, `edit_file`, MCP
write tools) are deliberately excluded — they are HITL-gated outside the
REPL, and PTC bypasses `interrupt_on`, so including them would silently
escalate privileges. Users who need network or subagent access from inside
the REPL must list those tools explicitly (which signals intent at config
time) or use `interpreter_ptc="all"` with the unsafe acknowledgement.
"""

INTERPRETER_PTC_ALL_SENTINEL = "all"
"""Sentinel string for `interpreter_ptc="all"` — resolved at agent-build time
from the live tool list. Requires `interpreter_ptc_acknowledge_unsafe=True`
when `auto_approve` is `False`."""

INTERPRETER_PTC_SAFE_SENTINEL = "safe"
"""Sentinel string for `interpreter_ptc="safe"` — expanded from
`INTERPRETER_PTC_SAFE_PRESET`."""


def _parse_interpreter_ptc(
    raw: Any,  # noqa: ANN401  # accepts TOML-shaped value
) -> str | bool | list[str]:
    """Coerce a raw `interpreter_ptc` value into the canonical shape.

    Args:
        raw: Value loaded from TOML or supplied by the CLI.

    Returns:
        `False` for `False`/`None`/`[]`, the string `"safe"`/`"all"` when
        either sentinel is given, otherwise a validated list of tool names.
        A list may include the `"safe"` preset (expanded at agent-build time)
        but never `"all"`.

    Raises:
        ValueError: If `raw` is a list with empty or non-string entries, a
            list containing `"all"`, or a string other than `"safe"`/`"all"`.
    """
    if raw is None or raw is False:
        return False
    if raw is True:
        msg = (
            "`interpreter_ptc` cannot be set to True; use 'safe', 'all', or "
            "an explicit list of tool names."
        )
        raise ValueError(msg)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {INTERPRETER_PTC_SAFE_SENTINEL, INTERPRETER_PTC_ALL_SENTINEL}:
            return normalized
        msg = (
            f"Invalid `interpreter_ptc` string {raw!r}; expected 'safe', 'all', "
            "or a list of tool names."
        )
        raise ValueError(msg)
    if isinstance(raw, list):
        if not raw:
            return False
        names: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                msg = (
                    "`interpreter_ptc` list entries must be non-empty strings; "
                    f"got {entry!r}."
                )
                raise ValueError(msg)
            cleaned = entry.strip()
            if cleaned.lower() == INTERPRETER_PTC_ALL_SENTINEL:
                msg = (
                    "`interpreter_ptc` list entries cannot include 'all'; use "
                    "'all' as a standalone value or list explicit tool names "
                    "(optionally with the 'safe' preset)."
                )
                raise ValueError(msg)
            names.append(cleaned)
        return names
    msg = (
        f"`interpreter_ptc` must be False, 'safe', 'all', or a list of tool "
        f"names; got {type(raw).__name__}."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class _ProviderRetryConfig:
    """Validated retry settings for one provider."""

    max_retries: int | None = None
    param: str | None = None


@dataclass(frozen=True)
class _RetryConfig:
    """Validated retry configuration and user-facing diagnostics."""

    max_retries: int | None = None
    providers: dict[str, _ProviderRetryConfig] = dataclass_field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _coerce_max_retries(raw: Any, *, source: str) -> tuple[int | None, str | None]:  # noqa: ANN401
    """Return a validated retry count and an optional diagnostic."""
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw, None
    return None, f"Ignoring {source}={raw!r} in config.toml (expected int >= 0)"


def _coerce_retry_param(raw: Any, *, source: str) -> tuple[str | None, str | None]:  # noqa: ANN401
    """Return a validated provider retry parameter and optional diagnostic."""
    if isinstance(raw, str) and raw.isidentifier() and not keyword.iskeyword(raw):
        return raw, None
    return (
        None,
        (
            f"Ignoring {source}={raw!r} in config.toml "
            "(expected Python identifier string)"
        ),
    )


def _parse_retry_config(
    section: dict[str, Any] | None,
    *,
    known_providers: set[str] | None = None,
    warnings: tuple[str, ...] = (),
) -> _RetryConfig:
    """Validate a raw `[retries]` table without logging side effects.

    Returns:
        Parsed retry configuration and diagnostics.
    """
    if not section:
        return _RetryConfig(warnings=warnings)

    diagnostics = list(warnings)
    global_retries: int | None = None
    providers: dict[str, _ProviderRetryConfig] = {}
    if "max_retries" in section:
        global_retries, warning = _coerce_max_retries(
            section["max_retries"], source="[retries].max_retries"
        )
        if warning:
            diagnostics.append(warning)

    for provider, raw in section.items():
        if provider == "max_retries":
            continue
        if not isinstance(raw, dict):
            diagnostics.append(f"Ignoring [retries].{provider}={raw!r} in config.toml")
            continue
        if (
            known_providers is not None
            and provider not in known_providers
            and "param" not in raw
        ):
            # Kept, not dropped: a provider dcode does not list can still be
            # the one the langchain registry builds, and discarding the table
            # would silently ignore a setting that does apply. Say that, rather
            # than claiming an override that never happens.
            diagnostics.append(
                f"[retries.{provider}] in config.toml names an unrecognized "
                f"provider; it applies only if {provider!r} is the provider "
                "dcode builds"
            )
        for key, value in raw.items():
            if key not in {"max_retries", "param"}:
                diagnostics.append(
                    f"Ignoring [retries.{provider}].{key}={value!r} in config.toml"
                )
        provider_retries: int | None = None
        retry_param: str | None = None
        if "max_retries" in raw:
            provider_retries, warning = _coerce_max_retries(
                raw["max_retries"], source=f"[retries.{provider}].max_retries"
            )
            if warning:
                diagnostics.append(warning)
        if "param" in raw:
            retry_param, warning = _coerce_retry_param(
                raw["param"], source=f"[retries.{provider}].param"
            )
            if warning:
                diagnostics.append(warning)
        providers[provider] = _ProviderRetryConfig(provider_retries, retry_param)

    return _RetryConfig(global_retries, providers, tuple(diagnostics))


def _read_retry_config() -> _RetryConfig:
    """Read and validate merged retry configuration.

    Returns:
        Parsed retry configuration and diagnostics.
    """
    from deepagents_code.configuration.service import get_config_sources
    from deepagents_code.model_config import (
        IMPLICIT_AUTH_PROVIDERS,
        NO_AUTH_REQUIRED_PROVIDERS,
        PROVIDER_API_KEY_ENV,
        RETRY_PARAM_BY_PROVIDER,
    )

    diagnostics: list[str] = []
    sources = get_config_sources()
    if not sources.user.status.usable:
        diagnostics.append(
            f"Could not read retries config from {sources.user.status.path}"
        )
    dropped = sources.dropped_managed_detail()
    if dropped is not None:
        diagnostics.append(
            f"Managed policy from {sources.managed.status.path} "
            f"is not being applied: {dropped}"
        )
    data, _ = sources.merged()
    section = data.get("retries")
    if not isinstance(section, dict):
        return _RetryConfig(warnings=tuple(diagnostics))

    known_providers = (
        set(PROVIDER_API_KEY_ENV)
        | set(NO_AUTH_REQUIRED_PROVIDERS)
        | set(IMPLICIT_AUTH_PROVIDERS)
        | set(RETRY_PARAM_BY_PROVIDER)
    )
    models = data.get("models")
    if isinstance(models, dict):
        providers = models.get("providers")
        if isinstance(providers, dict):
            known_providers.update(providers)
    return _parse_retry_config(
        section,
        known_providers=known_providers,
        warnings=tuple(diagnostics),
    )


def _resolve_config_retry_count(
    config: _RetryConfig,
    provider: str,
) -> int | None:
    """Return the provider-specific retry count over the global count."""
    provider_config = config.providers.get(provider)
    if provider_config is not None and provider_config.max_retries is not None:
        return provider_config.max_retries
    return config.max_retries


_warned_unknown_retry_providers: set[str] = set()
"""Providers already reported as having no identifiable retry control."""


def _resolve_retry_param(
    config: _RetryConfig, provider: str, model_kwargs: Mapping[str, Any]
) -> str | None:
    """Return the kwarg naming `provider`'s own retry count, if it is known.

    Args:
        config: Parsed retry configuration.
        provider: Effective model provider.
        model_kwargs: Constructor kwargs after all user overrides are merged.

    Returns:
        The kwarg dcode will force to the provider's disable value, or `None`
        when the provider's retry control cannot be identified.
    """
    from deepagents_code.model_config import RETRY_PARAM_BY_PROVIDER

    # An explicit `[retries.<provider>].param` outranks the built-in registry:
    # the user is correcting our knowledge of their provider's SDK, so honoring
    # the registry instead would silently discard the directive.
    provider_config = config.providers.get(provider)
    retry_param = provider_config.param if provider_config is not None else None
    if retry_param is None:
        retry_param = RETRY_PARAM_BY_PROVIDER.get(provider)

    # A custom provider that already exposes the conventional parameter has
    # positively identified its retry control through model configuration. The
    # value itself is replaced by the caller -- it is a detection signal, not a
    # setting.
    if retry_param is None and "max_retries" in model_kwargs:
        retry_param = "max_retries"
    return retry_param


def _provider_retry_disable_kwargs(
    config: _RetryConfig,
    provider: str,
    model_kwargs: dict[str, Any],
) -> dict[str, int]:
    """Return the constructor kwarg that disables provider-owned retries.

    Args:
        config: Parsed retry configuration.
        provider: Effective model provider.
        model_kwargs: Constructor kwargs after all user overrides are merged.

    Returns:
        A one-item mapping that disables provider retries when the provider's
        retry control is known, otherwise an empty mapping.
    """
    from deepagents_code.model_config import RETRY_DISABLE_VALUE_BY_PROVIDER

    retry_param = _resolve_retry_param(config, provider, model_kwargs)
    if retry_param is None:
        # A `None` entry records an integration checked and found to have no
        # retry-count kwarg, so there is no SDK loop to multiply and nothing to
        # warn about. Warning anyway pointed the user at
        # `[retries.<provider>].param`, which their integration would drop.
        from deepagents_code.model_config import RETRY_PARAM_BY_PROVIDER

        if provider in RETRY_PARAM_BY_PROVIDER:
            return {}
        # The provider's own SDK retry loop can't be identified, so it stays
        # active and may multiply the middleware's attempts. Register the
        # provider in `RETRY_PARAM_BY_PROVIDER` or set `[retries.<provider>].param`.
        # Said once per provider: nothing the user can do makes it stop, and
        # `create_model` runs again for every subagent, rubric model, and
        # runtime `/model` switch, so repeating it just buries the debug buffer.
        if provider not in _warned_unknown_retry_providers:
            _warned_unknown_retry_providers.add(provider)
            logger.warning(
                "No retry-disable kwarg known for provider %r, so its own SDK "
                "retries stay active and may multiply dcode's retry attempts. "
                "Set [retries.%s].param in config.toml to name the provider's "
                "retry-count kwarg.",
                provider,
                provider,
            )
        return {}

    disable_value = RETRY_DISABLE_VALUE_BY_PROVIDER.get(provider, 0)
    existing = model_kwargs.get(retry_param)
    if (
        isinstance(existing, int)
        and not isinstance(existing, bool)
        and existing != disable_value
    ):
        logger.warning(
            "Ignoring %s=%r for provider %r: dcode's model-node middleware owns "
            "the retry budget, so the provider's own retry loop is disabled. Use "
            "--max-retries or [retries.%s].max_retries to set the budget.",
            retry_param,
            existing,
            provider,
            provider,
        )
    return {retry_param: disable_value}


DEFAULT_MODEL_RETRIES = 5
"""Default model-node retry attempts after the first call when config is absent.

Canonical source for the dcode retry default; `model_retry` re-exports it so the
middleware and the config resolver never drift.
"""

MODEL_RETRIES_ATTR = "_deepagents_model_retries"
"""Private model attribute carrying the budget resolved when it was built."""


def _resolve_model_retries_from_section(
    config: _RetryConfig,
    provider: str,
    cli_max_retries: int | None,
) -> int:
    """Resolve a retry budget from an already-loaded `[retries]` section.

    Precedence (highest first):

    1. `cli_max_retries` (the `--max-retries` flag).
    2. `[retries.<provider>].max_retries` in `config.toml`.
    3. `[retries].max_retries` (global) in `config.toml`.
    4. `DEFAULT_MODEL_RETRIES`.

    A resolved value of `0` disables retries. The caller passes the section in
    so `create_model` reads `config.toml` once for both the budget and the
    provider disable kwarg.

    Args:
        config: Parsed retry configuration.
        provider: Effective model provider.
        cli_max_retries: Explicit CLI override, or `None` when unset.

    Returns:
        The effective retry count. Always `>= 0` for a `cli_max_retries` that
        came through `--max-retries`, which `non_negative_int` validates.
    """
    if cli_max_retries is not None:
        return cli_max_retries
    configured = _resolve_config_retry_count(config, provider)
    return configured if configured is not None else DEFAULT_MODEL_RETRIES


def collect_retry_config_startup(
    provider: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[str], set[str]]:
    """Return retry diagnostics and the retry kwarg dcode will force.

    Args:
        provider: Effective model provider, when it is already resolved.
        model_kwargs: Constructor kwargs the run will supply, used to detect a
            custom provider's retry control.

    Returns:
        User-facing diagnostics, and the retry kwarg names `create_model` will
        override for `provider`. The set is empty when the provider is unknown
        or its retry control cannot be identified, because a kwarg belonging to
        some other provider is forwarded to the constructor untouched.
    """
    retry_config = _read_retry_config()
    if not provider:
        return list(retry_config.warnings), set()
    retry_param = _resolve_retry_param(retry_config, provider, model_kwargs or {})
    return list(retry_config.warnings), {retry_param} if retry_param else set()


_extra_skills_path_base: ContextVar[Path | None] = ContextVar(
    "extra_skills_path_base",
    default=None,
)


@contextmanager
def _use_extra_skills_path_base(path: Path | None) -> Iterator[None]:
    """Resolve relative extra skill roots from an explicit project path."""
    token = _extra_skills_path_base.set(path)
    try:
        yield
    finally:
        _extra_skills_path_base.reset(token)


def _resolve_extra_skills_path(raw: str) -> Path:
    """Resolve one configured skill root from the active project path.

    Returns:
        The absolute, symlink-resolved skill root.
    """
    path = Path(raw).expanduser()
    base = _extra_skills_path_base.get()
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_extra_skills_dirs(
    env_raw: str | None,
    config_toml_dirs: list[str] | None = None,
) -> list[Path] | None:
    """Merge extra skill directories from env var and config.toml.

    Extra skills directories extend the containment allowlist used by
    `load_skill_content` to validate that a resolved skill path lives inside a
    trusted root. They do **not** add new skill discovery locations — skills are
    still discovered only from the standard directories. This exists so that
    symlinks inside standard skill directories can legitimately point to targets
    in user-specified locations without being rejected by the path
    containment check.

    The env var (`DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS`, colon-separated) takes
    precedence: when set, `config.toml` values are ignored.

    Args:
        env_raw: Value of `DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS` (colon-separated), or
            `None` if unset.
        config_toml_dirs: List of path strings from
            `[skills].extra_allowed_dirs` in `~/.deepagents/config.toml`.

    Returns:
        List of resolved `Path` objects, or `None` if not configured.
    """
    # Env var takes precedence when set
    if env_raw:
        dirs = [
            _resolve_extra_skills_path(p.strip())
            for p in env_raw.split(":")
            if p.strip()
        ]
        return dirs or None

    if config_toml_dirs:
        dirs = [
            _resolve_extra_skills_path(p)
            for p in config_toml_dirs
            if isinstance(p, str) and p.strip()
        ]
        return dirs or None

    return None


MANAGED_RELOAD_BLOCKED_PREFIX = "Kept previous settings: "
"""Lead-in of the notice a reload returns when managed policy blocked it.

`reload_from_environment` reports the block as the first entry of its change
list, so a caller that only counts changes reads "policy could not be
refreshed" as "nothing changed". Use `managed_reload_block` to recover it
instead of matching this text at each call site.
"""


def managed_reload_block(changes: Sequence[str]) -> str | None:
    """Return the managed-policy block notice a reload reported, if any.

    Args:
        changes: The list `reload_from_environment` or `preview_reload` returned.

    Scans the whole list rather than only its first entry. The notice is
    prepended today, so position would work -- but a caller that reads a
    blocked reload as success mounts "Restart complete." on the previous policy
    generation, and one future entry prepended ahead of the notice would cause
    that at all four call sites at once. Scanning cannot regress that way, and
    only this module produces the prefix.

    Returns:
        The notice, or `None` when managed policy did not block the reload.
    """
    for change in changes:
        if change.startswith(MANAGED_RELOAD_BLOCKED_PREFIX):
            return change
    return None


_RELOADABLE_FIELDS = (
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "nvidia_api_key",
    "tavily_api_key",
    "google_cloud_project",
    "google_cloud_location",
    "deepagents_langchain_project",
    "project_root",
)
"""Fields refreshed on `/reload` and cwd switches.

Runtime model metadata lives in `RuntimeState` and cannot be touched by a config
reload. The original user LangSmith project is intentionally excluded because it
is captured once at bootstrap.
"""

_API_KEY_FIELDS = frozenset(
    field for field in _RELOADABLE_FIELDS if field.endswith("_api_key")
)
"""Reloadable fields that hold API keys and must be masked in change reports.

Derived from `_RELOADABLE_FIELDS` so new `*_api_key` fields are picked up
automatically.
"""

_RESOLVER_RELOAD_FIELDS = (
    "shell_allow_list",
    "extra_skills_dirs",
)
"""Resolver-backed values included in reload previews and change reports."""

_RELOAD_CHANGE_FIELDS = (*_RELOADABLE_FIELDS, *_RESOLVER_RELOAD_FIELDS)
"""Stable display order for every reload-owned change report entry."""

_resolver_reload_snapshot: dict[str, object] = {
    "shell_allow_list": None,
    "extra_skills_dirs": None,
}
_resolver_reload_snapshot_lock = threading.Lock()


@dataclass(slots=True)
class _ReloadOverrideProvider:
    """Retain resolver values when one reload candidate cannot be applied."""

    name: str = "retained reload value"
    rank: int = 350
    _values: dict[str, object] = dataclass_field(default_factory=dict)
    _lock: threading.Lock = dataclass_field(default_factory=threading.Lock)

    @property
    def durable(self) -> bool:
        """The retained generation exists only for this process lifetime."""
        return False

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Return a retained typed value or leave the option unset."""
        from deepagents_code.configuration.resolver import RankedProviderValue
        from deepagents_code.configuration.types import (
            Found,
            ProviderHealth,
            ProviderStatus,
            Unset,
        )

        with self._lock:
            result = (
                Found(self._values[option.key])
                if option.key in self._values
                else Unset()
            )
        return RankedProviderValue(
            self.rank,
            self.durable,
            ProviderStatus(self.name, None, ProviderHealth.OK),
            result,
        )

    def status(self) -> ProviderStatus:
        """Return the health of the in-memory retained generation."""
        from deepagents_code.configuration.types import ProviderHealth, ProviderStatus

        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Keep the accepted in-memory generation unchanged."""

    def replace(self, values: Mapping[str, object]) -> None:
        """Atomically replace the options whose previous values stay in force."""
        with self._lock:
            self._values = dict(values)


_reload_override_provider = _ReloadOverrideProvider()


def _resolver_with_reload_overrides() -> ConfigResolver:
    """Return the shared resolver with the reload-retention tier installed."""
    from deepagents_code.configuration.resolver import (
        RELOAD_RANK,
        get_config_resolver,
    )

    resolver = get_config_resolver()
    if RELOAD_RANK not in resolver.provider_statuses():
        resolver.install_provider(_reload_override_provider)
    return resolver


def _sync_reload_overrides(
    values: Mapping[str, object], *, path_base: Path | None
) -> None:
    """Retain values that the refreshed resolver generation cannot reproduce."""
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.resolver import RELOAD_RANK

    resolver = _resolver_with_reload_overrides()
    retained: dict[str, object] = {}
    option_keys = {
        "shell_allow_list": "shell.allow_list",
        "extra_skills_dirs": "skills.extra_allowed_dirs",
    }
    with _use_extra_skills_path_base(path_base):
        for field, key in option_keys.items():
            option = get_option(key)
            if option is None:
                continue
            try:
                candidate = resolver.get_without_ranks(option, {RELOAD_RANK}).value
            except (OSError, RuntimeError, ValueError):
                candidate = object()
            if candidate != values[field]:
                retained[key] = values[field]
    _reload_override_provider.replace(retained)


def _remember_resolver_reload_values(values: Mapping[str, object]) -> None:
    """Remember the accepted resolver generation for future change reports."""
    with _resolver_reload_snapshot_lock:
        for field in _RESOLVER_RELOAD_FIELDS:
            _resolver_reload_snapshot[field] = values[field]


def _remembered_resolver_reload_values() -> dict[str, object]:
    """Return the resolver generation currently accepted by runtime reload."""
    with _resolver_reload_snapshot_lock:
        return dict(_resolver_reload_snapshot)


def _current_resolver_reload_values(*, path_base: Path | None) -> dict[str, object]:
    """Snapshot resolver-backed values before a preview or accepted reload.

    Returns:
        Resolver values keyed by their stable reload-report field names.

    Raises:
        RuntimeError: If either reload-owned option is absent from the manifest.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option

    options = tuple(
        option
        for key in ("shell.allow_list", "skills.extra_allowed_dirs")
        if (option := get_option(key)) is not None
    )
    if len(options) != len(_RESOLVER_RELOAD_FIELDS):
        msg = "reload options are missing from the configuration manifest"
        raise RuntimeError(msg)
    with _use_extra_skills_path_base(path_base):
        resolved = _resolver_with_reload_overrides().resolve_options(options)
    for option in options:
        _emit_ranked_diagnostics(option, resolved[option.key])
    return {
        "shell_allow_list": resolved["shell.allow_list"].value,
        "extra_skills_dirs": resolved["skills.extra_allowed_dirs"].value,
    }


@dataclass
class RuntimeState:
    """Mutable metadata for the model active in this process."""

    model_name: str | None = None
    """Currently active model name, set after model creation."""

    model_provider: str | None = None
    """Provider identifier (e.g., `openai`, `anthropic`, `google_genai`)."""

    model_context_limit: int | None = None
    """Maximum input token count from the model profile."""

    model_unsupported_modalities: frozenset[str] = frozenset()
    """Input modalities not indicated as supported by the model profile."""


@dataclass(frozen=True, slots=True)
class CredentialsSnapshot:
    """One complete generation of credentials and project context."""

    openai_api_key: str | None
    """OpenAI API key if available."""

    anthropic_api_key: str | None
    """Anthropic API key if available."""

    google_api_key: str | None
    """Google API key if available."""

    nvidia_api_key: str | None
    """NVIDIA API key if available."""

    tavily_api_key: str | None
    """Tavily API key if available."""

    google_cloud_project: str | None
    """Google Cloud project ID for VertexAI authentication."""

    google_cloud_location: str | None
    """Google Cloud region for Anthropic models on Vertex AI."""

    deepagents_langchain_project: str | None
    """LangSmith project name for deepagents agent tracing."""

    user_langchain_project: str | None
    """Original `LANGSMITH_PROJECT` from environment (for user code)."""

    project_root: Path | None = None
    """Current project root directory, or `None` if not in a git project."""

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None

    @property
    def has_vertex_ai(self) -> bool:
        """Check if VertexAI is available (Google Cloud project set, no API key)."""
        return self.google_cloud_project is not None and self.google_api_key is None

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None


_CREDENTIAL_FIELDS = frozenset(CredentialsSnapshot.__dataclass_fields__)


class Credentials:
    """Stable owner of the active credential and project-context snapshot.

    Reloads construct a complete immutable `CredentialsSnapshot` and publish it
    with one reference assignment. Callers that need a consistent multi-field
    view can retain `active` while ordinary field reads remain source compatible.
    """

    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    nvidia_api_key: str | None
    tavily_api_key: str | None
    google_cloud_project: str | None
    google_cloud_location: str | None
    deepagents_langchain_project: str | None
    user_langchain_project: str | None
    project_root: Path | None

    def __init__(self, active: CredentialsSnapshot) -> None:
        """Create a stable owner for one complete credential generation."""
        self._active = active

    @property
    def active(self) -> CredentialsSnapshot:
        """Complete credential generation currently in force."""
        return self._active

    def __getattr__(self, name: str) -> object:
        """Forward credential field reads to the active immutable snapshot.

        Returns:
            The requested credential field value.

        Raises:
            AttributeError: If `name` is not a credential field.
        """
        if name in _CREDENTIAL_FIELDS:
            return getattr(self._active, name)
        msg = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a replacement snapshot for compatibility field mutations."""
        if name in _CREDENTIAL_FIELDS:
            self._active = dataclass_replace(self._active, **{name: value})
            return
        object.__setattr__(self, name, value)

    @classmethod
    def snapshot_from_environment(
        cls,
        *,
        start_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> CredentialsSnapshot:
        """Resolve an immutable credential snapshot without publishing state.

        Returns:
            Workspace-local credential and project context values.
        """
        env = active_environment() if environ is None else environ
        openai_key = _resolve_env_var_from(env, "OPENAI_API_KEY")
        anthropic_key = _resolve_env_var_from(env, "ANTHROPIC_API_KEY")
        google_key = _resolve_env_var_from(env, "GOOGLE_API_KEY")
        nvidia_key = _resolve_env_var_from(env, "NVIDIA_API_KEY")
        tavily_key = _resolve_env_var_from(env, "TAVILY_API_KEY")
        google_cloud_project = _resolve_env_var_from(env, "GOOGLE_CLOUD_PROJECT")
        google_cloud_location = _resolve_env_var_from(env, "GOOGLE_CLOUD_LOCATION")
        from deepagents_code._env_vars import LANGSMITH_PROJECT
        from deepagents_code.project_utils import find_project_root

        return CredentialsSnapshot(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            nvidia_api_key=nvidia_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            google_cloud_location=google_cloud_location,
            deepagents_langchain_project=_resolve_env_var_from(env, LANGSMITH_PROJECT),
            user_langchain_project=_user_langsmith_project_from(env),
            project_root=find_project_root(start_path),
        )

    @classmethod
    def from_environment(
        cls,
        *,
        start_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Credentials:
        """Create credentials and publish legacy resolver reload state.

        Returns:
            Stable owner of the resolved credential snapshot.
        """
        snapshot = cls.snapshot_from_environment(
            start_path=start_path,
            environ=environ,
        )
        _reload_override_provider.replace({})
        _remember_resolver_reload_values(
            _current_resolver_reload_values(path_base=start_path)
        )
        return cls(snapshot)

    @staticmethod
    def _reload_values(
        *,
        start_path: Path | None,
        env: dict[str, str],
        previous: dict[str, object],
        refresh_managed: bool = True,
    ) -> tuple[dict[str, object], str | None]:
        """Resolve reloadable settings from an environment mapping.

        Managed policy outranks the environment for every field it declares. A
        managed source that is present but unenforceable keeps `previous`
        unchanged, so a reload can never drop policy that is already in force.

        Args:
            start_path: Directory to start project detection from.
            env: Environment mapping to resolve from.
            previous: Current values, kept for any field that cannot be resolved.
            refresh_managed: Re-read managed policy from disk. A preview passes
                `False`: re-reading swaps the process-wide snapshot that every
                other reader observes, which is not something a dry run may do.

        Returns:
            Reloadable setting values keyed by field name, and a notice when a
            source could not be applied (`None` when both applied cleanly).
            Managed policy that blocks the reload and a `config.toml` that
            fails to parse both keep the previous values in force, so both must
            say so rather than letting the caller report "no changes".
        """
        from deepagents_code._env_vars import (
            EXTRA_SKILLS_DIRS,
            LANGSMITH_PROJECT,
            SHELL_ALLOW_LIST,
        )
        from deepagents_code.configuration.service import (
            ManagedConfigError,
            get_healthy_managed_snapshot,
            managed_decided,
        )

        # Refresh in place rather than invalidating first: dropping the cached
        # snapshot before the reload would leave every other reader with an
        # empty managed table if the new file fails to parse, which reads as
        # "no policy" instead of "policy unchanged". `refresh=True` keeps the
        # last snapshot that parsed cleanly and still raises on the failure.
        #
        # A preview must not refresh at all: it is a dry run, and re-reading
        # replaces the snapshot that every other reader in the process observes
        # before the user has accepted anything.
        try:
            # Path-valued policy must be validated against the same project
            # base used when the candidate resolver applies it below. Without
            # this, a relative managed skill root can validate in the old cwd,
            # fail in the target cwd, and fall through to the user's env value.
            with _use_extra_skills_path_base(start_path):
                managed_snapshot = get_healthy_managed_snapshot(refresh=refresh_managed)
        except ManagedConfigError as exc:
            logger.error("Keeping previous settings: %s", exc)  # noqa: TRY400
            # Report the block to the caller. Returning only `previous` reads
            # as "nothing changed", so the user would be told the reload
            # succeeded while their environment edits were discarded.
            return dict(previous), f"{MANAGED_RELOAD_BLOCKED_PREFIX}{exc}"

        from deepagents_code.config_manifest import (
            _emit_ranked_diagnostics,
            _ranked_source,
            get_option,
        )
        from deepagents_code.configuration.resolver import (
            CLI_RANK,
            RELOAD_RANK,
            USER_RANK,
            get_config_resolver,
            resolver_from_snapshots,
        )
        from deepagents_code.configuration.types import Found

        # A real `/reload` exists to pick up file edits made since the shared
        # resolver's snapshot was taken, so this method and later
        # `get_config_resolver()` readers observe the same generation. Seed the
        # resolver with the snapshot just validated above; asking it to refresh
        # managed policy again would let one reload observe multiple files.
        resolver = get_config_resolver(
            refresh_managed=refresh_managed,
            managed_snapshot=managed_snapshot,
        )

        # A user file that fails to parse keeps the previous generation in
        # force, which is the right runtime behavior but silent: the only
        # signal is a `logger.warning` in the debug buffer, while the report
        # the user reads says "Configuration reloaded. No changes detected."
        # Managed corruption is already surfaced as a notice above; a
        # `config.toml` the user just edited deserves the same treatment, and
        # more so -- they are staring at the edit that did not take.
        user_notice: str | None = None
        provider_statuses = resolver.provider_statuses()
        user_status = provider_statuses.get(USER_RANK)
        if user_status is not None and not user_status.usable:
            detail = user_status.detail or user_status.health.value
            user_notice = f"Kept previous config.toml: {detail}"
            logger.error("Keeping previous config.toml: %s", detail)

        try:
            shell_allow_list = parse_shell_allow_list(env.get(SHELL_ALLOW_LIST))
        except ValueError:
            logger.warning(
                "Invalid %s during reload; keeping previous value",
                SHELL_ALLOW_LIST,
            )
            shell_allow_list = previous["shell_allow_list"]

        candidate_resolver = resolver
        if not refresh_managed:
            from deepagents_code.configuration.providers import TomlFileProvider
            from deepagents_code.model_config import DEFAULT_CONFIG_PATH

            user_candidate = TomlFileProvider("config.toml", DEFAULT_CONFIG_PATH).load()
            if user_candidate.status.usable:
                candidate_resolver = resolver_from_snapshots(
                    managed=managed_snapshot,
                    user=user_candidate,
                )
                user_notice = None
            else:
                detail = (
                    user_candidate.status.detail or user_candidate.status.health.value
                )
                user_notice = f"Kept previous config.toml: {detail}"

        shell_option = get_option("shell.allow_list")
        if shell_option is not None:
            shell_resolved = resolver.get_without_ranks(
                shell_option,
                {RELOAD_RANK},
            )
            _emit_ranked_diagnostics(shell_option, shell_resolved)
            shell_source = _ranked_source(shell_resolved)
            if (
                not managed_decided(shell_source)
                and CLI_RANK not in shell_resolved.ranks
                and candidate_resolver is not resolver
            ):
                shell_resolved = candidate_resolver.get(shell_option)
                _emit_ranked_diagnostics(shell_option, shell_resolved)
                shell_source = _ranked_source(shell_resolved)
            if (
                managed_decided(shell_source)
                or CLI_RANK in shell_resolved.ranks
                or shell_source == "config.toml"
            ):
                shell_allow_list = cast("list[str] | None", shell_resolved.value)

        try:
            from deepagents_code.project_utils import find_project_root

            project_root = find_project_root(start_path)
        except OSError:
            logger.warning(
                "Could not detect project root during reload; keeping previous value"
            )
            project_root = previous["project_root"]

        try:
            with _use_extra_skills_path_base(start_path):
                skills_option = get_option("skills.extra_allowed_dirs")
                resolved_skills: list[Path] | None = None
                skills_managed = False
                if skills_option is not None:
                    skills_resolved = (
                        candidate_resolver.get_without_ranks(
                            skills_option,
                            {RELOAD_RANK},
                        )
                        if candidate_resolver is resolver
                        else candidate_resolver.get(skills_option)
                    )
                    _emit_ranked_diagnostics(skills_option, skills_resolved)
                    if managed_decided(_ranked_source(skills_resolved)):
                        skills_managed = True
                        resolved_skills = cast(
                            "list[Path] | None", skills_resolved.value
                        )
                    else:
                        user_result = skills_resolved.tier_health[USER_RANK]
                        if isinstance(user_result, Found):
                            resolved_skills = cast(
                                "list[Path] | None", user_result.value
                            )
                env_skills = env.get(EXTRA_SKILLS_DIRS)
                extra_skills_dirs = (
                    resolved_skills
                    if skills_managed or not env_skills
                    else _parse_extra_skills_dirs(env_skills)
                )
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Could not resolve %s during reload; keeping previous value",
                EXTRA_SKILLS_DIRS,
                exc_info=True,
            )
            extra_skills_dirs = previous["extra_skills_dirs"]

        return {
            "openai_api_key": _resolve_env_var_from(env, "OPENAI_API_KEY"),
            "anthropic_api_key": _resolve_env_var_from(env, "ANTHROPIC_API_KEY"),
            "google_api_key": _resolve_env_var_from(env, "GOOGLE_API_KEY"),
            "nvidia_api_key": _resolve_env_var_from(env, "NVIDIA_API_KEY"),
            "tavily_api_key": _resolve_env_var_from(env, "TAVILY_API_KEY"),
            "google_cloud_project": _resolve_env_var_from(env, "GOOGLE_CLOUD_PROJECT"),
            "google_cloud_location": _resolve_env_var_from(
                env, "GOOGLE_CLOUD_LOCATION"
            ),
            "deepagents_langchain_project": _resolve_env_var_from(
                env,
                LANGSMITH_PROJECT,
            ),
            "project_root": project_root,
            "shell_allow_list": shell_allow_list,
            "extra_skills_dirs": extra_skills_dirs,
        }, user_notice

    @staticmethod
    def _format_reload_changes(
        previous: dict[str, object], refreshed: dict[str, object]
    ) -> list[str]:
        """Format changed reloadable settings for logs and messages.

        Returns:
            Human-readable change descriptions.
        """

        def display(field: str, value: object) -> str:
            if field in _API_KEY_FIELDS:
                return "set" if value else "unset"
            return str(value)

        changes: list[str] = []
        for field in _RELOAD_CHANGE_FIELDS:
            old_value = previous[field]
            new_value = refreshed[field]
            if old_value != new_value:
                changes.append(
                    f"{field}: {display(field, old_value)} -> "
                    f"{display(field, new_value)}"
                )
        return changes

    def preview_reload_from_environment(
        self, *, start_path: Path | None = None
    ) -> list[str]:
        """Preview runtime settings changes without applying them.

        Args:
            start_path: Directory to start project detection from (defaults to cwd).

        Returns:
            A list of human-readable change descriptions that would be produced by
            `reload_from_environment`.
        """
        active = self.active
        previous = {field: getattr(active, field) for field in _RELOADABLE_FIELDS}
        previous.update(_remembered_resolver_reload_values())
        env = _preview_dotenv_environ(start_path=start_path)
        refreshed, blocked = self._reload_values(
            start_path=start_path,
            env=env,
            previous=previous,
            refresh_managed=False,
        )
        changes = self._format_reload_changes(previous, refreshed)
        return [blocked, *changes] if blocked else changes

    def reload_from_environment(self, *, start_path: Path | None = None) -> list[str]:
        """Reload selected settings from environment variables and project files.

        This refreshes only fields that are expected to change at runtime
        (API keys, Google Cloud project, project root, and LangSmith tracing
        project). Resolver-backed configuration is refreshed separately by the
        shared `ConfigResolver` generation.

        Runtime model metadata lives in `RuntimeState` and is never touched by
        this method. The original user LangSmith project
        (`user_langchain_project`) is also intentionally preserved because it is
        not in `_RELOADABLE_FIELDS`.

        !!! note

            Managed config takes precedence over shell-exported variables for
            the fields it declares. Below managed policy, shell exports still
            outrank `.env` values. Values previously injected from `.env` files
            are refreshed so an accepted cwd switch can pick up the resumed
            project's `.env`.

        Args:
            start_path: Directory to start project detection from (defaults to cwd).

        Returns:
            A list of human-readable change descriptions. Empty when nothing
            changed; a single notice when managed policy blocked the reload. A
            notice also leads the list when the LangSmith carrier was unusable
            and the current settings were kept.
        """
        active = self.active
        previous = {field: getattr(active, field) for field in _RELOADABLE_FIELDS}
        previous.update(_remembered_resolver_reload_values())
        _resolver_with_reload_overrides()
        encoded = os.environ.get(_USER_LANGSMITH_ENV_CARRIER)
        carrier_notice: str | None = None
        if encoded is not None:
            carried = _decode_user_langsmith_env(encoded)
            if carried is None:
                # Applying the mapping anyway would reset this process's
                # LangSmith identity from data already known to be unusable.
                # Leave the environment alone and say so: a reload that
                # silently changes credentials is the hard kind to debug.
                carrier_notice = (
                    "Kept the current LangSmith settings: your launch settings "
                    "could not be read (restart dcode if this persists)"
                )
                logger.warning("%s", carrier_notice)
            else:
                launch, _ = carried
                _bootstrap_state.launch_langsmith_env = launch
        if carrier_notice is None:
            _apply_env_values(os.environ, _bootstrap_state.launch_langsmith_env)
        _load_dotenv(
            start_path=start_path,
            refresh_loaded=True,
            capture_user_langsmith=True,
        )
        # The refreshed values are not republished into `os.environ`. Reload runs
        # only in the client, where `_bootstrap_state` is already the source of
        # truth and `_build_server_env` re-encodes from it at every spawn. Writing
        # the carrier here bought nothing and left the user's plaintext API key,
        # inside a JSON blob no secret scrubber recognizes, in the environment
        # every client-spawned child inherits.
        apply_stored_langsmith_auth()
        refreshed, blocked = self._reload_values(
            start_path=start_path,
            env=dict(os.environ),
            previous=previous,
        )

        replacement = dataclass_replace(
            active,
            **{field: refreshed[field] for field in _RELOADABLE_FIELDS},
        )
        _remember_resolver_reload_values(refreshed)
        _sync_reload_overrides(refreshed, path_base=start_path)

        # Sync the LANGSMITH_PROJECT env var so LangSmith tracing picks up
        # the change
        new_project = refreshed["deepagents_langchain_project"]
        if new_project:
            os.environ["LANGSMITH_PROJECT"] = str(new_project)
        elif previous["deepagents_langchain_project"]:
            # Override was previously active but new value is unset; restore the
            # user's original project. With no original, drop the override and
            # re-apply the default so ingestion keeps matching the name
            # `get_langsmith_project_name` displays (the default is a no-op when
            # tracing is off, so a disabled setup is left unset).
            if _bootstrap_state.original_langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = (
                    _bootstrap_state.original_langsmith_project
                )
            else:
                os.environ.pop("LANGSMITH_PROJECT", None)
                _apply_default_langsmith_project()

        # A reload can repoint env resolution at a different .env (e.g. after a
        # cwd switch), so start a fresh diagnostics generation; otherwise the new
        # "Resolved X from ..." lines would be suppressed by the pre-reload dedup
        # set.
        from deepagents_code.model_config import reset_env_resolution_log

        reset_env_resolution_log()
        changes = self._format_reload_changes(previous, refreshed)
        if carrier_notice is not None:
            changes = [carrier_notice, *changes]
        if managed_reload_block([blocked] if blocked else []) is None:
            self._active = replacement
        return [blocked, *changes] if blocked else changes

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.active.has_anthropic

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return self.active.has_google

    @property
    def has_vertex_ai(self) -> bool:
        """Check if VertexAI is available (Google Cloud project set, no API key).

        VertexAI uses Application Default Credentials (ADC) for authentication,
        so if GOOGLE_CLOUD_PROJECT is set and GOOGLE_API_KEY is not, we assume
        VertexAI.
        """
        return self.active.has_vertex_ai

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.active.has_tavily


DANGEROUS_SHELL_PATTERNS = (
    "$(",  # Command substitution
    "`",  # Backtick command substitution
    "$'",  # ANSI-C quoting (can encode dangerous chars via escape sequences)
    "\n",  # Newline (command injection)
    "\r",  # Carriage return (command injection)
    "\t",  # Tab (can be used for injection in some shells)
    "<(",  # Process substitution (input)
    ">(",  # Process substitution (output)
    "<<<",  # Here-string
    "<<",  # Here-doc (can embed commands)
    ">>",  # Append redirect
    ">",  # Output redirect
    "<",  # Input redirect
    "${",  # Variable expansion with braces (can run commands via ${var:-$(cmd)})
)
"""Literal substrings that indicate shell injection risk.

Used by `contains_dangerous_patterns` to reject commands that embed arbitrary
execution via redirects, substitution operators, or control characters — even
when the base command is on the allow-list.
"""

RECOMMENDED_SAFE_SHELL_COMMANDS = (
    # Directory listing
    "ls",
    "dir",
    # File content viewing (read-only)
    "cat",
    "head",
    "tail",
    # Text searching (read-only)
    "grep",
    "wc",
    "strings",
    # Text processing (read-only, no shell execution)
    "cut",
    "tr",
    "diff",
    "md5sum",
    "sha256sum",
    # Path utilities
    "pwd",
    "which",
    # System info (read-only)
    "uname",
    "hostname",
    "whoami",
    "id",
    "groups",
    "uptime",
    "nproc",
    "lscpu",
    "lsmem",
    # Process viewing (read-only)
    "ps",
)
"""Read-only commands auto-approved in non-interactive mode.

Only includes readers and formatters — shells, editors, interpreters, package
managers, network tools, archivers, and anything on GTFOBins/LOOBins is
intentionally excluded. File-write and injection vectors are blocked separately
by `DANGEROUS_SHELL_PATTERNS`.
"""


def contains_dangerous_patterns(command: str) -> bool:
    """Check if a command contains dangerous shell patterns.

    These patterns can be used to bypass allow-list validation by embedding
    arbitrary commands within seemingly safe commands. The check includes
    both literal substring patterns (redirects, substitution operators, etc.)
    and regex patterns for bare variable expansion (`$VAR`) and the background
    operator (`&`).

    Args:
        command: The shell command to check.

    Returns:
        True if dangerous patterns are found, False otherwise.
    """
    if any(pattern in command for pattern in DANGEROUS_SHELL_PATTERNS):
        return True

    # Bare variable expansion ($VAR without braces) can leak sensitive paths.
    # We already block ${ and $( above; this catches plain $HOME, $IFS, etc.
    if re.search(r"\$[A-Za-z_]", command):
        return True

    # Standalone & (background execution) changes the execution model and
    # should not be allowed.  We check for & that is NOT part of &&.
    return bool(re.search(r"(?<![&])&(?![&])", command))


def is_shell_command_allowed(command: str, allow_list: list[str] | None) -> bool:
    """Check if a shell command is in the allow-list.

    The allow-list matches against the first token of the command (the executable
    name). This allows read-only commands like ls, cat, grep, etc. to be
    auto-approved.

    When `allow_list` is the `SHELL_ALLOW_ALL` sentinel, all non-empty commands
    are approved unconditionally — dangerous pattern checks are skipped.

    SECURITY: For regular allow-lists, this function rejects commands containing
    dangerous shell patterns (command substitution, redirects, process
    substitution, etc.) BEFORE parsing, to prevent injection attacks that could
    bypass the allow-list.

    Args:
        command: The full shell command to check.
        allow_list: List of allowed command names (e.g., `["ls", "cat", "grep"]`),
            the `SHELL_ALLOW_ALL` sentinel to allow any command, or `None`.

    Returns:
        `True` if the command is allowed, `False` otherwise.
    """
    if not allow_list or not command or not command.strip():
        return False

    # SHELL_ALLOW_ALL sentinel — skip pattern and token checks
    if isinstance(allow_list, _ShellAllowAll):
        return True

    # SECURITY: Check for dangerous patterns BEFORE any parsing
    # This prevents injection attacks like: ls "$(rm -rf /)"
    if contains_dangerous_patterns(command):
        return False

    allow_set = set(allow_list)

    # Extract the first command token
    # Handle pipes and other shell operators by checking each command in the pipeline
    # Split by compound operators first (&&, ||), then single-char operators (|, ;).
    # Note: standalone & (background) is blocked by contains_dangerous_patterns above.
    segments = re.split(r"&&|\|\||[|;]", command)

    # Track if we found at least one valid command
    found_command = False

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        try:
            # Try to parse as shell command to extract the executable name
            tokens = shlex.split(segment)
            if tokens:
                found_command = True
                cmd_name = tokens[0]
                # Check if this command is in the allow set
                if cmd_name not in allow_set:
                    return False
        except ValueError:
            # If we can't parse it, be conservative and require approval
            return False

    # All segments are allowed (and we found at least one command)
    return found_command


def get_langsmith_project_name() -> str | None:
    """Resolve the LangSmith project name if tracing is configured.

    Checks for the required API key and tracing environment variables.
    When both are present, resolves the project name with priority:
    `credentials.deepagents_langchain_project` (from
    `DEEPAGENTS_CODE_LANGSMITH_PROJECT`), then `LANGSMITH_PROJECT` from the
    environment (note: this may already have been overridden at bootstrap time
    to match `DEEPAGENTS_CODE_LANGSMITH_PROJECT`), then `'deepagents-code'`.

    Returns:
        Project name string when LangSmith tracing is active, None otherwise.
    """
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT
    from deepagents_code.model_config import resolve_env_var

    langsmith_key = resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var(
        "LANGCHAIN_API_KEY"
    )
    if not (langsmith_key and _tracing_enabled()):
        return None

    from deepagents_code._env_vars import LANGSMITH_PROJECT

    environ = active_environment()
    # The process-global credentials describe the launch environment, so a
    # workspace snapshot has to resolve the override itself.
    override = (
        _resolve_env_var_from(environ, LANGSMITH_PROJECT)
        if _environment_is_scoped()
        else _get_credentials().deepagents_langchain_project
    )
    return override or environ.get("LANGSMITH_PROJECT") or LANGSMITH_PROJECT_DEFAULT


@dataclass(frozen=True)
class LangsmithShadowResult:
    """Why `/trace` found no LangSmith key, when an empty override is involved.

    Distinguishes the three states the caller renders differently: a specific
    empty override is suppressing an available key (`shadowing_var`), the
    credential store could not be read so a stored key can't be ruled out
    (`store_unreadable`), or neither (both fields falsy -- the generic "not
    configured" hint applies).
    """

    shadowing_var: str | None = None
    """Prefixed env var whose empty value is suppressing an available key."""

    store_unreadable: bool = False
    """`True` when the `/auth` credential store raised while being checked."""


def langsmith_key_shadowed_by_empty_override() -> LangsmithShadowResult:
    """Report an empty prefixed override that is suppressing a LangSmith key.

    `/trace` shows a generic "not configured" hint whenever no key resolves, but
    a common footgun is exporting `DEEPAGENTS_CODE_LANGSMITH_API_KEY=` (empty).
    A present-but-empty prefixed variable suppresses a key two ways: per
    `resolve_env_var`'s precedence it shadows the canonical env variable
    directly, and -- because `apply_stored_service_credentials` skips the `/auth`
    bridge onto `LANGSMITH_API_KEY` whenever the prefixed var is present -- it
    also keeps a `/auth`-stored key from ever reaching the environment. Either
    way tracing silently stays off even though a key is available. Detecting this
    lets callers name the offending variable instead of sending the user to
    `/auth`.

    Only an override that actually gates the *effective* key is reported. If a
    key already resolves under the normal `LANGSMITH_API_KEY`-before-
    `LANGCHAIN_API_KEY` precedence, tracing is off for some other reason (a
    missing tracing flag), no override is to blame, and nothing is reported.
    Otherwise each override is checked against the specific key it suppresses, so
    the returned name is one that, once unset, actually lets a key resolve: its
    canonical variant carries a value, or -- for `LANGSMITH_API_KEY`, the var
    `/auth` bridges its stored key onto -- a stored key exists. When several
    overrides qualify, the first in `_TRACING_API_KEY_ENV_VARS` order is
    returned.

    Returns:
        A `LangsmithShadowResult`; see its fields for the three outcomes.
    """
    from deepagents_code import auth_store
    from deepagents_code.model_config import (
        LANGSMITH_SERVICE,
        resolve_env_var,
        resolved_env_var_name,
    )

    if resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var("LANGCHAIN_API_KEY"):
        # A key already resolves (matching `get_langsmith_project_name`'s key
        # precedence), so no empty override is what's keeping tracing off, and
        # unsetting one would change nothing. Defer to the generic hint.
        return LangsmithShadowResult()

    store_unreadable = False
    for name in _TRACING_API_KEY_ENV_VARS:
        resolved = resolved_env_var_name(name)
        if resolved == name or os.environ.get(resolved):
            # No prefixed override for this key, or the override carries a value:
            # either way it is not an empty override suppressing this key.
            continue
        if (os.environ.get(name) or "").strip():
            # The empty override is hiding a value on the canonical variable.
            return LangsmithShadowResult(shadowing_var=resolved)
        if name == "LANGSMITH_API_KEY":
            # `/auth` bridges its stored key onto `LANGSMITH_API_KEY`, so an
            # empty override for it also suppresses a stored key.
            try:
                if auth_store.get_stored_key(LANGSMITH_SERVICE):
                    return LangsmithShadowResult(shadowing_var=resolved)
            except RuntimeError as exc:
                # Can't confirm a stored key, but keep scanning: a later
                # override may still name a concrete shadow. Only if none does
                # do we surface the unreadable store to the caller.
                logger.warning(
                    "Could not read the stored LangSmith credential while "
                    "checking for an empty-override shadow: %s. The credential "
                    "file may be corrupt; re-add the key via /auth.",
                    exc,
                )
                store_unreadable = True
    return LangsmithShadowResult(store_unreadable=store_unreadable)


def is_langsmith_redaction_enabled() -> bool:
    """Return whether LangSmith secret redaction is enabled for agent traces."""
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("tracing.langsmith_redact")
    if option is None:
        return True
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value)


def is_memory_auto_save_enabled() -> bool:
    """Return whether the agent should proactively save learnings to memory.

    Resolves the `memory.auto_save` option from env/`config.toml`, defaulting to
    enabled. When disabled, memory is still loaded into context but the agent is
    told not to auto-save.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("memory.auto_save")
    if option is None:
        return True
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value)


def is_yolo_switcher_enabled() -> bool:
    """Return whether Shift+Tab may enter unrestricted YOLO mode.

    Resolves the `startup.yolo_switcher` option from env/`config.toml`,
    defaulting to enabled. When disabled, the interactive cycle stays Manual /
    Auto only (or Manual alone when Auto is ineligible). Sessions already in
    YOLO (for example via `--yolo`) can still leave it with Shift+Tab.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("startup.yolo_switcher")
    if option is None:
        return True
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value)


def is_openai_prompt_cache_key_enabled() -> bool:
    """Return whether OpenAI model calls should carry a per-thread cache key.

    Resolves the `models.openai_prompt_cache_key` option from env/`config.toml`,
    defaulting to enabled. When disabled, `ConfigurableModelMiddleware` stops
    injecting the thread ID as an OpenAI `prompt_cache_key` (a user-supplied key
    is still forwarded). This is the opt-out for OpenAI-compatible endpoints that
    reject unknown request fields.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("models.openai_prompt_cache_key")
    if option is None:
        return True
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value)


def resolve_auto_classifier_model_with_problem() -> tuple[str | None, str | None]:
    """Resolve the Auto classifier spec and any reason it was ignored.

    Reads the `models.auto_classifier` option from managed policy, then env,
    then `config.toml`. `None` means the classifier inherits the main agent
    model, which is the historical behavior and the default.

    A configured-but-unusable value (blank, or a non-string such as
    `auto_classifier = 3`, which coercion drops to the default) silently
    reverts authorization review to the main agent model — the agent grading its
    own actions. The caller gets a description so it can say so on a surface the
    user actually reads; a log line alone is not that surface.

    A present-but-blank env var is an explicit "inherit" and outranks
    `config.toml`, so it is detected before resolution rather than being skipped
    as unset the way the resolver treats every other option's blank env
    value. A managed value outranks that veto, so it is resolved first; a blank
    managed value also forces inherit, credited to managed policy. `dcode
    config` shares this order via `resolve_auto_classifier_model_with_source`,
    so the two surfaces cannot disagree about which model grades gated actions.

    Returns:
        `(spec, problem)`. `spec` is a `provider:model` spec, or `None` when the
            classifier should inherit. `problem` is a one-line description when a
            configured value was ignored, else `None`.
    """
    from deepagents_code.config_manifest import (
        _emit_ranked_diagnostics,
        _ranked_source,
        blank_auto_classifier_env_name,
        get_option,
    )
    from deepagents_code.configuration.resolver import USER_RANK, get_config_resolver
    from deepagents_code.configuration.types import Found, Invalid

    option = get_option("models.auto_classifier")
    if option is None:
        return None, None
    from deepagents_code.configuration.service import managed_decided

    managed_resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, managed_resolved)
    managed_value = managed_resolved.value
    managed_source = _ranked_source(managed_resolved)
    if managed_decided(managed_source):
        if isinstance(managed_value, str) and managed_value.strip():
            return managed_value.strip(), None
        # A blank managed entry is an explicit inherit; anything else blank or
        # malformed reverts to the main agent model, credited to the file that
        # declared it.
        problem = (
            f"Ignoring blank {managed_source} auto_classifier model; the Auto "
            "approval classifier will review with the main agent model."
        )
        logger.warning("%s", problem)
        return None, problem

    def resolve_user_tier() -> tuple[object, str]:
        """Return the user value from the shared resolution generation.

        The blank-env veto below names the user-level value it overrides, so
        it needs the user tier alone rather than the selected env value. Reading
        that result out of the shared resolution keeps the classifier on the
        same user snapshot as every other manifest consumer.

        Returns:
            The user value and its compatibility source label, or the default
                when the user tier did not supply a usable value.
        """
        user_result = managed_resolved.tier_health[USER_RANK]
        if isinstance(user_result, Found):
            return user_result.value, managed_resolved.provider_status[USER_RANK].name
        return option.default, "default"

    blank_env = blank_auto_classifier_env_name()
    if blank_env is not None:
        # Name the config.toml value being overridden: without it the warning
        # sends the user to a config file that still shows their setting.
        shadowed, shadowed_source = resolve_user_tier()
        overridden = (
            f" (overriding {shadowed_source} {shadowed!r})"
            if isinstance(shadowed, str) and shadowed.strip()
            else ""
        )
        problem = (
            f"Ignoring blank env ({blank_env}) auto_classifier model{overridden}; "
            "the Auto approval classifier will review with the main agent model."
        )
        logger.warning("%s", problem)
        return None, problem
    value, source = managed_resolved.value, managed_source
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    if isinstance(value, str) and source != "default":
        problem = (
            f"Ignoring blank {source} auto_classifier model; the Auto approval "
            "classifier will review with the main agent model."
        )
        logger.warning("%s", problem)
        return None, problem
    # TOML coercion drops a wrong-typed user value to the option default. The
    # shared resolution retains that rejection in its user-tier result, so it
    # can remain visible without reopening a potentially newer file generation.
    user_result = managed_resolved.tier_health[USER_RANK]
    if isinstance(user_result, Invalid):
        # `reason` names the rejected value ("Ignoring
        # [models].auto_classifier=42 in config.toml (expected str)"). Dropping
        # it left the user told their setting is malformed with no indication
        # of what they wrote, on a surface they actually read.
        problem = (
            f"{user_result.reason}; expected a provider:model string. The Auto "
            "approval classifier will review with the main agent model."
        )
        logger.warning("%s", problem)
        return None, problem
    return None, None


_DEFAULT_AUTO_CLASSIFIER_MODELS = {
    "anthropic": "anthropic:claude-sonnet-5",
    "google_genai": "google_genai:gemini-3.8-flash",
    "google_vertexai": "google_vertexai:gemini-3.8-flash",
    "openai": "openai:gpt-5.6-luna",
    "openai_codex": "openai_codex:gpt-5.6-luna",
}


def default_auto_classifier_model(provider: str) -> str | None:
    """Return the default Auto classifier model for a main-model provider."""
    return _DEFAULT_AUTO_CLASSIFIER_MODELS.get(provider)


def resolve_auto_classifier_model_for_provider(
    provider: str,
    classifier_model: str | None = None,
) -> str | None:
    """Resolve an unset Auto classifier after the main provider is known.

    Returns:
        The configured or provider-default classifier, or `None` to inherit.
    """
    if classifier_model is not None:
        return classifier_model

    from deepagents_code._cli_context import INHERIT_CLASSIFIER_MODEL
    from deepagents_code.model_config import ModelConfig

    configured, problem = resolve_auto_classifier_model_with_problem()
    if problem is not None:
        return INHERIT_CLASSIFIER_MODEL
    selected = configured or default_auto_classifier_model(provider)
    if selected is None:
        return None
    if ModelConfig.load().policy_error(selected, canonicalize=True) is not None:
        return INHERIT_CLASSIFIER_MODEL
    return selected


def resolve_auto_classifier_model() -> str | None:
    """Resolve the model spec the Auto approval classifier should use.

    Returns:
        A `provider:model` spec, or `None` when the classifier should inherit.
    """
    spec, _problem = resolve_auto_classifier_model_with_problem()
    return spec


def resolve_goal_auto_accept_criteria() -> tuple[bool, str]:
    """Resolve whether Auto mode applies generated goal criteria without review.

    Returns:
        The effective preference and its config source. The preference fails closed
        to disabled if the manifest entry is unavailable.
    """
    from deepagents_code.config_manifest import (
        _emit_ranked_diagnostics,
        _ranked_source,
        get_option,
    )
    from deepagents_code.configuration.resolver import get_config_resolver

    option = get_option("goals.auto_accept_criteria")
    if option is None:
        logger.warning(
            "Manifest option 'goals.auto_accept_criteria' is missing; goal "
            "criteria auto-accept is disabled and any saved preference is "
            "ignored.",
        )
        return False, "default"
    resolved = get_config_resolver().get(option)
    _emit_ranked_diagnostics(option, resolved)
    return bool(resolved.value), _ranked_source(resolved)


def configure_langsmith_secret_redaction() -> bool:
    """Install the LangSmith SDK secret anonymizer for active agent tracing.

    This is a fail-closed security control: when redaction is requested but the
    redacting client cannot be installed, tracing is disabled rather than risk
    uploading unredacted secrets to LangSmith.

    Returns:
        `True` when a redacting LangSmith client was configured, `False` when
        tracing is inactive, has no upload target, redaction is disabled, or the
        redacting client could not be installed (tracing is then disabled).
    """
    from deepagents_code._env_vars import LANGSMITH_REDACT

    env = active_environment()
    # The tracing checks come first so the common (tracing-off) startup path
    # skips the TOML read in `is_langsmith_redaction_enabled`. They stay outside
    # the fail-closed boundary: if there is no upload target, there is nothing
    # to protect.
    #
    # Resolved into locals, and short-circuited, because
    # `_tracing_can_upload_from` reads the LangSmith profile config from disk --
    # asking it twice, or asking it at all with the flag off, is startup I/O for
    # an answer that changes nothing.
    tracing_enabled = _tracing_enabled_from(env)
    if not (tracing_enabled and _tracing_can_upload_from(env)):
        # Distinguish "nothing to protect" from "declined": without this, a
        # missing redacting client looks the same either way after the fact.
        logger.debug(
            "Skipping secret redaction: tracing enabled=%s, upload target checked=%s",
            tracing_enabled,
            tracing_enabled,
        )
        return False

    # Everything from here on runs inside the fail-closed boundary: any
    # unexpected exception (including from the redaction-toggle lookup) disables
    # tracing rather than escaping and leaving tracing live but unredacted.
    try:
        if not is_langsmith_redaction_enabled():
            logger.warning(
                "LangSmith tracing is active without secret redaction; secrets may "
                "be uploaded to traces unredacted. Set %s=true to enable redaction.",
                LANGSMITH_REDACT,
            )
            return False

        from langsmith import Client, configure
        from langsmith.anonymizer import create_secret_anonymizer

        api_key = _resolve_env_var_from(
            env,
            "LANGSMITH_API_KEY",
        ) or _resolve_env_var_from(env, "LANGCHAIN_API_KEY")
        api_url = _tracing_endpoint_from(env)
        kwargs: dict[str, Any] = {"anonymizer": create_secret_anonymizer()}
        if api_key:
            kwargs["api_key"] = api_key
        if api_url:
            kwargs["api_url"] = api_url
        # Reinstall the redacting client on every call rather than caching it:
        # callers such as `/auth` re-authentication may rotate credentials, and
        # a cached client could leave a stale or non-redacting client in place —
        # a fail-open risk this control exists to prevent.
        configure(client=Client(**kwargs))
    except Exception:
        logger.exception(
            "Failed to install LangSmith secret redaction; disabling tracing so "
            "unredacted secrets are not uploaded.",
        )
        _fail_closed_disable_tracing()
        return False

    logger.info("LangSmith secret redaction enabled for agent traces.")
    return True


def _fail_closed_disable_tracing() -> None:
    """Best-effort disable LangSmith tracing after a redaction setup failure.

    The SDK's global tracing switch (`configure(enabled=False)`) is the primary,
    load-bearing control and is tried first. Clearing the canonical
    tracing-enable env vars (and their `DEEPAGENTS_CODE_`-prefixed forms) is only
    a last-resort fallback for the case where even that call fails (e.g. the
    `langsmith` import is broken): the LangChain tracer checks the global switch
    first but falls back to these env vars, so removing them helps prevent a
    newly created tracer from starting an unredacted upload. (It only helps —
    the SDK's env-var lookup is `lru_cache`d, so a value already read this
    process may still be served from cache; the global switch is the reliable
    stop.)
    """
    try:
        from langsmith import configure

        configure(enabled=False)
    except Exception:
        logger.exception(
            "Failed to disable LangSmith tracing via the SDK after a redaction "
            "setup failure; clearing tracing env vars as a fallback.",
        )
    else:
        return

    from deepagents_code.model_config import _ENV_PREFIX

    for var in _TRACING_ENABLE_ENV_VARS:
        os.environ.pop(var, None)
        os.environ.pop(f"{_ENV_PREFIX}{var}", None)


def get_langsmith_replica_projects() -> list[str]:
    """Extra LangSmith project names to dual-write agent traces to.

    Parses `DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS` (comma-separated) into a
    de-duplicated, order-preserving list.

    Returns:
        Project names, or `[]` when the env var is unset or empty.
    """
    return _get_langsmith_replica_projects_from(active_environment())


def _get_langsmith_replica_projects_from(env: Mapping[str, str]) -> list[str]:
    """Parse replica project names from an environment snapshot.

    Args:
        env: Environment mapping to read.

    Returns:
        Project names, or `[]` when the env var is unset or empty.
    """
    from deepagents_code._env_vars import LANGSMITH_REPLICA_PROJECTS

    raw = env.get(LANGSMITH_REPLICA_PROJECTS)
    if not raw:
        return []
    return list(dict.fromkeys(p.strip() for p in raw.split(",") if p.strip()))


def get_langsmith_replica_project() -> str | None:
    """The single extra LangSmith project to mirror agent runs to, if configured.

    dcode agent runs execute inside the LangGraph server subprocess, so the only
    way to mirror them to another project is the server's own replica path: the
    SDK forwards a `langsmith_tracing` project in the run-create request, and the
    server wraps the run in a `tracing_context` whose write replicas are that
    project plus the server's primary project. Client-side callbacks and
    `tracing_context(replicas=...)` cannot reach the run because it is created
    server-side, not in the app process.

    Implementation detail (subject to change): as of `langgraph-api` 0.10.0 this
    happens in `langgraph_api.stream` and `langgraph_api.models.run`.

    The server mirrors to exactly one extra project, so when
    `DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS` lists several, only the first is
    used and the rest are dropped with a warning.

    Returns:
        The first configured replica project name, or `None` when none are set.
    """
    extras = get_langsmith_replica_projects()
    return _get_first_langsmith_replica_project(extras)


def _get_first_langsmith_replica_project(extras: list[str]) -> str | None:
    """Return the first configured LangSmith replica project, if any.

    Args:
        extras: Parsed replica project names.

    Returns:
        The first configured replica project name, or `None` when none are set.
    """
    if not extras:
        return None
    if len(extras) > 1:
        logger.warning(
            "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS lists %d projects, but the "
            "LangGraph server mirrors runs to only one extra project; tracing to "
            "%r and ignoring %s.",
            len(extras),
            extras[0],
            extras[1:],
        )
    return extras[0]


def _tracing_enabled_from(env: Mapping[str, str]) -> bool:
    """Return whether tracing is (or will be) enabled, prefix-aware.

    Mirrors the runtime: `DEEPAGENTS_CODE_`-prefixed forms of the bridged flags
    count (bootstrap propagates them), while the non-bridged flags are honored
    only in their canonical form.

    Args:
        env: Environment mapping to read.
    """
    from deepagents_code._env_vars import classify_env_bool

    for var in _TRACING_BRIDGED_ENABLE_ENV_VARS:
        raw = _resolve_env_var_from(env, var)
        if raw is not None and classify_env_bool(raw):
            return True
    return any(
        classify_env_bool(env[var])
        for var in _TRACING_ENABLE_ENV_VARS
        if var not in _TRACING_BRIDGED_ENABLE_ENV_VARS and var in env
    )


def _tracing_explicitly_disabled_from(env: Mapping[str, str]) -> bool:
    """Return whether a tracing flag is explicitly set to a recognized off value.

    True only when tracing is not enabled and at least one tracing-enable flag
    carries a falsy token (`0`/`false`/`no`/`off`). An empty flag usually reads
    as "not configured" rather than "disabled", except when an empty prefixed
    bridged flag shadows a canonical truthy flag and therefore disables tracing.

    Args:
        env: Environment mapping to read.
    """
    from deepagents_code._env_vars import classify_env_bool
    from deepagents_code.model_config import _ENV_PREFIX

    if _tracing_enabled_from(env):
        return False

    def _is_off(raw: str | None) -> bool:
        if raw is None or not raw.strip():
            return False
        return classify_env_bool(raw) is False

    def _empty_prefixed_shadow_disables(var: str) -> bool:
        prefixed = f"{_ENV_PREFIX}{var}"
        if prefixed not in env or env[prefixed].strip():
            return False
        canonical = env.get(var)
        return canonical is not None and classify_env_bool(canonical) is True

    for var in _TRACING_BRIDGED_ENABLE_ENV_VARS:
        if _is_off(_resolve_env_var_from(env, var)) or _empty_prefixed_shadow_disables(
            var
        ):
            return True
    return any(
        _is_off(env.get(var))
        for var in _TRACING_ENABLE_ENV_VARS
        if var not in _TRACING_BRIDGED_ENABLE_ENV_VARS
    )


def _tracing_enabled() -> bool:
    """Return whether tracing is (or will be) enabled, prefix-aware."""
    return _tracing_enabled_from(active_environment())


def _tracing_has_credentials_from(env: Mapping[str, str]) -> bool:
    """Return whether a LangSmith API key (env or active profile) is available.

    Both API-key vars are bridged from a `DEEPAGENTS_CODE_` prefix at bootstrap,
    so resolve them prefix-aware to match what the runtime will see.

    Args:
        env: Environment mapping to read.
    """
    has_key = any(_resolve_env_var_from(env, var) for var in _TRACING_API_KEY_ENV_VARS)
    return has_key or _has_langsmith_profile_credentials(env)


def _langsmith_runs_endpoint_urls_from(env: Mapping[str, str]) -> tuple[str, ...]:
    """Return the replica trace ingestion URLs configured via runs-endpoints.

    Mirrors the LangSmith SDK's accepted `LANGSMITH_RUNS_ENDPOINTS` shapes: a
    JSON list of `{"api_url": "...", "api_key": "..."}` objects, or a JSON
    object mapping URL to API key. Invalid entries are ignored because the SDK
    ignores them too.

    Args:
        env: Environment mapping to read.

    Returns:
        The configured replica ingestion URLs, in configuration order.
    """
    raw = next(
        (
            env[var]
            for var in _TRACING_RUNS_ENDPOINTS_ENV_VARS
            if (env.get(var) or "").strip()
        ),
        None,
    )
    if raw is None:
        return ()

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return ()

    if isinstance(parsed, list):
        return tuple(
            item["api_url"]
            for item in parsed
            if isinstance(item, dict)
            and isinstance(item.get("api_url"), str)
            and isinstance(item.get("api_key"), str)
        )
    if isinstance(parsed, dict):
        return tuple(
            url
            for url, api_key in parsed.items()
            if isinstance(url, str) and isinstance(api_key, str)
        )
    return ()


def _has_langsmith_runs_endpoints_from(env: Mapping[str, str]) -> bool:
    """Return whether replica trace ingestion targets are configured.

    Args:
        env: Environment mapping to read.

    Returns:
        `True` when a valid runs-endpoints configuration is present.
    """
    return bool(_langsmith_runs_endpoint_urls_from(env))


def _tracing_can_upload_from(env: Mapping[str, str]) -> bool:
    """Return whether tracing has credentials or an ingestion endpoint.

    Custom and replica endpoints are supported as keyless ingestion targets, so
    redaction must be configured whenever tracing could still upload without an
    API key.

    Args:
        env: Environment mapping to read.

    Returns:
        `True` when tracing has credentials or any ingestion endpoint set.
    """
    return (
        _tracing_has_credentials_from(env)
        or _tracing_endpoint_from(env) is not None
        or _has_langsmith_runs_endpoints_from(env)
    )


def _tracing_endpoint_from(env: Mapping[str, str]) -> str | None:
    """Return a custom tracing endpoint (env or active profile), if configured.

    The endpoint vars are not bridged from a `DEEPAGENTS_CODE_` prefix and the
    LangSmith SDK reads them canonically, so only the canonical names (plus the
    active profile's `api_url`) are consulted here.

    Mirrors SDK resolution order (`env_api_url or profile_config.api_url`): a
    populated env endpoint wins over the profile. The SDK default US SaaS URL is
    not a custom ingest target — when env is that default, return `None` without
    falling through to the profile. When env is unset, a non-default profile
    `api_url` still counts.

    Args:
        env: Environment mapping to read.
    """
    for var in _TRACING_ENDPOINT_ENV_VARS:
        value = (env.get(var) or "").strip()
        if not value:
            continue
        if _is_langsmith_sdk_default_endpoint(value):
            # Non-empty env wins over profile, even when it is the SDK default.
            return None
        return value
    config = _load_langsmith_profile_config(env)
    if config is not None:
        api_url = (config.api_url or "").strip()
        if api_url and not _is_langsmith_sdk_default_endpoint(api_url):
            return api_url
    return None


def _resolve_tracing_project_from(env: Mapping[str, str]) -> tuple[str, bool]:
    """Resolve the project agent traces would route to, without bootstrap.

    The reported project matches the `tracing.langsmith_project` manifest
    option's env precedence: the prefixed `DEEPAGENTS_CODE_LANGSMITH_PROJECT`
    (skipped when empty), then bare `LANGSMITH_PROJECT`, then the default.
    Unlike `resolve_env_var`, an empty prefixed value does not shadow a real
    `LANGSMITH_PROJECT`.

    Args:
        env: Environment mapping to read.

    Returns:
        The resolved project name and whether it fell back to the default
            because no project was explicitly configured.
    """
    from deepagents_code._env_vars import LANGSMITH_PROJECT
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    for name in (LANGSMITH_PROJECT, "LANGSMITH_PROJECT"):
        value = env.get(name)
        if value:
            return value, False
    return LANGSMITH_PROJECT_DEFAULT, True


def _tracing_diagnostic_env() -> dict[str, str]:
    """Return the dotenv-aware environment snapshot for tracing diagnostics.

    Returns:
        Environment mapping with project/global dotenv values applied using the
        same precedence as bootstrap, without mutating `os.environ`.
    """
    from deepagents_code.project_utils import get_server_project_context

    ctx = get_server_project_context()
    return _preview_dotenv_environ(start_path=ctx.user_cwd if ctx else None)


@dataclass(frozen=True)
class TracingStatus:
    """Offline snapshot of LangSmith tracing configuration for diagnostics.

    Carries only presence/identity facts — never API keys or other secret
    values — so it is safe to render in `dcode doctor` output.
    """

    enabled: bool
    """Whether a tracing flag is truthy in the environment."""

    explicitly_disabled: bool
    """Whether a tracing flag is explicitly set to a falsy value (vs. unset)."""

    has_credentials: bool
    """Whether an API key or profile credential is resolvable."""

    endpoint: str | None
    """Custom (self-hosted/proxied) endpoint URL, if one is configured."""

    project: str | None
    """Resolved configured project name, independent of active trace ingestion."""

    project_is_default: bool
    """Whether `project` is the built-in default rather than an explicit setting."""

    replica_project: str | None
    """Extra project agent runs are mirrored to, if configured."""

    runs_endpoints: tuple[str, ...] = ()
    """Replica ingestion URLs from `LANGSMITH_RUNS_ENDPOINTS`, if any."""

    def __post_init__(self) -> None:
        """Reject the contradictory enabled/explicitly-disabled pair.

        `enabled` and `explicitly_disabled` model a tri-state (enabled /
        explicitly disabled / not configured), so both being true is
        meaningless. Fail loud at construction rather than letting the illegal
        state flow through to the `dcode doctor` renderer.

        Raises:
            ValueError: If both `enabled` and `explicitly_disabled` are true.
        """
        if self.enabled and self.explicitly_disabled:
            msg = "tracing cannot be both enabled and explicitly disabled"
            raise ValueError(msg)


def get_tracing_status() -> TracingStatus:
    """Summarize LangSmith tracing configuration for diagnostics.

    Reads only the local environment and the active LangSmith profile; never
    contacts the network and never exposes secret values. All fields are
    resolved prefix-/profile-aware so the report matches what the runtime does
    after bootstrap, even though `dcode doctor` runs before it.

    Returns:
        A `TracingStatus` snapshot describing the current tracing setup.
    """
    env = _tracing_diagnostic_env()
    enabled = _tracing_enabled_from(env)
    has_credentials = _tracing_has_credentials_from(env)
    endpoint = _tracing_endpoint_from(env)
    project, project_is_default = _resolve_tracing_project_from(env)
    return TracingStatus(
        enabled=enabled,
        explicitly_disabled=_tracing_explicitly_disabled_from(env),
        has_credentials=has_credentials,
        endpoint=endpoint,
        project=project,
        project_is_default=project_is_default,
        replica_project=_get_first_langsmith_replica_project(
            _get_langsmith_replica_projects_from(env)
        ),
        runs_endpoints=_langsmith_runs_endpoint_urls_from(env),
    )


class LangSmithLookupError(Exception):
    """Base class for typed LangSmith project URL lookup failures.

    Concrete subclasses (`LangSmithImportError`, `LangSmithLookupTimeoutError`,
    `LangSmithApiError`) let interactive callers like `/trace` show the user
    the actual cause instead of collapsing every failure into a generic
    "could not reach LangSmith" message.
    """


class LangSmithImportError(LangSmithLookupError):
    """The `langsmith` package is not installed."""


class LangSmithLookupTimeoutError(LangSmithLookupError):
    """The LangSmith project URL lookup exceeded its hard timeout."""


class LangSmithApiError(LangSmithLookupError):
    """The LangSmith SDK call raised — auth, 404, network, etc.

    Wraps the underlying SDK exception in `__cause__`.
    """


class LangSmithProjectNotFoundError(LangSmithApiError):
    """The LangSmith project does not exist yet (lookup returned 404).

    Projects are created lazily on the first ingested trace, so this is
    expected before any run has flushed and should be surfaced as an
    informational message rather than an error.
    """


def _is_langsmith_not_found(exc: Exception) -> bool:
    """Whether a LangSmith SDK error indicates the project does not exist.

    Returns:
        `True` for a `LangSmithNotFoundError` (404), `False` otherwise.
    """
    try:
        from langsmith.utils import LangSmithNotFoundError
    except ImportError:
        return False
    return isinstance(exc, LangSmithNotFoundError)


def _assemble_langsmith_thread_url(project_url: str, thread_id: str) -> str:
    """Format a LangSmith thread URL from a project URL prefix.

    Args:
        project_url: Project URL prefix from `fetch_langsmith_project_url`
            (e.g. `https://smith.langchain.com/o/<org>/projects/p/<proj>`).
        thread_id: Thread identifier to append.

    Returns:
        Full thread URL with the `deepagents-code` utm tag.
    """
    return f"{project_url.rstrip('/')}/t/{thread_id}?utm_source=deepagents-code"


def fetch_langsmith_project_url_or_raise(project_name: str) -> str:
    """Fetch the LangSmith project URL, raising on any failure.

    Successful results are cached at module level so repeated calls do not
    make additional network requests.

    The network call runs in a daemon thread with a hard timeout of
    `_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS`, so this function blocks the
    calling thread for at most that duration even if LangSmith is unreachable.

    Args:
        project_name: LangSmith project name to look up.

    Returns:
        Project URL string.

    Raises:
        LangSmithImportError: `langsmith` is not installed.
        LangSmithLookupTimeoutError: lookup exceeded the hard timeout.
        LangSmithProjectNotFoundError: the project does not exist yet (404).
        LangSmithApiError: the SDK call raised (auth, network, etc.);
            wraps the original exception in `__cause__`.
    """
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement

    if _langsmith_url_cache is not None:
        cached_name, cached_url = _langsmith_url_cache
        if cached_name == project_name:
            return cached_url
        # Different project name — fall through to fetch.

    try:
        from langsmith import Client
    except ImportError as exc:
        logger.debug(
            "langsmith package not installed; cannot fetch project URL for '%s'",
            project_name,
            exc_info=True,
        )
        msg = "langsmith package is not installed"
        raise LangSmithImportError(msg) from exc

    result: str | None = None
    lookup_error: Exception | None = None
    done = threading.Event()

    def _lookup_url() -> None:
        nonlocal result, lookup_error
        try:
            from deepagents_code.model_config import resolve_env_var

            # Explicit api_key because Client() reads os.environ directly
            # and doesn't know about the DEEPAGENTS_CODE_ prefix.
            api_key = resolve_env_var("LANGSMITH_API_KEY") or resolve_env_var(
                "LANGCHAIN_API_KEY"
            )
            project = Client(api_key=api_key).read_project(project_name=project_name)
            result = project.url or None
        except Exception as exc:  # noqa: BLE001  # LangSmith SDK error types are not stable
            lookup_error = exc
        finally:
            done.set()

    thread = threading.Thread(target=_lookup_url, daemon=True)
    thread.start()

    if not done.wait(_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS):
        logger.debug(
            "Timed out fetching LangSmith project URL for '%s' after %.1fs",
            project_name,
            _LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS,
        )
        msg = (
            f"LangSmith project URL lookup timed out after "
            f"{_LANGSMITH_URL_LOOKUP_TIMEOUT_SECONDS:.1f}s"
        )
        raise LangSmithLookupTimeoutError(msg)

    if lookup_error is not None:
        logger.debug(
            "Could not fetch LangSmith project URL for '%s'",
            project_name,
            exc_info=(
                type(lookup_error),
                lookup_error,
                lookup_error.__traceback__,
            ),
        )
        msg = str(lookup_error) or repr(lookup_error)
        if _is_langsmith_not_found(lookup_error):
            raise LangSmithProjectNotFoundError(msg) from lookup_error
        raise LangSmithApiError(msg) from lookup_error

    if not result:
        # SDK returned a project with an empty URL — treat as an API anomaly.
        msg = f"LangSmith returned no URL for project '{project_name}'"
        raise LangSmithApiError(msg)

    _langsmith_url_cache = (project_name, result)
    return result


def fetch_langsmith_project_url(project_name: str) -> str | None:
    """Fetch the LangSmith project URL, returning None on any failure.

    Thin back-compat wrapper around `fetch_langsmith_project_url_or_raise`
    for passive callers (status banners, non-interactive output) that just
    want a URL-or-nothing answer. Interactive callers that need to tell the
    user *why* the lookup failed should use the raising variant directly.

    Args:
        project_name: LangSmith project name to look up.

    Returns:
        Project URL string if found, None otherwise.
    """
    try:
        return fetch_langsmith_project_url_or_raise(project_name)
    except LangSmithLookupError:
        return None


def build_langsmith_thread_url(thread_id: str) -> str | None:
    """Build a full LangSmith thread URL if tracing is configured.

    Combines `get_langsmith_project_name` and `fetch_langsmith_project_url`
    into a single convenience helper.

    Args:
        thread_id: Thread identifier to build the URL for.

    Returns:
        Full thread URL string, or `None` if unavailable (LangSmith is not
            configured or the project URL cannot be resolved.)
    """
    project_name = get_langsmith_project_name()
    if not project_name:
        return None

    project_url = fetch_langsmith_project_url(project_name)
    if not project_url:
        return None

    return _assemble_langsmith_thread_url(project_url, thread_id)


def get_cached_langsmith_thread_url(thread_id: str) -> str | None:
    """Build a LangSmith thread URL only when its project URL is cached.

    This non-blocking variant lets transient UI surfaces render a previously
    resolved link immediately without repeating or scheduling another lookup.

    Args:
        thread_id: Thread identifier to build the URL for.

    Returns:
        Full thread URL string when the active project's URL is already cached,
        otherwise `None`.
    """
    project_name = get_langsmith_project_name()
    if not project_name or _langsmith_url_cache is None:
        return None

    cached_name, cached_url = _langsmith_url_cache
    if cached_name != project_name:
        return None
    return _assemble_langsmith_thread_url(cached_url, thread_id)


def reset_langsmith_url_cache() -> None:
    """Reset the LangSmith URL cache (for testing)."""
    global _langsmith_url_cache  # noqa: PLW0603  # Module-level cache requires global statement
    _langsmith_url_cache = None


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (AGENTS.md) is handled separately by the middleware.

    Returns:
        The default agent instructions as a string.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


_BEDROCK_REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.")
"""Cross-region inference-profile prefixes that front a vendor namespace.

E.g. `us.anthropic.claude-3-5-sonnet-20241022-v2:0`. Only stripped when a vendor
namespace follows, so a bare name merely starting with `us`/`eu` is untouched.
"""


def _is_bedrock_model_id(model_lower: str) -> bool:
    """Return whether *model_lower* is a bare Bedrock model ID.

    Bedrock IDs have the shape `[<region>.]<vendor>.<model>[:<version>]`, e.g.
    `meta.llama3-70b-instruct-v1:0` or the cross-region inference profile
    `us.anthropic.claude-3-5-sonnet-20241022-v2:0`. Rather than enumerate AWS's
    ever-growing vendor list, this keys off the structural signature: an
    alphanumeric vendor token immediately followed by a dot. Bare direct-API
    names don't fit -- they either have no dot (`mistral-large`, `command-r`),
    carry a hyphen before their version dot (`claude-3.5`, `gemini-2.5`), or are
    already claimed by an earlier prefix check (`gpt-4.1`). Case is folded by the
    caller, and the explicit `bedrock:<model>` syntax is handled upstream via
    `provider:model` parsing.
    """
    for region in _BEDROCK_REGION_PREFIXES:
        if model_lower.startswith(region):
            model_lower = model_lower.removeprefix(region)
            break
    vendor, dot, _ = model_lower.partition(".")
    return bool(dot) and vendor.isalnum()


def _detection_credentials() -> Credentials | CredentialsSnapshot:
    """Return the credential source that matches the active environment scope.

    Returns:
        A workspace-scoped snapshot when an environment is bound, otherwise the
        cached process-global credentials.
    """
    if _environment_is_scoped():
        return Credentials.snapshot_from_environment()
    return _get_credentials()


def detect_provider(model_name: str) -> str | None:
    """Auto-detect provider from model name.

    Intentionally duplicates a subset of LangChain's
    `_attempt_infer_model_provider` because we need to resolve the provider
    **before** calling `init_chat_model` in order to:

    1. Build provider-specific kwargs (API base URLs, headers, etc.) that are
       passed *into* `init_chat_model`.
    2. Validate credentials early to surface user-friendly errors.

    Args:
        model_name: Model name to detect provider from.

    Returns:
        Provider name inferred from the model name (some names, e.g. `claude`
            and `gemini`, are disambiguated using configured credentials), or
            `None` if the provider cannot be determined.
    """
    model_lower = model_name.lower()

    if model_lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt", "text-davinci")):
        return "openai"

    # Bedrock uses dotted, vendor-namespaced IDs. Match them before the bare
    # `mistral`/`deepseek` prefixes below (which would otherwise swallow
    # `mistral.`/`deepseek.` IDs) and before the fall-through `None`, so a
    # `:version` suffix is never misparsed as a `provider:model` separator.
    if _is_bedrock_model_id(model_lower):
        return "bedrock"

    if model_lower.startswith("command"):
        return "cohere"

    if model_lower.startswith(("mistral", "mixtral")):
        return "mistralai"

    if model_lower.startswith("deepseek"):
        return "deepseek"

    if model_lower.startswith("grok"):
        return "xai"

    if model_lower.startswith("sonar"):
        return "perplexity"

    if model_lower.startswith("claude"):
        credentials = _detection_credentials()
        if not credentials.has_anthropic and credentials.has_vertex_ai:
            return "google_anthropic_vertex"
        return "anthropic"

    if model_lower.startswith("gemini"):
        credentials = _detection_credentials()
        if credentials.has_vertex_ai and not credentials.has_google:
            return "google_vertexai"
        return "google_genai"

    if model_lower.startswith(("nemotron", "nvidia/")):
        return "nvidia"

    # Fireworks uses fully-qualified IDs like `accounts/fireworks/models/<name>`.
    # `init_chat_model` can infer the provider from this prefix, but the inferred
    # name is not exposed on the returned model, so resolving it here keeps the
    # provider visible to every downstream consumer of `detect_provider` (e.g.
    # the `/model` confirmation, the status bar, and the early credential check)
    # instead of leaving the raw ID unprefixed.
    if model_lower.startswith(FIREWORKS_PROVIDER_ID_PREFIX):
        return "fireworks"

    return None


def _expand_allowed_entry(entry: str) -> list[str]:
    """Resolve one `models.allowed` entry to the model specs it admits.

    An exact `provider:model` entry yields itself. A `provider:*` wildcard
    yields the provider's discovered model lineup (registry discovery merged
    with the config's explicit list), read before allowlist filtering:
    `get_available_models` applies this policy itself, so calling it here
    would recurse.

    Args:
        entry: One entry from `ModelConfig.allowed_models`.

    Returns:
        Exact specs the entry contributes as default candidates, empty when a
        wildcarded provider has no discovered or configured models.
    """
    if not entry.endswith(":*"):
        return [entry]
    provider = entry[:-2]
    from deepagents_code.model_config import get_discovered_models

    return [f"{provider}:{model}" for model in get_discovered_models(provider)]


def _get_default_model_spec() -> str:
    """Get default model specification based on available credentials.

    Checks in order:

    1. `[models].default` in config file (user's intentional preference).
    2. `[models].recent` in config file (last `/model` switch).
    3. When `models.allowed` is active, the first entry in it whose provider
       does not have a definitively missing credential. A `provider:*` entry
       expands to the provider's discovered models rather than being a
       selectable candidate itself. Steps 1 and 2 are skipped with a warning
       when the stored value is outside the policy, and step 4 is never
       reached -- a policy declares the whole candidate set.
    4. Auto-detection based on available API credentials.

    Returns:
        Model specification in `provider:model` format.

    Raises:
        NoCredentialsConfiguredError: If no credentials are configured for any
            of the auto-detectable providers. Callers may catch this to defer
            startup and prompt for credentials interactively.
        NoAllowedModelCredentialsError: If `models.allowed` is active and no
            model in it has usable credentials. A `NoCredentialsConfiguredError`
            subclass, but callers should report it rather than silently retry:
            only a credential for an allowlisted provider can resolve it.
        ModelNotAllowedError: If `models.allowed` is active but empty, so no
            model can be resolved. Unlike the error above this is **not**
            recoverable by adding credentials, so the deferred-start path must
            not treat it as a prompt-for-credentials signal.
    """  # noqa: DOC502 - `ModelNotAllowedError` propagates from `ModelConfig.policy_error`
    from deepagents_code.model_config import (
        ModelConfig,
        ModelSpec,
        NoAllowedModelCredentialsError,
        NoCredentialsConfiguredError,
        ProviderAuthState,
        get_provider_auth_status,
    )

    config = ModelConfig.load()
    for label, candidate in (
        ("default", config.default_model),
        ("recent", config.recent_model),
    ):
        if candidate and config.is_model_allowed(candidate):
            return candidate
        if candidate:
            logger.warning(
                "Ignoring [models].%s=%r because it is outside models.allowed",
                label,
                candidate,
            )

    if config.allowed_models is not None:
        if not config.allowed_models:
            # No spec to name -- the user asked for nothing in particular, so
            # `policy_error(None)` reports the empty policy rather than
            # inventing a placeholder spec the user never typed.
            deny_all = config.policy_error(None)
            if deny_all is not None:
                raise deny_all
        # A `provider:*` wildcard cannot be selected itself -- no model is
        # named -- but every configured model it admits is a candidate in the
        # provider's declaration order.
        candidates = [
            spec
            for entry in config.allowed_models
            for spec in _expand_allowed_entry(entry)
        ]
        for candidate in candidates:
            parsed = ModelSpec.parse(candidate)
            auth = get_provider_auth_status(parsed.provider)
            # Only a definitively missing credential disqualifies a candidate.
            # UNKNOWN covers remote no-auth providers (e.g., a LAN/hosted
            # Ollama endpoint) that may not require auth at all; rejecting
            # them here would block startup even though create_model()
            # deliberately permits that state.
            if auth.state is not ProviderAuthState.MISSING:
                return candidate
        if not candidates:
            # Every entry is a wildcard for a provider with no discoverable
            # models, so there is nothing to credential.
            allowed = ", ".join(config.allowed_models)
            msg = (
                "No discoverable models match models.allowed "
                f"({allowed}). Name an exact provider:model spec or configure "
                "models for a wildcarded provider."
            )
            raise NoAllowedModelCredentialsError(msg)
        allowed = ", ".join(candidates)
        msg = (
            "No credentials are configured for any model in models.allowed. "
            f"Add credentials for one of: {allowed}."
        )
        raise NoAllowedModelCredentialsError(msg)

    # `is True` deliberately excludes `ProviderAuthState.UNKNOWN` (which maps
    # to `as_legacy_bool() -> None`). For the three explicit-credential
    # providers below, an UNKNOWN result means we cannot prove auth works, so
    # we fall through rather than pick an unverifiable default. If an
    # implicit-auth provider (e.g., Vertex ADC) is added to this fallback
    # list, switch to checking `state` against the relevant
    # `ProviderAuthState` members directly.
    if get_provider_auth_status("openai").as_legacy_bool() is True:
        return "openai:gpt-5.6-terra"
    if get_provider_auth_status("anthropic").as_legacy_bool() is True:
        return "anthropic:claude-opus-5"
    if get_provider_auth_status("google_genai").as_legacy_bool() is True:
        return "google_genai:gemini-3.1-pro-preview"

    msg = (
        "No credentials configured. Please set one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY"
    )
    raise NoCredentialsConfiguredError(msg)


_OPENROUTER_APP_URL = "https://pypi.org/project/deepagents-code/"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents Code"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""

_OPENROUTER_APP_CATEGORIES: list[str] = ["cli-agent"]
"""Default `app_categories` (maps to `X-OpenRouter-Categories`) for OpenRouter."""

_cli_openrouter_profile_registered = False
"""Process-wide guard so the app's OpenRouter profile is registered exactly once."""

AWS_CREDENTIAL_ENV_SOURCES: dict[str, tuple[str, ...]] = {
    "profile_name": ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"),
    "aws_access_key_id": ("AWS_ACCESS_KEY_ID",),
    "aws_secret_access_key": ("AWS_SECRET_ACCESS_KEY",),
    "aws_session_token": ("AWS_SESSION_TOKEN",),
}
"""Canonical AWS credential sources, keyed by the `boto3.Session` argument.

Shared with `integrations.sandbox_factory` so a new AWS credential source is
added in one place rather than drifting between the model and sandbox paths.
"""

AWS_REGION_ENV_SOURCES = ("AWS_REGION", "AWS_DEFAULT_REGION")
"""Canonical AWS region sources, in resolution order.

Shared with `integrations.sandbox_factory` for the same reason as
`AWS_CREDENTIAL_ENV_SOURCES`: one place to add a source.
"""

_AWS_MODEL_SDK_ENV_KWARGS = {
    "region_name": AWS_REGION_ENV_SOURCES,
    # LangChain constructors spell the profile argument differently to boto3.
    "credentials_profile_name": AWS_CREDENTIAL_ENV_SOURCES["profile_name"],
    **{
        argument: names
        for argument, names in AWS_CREDENTIAL_ENV_SOURCES.items()
        if argument != "profile_name"
    },
}
"""AWS model environment values accepted directly by LangChain constructors."""


def resolve_env_kwargs(
    table: Mapping[str, tuple[str, ...]],
    lookup: Callable[[str], str | None],
) -> dict[str, str]:
    """Translate an argument/env-name table into explicit constructor kwargs.

    The first name that `lookup` resolves to a non-empty value wins.

    Args:
        table: Constructor argument name mapped to candidate env var names.
        lookup: Resolver for a single env var name.

    Returns:
        Populated keyword arguments, omitting arguments with no value.
    """
    resolved: dict[str, str] = {}
    for argument, env_names in table.items():
        value = next((value for name in env_names if (value := lookup(name))), None)
        if value:
            resolved[argument] = value
    return resolved


_PROVIDER_SDK_ENV_KWARGS: dict[str, dict[str, tuple[str, ...]]] = {
    "anthropic_bedrock": _AWS_MODEL_SDK_ENV_KWARGS,
    "azure_openai": {
        "api_version": ("OPENAI_API_VERSION",),
        "azure_ad_token": ("AZURE_OPENAI_AD_TOKEN",),
    },
    "bedrock": _AWS_MODEL_SDK_ENV_KWARGS,
    "bedrock_converse": _AWS_MODEL_SDK_ENV_KWARGS,
}
"""SDK-read environment values that must be explicit in workspace runtimes."""


def _cli_openrouter_attribution_kwargs() -> dict[str, Any]:
    """App-specific OpenRouter attribution kwargs.

    Layered on top of the SDK's built-in factory via profile stacking; these
    values override the SDK defaults but still sit beneath any caller-supplied
    `kwargs` (i.e. `config.toml`-resolved values), preserving the precedence
    documented on `apply_provider_profile`.

    Returns:
        Mapping of `app_url` and `app_title` to spread into `init_chat_model`.
    """
    return {
        "app_url": _OPENROUTER_APP_URL,
        "app_title": _OPENROUTER_APP_TITLE,
    }


def _ensure_cli_openrouter_profile_registered() -> None:
    """Stack the app's OpenRouter attribution onto the SDK's built-in profile.

    Stacking (vs. duplicating the inline `_get_provider_kwargs` path) means the
    SDK's `pre_init` version check fires exactly once and the app's app-
    attribution defaults are composed via the same `apply_provider_profile`
    path used for every other provider. `register_provider_profile` merges on
    top of the existing built-in registration: the app's `init_kwargs` and
    factory output win on shared keys, while the built-in's `pre_init` and
    factory still chain.
    """
    global _cli_openrouter_profile_registered  # noqa: PLW0603
    if _cli_openrouter_profile_registered:
        return

    from deepagents.profiles.provider import ProviderProfile, register_provider_profile

    register_provider_profile(
        "openrouter",
        ProviderProfile(
            init_kwargs={"app_categories": _OPENROUTER_APP_CATEGORIES},
            init_kwargs_factory=_cli_openrouter_attribution_kwargs,
        ),
    )
    _cli_openrouter_profile_registered = True


def _apply_azure_sdk_endpoint(kwargs: dict[str, Any]) -> None:
    """Forward the Azure endpoint under the constructor's current field name.

    Reads the endpoint through `get_base_url_env_vars` rather than a literal
    name so the user's `base_url_env` override is honored here too.
    """
    from deepagents_code.model_config import get_base_url_env_vars, resolve_env_var

    endpoint = next(
        (
            value
            for name in get_base_url_env_vars("azure_openai")
            if (value := resolve_env_var(name))
        ),
        None,
    )
    if endpoint and kwargs.get("base_url") == endpoint:
        kwargs.pop("base_url")
    if endpoint and "base_url" not in kwargs:
        kwargs.setdefault("azure_endpoint", endpoint)


def _apply_provider_sdk_environment(
    provider: str,
    kwargs: dict[str, Any],
) -> None:
    """Apply SDK-only environment defaults without replacing explicit kwargs."""
    from deepagents_code.model_config import resolve_env_var

    if provider == "azure_openai":
        _apply_azure_sdk_endpoint(kwargs)

    table = {
        argument: env_names
        for argument, env_names in _PROVIDER_SDK_ENV_KWARGS.get(provider, {}).items()
        if argument not in kwargs
    }
    kwargs.update(resolve_env_kwargs(table, resolve_env_var))


def _get_provider_kwargs(
    provider: str, *, model_name: str | None = None
) -> dict[str, Any]:
    """Get provider-specific kwargs from the config file.

    Reads `base_url`, `api_key_env`, and the `params` table from the user's
    `config.toml` for the given provider.

    When `model_name` is provided, per-model overrides from the `params`
    sub-table are shallow-merged on top.

    Args:
        provider: Provider name (e.g., openai, anthropic, fireworks, ollama).
        model_name: Optional model name for per-model overrides.

    Returns:
        Dictionary of provider-specific kwargs.
    """
    from deepagents_code.model_config import ModelConfig

    config = ModelConfig.load()
    result = config.get_effective_kwargs(provider, model_name=model_name)
    from deepagents_code.model_config import (
        OPTIONAL_AUTH_ENV,
        PROVIDER_API_KEY_ENV,
        resolve_env_var,
    )

    api_key_env = config.get_api_key_env(provider)
    if not api_key_env:
        api_key_env = PROVIDER_API_KEY_ENV.get(provider)
        if api_key_env:
            logger.debug(
                "No api_key_env in config.toml for '%s';"
                " using hardcoded provider env var",
                provider,
            )
    if api_key_env:
        api_key = resolve_env_var(api_key_env)
        if api_key and provider != "google_anthropic_vertex":
            result["api_key"] = api_key

    # `langchain-ollama` has no `api_key` kwarg; hosted Ollama (Cloud or
    # gateway) needs the bearer token threaded through `client_kwargs.headers`.
    if provider == "ollama":
        optional_env = OPTIONAL_AUTH_ENV.get(provider)
        optional_key = resolve_env_var(optional_env) if optional_env else None
        if optional_key:
            client_kwargs = result.get("client_kwargs")
            if client_kwargs is not None and not isinstance(client_kwargs, dict):
                logger.warning(
                    "Provider 'ollama' has non-mapping client_kwargs (%s);"
                    " skipping Authorization header injection",
                    type(client_kwargs).__name__,
                )
            else:
                client_kwargs = dict(client_kwargs) if client_kwargs else {}
                headers = client_kwargs.get("headers")
                if headers is not None and not isinstance(headers, dict):
                    logger.warning(
                        "Provider 'ollama' has non-mapping client_kwargs.headers"
                        " (%s); skipping Authorization header injection",
                        type(headers).__name__,
                    )
                else:
                    headers = dict(headers) if headers else {}
                    has_auth_header = any(
                        isinstance(k, str) and k.lower() == "authorization"
                        for k in headers
                    )
                    if not has_auth_header:
                        headers["Authorization"] = f"Bearer {optional_key}"
                        client_kwargs["headers"] = headers
                        result["client_kwargs"] = client_kwargs

    _apply_provider_sdk_environment(provider, result)
    return result


def _apply_scoped_stored_endpoint(provider: str, kwargs: dict[str, Any]) -> None:
    """Pair stored credentials with their endpoint without mutating the process."""
    from deepagents_code.model_config import (
        PROVIDER_CUSTOM_HEADERS_ENV,
        ModelConfig,
        _configured_base_url_survives_env_clear,
        auth_store,
    )

    try:
        stored_key = auth_store.get_stored_key(provider)
        stored_base_url = auth_store.get_stored_base_url(provider)
    except RuntimeError:
        # On the scoped path this function is the only thing enforcing the
        # key/endpoint pairing, because `apply_stored_credentials` is skipped.
        # `resolve_provider_credential` still falls back to the environment key,
        # so fail closed: drop an inherited gateway endpoint and its auth
        # headers rather than shipping that key to it.
        logger.warning(
            "Could not read the stored credential for %r; the credential file "
            "may be corrupt. Clearing any inherited endpoint so the resolved "
            "key is not sent to it. Re-add the key via /auth.",
            provider,
        )
        stored_key = None
        stored_base_url = None
    else:
        if not stored_key:
            return
    provider_config = ModelConfig.load().providers.get(provider)
    configured_url = provider_config.get("base_url") if provider_config else None
    if configured_url or _configured_base_url_survives_env_clear(provider):
        return
    if stored_base_url:
        kwargs["base_url"] = stored_base_url
        return
    kwargs.pop("base_url", None)
    if provider == "anthropic":
        kwargs["base_url"] = "https://api.anthropic.com"
    custom_headers = PROVIDER_CUSTOM_HEADERS_ENV.get(provider)
    if custom_headers:
        kwargs["default_headers"] = {}


def _apply_google_anthropic_vertex_kwargs(
    provider: str, kwargs: dict[str, Any]
) -> None:
    """Apply required Claude-on-Vertex project and location defaults.

    Raises:
        ModelConfigError: If no location is configured.
    """
    if provider != "google_anthropic_vertex":
        return
    from deepagents_code.model_config import resolve_env_var

    project = resolve_env_var("GOOGLE_CLOUD_PROJECT")
    location = resolve_env_var("GOOGLE_CLOUD_LOCATION")
    if _environment_is_scoped():
        from deepagents_code.model_config import auth_store

        try:
            if not project:
                project = auth_store.get_stored_key(provider)
        except RuntimeError:
            # `google_anthropic_vertex` stores the GCP project id in the key
            # slot. It is also an implicit-auth provider, so the early
            # credential check never fires — without this log a corrupt store
            # surfaces only as an opaque ADC project-inference error.
            logger.warning(
                "Could not read the stored Google Cloud project for %r; the "
                "credential file may be corrupt. Falling back to ADC project "
                "inference. Re-add the project via /auth.",
                provider,
            )
    if project:
        kwargs.setdefault("project", project)
    if location:
        kwargs.setdefault("location", location)
    if not kwargs.get("location"):
        from deepagents_code.model_config import ModelConfigError

        msg = (
            "Google Cloud location is required for provider "
            "'google_anthropic_vertex'. Set GOOGLE_CLOUD_LOCATION or "
            "DEEPAGENTS_CODE_GOOGLE_CLOUD_LOCATION, or pass 'location' in model params."
        )
        raise ModelConfigError(msg)


def _compose_openai_reasoning_effort(
    provider: str,
    kwargs: dict[str, Any],
    effort_override: object,
    reasoning_override: object,
) -> dict[str, Any]:
    """Compose a session effort override with an OpenAI reasoning mapping.

    Args:
        provider: Resolved model provider.
        kwargs: Layered model constructor parameters.
        effort_override: High-priority `reasoning_effort` from session params.
        reasoning_override: High-priority native `reasoning` from session params.

    Returns:
        Constructor parameters with one native `reasoning` mapping when
        composition is needed.
    """
    if provider not in {"openai", "openai_codex"} or not isinstance(
        effort_override, str
    ):
        return kwargs
    reasoning = kwargs.get("reasoning")
    if not isinstance(reasoning, dict):
        return kwargs
    composed = dict(kwargs)
    if isinstance(reasoning_override, dict) and "effort" in reasoning_override:
        composed.pop("reasoning_effort", None)
        return composed
    composed["reasoning"] = {**reasoning, "effort": effort_override}
    composed.pop("reasoning_effort", None)
    return composed


def _create_model_from_class(
    class_path: str,
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """Import and instantiate a custom `BaseChatModel` class.

    Args:
        class_path: Fully-qualified class in `module.path:ClassName` format.
        model_name: Model identifier to pass as `model` kwarg.
        provider: Provider name (for error messages).
        kwargs: Additional keyword arguments for the constructor.

    Returns:
        Instantiated `BaseChatModel`.

    Raises:
        ModelConfigError: If the class cannot be imported, is not a
            `BaseChatModel` subclass, or fails to instantiate.
    """
    from langchain_core.language_models import (
        BaseChatModel as _BaseChatModel,  # Runtime import; module level is typing only
    )

    from deepagents_code.model_config import ModelConfigError

    if ":" not in class_path:
        msg = (
            f"Invalid class_path '{class_path}' for provider '{provider}': "
            "must be in module.path:ClassName format"
        )
        raise ModelConfigError(msg)

    module_path, class_name = class_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Could not import module '{module_path}' for provider '{provider}': {e}"
        raise ModelConfigError(msg) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        msg = (
            f"Class '{class_name}' not found in module '{module_path}' "
            f"for provider '{provider}'"
        )
        raise ModelConfigError(msg)

    if not (isinstance(cls, type) and issubclass(cls, _BaseChatModel)):
        msg = (
            f"'{class_path}' is not a BaseChatModel subclass (got {type(cls).__name__})"
        )
        raise ModelConfigError(msg)

    try:
        return cls(model=model_name, **kwargs)
    except Exception as e:
        msg = f"Failed to instantiate '{class_path}' for '{provider}:{model_name}': {e}"
        raise ModelConfigError(msg) from e


def _create_model_via_init(
    model_name: str,
    provider: str,
    kwargs: dict[str, Any],
) -> BaseChatModel:
    """Create a model using langchain's `init_chat_model`.

    Args:
        model_name: Model identifier.
        provider: Provider name (may be empty for auto-detection).
        kwargs: Additional keyword arguments.

    Returns:
        Instantiated `BaseChatModel`.

    Raises:
        UnknownProviderError: When `provider` is empty and
            `init_chat_model` also fails to infer one. Carries the
            model spec and docs URL as attributes so the UI can render
            a clickable link.
        MissingProviderPackageError: When the provider's LangChain package
            is not installed. Carries the `provider` and `package` to install
            so the UI can render a targeted recovery hint.
        ModelConfigError: On other import, value, or runtime errors.
    """
    from langchain.chat_models import init_chat_model

    from deepagents_code.model_config import (
        MissingProviderPackageError,
        ModelConfigError,
        UnknownProviderError,
    )

    try:
        if provider:
            return init_chat_model(model_name, model_provider=provider, **kwargs)
        return init_chat_model(model_name, **kwargs)
    except ImportError as e:
        import importlib.util

        package_map = {
            "anthropic": "langchain-anthropic",
            "openai": "langchain-openai",
            "google_anthropic_vertex": "langchain-google-vertexai",
            "google_genai": "langchain-google-genai",
            "google_vertexai": "langchain-google-vertexai",
            "nvidia": "langchain-nvidia-ai-endpoints",
        }
        package = package_map.get(provider, f"langchain-{provider}")
        # Convert pip package name to Python module name for import check.
        module_name = package.replace("-", "_")
        try:
            spec_found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError) as spec_exc:
            # A broken finder is indistinguishable from "not installed" here;
            # log so a real corruption doesn't masquerade as the missing-package
            # hint without leaving a trail.
            logger.debug(
                "find_spec failed for %s; treating provider package as missing: %s",
                module_name,
                spec_exc,
            )
            spec_found = False
        if spec_found:
            # Package is installed but an internal import failed — surface
            # the real error instead of the misleading "missing package" hint.
            msg = (
                f"Provider package '{package}' is installed but failed to "
                f"import for provider '{provider}': {e}"
            )
        else:
            from deepagents_code.extras_info import resolve_install_hint

            hint = resolve_install_hint(package)
            if hint.extra is not None:
                msg = (
                    f"Missing package for provider '{provider}'. "
                    f"Install: /install {hint.extra}"
                )
            else:
                if hint.command is not None:
                    install_hint = f"Install with: {hint.command}"
                else:
                    install_hint = f"Install the '{package}' package manually"
                msg = (
                    f"Missing package for provider '{provider}'. "
                    f"{install_hint}, then retry with `/model`."
                )
            raise MissingProviderPackageError(
                msg, provider=provider, package=package
            ) from e
        raise ModelConfigError(msg) from e
    except (ValueError, TypeError) as e:
        if not provider:
            # Both app auto-detection and `init_chat_model`'s own inference
            # failed; surface a structured error so the UI can render the
            # docs URL as a clickable link.
            raise UnknownProviderError(model_spec=model_name) from e
        spec = f"{provider}:{model_name}"
        msg = f"Invalid model configuration for '{spec}': {e}"
        raise ModelConfigError(msg) from e
    except Exception as e:  # provider SDK auth/network errors
        spec = f"{provider}:{model_name}" if provider else model_name
        msg = f"Failed to initialize model '{spec}': {e}"
        raise ModelConfigError(msg) from e


@dataclass(frozen=True)
class ModelResult:
    """Result of creating a chat model, bundling the model with its metadata.

    This separates model creation from runtime-state mutation so callers can
    decide when to commit the metadata to process-wide state.

    Attributes:
        model: The instantiated chat model.
        model_name: Resolved model name.
        provider: Resolved provider name.
        context_limit: Max input tokens from the model profile, or `None`.
        unsupported_modalities: Input modalities not indicated as supported by
            the model profile (e.g. `{"audio", "video"}`).
        model_retries: Effective model-node retry count for the resolved
            provider (see `_resolve_model_retries_from_section`). `0` disables
            retries.
        cli_max_retries: The `--max-retries` flag value, or `None` when the user
            did not set it. Kept distinct from `model_retries` so a model built
            for a different provider can resolve its own configured budget
            instead of inheriting this one.
    """

    model: BaseChatModel
    model_name: str
    provider: str
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()
    model_retries: int = DEFAULT_MODEL_RETRIES
    cli_max_retries: int | None = None

    def __post_init__(self) -> None:
        """Enforce the middleware's non-negative retry-budget invariant.

        Non-negativity is enforced upstream by `non_negative_int` on
        `--max-retries` and by `_coerce_max_retries` for config values, not by
        the retry resolver itself, so a bad value here signals a caller
        constructing `ModelResult` by hand with a budget the retry middleware
        could not honor. `bool` is rejected for the same reason
        `_model_max_retries` and `CodeModelRetryMiddleware.__init__` reject it:
        `True` would silently read as a budget of one.

        `cli_max_retries` gets the same gate. It is the field that carries the
        explicit flag onward to a re-resolution for a different provider, and
        it was the one budget field in dcode without the check -- which is the
        argument for a single validated budget type rather than a ninth copy of
        this predicate.

        Raises:
            TypeError: If `model_retries` or `cli_max_retries` is a `bool`.
            ValueError: If `model_retries` or `cli_max_retries` is negative.
        """
        if isinstance(self.model_retries, bool):
            msg = f"model_retries must be an int, got {self.model_retries!r}"
            raise TypeError(msg)
        if self.model_retries < 0:
            msg = f"model_retries must be >= 0, got {self.model_retries}"
            raise ValueError(msg)
        if isinstance(self.cli_max_retries, bool):
            msg = (
                f"cli_max_retries must be None or an int, got {self.cli_max_retries!r}"
            )
            raise TypeError(msg)
        if self.cli_max_retries is not None and self.cli_max_retries < 0:
            msg = f"cli_max_retries must be >= 0, got {self.cli_max_retries}"
            raise ValueError(msg)

    def apply_to_runtime_state(self) -> None:
        """Commit this result's metadata to global `runtime_state`."""
        state = _get_runtime_state()
        state.model_name = self.model_name
        state.model_provider = self.provider
        state.model_context_limit = self.context_limit
        state.model_unsupported_modalities = self.unsupported_modalities


def _apply_profile_overrides(
    model: BaseChatModel,
    overrides: dict[str, Any],
    model_name: str,
    *,
    label: str,
    raise_on_failure: bool = False,
) -> None:
    """Merge `overrides` into `model.profile`.

    If the model already has a dict profile, overrides are layered on top
    so existing keys (e.g., `tool_calling`) are preserved unchanged.

    Args:
        model: The chat model whose profile will be updated.
        overrides: Key/value pairs to merge into the profile.
        model_name: Model name used in log/error messages.
        label: Human-readable source label for messages
            (e.g., `"config.toml"`, `"CLI --profile-override"`).
        raise_on_failure: When `True`, raise `ModelConfigError` instead
            of logging a warning if assignment fails.

    Raises:
        ModelConfigError: If `raise_on_failure` is `True` and the model
            rejects profile assignment.
    """
    from deepagents_code.model_config import ModelConfigError

    logger.debug("Applying %s profile overrides: %s", label, overrides)
    profile = getattr(model, "profile", None)
    merged = {**profile, **overrides} if isinstance(profile, dict) else overrides
    try:
        model.profile = merged  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError, ValueError) as exc:
        if raise_on_failure:
            msg = (
                f"Could not apply {label} to model '{model_name}': {exc}. "
                f"The model may not support profile assignment."
            )
            raise ModelConfigError(msg) from exc
        logger.warning(
            "Could not apply %s profile overrides to model '%s': %s. "
            "Overrides will be ignored.",
            label,
            model_name,
            exc,
        )


def create_model(
    model_spec: str | None = None,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    profile_overrides: dict[str, Any] | None = None,
    cli_max_retries: int | None = None,
) -> ModelResult:
    """Create a chat model.

    Uses `init_chat_model` for standard providers, or imports a custom
    `BaseChatModel` subclass when the provider has a `class_path` in config.

    Supports `provider:model` format (e.g., `'openai:gpt-5.5'`)
    for explicit provider selection, or bare model names for auto-detection.

    Args:
        model_spec: Model specification in `provider:model` format (e.g.,
            `'anthropic:claude-sonnet-4-5'`, `'openai:gpt-5.5'`) or just the model
            name for auto-detection (e.g., `'claude-sonnet-4-5'`).

                If not provided, uses environment-based defaults.
        extra_kwargs: Additional kwargs to pass to the model constructor.

            These take highest priority, overriding values from the config file,
            except that provider-owned retry loops are disabled when known.

            The provider's own retry-count kwarg (`RETRY_PARAM_BY_PROVIDER`,
            usually `max_retries`) is forced off after this merge whenever dcode
            can identify it, because the model-node middleware owns the retry
            budget and nested SDK retries would multiply its attempts. Supplying
            that kwarg here logs a warning; use `--max-retries` or `[retries]`
            instead. A provider dcode cannot identify (absent from the registry,
            no `[retries.<provider>].param`, and no `max_retries` already in the
            kwargs) keeps its own retry loop and logs a warning saying so.
        profile_overrides: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides (dcode wins).
        cli_max_retries: Explicit `--max-retries` value. When absent, the
            provider-specific or global config value applies.

    Returns:
        A `ModelResult` containing the model and its metadata.

    Raises:
        ModelConfigError: If provider cannot be determined from the model name
            or required provider package is not installed.
        ModelNotAllowedError: If the resolved spec is outside `models.allowed`.
            A `ModelConfigError` subclass, so a bare `except ModelConfigError`
            swallows a policy denial -- handlers that fall back to another
            model must re-raise it (see `configurable_model._apply_overrides`).
        MissingCredentialsError: If no credentials are configured for the
            resolved provider.
        TypeError: If `cli_max_retries` is not an integer.
        ValueError: If `cli_max_retries` is negative.

    Examples:
        >>> model = create_model("anthropic:claude-sonnet-4-5")
        >>> model = create_model("openai:gpt-5.5")
        >>> model = create_model("gpt-5.5")  # Auto-detects openai
        >>> model = create_model()  # Uses environment defaults
    """  # noqa: DOC502 - `ModelNotAllowedError` propagates from `require_model_allowed`
    from deepagents_code.model_config import (
        IMPLICIT_AUTH_PROVIDERS,
        ModelConfig,
        ModelConfigError,
        ModelSpec,
        apply_stored_credentials,
        get_credential_env_var,
        has_provider_credentials,
        resolve_provider_credential,
        warn_on_split_credential_source,
    )

    if not model_spec:
        model_spec = _get_default_model_spec()

    # Parse provider:model syntax. Bedrock model IDs can include a version suffix
    # such as `:0`, so resolve their distinctive bare-ID prefixes unless the
    # parsed provider is explicitly configured.
    provider: str
    model_name: str
    config = ModelConfig.load()
    inferred_provider = detect_provider(model_spec)
    parsed = ModelSpec.try_parse(model_spec)
    if parsed and parsed.provider in config.providers:
        provider, model_name = parsed.provider, parsed.model
    elif inferred_provider == "bedrock":
        provider, model_name = inferred_provider, model_spec
    elif parsed:
        # Explicit provider:model (e.g., "anthropic:claude-sonnet-4-5")
        provider, model_name = parsed.provider, parsed.model
    elif ":" in model_spec:
        # Contains colon but ModelSpec rejected it (empty provider or model)
        _, _, after = model_spec.partition(":")
        if after:
            # Leading colon (e.g., ":claude-opus-4-6") — treat as bare model name
            model_name = after
            provider = detect_provider(model_name) or ""
        else:
            msg = (
                f"Invalid model spec '{model_spec}': model name is required "
                "(e.g., 'anthropic:claude-sonnet-4-5' or 'claude-sonnet-4-5')"
            )
            raise ModelConfigError(msg)
    else:
        # Bare model name — auto-detect provider or let init_chat_model infer
        model_name = model_spec
        provider = inferred_provider or ""

    if provider == "google_vertexai" and model_name.lower().startswith("claude-"):
        msg = (
            f"Claude model '{model_name}' uses the Anthropic Messages API on "
            "Vertex AI. Use "
            f"'google_anthropic_vertex:{model_name}' instead of "
            f"'google_vertexai:{model_name}'."
        )
        raise ModelConfigError(msg)

    resolved_spec = f"{provider}:{model_name}" if provider else model_spec
    # The authoritative policy gate, and its position is load-bearing: it runs
    # after provider inference (so a bare name is matched in canonical form)
    # but before credential bridging, provider profiles, and provider imports.
    # A blocked spec must not copy stored keys onto env vars or run a provider
    # `pre_init` hook. `test_rejects_before_credential_side_effects` pins this.
    config.require_model_allowed(resolved_spec)

    # Stored API keys (added via `/auth`) take effect by being copied onto
    # the env var name LangChain reads. Apply before the credential check so
    # `has_provider_credentials` and the downstream SDK see the same value.
    #
    # Bound unconditionally: both names are read again further down, where a
    # binding that existed only on the `if provider:` branch would be a latent
    # `UnboundLocalError`.
    scoped_environment = _environment_is_scoped()
    stored_credential: str | None = None
    if provider:
        # Flag a key/endpoint resolved from different env tiers *before*
        # `apply_stored_credentials` bridges stored values onto plain env vars,
        # so the check sees the user's raw env intent rather than post-bridge
        # state. Diagnostic only -- never alters resolution.
        warn_on_split_credential_source(provider)
        if not scoped_environment:
            apply_stored_credentials(provider)
        stored_credential = resolve_provider_credential(provider)

    from deepagents_code.model_config import CODEX_PROVIDER

    # Early credential check — fail fast with an actionable message instead of
    # letting the provider SDK raise an opaque auth error on first invocation.
    # Providers that support implicit auth (e.g., Vertex AI ADC) are excluded
    # because their env-var mapping is not a reliable indicator.
    if provider and provider not in IMPLICIT_AUTH_PROVIDERS:
        cred_status = has_provider_credentials(provider)
        if cred_status is False:
            from deepagents_code.model_config import MissingCredentialsError

            if provider == CODEX_PROVIDER:
                # No env var to set; point the user at `/auth` instead.
                msg = (
                    "Not signed in to ChatGPT. Run `/auth` and select "
                    "openai_codex to sign in with your ChatGPT account."
                )
                raise MissingCredentialsError(msg, provider=provider, env_var=None)
            env_var = get_credential_env_var(provider)
            display_env = env_var or f"<{provider} API key>"
            msg = (
                f"No credentials found for provider '{provider}'. "
                f"Please set the {display_env} environment variable."
            )
            raise MissingCredentialsError(msg, provider=provider, env_var=env_var)

    # Provider-specific kwargs (with per-model overrides)
    kwargs = _get_provider_kwargs(provider, model_name=model_name)
    if provider and stored_credential and provider != "google_anthropic_vertex":
        kwargs["api_key"] = stored_credential

    # Compose under existing kwargs: profile < config.toml < --model-params
    # (applied below). The app's OpenRouter profile is stacked on top of the
    # built-in SDK profile so its `pre_init` (version check) and factory
    # (app attribution) compose into a single `apply_provider_profile` call.
    if provider:
        from deepagents.profiles.provider import apply_provider_profile

        if provider == "openrouter":
            _ensure_cli_openrouter_profile_registered()

        spec = f"{provider}:{model_name}" if model_name else provider
        try:
            kwargs = apply_provider_profile(spec, kwargs)
        except ModelConfigError:
            raise
        except Exception as exc:
            # `pre_init` and `init_kwargs_factory` callables registered on a
            # `ProviderProfile` may raise arbitrary exceptions (e.g. an
            # `ImportError` from the OpenRouter min-version check). Surface
            # them as `ModelConfigError` so the app's error path renders an
            # actionable message instead of a raw stack trace.
            logger.debug(
                "ProviderProfile resolution for %r failed.", spec, exc_info=True
            )
            msg = (
                f"Failed to apply provider profile for '{spec}': {exc}. "
                f"Check that the provider package is installed and up to date, "
                f"or set explicit kwargs via `--model-params`."
            )
            raise ModelConfigError(msg) from exc

    # App --model-params take highest priority.
    reasoning_effort_override: object = None
    reasoning_override: object = None
    if extra_kwargs:
        extra_kwargs = dict(extra_kwargs)
        reasoning_effort_override = extra_kwargs.get("reasoning_effort")
        reasoning_override = extra_kwargs.get("reasoning")
        kwargs.update(extra_kwargs)
    if provider and scoped_environment:
        if extra_kwargs and "api_key" in extra_kwargs:
            if "base_url" not in extra_kwargs:
                from deepagents_code.model_config import auth_store

                try:
                    stored_base_url = auth_store.get_stored_base_url(provider)
                except RuntimeError:
                    # The caller passed an explicit `api_key` with no
                    # `base_url`. Without the store we cannot tell whether the
                    # inherited endpoint was paired with the *stored* key, so
                    # fail closed and drop it: sending an explicitly supplied
                    # key to a gateway it was not issued for is the worse
                    # outcome.
                    logger.warning(
                        "Could not read the stored endpoint for %r; the "
                        "credential file may be corrupt. Dropping the "
                        "inherited base URL so the explicitly supplied key is "
                        "not sent to it. Pass `base_url` via `--model-params` "
                        "to target an endpoint explicitly.",
                        provider,
                    )
                    kwargs.pop("base_url", None)
                else:
                    if stored_base_url and kwargs.get("base_url") == stored_base_url:
                        kwargs.pop("base_url", None)
        else:
            _apply_scoped_stored_endpoint(provider, kwargs)
            if extra_kwargs and "base_url" in extra_kwargs:
                kwargs["base_url"] = extra_kwargs["base_url"]
    kwargs = _compose_openai_reasoning_effort(
        provider,
        kwargs,
        reasoning_effort_override,
        reasoning_override,
    )

    # dcode's model-node middleware owns the user-visible retry budget. Resolve
    # that budget separately, then force the provider's own retry loop off so
    # nested SDK retries cannot multiply the configured attempt count.
    if cli_max_retries is not None and (
        not isinstance(cli_max_retries, int) or isinstance(cli_max_retries, bool)
    ):
        msg = "cli_max_retries must be None or an int >= 0"
        raise TypeError(msg)
    if cli_max_retries is not None and cli_max_retries < 0:
        msg = "cli_max_retries must be None or an int >= 0"
        raise ValueError(msg)
    retry_config = _read_retry_config()
    for warning in retry_config.warnings:
        logger.warning("%s", warning)
    model_retries = _resolve_model_retries_from_section(
        retry_config,
        provider,
        cli_max_retries,
    )
    kwargs.update(_provider_retry_disable_kwargs(retry_config, provider, kwargs))

    _apply_google_anthropic_vertex_kwargs(provider, kwargs)

    # Check if this provider uses a custom BaseChatModel class
    class_path = config.get_class_path(provider) if provider else None

    if provider == CODEX_PROVIDER:
        # Codex models are constructed directly via `_ChatOpenAICodex` so the
        # `token_provider=` kwarg is wired to the on-disk OAuth token store
        # before any request goes out. `init_chat_model` does not know about
        # this class and would route through API-key `ChatOpenAI` instead.
        from deepagents_code.integrations import openai_codex as _codex
        from deepagents_code.model_config import (
            MissingCredentialsError,
            ModelConfigError,
        )

        # Drop any `api_key` left in kwargs (e.g. from a config-level
        # `api_key_env` set on the codex provider, or a `--model-params
        # api_key=...`) so the bearer always comes from the OAuth
        # `token_provider` rather than a static key.
        kwargs.pop("api_key", None)
        try:
            model = _codex.build_chat_model(model_name, **kwargs)
        except FileNotFoundError as exc:
            msg = (
                "Not signed in to ChatGPT. Run `/auth` and select "
                "openai_codex to sign in with your ChatGPT account."
            )
            raise MissingCredentialsError(msg, provider=provider, env_var=None) from exc
        except _codex.CodexAuthExpiredError as exc:
            # A token exists but its refresh token is dead. Route through the
            # same `MissingCredentialsError` recovery path as a missing token
            # (which the retry flow re-attempts after `/auth`) instead of the
            # generic `ModelConfigError` below, which would not offer sign-in.
            msg = (
                "ChatGPT session expired. Run `/auth` and select openai_codex "
                "to sign in again."
            )
            raise MissingCredentialsError(msg, provider=provider, env_var=None) from exc
        except Exception as exc:
            spec = f"{provider}:{model_name}"
            msg = f"Failed to initialize Codex model '{spec}': {exc}"
            raise ModelConfigError(msg) from exc
    elif class_path:
        model = _create_model_from_class(class_path, model_name, provider, kwargs)
    else:
        model = _create_model_via_init(model_name, provider, kwargs)

    resolved_provider = provider or getattr(model, "_model_provider", provider)
    from deepagents_code.cost_tracking import _set_configured_model_metadata

    _set_configured_model_metadata(model, model_name, resolved_provider)

    # Apply profile overrides from config.toml (e.g., max_input_tokens)
    if provider:
        config_profile_overrides = config.get_profile_overrides(
            provider, model_name=model_name
        )
        if config_profile_overrides:
            _apply_profile_overrides(
                model,
                config_profile_overrides,
                model_name,
                label=f"config.toml (provider '{provider}')",
            )

    # App --profile-override takes highest priority (on top of config.toml)
    if profile_overrides:
        _apply_profile_overrides(
            model,
            profile_overrides,
            model_name,
            label="CLI --profile-override",
            raise_on_failure=True,
        )

    # Keep retry policy metadata on the concrete model selected for the request.
    # Runtime `/model` switches replace `request.model`, so the downstream retry
    # middleware can read the matching budget without mutating shared middleware
    # state or forwarding an internal key to a provider API.
    try:
        object.__setattr__(  # noqa: PLC2801  # Pydantic models reject unknown fields through normal setattr
            model, MODEL_RETRIES_ATTR, model_retries
        )
    except AttributeError:
        # A custom provider class using `__slots__` rejects the write. The
        # metadata is advisory, so the middleware falls back to its startup
        # budget rather than failing an otherwise usable model.
        logger.warning(
            "Could not attach the retry budget to %r; the model-node middleware "
            "will use its startup budget instead",
            model_name,
        )

    # Extract context limit and modality support from model profile
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        if isinstance(profile.get("max_input_tokens"), int):
            context_limit = profile["max_input_tokens"]

        modality_keys = {
            "image_inputs": "image",
            "audio_inputs": "audio",
            "video_inputs": "video",
            "pdf_inputs": "pdf",
        }
        unsupported_modalities = frozenset(
            label for key, label in modality_keys.items() if profile.get(key) is False
        )

    return ModelResult(
        model=model,
        model_name=model_name,
        provider=resolved_provider,
        context_limit=context_limit,
        unsupported_modalities=unsupported_modalities,
        model_retries=model_retries,
        cli_max_retries=cli_max_retries,
    )


def validate_model_capabilities(model: BaseChatModel, model_name: str) -> None:
    """Validate that the model has required capabilities for `deepagents`.

    Checks the model's profile (if available) to ensure it supports tool calling, which
    is required for agent functionality. Issues warnings for models without profiles or
    with limited context windows.

    Args:
        model: The instantiated model to validate.
        model_name: Model name for error/warning messages.

    Note:
        This validation is best-effort. Models without profiles will pass with
        a warning. Calls `sys.exit(1)` if the model's profile explicitly
        indicates `tool_calling=False`.
    """
    console = _get_console()
    profile = getattr(model, "profile", None)

    if profile is None:
        # Model doesn't have profile data - warn but allow
        console.print(
            f"[dim][yellow]Note:[/yellow] No capability profile for "
            f"'{model_name}'. Cannot verify tool calling support.[/dim]"
        )
        return

    if not isinstance(profile, dict):
        return

    # Check required capability: tool_calling
    tool_calling = profile.get("tool_calling")
    if tool_calling is False:
        console.print(
            f"[bold red]Error:[/bold red] Model '{model_name}' "
            "does not support tool calling."
        )
        console.print(
            "\nDeep Agents requires tool calling for agent functionality. "
            "Please choose a model that supports tool calling."
        )
        console.print("\nSee MODELS.md for supported models.")
        sys.exit(1)

    # Warn about potentially limited context (< 8k tokens)
    max_input_tokens = profile.get("max_input_tokens")
    if max_input_tokens and max_input_tokens < 8000:  # noqa: PLR2004  # Model context window default
        console.print(
            f"[dim][yellow]Warning:[/yellow] Model '{model_name}' has limited context "
            f"({max_input_tokens:,} tokens). Agent performance may be affected.[/dim]"
        )


_console_instance: Console | None = None
_credentials_instance: Credentials | None = None
_runtime_state_instance: RuntimeState | None = None


def _get_console() -> Console:
    """Return the lazily-initialized global `Console` instance.

    Defers the `rich.console` import until console output is actually
    needed. The result is cached in `globals()["console"]`.

    Returns:
        The global Rich `Console` singleton.
    """
    global _console_instance  # noqa: PLW0603  # lazy process singleton
    cached = _console_instance
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = _console_instance
        if cached is not None:
            return cached
        from rich.console import Console

        inst = Console(highlight=False)
        _console_instance = inst
        return inst


def _get_credentials() -> Credentials:
    """Return the lazily initialized process-wide `Credentials` instance.

    Bootstrap runs before credentials are read.

    Returns:
        The global credentials singleton.
    """
    global _credentials_instance  # noqa: PLW0603  # lazy process singleton
    cached = _credentials_instance
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = _credentials_instance
        if cached is not None:
            return cached
        _ensure_bootstrap()
        try:
            inst = Credentials.from_environment(start_path=_bootstrap_state.start_path)
        except Exception:
            logger.exception(
                "Failed to initialize credentials from environment (start_path=%s)",
                _bootstrap_state.start_path,
            )
            raise
        _credentials_instance = inst
        return inst


def _get_runtime_state() -> RuntimeState:
    """Return the lazily initialized process-wide `RuntimeState` instance."""
    global _runtime_state_instance  # noqa: PLW0603  # lazy process singleton
    cached = _runtime_state_instance
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = _runtime_state_instance
        if cached is not None:
            return cached
        state = RuntimeState()
        _runtime_state_instance = state
        return state


class _LazyProxy:
    """Defer singleton construction until an attribute is first used."""

    __slots__ = ("_factory",)
    _factory: Callable[[], object]

    def __init__(self, factory: Callable[[], object]) -> None:
        object.__setattr__(self, "_factory", factory)

    def __getattr__(self, name: str) -> object:
        """Forward reads to the initialized singleton.

        Returns:
            The requested attribute.
        """
        return getattr(self._factory(), name)

    def __setattr__(self, name: str, value: object) -> None:
        """Forward mutations to the initialized singleton."""
        setattr(self._factory(), name, value)

    def __delattr__(self, name: str) -> None:
        """Forward patch cleanup to the initialized singleton."""
        delattr(self._factory(), name)


credentials = cast("Credentials", _LazyProxy(_get_credentials))
runtime_state = cast("RuntimeState", _LazyProxy(_get_runtime_state))
console = cast("Console", _LazyProxy(_get_console))
