"""Configuration, constants, and model creation."""

from __future__ import annotations

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
from dataclasses import dataclass, field as dataclass_field
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from deepagents_code._constants import FIREWORKS_PROVIDER_ID_PREFIX
from deepagents_code._env_vars import (
    DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
    DISABLED_PROJECT_MCP_SERVERS,
    HIDE_SPLASH_VERSION,
    is_env_truthy,
)
from deepagents_code._git import resolve_git_branch
from deepagents_code._version import __version__
from deepagents_code.config_manifest import (
    INTERPRETER_ENABLE_DEFAULT,
    INTERPRETER_MAX_PTC_CALLS_DEFAULT,
    INTERPRETER_MAX_RESULT_CHARS_DEFAULT,
    INTERPRETER_MEMORY_LIMIT_MB_DEFAULT,
    INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE_DEFAULT,
    INTERPRETER_PTC_DEFAULT,
    INTERPRETER_TIMEOUT_SECONDS_DEFAULT,
    RECURSION_LIMIT_DEFAULT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy bootstrap: dotenv loading, LANGSMITH_PROJECT override, and start-path
# detection are deferred until first access of `settings` (via module
# `__getattr__`).  This avoids disk I/O and path traversal during import for
# callers that never touch `settings` (e.g. `deepagents --help`).
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

    original_tracing_env: dict[str, str | None] = dataclass_field(default_factory=dict)
    """Caller's tracing-enable env before Deep Agents Code mutates flags."""

    original_tracing_api_keys: dict[str, str | None] = dataclass_field(
        default_factory=dict
    )
    """Caller's tracing API keys before Deep Agents Code overwrites them.

    Two bootstrap steps can overwrite the canonical `LANGSMITH_API_KEY` (and
    its `LANGCHAIN_API_KEY` alias): the `DEEPAGENTS_CODE_`-prefixed override and
    the `/auth`-stored key bridged on by `apply_stored_langsmith_auth`. Both run
    after this snapshot is captured. Without saving the originals, shell
    subprocesses inherit the agent's session key and the caller's own value is
    irrecoverable in-process. This mirrors the save/restore pattern used for
    tracing flags (`original_tracing_env`).
    """


_bootstrap_state = _BootstrapState()
"""State captured and mutated by lazy bootstrap."""

_bootstrap_lock = threading.Lock()
"""Guards `_ensure_bootstrap()` against concurrent access from the main thread
and the prewarm worker thread."""

_singleton_lock = threading.Lock()
"""Guards lazy singleton construction in `_get_console` / `_get_settings`."""

_dotenv_loaded_values: dict[str, str] = {}
"""Environment values injected by our dotenv loader and safe to refresh later."""

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

_DOTENV_DENIED_ENV_KEYS = frozenset(
    {
        "BASH_ENV",
        "BASHOPTS",
        "CDPATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GIT_ASKPASS",
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
        _INHERITED_PYTHONPATH_ENV,
    }
)
"""Environment keys that project `.env` files must not inject.

A project `.env` is untrusted (it travels with a cloned repo), so it must not be
able to set variables that turn loading the `.env` into code execution in the
subprocesses Deep Agents Code spawns. The set spans four threat categories;
every entry is here for one of these reasons, so do not remove one without
checking which category it belongs to:

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

`_INHERITED_PYTHONPATH_ENV` is denied so a project `.env` cannot smuggle a
`PYTHONPATH` into agent `execute` commands through the carrier var; the carrier
is only meant to relay a value the user set in their launch environment.

Matching is exact and case-sensitive: the protected consumers (the dynamic
linker, bash, CPython) read these names only in their canonical case, so a
lowercase `bash_env` injected into the environment is inert. Any future entry
that some consumer reads case-insensitively would need a different check.
"""

_PROJECT_DOTENV_DENIED_ENV_KEYS = frozenset(
    {
        DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        DISABLED_PROJECT_MCP_SERVERS,
    }
)
"""Env keys a *project* `.env` must not inject, even though they are otherwise
safe process-env inputs.

These two vars are the env form of the user-level project-MCP allow/deny lists
(`model_config.load_mcp_server_trust_lists`). Their whole purpose is to be a
*user-level* decision: naming a project MCP server here pre-approves it from an
untrusted `.mcp.json` (stdio → local command execution; remote → SSRF and
`${VAR}` header exfiltration during the discovery preflight). A project `.env`
travels with a cloned repo, so honoring it would let an attacker commit
`.mcp.json` + `.env` and self-approve their own servers — exactly the trust
boundary the feature exists to hold.

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


# Global user-level .env (~/.deepagents/.env); sentinel when Path.home() fails.
try:
    _GLOBAL_DOTENV_PATH = Path.home() / ".deepagents" / ".env"
except RuntimeError:
    _GLOBAL_DOTENV_PATH = Path("/nonexistent/.deepagents/.env")


def _preview_dotenv_environ(*, start_path: Path | None = None) -> dict[str, str]:
    """Return the environment after dotenv loading without mutating `os.environ`.

    Args:
        start_path: Directory to use for project `.env` discovery.

    Returns:
        Environment mapping with project and global dotenv values applied using
        the same first-write-wins precedence as `_load_dotenv`.
    """
    import dotenv

    env = dict(os.environ)
    for key, value in _dotenv_loaded_values.items():
        if env.get(key) == value:
            env.pop(key)

    def apply_dotenv(dotenv_path: Path | None, *, is_project: bool) -> None:
        if dotenv_path is None:
            return
        try:
            values = dotenv.dotenv_values(dotenv_path=dotenv_path)
        except (OSError, ValueError):
            logger.warning(
                "Could not read dotenv at %s; previewed project env vars may be "
                "incomplete",
                dotenv_path,
                exc_info=True,
            )
            return
        for key, value in values.items():
            if value is None or key in env:
                continue
            if key in _DOTENV_DENIED_ENV_KEYS:
                # Log the key only — the value is attacker-controlled.
                logger.debug("Ignoring denied env key %r from %s", key, dotenv_path)
                continue
            if is_project and key in _PROJECT_DOTENV_DENIED_ENV_KEYS:
                # Mirror `_load_dotenv`: a project `.env` cannot preview-set a
                # user-level MCP trust decision (the global `.env`/shell can).
                logger.debug(
                    "Ignoring project-denied env key %r from %s", key, dotenv_path
                )
                continue
            env[key] = value

    project_dotenv: Path | None = None
    try:
        project_dotenv = (
            _find_dotenv_from_start_path(start_path)
            if start_path is not None
            else _find_dotenv_from_start_path(Path.cwd())
        )
    except OSError:
        logger.warning(
            "Could not inspect project dotenv at %s; previewed project env vars may "
            "be incomplete",
            start_path or "cwd",
            exc_info=True,
        )
    apply_dotenv(project_dotenv, is_project=True)

    try:
        global_dotenv = _GLOBAL_DOTENV_PATH if _GLOBAL_DOTENV_PATH.is_file() else None
    except OSError:
        logger.warning(
            "Could not inspect global dotenv at %s; previewed global defaults may "
            "be incomplete",
            _GLOBAL_DOTENV_PATH,
            exc_info=True,
        )
        global_dotenv = None
    apply_dotenv(global_dotenv, is_project=False)

    return env


def _resolve_env_var_from(env: dict[str, str], name: str) -> str | None:
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


def _load_dotenv(
    *, start_path: Path | None = None, refresh_loaded: bool = False
) -> bool:
    """Load environment variables from project and global `.env` files.

    Loads in order (first write wins, `override=False`):

    1. Project/CWD `.env` — project-specific values
    2. `~/.deepagents/.env` — global user defaults

    Both layers use `override=False` (the python-dotenv default) so that
    shell-exported variables always take precedence over dotenv files.
    Because project loads first, the effective precedence is:

    ```text
    shell env (incl. inline `VAR=x`)  >  project `.env`  >  global `.env`
    ```

    !!! note

        To scope credentials to the app without colliding with
        identically-named shell exports, use the `DEEPAGENTS_CODE_` env-var
        prefix (see `resolve_env_var` in `deepagents_code.model_config`).

    Args:
        start_path: Directory to use for project `.env` discovery.
        refresh_loaded: Remove values previously injected by this loader before
            applying the current project/global dotenv stack. Values modified
            after loading are preserved.

    Returns:
        `True` when at least one dotenv file was loaded, `False` otherwise.
    """
    import dotenv

    loaded = False

    if refresh_loaded:
        for key, value in list(_dotenv_loaded_values.items()):
            if os.environ.get(key) == value:
                os.environ.pop(key)
        _dotenv_loaded_values.clear()

    def apply_dotenv(dotenv_path: Path, *, is_project: bool) -> bool:
        values = dotenv.dotenv_values(dotenv_path=dotenv_path)
        applied = False
        for key, value in values.items():
            if value is None or key in os.environ:
                continue
            if key in _DOTENV_DENIED_ENV_KEYS:
                # Log the key only — the value is attacker-controlled.
                logger.debug("Ignoring denied env key %r from %s", key, dotenv_path)
                continue
            if is_project and key in _PROJECT_DOTENV_DENIED_ENV_KEYS:
                # A committed project `.env` must not set a user-level MCP trust
                # decision; the global `.env` and shell may (is_project=False).
                logger.debug(
                    "Ignoring project-denied env key %r from %s", key, dotenv_path
                )
                continue
            os.environ[key] = value
            _dotenv_loaded_values[key] = value
            applied = True
        return applied

    # 1. Project/CWD .env — loads first so project values are set before the
    # global file, which can only fill in vars not already present.
    dotenv_path: Path | str | None = None
    try:
        if start_path is None:
            found = dotenv.find_dotenv(usecwd=True)
            if found:
                dotenv_path = found
                loaded = apply_dotenv(Path(found), is_project=True) or loaded
        else:
            dotenv_path = _find_dotenv_from_start_path(start_path)
            if dotenv_path is not None:
                loaded = apply_dotenv(dotenv_path, is_project=True) or loaded
    except (OSError, ValueError):
        logger.warning(
            "Could not read project dotenv at %s; project env vars will not be loaded",
            dotenv_path or start_path or "cwd",
            exc_info=True,
        )

    # 2. Global (~/.deepagents/.env) — fills in any vars not already set by
    # the shell or the project dotenv.
    # try/except wraps both is_file() and load_dotenv() to cover the TOCTOU
    # window where the file can vanish between stat and open.
    try:
        if _GLOBAL_DOTENV_PATH.is_file() and apply_dotenv(
            _GLOBAL_DOTENV_PATH, is_project=False
        ):
            loaded = True
            logger.debug("Loaded global dotenv: %s", _GLOBAL_DOTENV_PATH)
    except (OSError, ValueError):
        logger.warning(
            "Could not read global dotenv at %s; global defaults will not be applied",
            _GLOBAL_DOTENV_PATH,
            exc_info=True,
        )

    return loaded


_TRACING_ENABLE_ENV_VARS = (
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING",
)
"""Env vars LangChain/LangSmith read to decide whether tracing is enabled."""

_TRACING_API_KEY_ENV_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
"""Env vars that hold the LangSmith API key used for trace ingestion."""

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


_TRACING_RUNS_ENDPOINTS_ENV_VARS = (
    "LANGSMITH_RUNS_ENDPOINTS",
    "LANGCHAIN_RUNS_ENDPOINTS",
)
"""Env vars the LangSmith SDK parses into replica trace ingestion targets."""


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
    "langchain",
    "langsmith",
)


def _quiet_sdk_logging() -> None:
    """Keep non-actionable SDK diagnostics off the terminal.

    The harness-profile resolver and tracing SDKs emit diagnostics on their own
    logger hierarchies. With no handler attached, warnings reach Python's
    last-resort stderr handler and can bleed into command output or the
    alternate-screen TUI. Route them to the debug log when
    `DEEPAGENTS_CODE_DEBUG` is set, otherwise attach a `NullHandler` so they stay
    off the terminal. Other Deep Agents loggers remain untouched so actionable
    runtime warnings are still visible.
    """
    from deepagents_code._debug import configure_debug_logging
    from deepagents_code._env_vars import DEBUG, is_env_truthy

    debug_enabled = is_env_truthy(DEBUG)
    for name in _QUIET_SDK_LOGGER_NAMES:
        sdk_logger = logging.getLogger(name)
        if debug_enabled:
            configure_debug_logging(sdk_logger)
        if not sdk_logger.handlers:
            sdk_logger.addHandler(logging.NullHandler())


def _load_langsmith_profile_config(
    env: dict[str, str] | None = None,
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


def _has_langsmith_profile_credentials(env: dict[str, str] | None = None) -> bool:
    """Return whether the LangSmith profile config has usable auth material."""
    config = _load_langsmith_profile_config(env)
    if config is None:
        return False

    return bool(
        config.api_key or config.oauth_access_token or config.oauth_refresh_token
    )


def _has_langsmith_profile_custom_endpoint(env: dict[str, str] | None = None) -> bool:
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


def _tracing_enabled() -> bool:
    """Whether any LangSmith/LangChain tracing flag is truthy in the environment.

    Reads the canonical tracing-enable vars (`_TRACING_ENABLE_ENV_VARS`) and
    classifies each present value with `classify_env_bool`, mirroring how the
    LangChain/LangSmith SDKs decide whether to start tracing. Shared by
    `_disable_orphaned_tracing` and `_apply_default_langsmith_project` so both
    read the flags identically.

    Returns:
        `True` if at least one tracing flag is set to a truthy value,
            else `False`.
    """
    from deepagents_code._env_vars import classify_env_bool

    return any(
        classify_env_bool(os.environ[var])
        for var in _TRACING_ENABLE_ENV_VARS
        if var in os.environ
    )


def _disable_set_tracing_flags() -> list[str]:
    """Set every configured tracing-enable flag to `false`.

    Returns:
        Env var names that were disabled.
    """
    disabled = [var for var in _TRACING_ENABLE_ENV_VARS if var in os.environ]
    for var in disabled:
        os.environ[var] = "false"
    return disabled


def restore_user_tracing_env(env: dict[str, str]) -> None:
    """Restore caller tracing flags in an environment passed to user code.

    Args:
        env: Environment mapping prepared for a child/user subprocess.
    """
    for var, value in _bootstrap_state.original_tracing_env.items():
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value


def restore_user_tracing_api_keys(env: dict[str, str]) -> None:
    """Restore caller tracing API keys in an environment passed to user code.

    Reverts both bootstrap overwrites of the canonical LangSmith key — the
    `DEEPAGENTS_CODE_`-prefixed override and the `/auth`-stored key — so shell
    subprocesses receive the caller's own key rather than the agent's session
    key. See `original_tracing_api_keys` for the rationale; this mirrors
    `restore_user_tracing_env`, which does the same for tracing flags.

    Args:
        env: Environment mapping prepared for a child/user subprocess.
    """
    for var, value in _bootstrap_state.original_tracing_api_keys.items():
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value


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

    apply_stored_service_credentials()
    _apply_stored_langsmith_tracing(replace_project=replace_project)
    _disable_orphaned_tracing()
    _apply_default_langsmith_project()
    configure_langsmith_secret_redaction()


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
    prefixed_names = [f"DEEPAGENTS_CODE_{name}" for name in _TRACING_API_KEY_ENV_VARS]
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
    automatically by `_get_settings()` when `settings` is first accessed.

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
            from deepagents_code.project_utils import (
                get_server_project_context as _get_server_project_context,
            )

            ctx = _get_server_project_context()
            _bootstrap_state.start_path = ctx.user_cwd if ctx else None
            _load_dotenv(start_path=_bootstrap_state.start_path)

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
            _bootstrap_state.original_tracing_env = {
                var: os.environ.get(var) for var in _TRACING_ENABLE_ENV_VARS
            }
            _bootstrap_state.original_tracing_api_keys = {
                var: os.environ.get(var) for var in _TRACING_API_KEY_ENV_VARS
            }

            # CRITICAL: Override LANGSMITH_PROJECT to route agent traces to a
            # separate project. LangSmith reads LANGSMITH_PROJECT at invocation
            # time, so we override it here and preserve the user's original
            # value for shell commands.
            from deepagents_code._env_vars import LANGSMITH_PROJECT

            deepagents_project = os.environ.get(LANGSMITH_PROJECT)
            if deepagents_project:
                os.environ["LANGSMITH_PROJECT"] = deepagents_project

            # Propagate prefixed LangSmith env vars to canonical names.
            # The app resolves prefixed vars via resolve_env_var(), but the
            # LangSmith SDK reads os.environ directly and has no knowledge
            # of the DEEPAGENTS_CODE_ prefix. Setting canonical vars here
            # bridges that gap.
            from deepagents_code._env_vars import SUPPRESS_ENV_OVERRIDE_WARNING
            from deepagents_code.model_config import _ENV_PREFIX

            suppress_override_warning = is_env_truthy(SUPPRESS_ENV_OVERRIDE_WARNING)

            for canonical in (
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
            ):
                prefixed = f"{_ENV_PREFIX}{canonical}"
                if prefixed not in os.environ:
                    continue
                prefixed_val = os.environ[prefixed]
                if canonical not in os.environ:
                    # Propagate (including empty string for explicit disable).
                    os.environ[canonical] = prefixed_val
                elif os.environ[canonical] != prefixed_val:
                    os.environ[canonical] = prefixed_val
                    if not suppress_override_warning:
                        logger.warning(
                            "%s and %s are both set to different values. Deep "
                            "Agents Code uses %s for this session (the "
                            "%s-prefixed value takes precedence). The %s you "
                            "exported in your own shell is unaffected. This is "
                            "expected. To silence this warning, unset %s or set "
                            "%s=1.",
                            canonical,
                            prefixed,
                            prefixed,
                            _ENV_PREFIX,
                            canonical,
                            canonical,
                            SUPPRESS_ENV_OVERRIDE_WARNING,
                        )

            # Bridge stored service keys, apply stored LangSmith tracing defaults,
            # disable orphaned tracing, and route active tracing to the displayed
            # project. Keeping this in one helper lets `/auth` save apply the same
            # state immediately inside an already-running TUI session.
            apply_stored_langsmith_auth()
        except Exception:
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

    # Static type stubs for lazy module attributes resolved by __getattr__.
    # At runtime these are created on first access by _get_settings() /
    # _get_console() and cached in globals().
    settings: Settings
    console: Console

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
    output_prefix: str  # ⎿ vs L
    spinner_frames: tuple[str, ...]  # Braille vs ASCII spinner
    pause: str  # ⏸ vs ||
    newline: str  # ⏎ vs \\n
    warning: str  # ⚠ vs [!]
    question: str  # ? vs [?]
    hourglass: str  # ⏳ vs [~]
    retry: str  # ↻ vs [R]
    arrow_up: str  # up arrow vs ^
    arrow_down: str  # down arrow vs v
    bullet: str  # bullet vs -
    cursor: str  # cursor vs >
    disclosure_collapsed: str  # ▸ vs >
    disclosure_expanded: str  # ▾ vs v

    # Box-drawing characters
    box_vertical: str  # │ vs |
    box_horizontal: str  # ─ vs -
    box_double_horizontal: str  # ═ vs =

    # Diff-specific
    gutter_bar: str  # ▌ vs |

    # Status bar
    git_branch: str  # "↗" vs "git:"


UNICODE_GLYPHS = Glyphs(
    tool_prefix="⏺",
    ellipsis="…",
    checkmark="✓",
    error="✗",
    circle_empty="○",
    circle_filled="●",
    output_prefix="⎿",
    spinner_frames=("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"),
    pause="⏸",
    newline="⏎",
    warning="⚠",
    question="?",
    hourglass="⏳",
    retry="↻",
    arrow_up="↑",
    arrow_down="↓",
    bullet="•",
    cursor="›",  # noqa: RUF001  # Intentional Unicode glyph
    disclosure_collapsed="▸",
    disclosure_expanded="▾",
    # Box-drawing characters
    box_vertical="│",
    box_horizontal="─",
    box_double_horizontal="═",
    gutter_bar="▌",
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
    output_prefix="L",
    spinner_frames=("(-)", "(\\)", "(|)", "(/)"),
    pause="||",
    newline="\\n",
    warning="[!]",
    question="[?]",
    hourglass="[~]",
    retry="[R]",
    arrow_up="^",
    arrow_down="v",
    bullet="-",
    cursor=">",
    disclosure_collapsed=">",
    disclosure_expanded="v",
    # Box-drawing characters
    box_vertical="|",
    box_horizontal="-",
    box_double_horizontal="=",
    gutter_bar="|",
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
    this uses the shared resolver that prefers the editable source marker. A
    sibling monorepo workspace whose marker trails dcode's exact pin reports the
    pinned release baseline with a `+editable` suffix.

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

    Args:
        base_version: The base version string.
        editable: Whether the distribution is installed in editable mode.

    Returns:
        The version string, suffixed with ` (editable)` when `editable`.
    """
    return f"{base_version} (editable)" if editable else base_version


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
                    path = url2pathname(urlparse(url).path)
                    home = str(Path.home())
                    if path.startswith(home):
                        path = "~" + path[len(home) :]
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
    from deepagents_code.model_config import resolve_env_var

    env_mode = (resolve_env_var("UI_CHARSET_MODE") or "auto").lower()
    if env_mode == "unicode":
        return CharsetMode.UNICODE
    if env_mode == "ascii":
        return CharsetMode.ASCII

    # Auto: check stdout encoding and LANG
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return CharsetMode.UNICODE
    lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    if "utf" in lang.lower():
        return CharsetMode.UNICODE
    return CharsetMode.ASCII


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

config: RunnableConfig = {
    "recursion_limit": RECURSION_LIMIT_DEFAULT,
}
"""Default LangGraph runnable config for the main agent.

Sets `recursion_limit` to `RECURSION_LIMIT_DEFAULT` (2000) to accommodate deeply
nested agent graphs in long-running sessions without hitting the default
LangGraph ceiling. The literal lives in `config_manifest` so the default is
defined in exactly one place. This value is the fallback: `create_cli_agent`
resolves the effective limit at agent-build time via `resolve_recursion_limit`,
which honors the `--recursion-limit` CLI flag, the `DEEPAGENTS_CODE_RECURSION_LIMIT`
env var, and `[runtime].recursion_limit` in `config.toml`.
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
    can differ from a remote graph's Deep Agents runtime version. For sibling
    monorepo packages, a `+editable` suffix identifies workspace HEAD relative
    to the pinned published SDK baseline.

    Also records `dcode_experimental=True` when `DEEPAGENTS_CODE_EXPERIMENTAL`
    is enabled, so experimental runs are filterable in trace metadata.

    Also records `dcode_auto_approve=True` when auto-approve ("YOLO") mode is
    active, so runs that ran tools without HITL approval are filterable in trace
    metadata. This is a diagnostic key, not the contract-scoped `approval_policy`
    key (see above), so it is safe to stamp trace-wide.

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

    Returns:
        Config dict with `configurable` and `metadata` keys.
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

    # Legacy / diagnostic keys preserved for backward-compatibility during the
    # coding-agent-v1 rollout (not part of the contract).
    metadata["lc_versions"] = {
        "deepagents-code": _format_lc_version(
            __version__, editable=_is_editable_install()
        )
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

    return {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
    }


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
        ValueError: If `'all'` is combined with other commands.
    """
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


def _read_config_toml_retries() -> dict[str, Any] | None:
    """Read and lightly validate `[retries]` from `~/.deepagents/config.toml`.

    Provider sub-table names are checked against the set of providers the app
    knows how to authenticate so a mistyped provider (e.g. `[retries.fireorks]`)
    surfaces a warning rather than being silently dropped. Value validation is
    deferred to `_resolve_retry_kwargs`, which runs per active provider.

    Returns:
        The raw `[retries]` mapping, or `None` when the section is absent or the
            file cannot be read.
    """
    import tomllib

    from deepagents_code.model_config import (
        DEFAULT_CONFIG_PATH,
        IMPLICIT_AUTH_PROVIDERS,
        NO_AUTH_REQUIRED_PROVIDERS,
        PROVIDER_API_KEY_ENV,
        RETRY_PARAM_BY_PROVIDER,
    )

    try:
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not read retries config from %s",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
        return None

    section = data.get("retries")
    if not isinstance(section, dict):
        return None

    known_providers = (
        set(PROVIDER_API_KEY_ENV)
        | set(NO_AUTH_REQUIRED_PROVIDERS)
        | set(IMPLICIT_AUTH_PROVIDERS)
        | set(RETRY_PARAM_BY_PROVIDER)
    )
    for key, value in section.items():
        if (
            isinstance(value, dict)
            and key not in known_providers
            and "param" not in value
        ):
            logger.warning(
                "Ignoring [retries.%s] in config.toml; %r is not a known provider",
                key,
                key,
            )
    return section


def _coerce_max_retries(raw: Any, *, source: str) -> int | None:  # noqa: ANN401
    """Validate a TOML retry count.

    Args:
        raw: Value loaded from TOML.
        source: Human-readable config path for warnings.

    Returns:
        The retry count, or `None` when invalid.
    """
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    logger.warning("Ignoring %s=%r in config.toml (expected int >= 0)", source, raw)
    return None


def _coerce_retry_param(raw: Any, *, source: str) -> str | None:  # noqa: ANN401
    """Validate a constructor kwarg name for retry configuration.

    Args:
        raw: Value loaded from TOML.
        source: Human-readable config path for warnings.

    Returns:
        The retry parameter name, or `None` when invalid.
    """
    if isinstance(raw, str) and raw.isidentifier() and not keyword.iskeyword(raw):
        return raw
    logger.warning(
        "Ignoring %s=%r in config.toml (expected Python identifier string)",
        source,
        raw,
    )
    return None


def _resolve_retry_kwargs(
    section: dict[str, Any] | None,
    provider: str,
) -> dict[str, int]:
    """Resolve the retry-count kwarg for `provider` from a `[retries]` section.

    A per-provider `[retries.<provider>].max_retries` overrides the global
    `[retries].max_retries`. Known providers use `RETRY_PARAM_BY_PROVIDER`;
    arbitrary providers can opt in with `[retries.<provider>].param`.
    Unknown providers without a configured parameter receive nothing, and
    unknown or malformed keys are dropped with a warning.

    Args:
        section: Raw `[retries]` mapping from `config.toml`, or `None`.
        provider: Provider the kwargs are being resolved for.

    Returns:
        `{retry_param_name: count}` when a valid retry count resolves, else an
            empty dict.
    """
    if not section:
        return {}

    from deepagents_code.model_config import RETRY_PARAM_BY_PROVIDER

    for key, value in section.items():
        if key == "max_retries" or isinstance(value, dict):
            continue
        logger.warning("Ignoring [retries].%s=%r in config.toml", key, value)

    retry_param = RETRY_PARAM_BY_PROVIDER.get(provider)
    resolved: int | None = None
    if "max_retries" in section:
        resolved = _coerce_max_retries(
            section["max_retries"], source="[retries].max_retries"
        )

    provider_section = section.get(provider)
    if provider_section is not None and not isinstance(provider_section, dict):
        logger.warning(
            "Ignoring [retries].%s=%r in config.toml (expected table)",
            provider,
            provider_section,
        )
    elif provider_section:
        for key, value in provider_section.items():
            if key not in {"max_retries", "param"}:
                logger.warning(
                    "Ignoring [retries.%s].%s=%r in config.toml",
                    provider,
                    key,
                    value,
                )
        if "max_retries" in provider_section:
            provider_value = _coerce_max_retries(
                provider_section["max_retries"],
                source=f"[retries.{provider}].max_retries",
            )
            if provider_value is not None:
                resolved = provider_value
        if "param" in provider_section:
            provider_param = _coerce_retry_param(
                provider_section["param"],
                source=f"[retries.{provider}].param",
            )
            if provider_param is not None:
                retry_param = provider_param

    if retry_param is None:
        logger.warning(
            "Ignoring [retries] config for provider %r; provider does not support "
            "a registered or configured retry parameter",
            provider,
        )
        return {}

    if resolved is None:
        return {}
    return {retry_param: resolved}


CLI_MAX_RETRIES_KEY = "__deepagents_cli_max_retries__"
"""Internal carrier key for the `--max-retries` CLI flag.

`cli_main` stashes the flag value under this key in the `model_params` dict it
forwards to the run, and `create_model` pops it before constructing the model.
This lets the CLI value ride the existing `model_params`/`extra_kwargs` carrier
to the one place that authoritatively resolves the provider, where it can be
folded under the provider's *resolved* retry-param name (see
`_resolve_retry_param_name`) rather than a hardcoded `max_retries`.

The key is internal-only: it is popped before reaching any model constructor and
is never serialized or surfaced to users. It is deliberately unlikely to collide
with a real constructor kwarg name.
"""


def _resolve_retry_param_name(provider: str) -> str:
    """Resolve the constructor kwarg name that sets `provider`'s retry count.

    Honors a `[retries.<provider>].param` override in `config.toml`, then the
    registered `RETRY_PARAM_BY_PROVIDER` mapping, and finally falls back to
    `max_retries` -- the near-universal LangChain chat-model kwarg -- for
    providers that are neither registered nor configured.

    Args:
        provider: Provider the retry kwarg name is being resolved for.

    Returns:
        The constructor kwarg name to use for the retry count.
    """
    from deepagents_code.model_config import RETRY_PARAM_BY_PROVIDER

    section = _read_config_toml_retries()
    if section:
        provider_section = section.get(provider)
        if isinstance(provider_section, dict) and "param" in provider_section:
            configured = _coerce_retry_param(
                provider_section["param"],
                source=f"[retries.{provider}].param",
            )
            if configured is not None:
                return configured

    return RETRY_PARAM_BY_PROVIDER.get(provider, "max_retries")


def _read_config_toml_skills_dirs() -> list[str] | None:
    """Read `[skills].extra_allowed_dirs` from `~/.deepagents/config.toml`.

    Returns:
        List of path strings, or `None` if the key is absent or the file
            cannot be read.
    """
    import tomllib

    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    try:
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not read skills config from %s",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
        return None

    skills_section = data.get("skills", {})
    dirs = skills_section.get("extra_allowed_dirs")
    if isinstance(dirs, list):
        return dirs
    return None


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
            Path(p.strip()).expanduser().resolve()
            for p in env_raw.split(":")
            if p.strip()
        ]
        return dirs or None

    if config_toml_dirs:
        dirs = [
            Path(p).expanduser().resolve()
            for p in config_toml_dirs
            if isinstance(p, str) and p.strip()
        ]
        return dirs or None

    return None


_RELOADABLE_FIELDS = (
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "nvidia_api_key",
    "tavily_api_key",
    "google_cloud_project",
    "deepagents_langchain_project",
    "project_root",
    "shell_allow_list",
    "extra_skills_dirs",
)
"""Fields refreshed on `/reload` and cwd switches.

Runtime model state (`model_name`, `model_provider`, `model_context_limit`) and
the original user LangSmith project are intentionally excluded -- they are set
once and should not change across reloads.
"""

_API_KEY_FIELDS = frozenset(
    field for field in _RELOADABLE_FIELDS if field.endswith("_api_key")
)
"""Reloadable fields that hold API keys and must be masked in change reports.

Derived from `_RELOADABLE_FIELDS` so new `*_api_key` fields are picked up
automatically.
"""


@dataclass
class Settings:
    """Global settings and environment detection for deepagents-code.

    This class is initialized once at startup and provides access to:
    - Available models and API keys
    - Current project information
    - Tool availability (e.g., Tavily)
    - File system paths
    """

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

    deepagents_langchain_project: str | None
    """LangSmith project name for deepagents agent tracing."""

    user_langchain_project: str | None
    """Original `LANGSMITH_PROJECT` from environment (for user code)."""

    model_name: str | None = None
    """Currently active model name, set after model creation."""

    model_provider: str | None = None
    """Provider identifier (e.g., `openai`, `anthropic`, `google_genai`)."""

    model_context_limit: int | None = None
    """Maximum input token count from the model profile."""

    model_unsupported_modalities: frozenset[str] = frozenset()
    """Input modalities not indicated as supported by the model profile."""

    project_root: Path | None = None
    """Current project root directory, or `None` if not in a git project."""

    shell_allow_list: list[str] | None = None
    """Shell commands that don't require user approval."""

    extra_skills_dirs: list[Path] | None = None
    """Extra directories added to the skill path containment allowlist.

    These do NOT add new skill discovery locations — skills are still only
    discovered from the standard directories. They exist so that symlinks inside
    standard skill directories can point to targets in these additional
    locations without being rejected by the containment check
    in `load_skill_content`.

    Set via `DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS` env var (colon-separated) or
    `[skills].extra_allowed_dirs` in `~/.deepagents/config.toml`.
    """

    enable_interpreter: bool = INTERPRETER_ENABLE_DEFAULT
    """Wire `CodeInterpreterMiddleware` from `langchain-quickjs` into the main
    agent. Local-mode only; raises `ValueError` at agent-build time when a
    remote sandbox is active. Subagents never receive the interpreter in v1.

    `langchain-quickjs` is installed as a core dependency.

    Defaults are owned by `config_manifest` (the canonical config surface) so
    they are defined in exactly one place.
    """

    interpreter_timeout_seconds: float = INTERPRETER_TIMEOUT_SECONDS_DEFAULT
    """Per-`js_eval`-call wall-clock timeout (seconds) for the QuickJS REPL."""

    interpreter_memory_limit_mb: int = INTERPRETER_MEMORY_LIMIT_MB_DEFAULT
    """QuickJS heap memory cap (MB), shared across all calls within a session."""

    interpreter_max_ptc_calls: int = INTERPRETER_MAX_PTC_CALLS_DEFAULT
    """Maximum `tools.*` host-bridge invocations allowed per `js_eval` call.

    PTC calls bypass `interrupt_on`/HITL approval — this budget is the only
    runtime limiter on bursty tool fan-out from inside the REPL.
    """

    interpreter_max_result_chars: int = INTERPRETER_MAX_RESULT_CHARS_DEFAULT
    """Independent cap (chars) on `js_eval` result and stdout blocks before
    truncation."""

    interpreter_ptc: str | bool | list[str] = INTERPRETER_PTC_DEFAULT
    """Programmatic tool calling allowlist for `js_eval`.

    Accepted values:

    - `False` or `[]`: pure REPL, no `tools.*` bridge.
    - `"safe"`: expand to `INTERPRETER_PTC_SAFE_PRESET` (the default).
    - `"all"`: every tool passed to `create_cli_agent` is exposed. Requires
        `interpreter_ptc_acknowledge_unsafe=True` when `auto_approve` is `False`.
    - `list[str]`: explicit tool names. The list may also include the `"safe"`
        preset (expanded to `INTERPRETER_PTC_SAFE_PRESET`); `"all"` is rejected
        inside a list. Names are matched against the live tool registry at
        runtime, so names not present are simply not exposed.
    """

    interpreter_ptc_acknowledge_unsafe: bool = (
        INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE_DEFAULT
    )
    """Explicit acknowledgement required when `interpreter_ptc="all"` is set
    without `auto_approve`.

    `"all"` exposes every host tool to `tools.*` calls from inside the REPL,
    bypassing HITL approval — this flag is a deliberate sanity gate, not a
    feature toggle.
    """

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> Settings:
        """Create settings by detecting the current environment.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            Settings instance with detected configuration
        """
        # Detect API keys (normalize empty strings to None).
        from deepagents_code.model_config import resolve_env_var

        openai_key = resolve_env_var("OPENAI_API_KEY")
        anthropic_key = resolve_env_var("ANTHROPIC_API_KEY")
        google_key = resolve_env_var("GOOGLE_API_KEY")
        nvidia_key = resolve_env_var("NVIDIA_API_KEY")
        tavily_key = resolve_env_var("TAVILY_API_KEY")
        google_cloud_project = resolve_env_var("GOOGLE_CLOUD_PROJECT")

        # Detect LangSmith configuration
        # DEEPAGENTS_CODE_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # When accessed via the module-level `settings` singleton,
        # _ensure_bootstrap() has already run and may have overridden
        # LANGSMITH_PROJECT. We use the saved original value, not the
        # current os.environ value. Direct callers should ensure
        # bootstrap has run if they depend on the override.
        from deepagents_code._env_vars import (
            EXTRA_SKILLS_DIRS,
            LANGSMITH_PROJECT,
            SHELL_ALLOW_LIST,
        )

        deepagents_langchain_project = resolve_env_var(LANGSMITH_PROJECT)
        # Use the saved original, not the current `LANGSMITH_PROJECT` that
        # bootstrap may have overridden for agent traces.
        user_langchain_project = _bootstrap_state.original_langsmith_project

        # Detect project
        from deepagents_code.project_utils import find_project_root

        project_root = find_project_root(start_path)

        # Parse shell command allow-list from environment
        # Format: comma-separated list of commands (e.g., "ls,cat,grep,pwd")

        shell_allow_list_str = os.environ.get(SHELL_ALLOW_LIST)
        shell_allow_list = parse_shell_allow_list(shell_allow_list_str)

        # Parse extra skill containment roots from env var or config.toml.
        # These extend the path allowlist for load_skill_content but do not
        # add new skill discovery locations.
        extra_skills_dirs = _parse_extra_skills_dirs(
            os.environ.get(EXTRA_SKILLS_DIRS),
            _read_config_toml_skills_dirs(),
        )

        from deepagents_code.config_manifest import resolve_interpreter_kwargs

        interpreter_kwargs = resolve_interpreter_kwargs()

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            nvidia_api_key=nvidia_key,
            tavily_api_key=tavily_key,
            google_cloud_project=google_cloud_project,
            deepagents_langchain_project=deepagents_langchain_project,
            user_langchain_project=user_langchain_project,
            project_root=project_root,
            shell_allow_list=shell_allow_list,
            extra_skills_dirs=extra_skills_dirs,
            **interpreter_kwargs,
        )

    @staticmethod
    def _reload_values(
        *,
        start_path: Path | None,
        env: dict[str, str],
        previous: dict[str, object],
    ) -> dict[str, object]:
        """Resolve reloadable settings from an environment mapping.

        Returns:
            Reloadable setting values keyed by field name.
        """
        from deepagents_code._env_vars import (
            EXTRA_SKILLS_DIRS,
            LANGSMITH_PROJECT,
            SHELL_ALLOW_LIST,
        )

        try:
            shell_allow_list = parse_shell_allow_list(env.get(SHELL_ALLOW_LIST))
        except ValueError:
            logger.warning(
                "Invalid %s during reload; keeping previous value",
                SHELL_ALLOW_LIST,
            )
            shell_allow_list = previous["shell_allow_list"]

        try:
            from deepagents_code.project_utils import find_project_root

            project_root = find_project_root(start_path)
        except OSError:
            logger.warning(
                "Could not detect project root during reload; keeping previous value"
            )
            project_root = previous["project_root"]

        try:
            extra_skills_dirs = _parse_extra_skills_dirs(
                env.get(EXTRA_SKILLS_DIRS),
                _read_config_toml_skills_dirs(),
            )
        except (OSError, ValueError):
            # Path resolution can fail (e.g. broken symlink loop). Keep the
            # previous value rather than letting the failure escape reload --
            # callers such as the cwd switch run this after `os.chdir`, where an
            # uncaught error would strand the process in a half-applied cwd.
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
            "deepagents_langchain_project": _resolve_env_var_from(
                env,
                LANGSMITH_PROJECT,
            ),
            "project_root": project_root,
            "shell_allow_list": shell_allow_list,
            "extra_skills_dirs": extra_skills_dirs,
        }

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
        for field in _RELOADABLE_FIELDS:
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
        previous = {field: getattr(self, field) for field in _RELOADABLE_FIELDS}
        env = _preview_dotenv_environ(start_path=start_path)
        refreshed = self._reload_values(
            start_path=start_path,
            env=env,
            previous=previous,
        )
        return self._format_reload_changes(previous, refreshed)

    def reload_from_environment(self, *, start_path: Path | None = None) -> list[str]:
        """Reload selected settings from environment variables and project files.

        This refreshes only fields that are expected to change at runtime
        (API keys, Google Cloud project, project root, shell allow-list, and
        LangSmith tracing project).

        Runtime model state (`model_name`, `model_provider`,
        `model_context_limit`) and the original user LangSmith project
        (`user_langchain_project`) are intentionally preserved -- they are
        not in `_RELOADABLE_FIELDS` and are never touched by this method.

        !!! note

            Shell-exported variables always take precedence. Values previously
            injected from `.env` files are refreshed so an accepted cwd switch
            can pick up the resumed project's `.env`.

        Args:
            start_path: Directory to start project detection from (defaults to cwd).

        Returns:
            A list of human-readable change descriptions.
        """
        _load_dotenv(start_path=start_path, refresh_loaded=True)

        previous = {field: getattr(self, field) for field in _RELOADABLE_FIELDS}
        refreshed = self._reload_values(
            start_path=start_path,
            env=dict(os.environ),
            previous=previous,
        )

        for field, value in refreshed.items():
            setattr(self, field, value)

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
        return self._format_reload_changes(previous, refreshed)

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
        """Check if VertexAI is available (Google Cloud project set, no API key).

        VertexAI uses Application Default Credentials (ADC) for authentication,
        so if GOOGLE_CLOUD_PROJECT is set and GOOGLE_API_KEY is not, we assume
        VertexAI.
        """
        return self.google_cloud_project is not None and self.google_api_key is None

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """Base user-level `.deepagents` directory.

        Returns:
            Path to `~/.deepagents`
        """
        return Path.home() / ".deepagents"

    @staticmethod
    def get_user_agent_md_path(agent_name: str) -> Path:
        """Get user-level AGENTS.md path for a specific agent.

        Returns path regardless of whether the file exists.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/AGENTS.md
        """
        return Path.home() / ".deepagents" / agent_name / "AGENTS.md"

    def get_project_agent_md_path(self) -> list[Path]:
        """Get project-level AGENTS.md paths.

        Checks both `{project_root}/.deepagents/AGENTS.md` and
        `{project_root}/AGENTS.md`, returning all that exist. If both are
        present, both are loaded and their instructions are combined, with
        `.deepagents/AGENTS.md` first.

        Returns:
            Existing AGENTS.md paths.

                Empty if neither file exists or not in a project, one entry if
                only one is present, or two entries if both locations have the
                file.
        """
        if not self.project_root:
            return []
        from deepagents_code.project_utils import find_project_agent_md

        return find_project_agent_md(self.project_root)

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """Validate to prevent invalid filesystem paths and security issues.

        Returns:
            True if the agent name is valid, False otherwise.
        """
        if not agent_name or not agent_name.strip():
            return False
        # Allow only alphanumeric, hyphens, underscores, and whitespace
        return bool(re.match(r"^[a-zA-Z0-9_\-\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the global agent directory path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        return Path.home() / ".deepagents" / agent_name

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """Ensure the global agent directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}

        Raises:
            ValueError: If the agent name contains invalid characters.
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. Agent names can only "
                "contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """Get user-level skills directory path for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """Ensure user-level skills directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        skills_dir = self.get_user_skills_dir(agent_name)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_project_skills_dir(self) -> Path | None:
        """Get project-level skills directory path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """Ensure project-level skills directory exists and return its path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        if skills_dir is None:
            return None
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_user_agents_dir(self, agent_name: str) -> Path:
        """Get user-level agents directory path for custom subagent definitions.

        Args:
            agent_name: Name of the agent (e.g., "deepagents")

        Returns:
            Path to ~/.deepagents/{agent_name}/agents/
        """
        return self.get_agent_dir(agent_name) / "agents"

    def get_project_agents_dir(self) -> Path | None:
        """Get project-level agents directory path for custom subagent definitions.

        Returns:
            Path to {project_root}/.deepagents/agents/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agents"

    @property
    def user_agents_dir(self) -> Path:
        """Base user-level `.agents` directory (`~/.agents`).

        Returns:
            Path to `~/.agents`
        """
        return Path.home() / ".agents"

    def get_user_agent_skills_dir(self) -> Path:
        """Get user-level `~/.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `~/.agents/skills/`
        """
        return self.user_agents_dir / "skills"

    def get_project_agent_skills_dir(self) -> Path | None:
        """Get project-level `.agents/skills/` directory.

        This is a generic alias path for skills that is tool-agnostic.

        Returns:
            Path to `{project_root}/.agents/skills/`, or `None` if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".agents" / "skills"

    @staticmethod
    def get_user_claude_skills_dir() -> Path:
        """Get user-level `~/.claude/skills/` directory (experimental).

        Convenience bridge for cross-tool skill sharing with Claude Code.
        This is experimental and may be removed.

        Returns:
            Path to `~/.claude/skills/`
        """
        return Path.home() / ".claude" / "skills"

    def get_project_claude_skills_dir(self) -> Path | None:
        """Get project-level `.claude/skills/` directory (experimental).

        Convenience bridge for cross-tool skill sharing with Claude Code.
        This is experimental and may be removed.

        Returns:
            Path to `{project_root}/.claude/skills/`, or `None` if not in a project.
        """
        if not self.project_root:
            return None
        return self.project_root / ".claude" / "skills"

    @staticmethod
    def get_built_in_skills_dir() -> Path:
        """Get the directory containing built-in skills that ship with the app.

        Returns:
            Path to the `built_in_skills/` directory within the package.
        """
        return Path(__file__).parent / "built_in_skills"

    def get_extra_skills_dirs(self) -> list[Path]:
        """Get user-configured extra skill directories.

        Set via `DEEPAGENTS_CODE_EXTRA_SKILLS_DIRS` (colon-separated paths) or
        `[skills].extra_allowed_dirs` in `~/.deepagents/config.toml`.

        Returns:
            List of extra skill directory paths, or empty list if not configured.
        """
        return self.extra_skills_dirs or []


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
    `settings.deepagents_langchain_project` (from
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
    langsmith_tracing = resolve_env_var("LANGSMITH_TRACING") or resolve_env_var(
        "LANGCHAIN_TRACING_V2"
    )
    if not (langsmith_key and langsmith_tracing):
        return None

    return (
        _get_settings().deepagents_langchain_project
        or os.environ.get("LANGSMITH_PROJECT")
        or LANGSMITH_PROJECT_DEFAULT
    )


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
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("tracing.langsmith_redact")
    if option is None:
        return False
    value, _ = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)


def is_memory_auto_save_enabled() -> bool:
    """Return whether the agent should proactively save learnings to memory.

    Resolves the `memory.auto_save` option from env/`config.toml`, defaulting to
    enabled. When disabled, memory is still loaded into context but the agent is
    told not to auto-save.
    """
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("memory.auto_save")
    if option is None:
        return True
    value, _ = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)


def is_openai_prompt_cache_key_enabled() -> bool:
    """Return whether OpenAI model calls should carry a per-thread cache key.

    Resolves the `models.openai_prompt_cache_key` option from env/`config.toml`,
    defaulting to enabled. When disabled, `ConfigurableModelMiddleware` stops
    injecting the thread ID as an OpenAI `prompt_cache_key` (a user-supplied key
    is still forwarded). This is the opt-out for OpenAI-compatible endpoints that
    reject unknown request fields.
    """
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("models.openai_prompt_cache_key")
    if option is None:
        return True
    value, _ = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)


def resolve_goal_auto_accept_criteria() -> tuple[bool, str]:
    """Resolve whether Auto mode applies generated goal criteria without review.

    Returns:
        The effective preference and its config source. The preference fails closed
        to disabled if the manifest entry is unavailable.
    """
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("goals.auto_accept_criteria")
    if option is None:
        logger.warning(
            "Manifest option 'goals.auto_accept_criteria' is missing; goal "
            "criteria auto-accept is disabled and any saved preference is "
            "ignored.",
        )
        return False, "default"
    value, source = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value), source


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

    env = dict(os.environ)
    # Cheap env-var checks first so the common (tracing-off) startup path skips
    # the TOML read in `is_langsmith_redaction_enabled`. These are plain env
    # reads with no failure mode of their own, so they stay outside the
    # fail-closed boundary: if there is no upload target, there is nothing to
    # protect.
    if not (_tracing_enabled_from(env) and _tracing_can_upload_from(env)):
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
    return _get_langsmith_replica_projects_from(dict(os.environ))


def _get_langsmith_replica_projects_from(env: dict[str, str]) -> list[str]:
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


_TRACING_BRIDGED_ENABLE_ENV_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
"""Tracing flags bootstrap propagates from a `DEEPAGENTS_CODE_` prefix.

`dcode doctor` runs before `_ensure_bootstrap` bridges these to their canonical
names, so it must resolve them prefix-aware (via `resolve_env_var`) to predict
the runtime's effective state. The remaining flags in `_TRACING_ENABLE_ENV_VARS`
are not bridged, so only their canonical form takes effect.
"""


def _tracing_enabled_from(env: dict[str, str]) -> bool:
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


def _tracing_explicitly_disabled_from(env: dict[str, str]) -> bool:
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
    return _tracing_enabled_from(dict(os.environ))


def _tracing_has_credentials_from(env: dict[str, str]) -> bool:
    """Return whether a LangSmith API key (env or active profile) is available.

    Both API-key vars are bridged from a `DEEPAGENTS_CODE_` prefix at bootstrap,
    so resolve them prefix-aware to match what the runtime will see.

    Args:
        env: Environment mapping to read.
    """
    has_key = any(_resolve_env_var_from(env, var) for var in _TRACING_API_KEY_ENV_VARS)
    return has_key or _has_langsmith_profile_credentials(env)


def _langsmith_runs_endpoint_urls_from(env: dict[str, str]) -> tuple[str, ...]:
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


def _has_langsmith_runs_endpoints_from(env: dict[str, str]) -> bool:
    """Return whether replica trace ingestion targets are configured.

    Args:
        env: Environment mapping to read.

    Returns:
        `True` when a valid runs-endpoints configuration is present.
    """
    return bool(_langsmith_runs_endpoint_urls_from(env))


def _tracing_can_upload_from(env: dict[str, str]) -> bool:
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


def _tracing_endpoint_from(env: dict[str, str]) -> str | None:
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


def _resolve_tracing_project_from(env: dict[str, str]) -> tuple[str, bool]:
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
        s = _get_settings()
        if not s.has_anthropic and s.has_vertex_ai:
            return "google_vertexai"
        return "anthropic"

    if model_lower.startswith("gemini"):
        s = _get_settings()
        if s.has_vertex_ai and not s.has_google:
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


def _get_default_model_spec() -> str:
    """Get default model specification based on available credentials.

    Checks in order:

    1. `[models].default` in config file (user's intentional preference).
    2. `[models].recent` in config file (last `/model` switch).
    3. Auto-detection based on available API credentials.

    Returns:
        Model specification in `provider:model` format.

    Raises:
        NoCredentialsConfiguredError: If no credentials are configured for any
            of the auto-detectable providers. Callers may catch this to defer
            startup and prompt for credentials interactively.
    """
    from deepagents_code.model_config import (
        ModelConfig,
        NoCredentialsConfiguredError,
        get_provider_auth_status,
    )

    config = ModelConfig.load()
    if config.default_model:
        return config.default_model

    if config.recent_model:
        return config.recent_model

    # `is True` deliberately excludes `ProviderAuthState.UNKNOWN` (which maps
    # to `as_legacy_bool() -> None`). For the three explicit-credential
    # providers below, an UNKNOWN result means we cannot prove auth works, so
    # we fall through rather than pick an unverifiable default. If an
    # implicit-auth provider (e.g., Vertex ADC) is added to this fallback
    # list, switch to checking `state` against the relevant
    # `ProviderAuthState` members directly.
    if get_provider_auth_status("openai").as_legacy_bool() is True:
        return "openai:gpt-5.5"
    if get_provider_auth_status("anthropic").as_legacy_bool() is True:
        return "anthropic:claude-opus-4-7"
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
    result: dict[str, Any] = config.get_kwargs(provider, model_name=model_name)
    base_url = config.get_base_url(provider)
    if base_url:
        result["base_url"] = base_url
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
        if api_key:
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

    retry_section = _read_config_toml_retries()
    retry_kwargs = _resolve_retry_kwargs(retry_section, provider)
    for key, value in retry_kwargs.items():
        result.setdefault(key, value)

    return result


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
            from deepagents_code.extras_info import extra_for_package

            extra = extra_for_package(package)
            if extra is not None:
                msg = (
                    f"Missing package for provider '{provider}'. "
                    f"Install: /install {extra}"
                )
            else:
                from deepagents_code.extras_info import ExtrasIntrospectionError
                from deepagents_code.update_check import (
                    ToolRequirementIntrospectionError,
                    install_package_command,
                )

                try:
                    install_cmd = install_package_command(package)
                except (
                    ValueError,
                    ExtrasIntrospectionError,
                    ToolRequirementIntrospectionError,
                ) as exc:
                    logger.debug(
                        "install_package_command failed; falling back to "
                        "manual hint: %s",
                        exc,
                    )
                    install_hint = f"Install the '{package}' package manually"
                else:
                    install_hint = f"Install with: {install_cmd}"
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

    This separates model creation from settings mutation so callers can decide
    when to commit the metadata to global settings.

    Attributes:
        model: The instantiated chat model.
        model_name: Resolved model name.
        provider: Resolved provider name.
        context_limit: Max input tokens from the model profile, or `None`.
        unsupported_modalities: Input modalities not indicated as supported by
            the model profile (e.g. `{"audio", "video"}`).
    """

    model: BaseChatModel
    model_name: str
    provider: str
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()

    def apply_to_settings(self) -> None:
        """Commit this result's metadata to global `settings`."""
        s = _get_settings()
        s.model_name = self.model_name
        s.model_provider = self.provider
        s.model_context_limit = self.context_limit
        s.model_unsupported_modalities = self.unsupported_modalities


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

            These take highest priority, overriding values from the config file.

            A `CLI_MAX_RETRIES_KEY` entry (set by the `--max-retries` flag) is
            treated specially: it is popped here and re-applied under the
            provider's resolved retry-param name with top precedence, rather than
            being forwarded verbatim to the constructor.
        profile_overrides: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides (dcode wins).

    Returns:
        A `ModelResult` containing the model and its metadata.

    Raises:
        ModelConfigError: If provider cannot be determined from the model name
            or required provider package is not installed.
        MissingCredentialsError: If no credentials are configured for the
            resolved provider.

    Examples:
        >>> model = create_model("anthropic:claude-sonnet-4-5")
        >>> model = create_model("openai:gpt-5.5")
        >>> model = create_model("gpt-5.5")  # Auto-detects openai
        >>> model = create_model()  # Uses environment defaults
    """
    from deepagents_code.model_config import (
        IMPLICIT_AUTH_PROVIDERS,
        ModelConfig,
        ModelConfigError,
        ModelSpec,
        apply_stored_credentials,
        get_credential_env_var,
        has_provider_credentials,
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

    # Stored API keys (added via `/auth`) take effect by being copied onto
    # the env var name LangChain reads. Apply before the credential check so
    # `has_provider_credentials` and the downstream SDK see the same value.
    if provider:
        # Flag a key/endpoint resolved from different env tiers *before*
        # `apply_stored_credentials` bridges stored values onto plain env vars,
        # so the check sees the user's raw env intent rather than post-bridge
        # state. Diagnostic only -- never alters resolution.
        warn_on_split_credential_source(provider)
        apply_stored_credentials(provider)

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

    # App --model-params take highest priority. Copy defensively before popping
    # the CLI sentinel so a caller that retains and reuses this dict (e.g. the
    # app re-creating the model on a runtime `/model` switch) keeps the sentinel
    # for the next provider's resolution.
    cli_max_retries: int | None = None
    reasoning_effort_override: object = None
    reasoning_override: object = None
    if extra_kwargs:
        extra_kwargs = dict(extra_kwargs)
        cli_max_retries = extra_kwargs.pop(CLI_MAX_RETRIES_KEY, None)
        reasoning_effort_override = extra_kwargs.get("reasoning_effort")
        reasoning_override = extra_kwargs.get("reasoning")
        kwargs.update(extra_kwargs)
    kwargs = _compose_openai_reasoning_effort(
        provider,
        kwargs,
        reasoning_effort_override,
        reasoning_override,
    )

    # `--max-retries` outranks everything: fold it under the provider's resolved
    # retry-param name (honoring `[retries.<provider>].param`) so a custom
    # provider whose kwarg is not `max_retries` is still served. Applied after
    # the `extra_kwargs` merge so it wins over a `max_retries` in `--model-params`.
    if cli_max_retries is not None:
        kwargs[_resolve_retry_param_name(provider)] = cli_max_retries

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


def _get_console() -> Console:
    """Return the lazily-initialized global `Console` instance.

    Defers the `rich.console` import until console output is actually
    needed. The result is cached in `globals()["console"]`.

    Returns:
        The global Rich `Console` singleton.
    """
    cached = globals().get("console")
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = globals().get("console")
        if cached is not None:
            return cached
        from rich.console import Console

        inst = Console(highlight=False)
        globals()["console"] = inst
        return inst


def _get_settings() -> Settings:
    """Return the lazily-initialized global `Settings` instance.

    Ensures bootstrap has run before constructing settings. The result is cached
    in `globals()["settings"]` so subsequent access — including
    `from config import settings` in other modules — resolves instantly.

    Returns:
        The global `Settings` singleton.
    """
    cached = globals().get("settings")
    if cached is not None:
        return cached
    with _singleton_lock:
        cached = globals().get("settings")
        if cached is not None:
            return cached
        _ensure_bootstrap()
        try:
            inst = Settings.from_environment(start_path=_bootstrap_state.start_path)
        except Exception:
            logger.exception(
                "Failed to initialize settings from environment (start_path=%s)",
                _bootstrap_state.start_path,
            )
            raise
        globals()["settings"] = inst
        return inst


def __getattr__(name: str) -> Settings | Console:
    """Lazy module attributes for `settings` and `console`.

    Defers heavy initialization until first access. Subsequent accesses hit
    the module-level attribute directly (no `__getattr__` overhead).

    Returns:
        The requested lazy singleton.

    Raises:
        AttributeError: If *name* is not a lazily-provided attribute.
    """
    if name == "settings":
        return _get_settings()
    if name == "console":
        return _get_console()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
