"""Reloadable API keys, Google Cloud routing, and project root ownership.

`Credentials` is the holder; `CredentialsOwner` is the process-wide owner that
`get_credentials()` hands out lazily. All `Settings` reads for these values
moved here so `Settings` can shrink to an empty shell.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

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
"""Credential fields refreshed on `/reload` and cwd switches.

`shell.allow_list` and `skills.extra_allowed_dirs` left `Settings` when their
consumers moved to callsite resolution; the reload still refreshes the shared
resolver's tiers so those callsites see the new generation. Runtime model
state (`model_name`, `model_provider`, `model_context_limit`) and the original
user LangSmith project are intentionally excluded -- they are set once and
should not change across reloads.
"""

_RELOAD_REPORT_OPTIONS = ("shell.allow_list", "skills.extra_allowed_dirs")
"""Manifest keys the reload report still diffs after they left `Credentials`.

Their values are callsite-resolved now, so a `/reload` diff has to read the
shared resolver's current generation for the "before" side and the refreshed
generation for the "after" side.
"""

_API_KEY_FIELDS = frozenset(
    field for field in _RELOADABLE_FIELDS if field.endswith("_api_key")
)
"""Reloadable fields that hold API keys and must be masked in change reports.

Derived from `_RELOADABLE_FIELDS` so new `*_api_key` fields are picked up
automatically.
"""


@dataclass
class Credentials:
    """Reloadable API keys, Google Cloud routing, and project root."""

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


@dataclass
class CredentialsOwner:
    """Process-wide owner of the active credentials and reload state."""

    active: Credentials
    """The credentials in force."""

    user_langchain_project: str | None
    """Original `LANGSMITH_PROJECT` from environment (for user code).

    Set once at construction from the bootstrap snapshot; reloads never touch
    it.
    """

    _previewed_reload_before: dict[str, object] | None = dataclass_field(
        default=None, repr=False
    )
    """Manifest "before" snapshot a preview took for the reload that follows.

    The cwd-switch flow previews the target's settings and only then applies
    them; the applied report must diff against what the user accepted, not
    whatever the tiers resolve to by the time `reload_from_environment` runs
    (the preview itself already advanced the shared resolver's user tier).
    """

    _manifest_report_in_force: dict[str, object] | None = dataclass_field(
        default=None, repr=False
    )
    """Manifest values in force for the reload report's "before" side.

    `shell.allow_list` and `skills.extra_allowed_dirs` are callsite-resolved
    rather than stored here, but `/reload` still reports their changes. The
    environment can already carry the new value when a reload starts, so the
    diff reads the last *applied* generation, recorded at construction and
    after each unblocked reload.
    """

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> CredentialsOwner:
        """Detect the current environment and build the owner.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            CredentialsOwner with detected configuration
        """
        # Detect API keys (normalize empty strings to None).
        from deepagents_code.model_config import resolve_env_var

        openai_key = resolve_env_var("OPENAI_API_KEY")
        anthropic_key = resolve_env_var("ANTHROPIC_API_KEY")
        google_key = resolve_env_var("GOOGLE_API_KEY")
        nvidia_key = resolve_env_var("NVIDIA_API_KEY")
        tavily_key = resolve_env_var("TAVILY_API_KEY")
        google_cloud_project = resolve_env_var("GOOGLE_CLOUD_PROJECT")
        google_cloud_location = resolve_env_var("GOOGLE_CLOUD_LOCATION")

        # Detect LangSmith configuration
        # DEEPAGENTS_CODE_LANGSMITH_PROJECT: Project for deepagents agent tracing
        # user_langchain_project: User's ORIGINAL LANGSMITH_PROJECT (before override)
        # When accessed via `get_credentials()`, `_ensure_bootstrap()` has
        # already run and may have overridden LANGSMITH_PROJECT. We use the
        # saved original value, not the current os.environ value. Direct
        # callers should ensure bootstrap has run if they depend on the
        # override.
        from deepagents_code import config as _config
        from deepagents_code._env_vars import (
            LANGSMITH_PROJECT,
        )

        deepagents_langchain_project = resolve_env_var(LANGSMITH_PROJECT)
        # Use the saved original, not the current `LANGSMITH_PROJECT` that
        # bootstrap may have overridden for agent traces.
        user_langchain_project = _config._bootstrap_state.original_langsmith_project

        # Detect project
        from deepagents_code.project_utils import find_project_root

        project_root = find_project_root(start_path)

        # Resolve the manifest-backed report options once so the shared
        # resolver's snapshots are loaded here rather than lazily at the
        # first callsite read, and record them as the state in force for the
        # reload report's "before" side.
        manifest_in_force = dict(
            _config._resolve_manifest_options(
                _RELOAD_REPORT_OPTIONS, start_path=start_path
            )
        )

        instance = cls(
            active=Credentials(
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                google_api_key=google_key,
                nvidia_api_key=nvidia_key,
                tavily_api_key=tavily_key,
                google_cloud_project=google_cloud_project,
                google_cloud_location=google_cloud_location,
                deepagents_langchain_project=deepagents_langchain_project,
                project_root=project_root,
            ),
            user_langchain_project=user_langchain_project,
        )
        instance._manifest_report_in_force = manifest_in_force
        return instance

    # Convenience pass-throughs so `get_credentials().has_tavily`-style reads
    # keep working while consumers settle on `get_credentials().active`.
    @property
    def openai_api_key(self) -> str | None:
        """OpenAI API key if available."""
        return self.active.openai_api_key

    @property
    def anthropic_api_key(self) -> str | None:
        """Anthropic API key if available."""
        return self.active.anthropic_api_key

    @property
    def google_api_key(self) -> str | None:
        """Google API key if available."""
        return self.active.google_api_key

    @property
    def nvidia_api_key(self) -> str | None:
        """NVIDIA API key if available."""
        return self.active.nvidia_api_key

    @property
    def tavily_api_key(self) -> str | None:
        """Tavily API key if available."""
        return self.active.tavily_api_key

    @property
    def google_cloud_project(self) -> str | None:
        """Google Cloud project ID for VertexAI authentication."""
        return self.active.google_cloud_project

    @property
    def google_cloud_location(self) -> str | None:
        """Google Cloud region for Anthropic models on Vertex AI."""
        return self.active.google_cloud_location

    @property
    def deepagents_langchain_project(self) -> str | None:
        """LangSmith project name for deepagents agent tracing."""
        return self.active.deepagents_langchain_project

    @property
    def project_root(self) -> Path | None:
        """Current project root directory, or `None` if not in a git project."""
        return self.active.project_root

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
        """Check if VertexAI is available (Google Cloud project set, no API key)."""
        return self.active.has_vertex_ai

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.active.has_tavily

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
        from deepagents_code import config as _config
        from deepagents_code._env_vars import LANGSMITH_PROJECT
        from deepagents_code.configuration.service import (
            ManagedConfigError,
            get_healthy_managed_snapshot,
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
            # base used when the refreshed resolver applies it below. Without
            # this, a relative managed skill root can validate in the old cwd
            # and fail in the target cwd.
            with _config._use_extra_skills_path_base(start_path):
                managed_snapshot = get_healthy_managed_snapshot(refresh=refresh_managed)
        except ManagedConfigError as exc:
            _config.logger.error("Keeping previous settings: %s", exc)
            # Report the block to the caller. Returning only `previous` reads
            # as "nothing changed", so the user would be told the reload
            # succeeded while their environment edits were discarded.
            return dict(previous), f"{_config.MANAGED_RELOAD_BLOCKED_PREFIX}{exc}"

        from deepagents_code.configuration.resolver import (
            MANAGED_RANK,
            USER_RANK,
            get_config_resolver,
        )

        # A real `/reload` exists to pick up file edits made since the shared
        # resolver's snapshot was taken, so this method and later
        # `get_config_resolver()` readers observe the same generation. Seed the
        # resolver with the snapshot just validated above; asking it to refresh
        # managed policy again would let one reload observe multiple files.
        # Manifest-backed values (`shell.allow_list`, `skills.extra_allowed_dirs`,
        # `interpreter.*`) no longer live on `Settings`: their consumers resolve
        # them through this refreshed resolver, so advancing the generation here
        # is what applies their reloads.
        resolver = get_config_resolver(
            refresh_managed=refresh_managed,
            managed_snapshot=managed_snapshot,
        )
        if refresh_managed:
            # `get_config_resolver(refresh_managed=True)` replaces only the
            # managed provider; the user provider keeps the generation it was
            # built with until a reload propagates the refresh. Replacing
            # rather than plain-reloading keeps this generation to the single
            # managed file read above. Both providers here are loader-free:
            # a loader would re-read at the next reload, and the managed
            # loader (`_reload_enforceable_managed_snapshot`) would double
            # this reload's managed reads.
            from deepagents_code.configuration.providers import TomlFileProvider
            from deepagents_code.model_config import DEFAULT_CONFIG_PATH

            installed_user = resolver.toml_snapshot(USER_RANK)
            replacement_user = TomlFileProvider(
                name="config.toml",
                path=DEFAULT_CONFIG_PATH,
                snapshot=installed_user,
            )
            # Install through `reload_from_snapshot` so a file that fails to
            # parse keeps serving the previous generation (and records the
            # failure status) instead of being rejected by
            # `reload_with_replacements`' usable-snapshot gate.
            replacement_user.reload_from_snapshot(
                TomlFileProvider(name="config.toml", path=DEFAULT_CONFIG_PATH).load()
            )
            installed_managed = resolver.toml_snapshot(MANAGED_RANK)
            replacement_managed = TomlFileProvider(
                name=managed_snapshot.status.name,
                path=managed_snapshot.status.path,
                rank=MANAGED_RANK,
                durable=True,
                snapshot=installed_managed,
            )
            replacement_managed.reload_from_snapshot(managed_snapshot)
            resolver.reload_with_replacements(
                {
                    MANAGED_RANK: replacement_managed,
                    USER_RANK: replacement_user,
                }
            )

        # A user file that fails to parse keeps the previous generation in
        # force, which is the right runtime behavior but silent: the only
        # signal is a `logger.warning` in the debug buffer, while the report
        # the user reads says "Configuration reloaded. No changes detected."
        # Managed corruption is already surfaced as a notice above; a
        # `config.toml` the user just edited deserves the same treatment, and
        # more so -- they are staring at the edit that did not take.
        user_notice: str | None = None
        if refresh_managed:
            # The user provider was just replaced with a fresh read above;
            # its status is the read this reload performed. Reading it from
            # `provider_statuses()` instead would race nothing here, but the
            # replacement is the object this code just built, so keep the
            # reference local.
            user_status = replacement_user.status()
        else:
            user_status = resolver.provider_statuses().get(USER_RANK)
        if user_status is not None and not user_status.usable:
            detail = user_status.detail or user_status.health.value
            user_notice = f"Kept previous config.toml: {detail}"
            _config.logger.error("Keeping previous config.toml: %s", detail)

        try:
            from deepagents_code.project_utils import find_project_root

            project_root = find_project_root(start_path)
        except OSError:
            _config.logger.warning(
                "Could not detect project root during reload; keeping previous value"
            )
            project_root = previous["project_root"]

        refreshed = {
            "openai_api_key": _config._resolve_env_var_from(env, "OPENAI_API_KEY"),
            "anthropic_api_key": _config._resolve_env_var_from(
                env, "ANTHROPIC_API_KEY"
            ),
            "google_api_key": _config._resolve_env_var_from(env, "GOOGLE_API_KEY"),
            "nvidia_api_key": _config._resolve_env_var_from(env, "NVIDIA_API_KEY"),
            "tavily_api_key": _config._resolve_env_var_from(env, "TAVILY_API_KEY"),
            "google_cloud_project": _config._resolve_env_var_from(
                env, "GOOGLE_CLOUD_PROJECT"
            ),
            "google_cloud_location": _config._resolve_env_var_from(
                env, "GOOGLE_CLOUD_LOCATION"
            ),
            "deepagents_langchain_project": _config._resolve_env_var_from(
                env,
                LANGSMITH_PROJECT,
            ),
            "project_root": project_root,
        }

        # Manifest-backed values the report still previews even though no
        # Credentials field receives them. A real reload reads the shared
        # resolver just refreshed above; a preview reads an ad-hoc resolver
        # that substitutes the `env` mapping for the live environment tier --
        # the shared resolver's env provider reads `os.environ`, which would
        # report the value live in the process instead of the `.env` edit
        # being previewed.
        manifest_report = CredentialsOwner._resolve_report_options(
            resolver, start_path=start_path
        )
        if not refresh_managed:
            from deepagents_code.configuration.providers import TomlFileProvider
            from deepagents_code.model_config import DEFAULT_CONFIG_PATH

            user_candidate = TomlFileProvider("config.toml", DEFAULT_CONFIG_PATH).load()
            if user_candidate.status.usable:
                preview_resolver = _config._preview_resolver(
                    managed_snapshot=managed_snapshot,
                    user_snapshot=user_candidate,
                    env=env,
                )
                user_notice = None
                manifest_report = CredentialsOwner._resolve_report_options(
                    preview_resolver, start_path=start_path
                )
            else:
                # The preview's own read failed, so report that rather than
                # the shared resolver's status: the file on disk right now is
                # what the user would be accepting.
                detail = (
                    user_candidate.status.detail or user_candidate.status.health.value
                )
                user_notice = f"Kept previous config.toml: {detail}"

        refreshed.update(manifest_report)
        return refreshed, user_notice

    @staticmethod
    def _resolve_report_options(
        resolver: Any,  # noqa: ANN401  # `ConfigResolver`; kept lazy, see below
        *,
        start_path: Path | None = None,
    ) -> dict[str, object]:
        """Resolve the manifest options the reload report still covers.

        These values no longer live on `Credentials`; they are read here purely
        so `/reload` and its preview keep reporting the tier changes a user
        just made to `[shell]` or `[skills]`.

        Args:
            resolver: Resolver generation to read. A preview passes an ad-hoc
                resolver bound to the `.env` mapping being previewed; a real
                reload passes the shared resolver it just refreshed.
            start_path: Project base for relative skill-root resolution.

        Returns:
            Manifest-keyed values for the report.
        """
        from deepagents_code import config as _config
        from deepagents_code.config_manifest import (
            _emit_ranked_diagnostics,
            get_option,
        )

        values: dict[str, object] = {}
        with _config._use_extra_skills_path_base(start_path):
            for key in ("shell.allow_list", "skills.extra_allowed_dirs"):
                option = get_option(key)
                if option is None:
                    continue
                resolved = resolver.get(option)
                _emit_ranked_diagnostics(option, resolved)
                values[key] = resolved.value
        return values

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
        for field in _RELOAD_REPORT_OPTIONS:
            if field not in refreshed or field not in previous:
                continue
            old_value = previous[field]
            new_value = refreshed[field]
            if old_value != new_value:
                changes.append(f"{field}: {old_value} -> {new_value}")
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
        from deepagents_code import config as _config

        previous = {field: getattr(self.active, field) for field in _RELOADABLE_FIELDS}
        # The "before" side of the report must be captured before
        # `_reload_values` touches the shared resolver: an applied reload
        # advances the generation it reads, and the read would otherwise
        # report the after state on both sides of the arrow. A preview changes
        # nothing, but the accepted reload that follows must diff against the
        # state the user was shown, so the snapshot is kept for it.
        previous.update(_config._resolve_manifest_options(_RELOAD_REPORT_OPTIONS))
        env = _config._preview_dotenv_environ(start_path=start_path)
        refreshed, blocked = self._reload_values(
            start_path=start_path,
            env=env,
            previous=previous,
            refresh_managed=False,
        )
        self._previewed_reload_before = previous
        changes = self._format_reload_changes(previous, refreshed)
        return [blocked, *changes] if blocked else changes

    def reload_from_environment(self, *, start_path: Path | None = None) -> list[str]:
        """Reload selected settings from environment variables and project files.

        This refreshes only fields that are expected to change at runtime
        (API keys, Google Cloud project, project root, and the LangSmith
        tracing project) and advances the shared config resolver's generation
        so callsite-resolved values (`shell.allow_list`,
        `skills.extra_allowed_dirs`, `interpreter.*`) observe the reload.

        Runtime model state (`model_name`, `model_provider`,
        `model_context_limit`) and the original user LangSmith project
        (`user_langchain_project`) are intentionally preserved -- they are
        not in `_RELOADABLE_FIELDS` and are never touched by this method.

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
            changed; a single notice when managed policy blocked the reload.
        """
        from deepagents_code import config as _config

        previous = {field: getattr(self.active, field) for field in _RELOADABLE_FIELDS}
        # The report's "before" side for manifest-backed values is the state
        # in force since the last application (tracked on the instance): the
        # environment may already carry the new value by the time this method
        # runs, so re-resolving here would report the after state on both
        # sides of the arrow. When a preview ran immediately before (the
        # cwd-switch flow), its snapshot is the state the user accepted, and
        # wins.
        manifest_before = self._previewed_reload_before
        self._previewed_reload_before = None
        if manifest_before is None:
            manifest_before = self._manifest_report_in_force
            if manifest_before is None:
                manifest_before = dict(
                    _config._resolve_manifest_options(_RELOAD_REPORT_OPTIONS)
                )
        previous.update(manifest_before)

        _config._load_dotenv(start_path=start_path, refresh_loaded=True)
        refreshed, blocked = self._reload_values(
            start_path=start_path,
            env=dict(os.environ),
            previous=previous,
        )

        if blocked is None:
            self._manifest_report_in_force = {
                key: refreshed.get(key) for key in _RELOAD_REPORT_OPTIONS
            }
        # Atomic swap: build the replacement holder first so a failure leaves
        # the previous credentials in force for every reader.
        self.active = Credentials(
            openai_api_key=cast("str | None", refreshed["openai_api_key"]),
            anthropic_api_key=cast("str | None", refreshed["anthropic_api_key"]),
            google_api_key=cast("str | None", refreshed["google_api_key"]),
            nvidia_api_key=cast("str | None", refreshed["nvidia_api_key"]),
            tavily_api_key=cast("str | None", refreshed["tavily_api_key"]),
            google_cloud_project=cast("str | None", refreshed["google_cloud_project"]),
            google_cloud_location=cast(
                "str | None", refreshed["google_cloud_location"]
            ),
            deepagents_langchain_project=cast(
                "str | None", refreshed["deepagents_langchain_project"]
            ),
            project_root=cast("Path | None", refreshed["project_root"]),
        )

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
            if _config._bootstrap_state.original_langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = (
                    _config._bootstrap_state.original_langsmith_project
                )
            else:
                os.environ.pop("LANGSMITH_PROJECT", None)
                _config._apply_default_langsmith_project()

        # A reload can repoint env resolution at a different .env (e.g. after a
        # cwd switch), so start a fresh diagnostics generation; otherwise the new
        # "Resolved X from ..." lines would be suppressed by the pre-reload dedup
        # set.
        from deepagents_code.model_config import reset_env_resolution_log

        reset_env_resolution_log()
        changes = self._format_reload_changes(previous, refreshed)
        return [blocked, *changes] if blocked else changes


_credentials_owner: CredentialsOwner | None = None
_credentials_lock = threading.Lock()


def get_credentials() -> CredentialsOwner:
    """Return the lazily-initialized process-wide credentials owner.

    Ensures bootstrap has run before building credentials. The result is
    cached so all readers observe the same reloadable state.

    Returns:
        The global `CredentialsOwner` singleton.
    """
    global _credentials_owner  # noqa: PLW0603
    if _credentials_owner is not None:
        return _credentials_owner
    with _credentials_lock:
        if _credentials_owner is not None:
            return _credentials_owner
        from deepagents_code import config as _config

        _config._ensure_bootstrap()
        try:
            _credentials_owner = CredentialsOwner.from_environment(
                start_path=_config._bootstrap_state.start_path
            )
        except Exception:
            _config.logger.exception(
                "Failed to initialize credentials from environment (start_path=%s)",
                _config._bootstrap_state.start_path,
            )
            raise
        return _credentials_owner
