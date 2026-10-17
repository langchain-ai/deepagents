"""Built-in OpenRouter provider profile and helpers.

Enforces the minimum `langchain-openrouter` version and injects default
app-attribution headers when the corresponding environment variables are not
set. Users may layer additional kwargs on top via
`register_provider_profile("openrouter", ...)`.

Registered directly by `_ensure_builtin_profiles_loaded` during the
first profile-registry access. Not exposed as an
`importlib.metadata` entry point — built-ins ship with the SDK and
should not depend on install-time metadata to activate.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import Any

from packaging.version import InvalidVersion, Version

from deepagents.profiles.provider.provider_profiles import ProviderProfile, _register_provider_profile_impl

logger = logging.getLogger(__name__)

OPENROUTER_MIN_VERSION = "0.2.0"  # app attribution support added
"""Minimum required version of `langchain-openrouter`.

Used to enforce a consistent version floor at runtime.
"""

_OPENROUTER_APP_URL = "https://github.com/langchain-ai/deepagents"
"""Default `app_url` (maps to `HTTP-Referer`) for OpenRouter attribution.

See https://openrouter.ai/docs/app-attribution for details.
"""

_OPENROUTER_APP_TITLE = "Deep Agents"
"""Default `app_title` (maps to `X-Title`) for OpenRouter attribution."""

_OPENROUTER_ALLOW_AZURE_ENV = "DEEPAGENTS_OPENROUTER_ALLOW_AZURE"
"""Env var that opts back into Azure as an OpenRouter upstream provider.

By default the SDK injects `openrouter_provider={"ignore": ["azure"]}` to keep
OpenRouter from routing reasoning-model calls through Azure. Azure's Responses
API support is fine in isolation, but OpenRouter's `/responses` beta is
documented stateless — it does not propagate `store=true` or `previous_response_id`
plumbing — so any prior `rs_*` reasoning item the client replays cannot be looked
up upstream and the request fails with `"Item with id 'rs_...' not found"`.
Set this var to a truthy value (`1`, `true`, `yes`, `on`) to allow Azure back
into the provider pool, accepting the multi-turn reasoning breakage as the
trade-off.
"""


def _openrouter_attribution_kwargs() -> dict[str, Any]:
    """Build default OpenRouter kwargs, deferring to env var overrides.

    `ChatOpenRouter` reads `OPENROUTER_APP_URL` and `OPENROUTER_APP_TITLE` via
    `from_env()` defaults. Explicit kwargs passed to the constructor take
    precedence over those env-var defaults, so we only inject our SDK defaults
    when the corresponding env var is **not** set — otherwise the user's env var
    would be overridden.

    An explicitly empty string (`OPENROUTER_APP_URL=""`) is treated as "set"
    and suppresses the SDK default. This lets a caller opt out of app
    attribution without unsetting the variable.

    Also injects `openrouter_provider={"ignore": ["azure"]}` so OpenRouter never
    routes to Azure as the upstream provider. Opt out by setting
    `DEEPAGENTS_OPENROUTER_ALLOW_AZURE` to a truthy value. The ignore is a
    no-op for non-OpenAI models (Azure is not a candidate provider for them),
    so the rule is safe to apply globally.

    Returns:
        Dictionary of kwargs to spread into `init_chat_model`.
    """
    kwargs: dict[str, Any] = {}
    if os.environ.get("OPENROUTER_APP_URL") is None:
        kwargs["app_url"] = _OPENROUTER_APP_URL
    if os.environ.get("OPENROUTER_APP_TITLE") is None:
        kwargs["app_title"] = _OPENROUTER_APP_TITLE
    if os.environ.get(_OPENROUTER_ALLOW_AZURE_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        kwargs["openrouter_provider"] = {"ignore": ["azure"]}
    return kwargs


def check_openrouter_version() -> None:
    """Raise if the installed `langchain-openrouter` is below the minimum.

    If the package is not installed at all the check is skipped;
    `init_chat_model` will surface its own missing-dependency error downstream.

    Raises:
        ImportError: If the installed version is too old.
    """
    try:
        installed = pkg_version("langchain-openrouter")
    except PackageNotFoundError:
        return
    try:
        is_old = Version(installed) < Version(OPENROUTER_MIN_VERSION)
    except InvalidVersion:
        # Non-PEP-440 version (dev build, fork, etc.) — skip the check but
        # leave a breadcrumb so an unexpected downstream failure is traceable.
        logger.warning(
            "Skipping langchain-openrouter version check: installed version %r is not PEP 440. Minimum required is %s.",
            installed,
            OPENROUTER_MIN_VERSION,
        )
        return
    if is_old:
        msg = (
            f"deepagents requires langchain-openrouter>={OPENROUTER_MIN_VERSION}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-openrouter>={OPENROUTER_MIN_VERSION}'"
        )
        raise ImportError(msg)


def register() -> None:
    """Register the built-in OpenRouter provider profile."""
    _register_provider_profile_impl(
        "openrouter",
        ProviderProfile(
            pre_init=lambda _spec: check_openrouter_version(),
            init_kwargs_factory=_openrouter_attribution_kwargs,
        ),
    )
