"""Built-in Google GenAI provider profile and helpers.

Enforces the minimum `langchain-google-genai` version before model
construction. Users may layer additional kwargs on top via
`register_provider_profile("google_genai", ...)`.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version as pkg_version

from packaging.version import InvalidVersion, Version

from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    register_provider_profile,
)

logger = logging.getLogger(__name__)

GOOGLE_GENAI_MIN_VERSION = "4.2.1"
"""Minimum required version of `langchain-google-genai`.

Used to enforce a consistent version floor at runtime.
"""


def check_google_genai_version() -> None:
    """Raise if the installed `langchain-google-genai` is below the minimum.

    If the package is not installed at all the check is skipped;
    `init_chat_model` will surface its own missing-dependency error downstream.

    Raises:
        ImportError: If the installed version is too old.
    """
    try:
        installed = pkg_version("langchain-google-genai")
    except PackageNotFoundError:
        return
    try:
        is_old = Version(installed) < Version(GOOGLE_GENAI_MIN_VERSION)
    except InvalidVersion:
        # Non-PEP-440 version (dev build, fork, etc.) — skip the check but
        # leave a breadcrumb so an unexpected downstream failure is traceable.
        logger.warning(
            "Skipping langchain-google-genai version check: installed version %r is not PEP 440. Minimum required is %s.",
            installed,
            GOOGLE_GENAI_MIN_VERSION,
        )
        return
    if is_old:
        msg = (
            f"deepagents requires langchain-google-genai>={GOOGLE_GENAI_MIN_VERSION}, "
            f"but {installed} is installed. "
            f"Run: pip install 'langchain-google-genai>={GOOGLE_GENAI_MIN_VERSION}'"
        )
        raise ImportError(msg)


def register() -> None:
    """Register the built-in Google GenAI provider profile."""
    register_provider_profile(
        "google_genai",
        ProviderProfile(pre_init=lambda _spec: check_google_genai_version()),
    )
