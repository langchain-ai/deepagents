"""LangSmith sandboxes authenticated with their own key.

Harbor's LangSmith environment falls back to `LANGSMITH_API_KEY`, which the eval
workflow also uses for tracing and experiments. Reading the sandbox key here, rather
than passing it with `--ek`, keeps it out of the job config Harbor writes to disk.
"""

from __future__ import annotations

import os
from typing import Any

from harbor.environments.langsmith import LangSmithEnvironment

SANDBOX_KEY_ENV = "LANGSMITH_SANDBOX_API_KEY"


def sandbox_api_key() -> str | None:
    """The sandbox key, or `None` to fall back to Harbor's `LANGSMITH_API_KEY` default."""
    return os.environ.get(SANDBOX_KEY_ENV) or None


class SandboxKeyLangSmithEnvironment(LangSmithEnvironment):
    """Harbor's LangSmith environment, using `LANGSMITH_SANDBOX_API_KEY` when set."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Default `api_key` to the sandbox key, keeping any explicit one."""
        kwargs.setdefault("api_key", sandbox_api_key())
        super().__init__(*args, **kwargs)
