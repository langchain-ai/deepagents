"""Activation-gated browser middleware for Deep Agents."""

from deepagents_browser._version import __version__
from deepagents_browser.errors import (
    BrowserAccessError,
    BrowserError,
    BrowserErrorCode,
    BrowserRuntimeError,
    NetworkPolicyError,
)
from deepagents_browser.middleware import BrowserMiddleware
from deepagents_browser.runtime import BrowserLimits
from deepagents_browser.state import BrowserState

__all__ = [
    "BrowserAccessError",
    "BrowserError",
    "BrowserErrorCode",
    "BrowserLimits",
    "BrowserMiddleware",
    "BrowserRuntimeError",
    "BrowserState",
    "NetworkPolicyError",
    "__version__",
]
