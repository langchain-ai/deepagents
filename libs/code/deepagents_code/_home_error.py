"""The `DEEPAGENTS_HOME` resolution error, isolated from path capture.

`_paths` resolves `DEEPAGENTS_HOME` at import time, so a failure there raises
while `_paths` itself is still initializing. Any handler that wants to catch it
therefore cannot import it from `_paths`. This module holds nothing but the
exception class and runs no code at import, so `__init__` can import it, catch
the error by type, and print a friendly message instead of a traceback.

Keep this module free of imports and side effects.
"""

from __future__ import annotations


class DeepAgentsHomeError(ValueError):
    """Raised when the launch-time profile location cannot be resolved.

    Covers an unsupported `DEEPAGENTS_HOME` value and a launch home that
    cannot be determined or is itself unusable. It is a `ValueError` subclass
    so existing handlers that expect a bad-value error keep working.
    """
