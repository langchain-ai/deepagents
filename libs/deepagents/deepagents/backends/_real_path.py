"""Internal hook: resolves a path to the target a backend would actually touch.

Backends that can alias paths (e.g. via symlinks) expose it so filesystem
permissions can be checked on the resolved target; backends without it are
checked on the requested path only. Not part of the public backend protocol.
"""

from collections.abc import Callable

_REAL_PATH_ATTR = "_deepagents_real_path"

RealPathFn = Callable[[str], str]


def get_real_path(backend: object) -> RealPathFn | None:
    """Return the backend's real-path hook, if it has one."""
    fn = getattr(backend, _REAL_PATH_ATTR, None)
    return fn if callable(fn) else None
