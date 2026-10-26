"""Run untraced `@traceable` bookkeeping inline on the server loop.

LangChain wraps every middleware `awrap_tool_call` and `awrap_model_call` hook
in `langsmith.traceable`. Its async wrapper sends two calls to a worker thread
per hook: one to set up the run and one to end it. It does this even when
tracing is off and both calls only build an empty container. About 20 traced
hooks run per tool round trip, so each round trip pays for about 40 thread
hops that do no work.

`install_untraced_inline_override` installs a LangSmith runtime override. The
override runs those two calls inline when they cannot trace, and gives every
other call to LangSmith's default thread hop.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Awaitable, Callable

    DefaultAioToThread = Callable[..., Awaitable[Any]]

logger = logging.getLogger(__name__)


def _target(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the function a `functools.partial` wraps, else `func` itself.

    Returns:
        The underlying callable.
    """
    return func.func if isinstance(func, functools.partial) else func


def _runs_inline(
    ctx: contextvars.Context,
    func: Callable[..., Any],
    args: tuple[Any, ...],
) -> bool:
    """Whether a `traceable` bookkeeping call cannot trace, so it is cheap.

    Only run setup and run end qualify. Any other call (for example an async
    `Client` API method) can do network I/O, so it keeps the thread hop.

    Returns:
        `True` when the call is safe to run on the event loop.
    """
    from langsmith import run_helpers
    from langsmith.utils import tracing_is_enabled

    target = _target(func)
    if target is getattr(run_helpers, "_setup_run", None):
        return not ctx.run(tracing_is_enabled)
    if target is getattr(run_helpers, "_handle_container_end", None):
        container = args[0] if args else None
        return isinstance(container, dict) and container.get("new_run") is None
    return False


async def _untraced_inline_aio_to_thread(
    default: DefaultAioToThread,
    ctx: contextvars.Context,
    func: Callable[..., Any],
    /,
    *args: Any,  # forwards arbitrary LangSmith call arguments
    **kwargs: Any,  # forwards arbitrary LangSmith call arguments
) -> Any:  # noqa: ANN401  # returns whatever `func` returns
    """Run untraced bookkeeping inline; hop to a thread for everything else.

    Returns:
        The result of `func`.
    """
    if _runs_inline(ctx, func, args):
        return ctx.run(func, *args, **kwargs)
    return await default(ctx, func, *args, **kwargs)


def install_untraced_inline_override() -> None:
    """Install the override for this process.

    The override is process-global. Call it only in the server process, which
    owns its event loop and pins one tracing configuration for its lifetime.
    """
    import langsmith

    langsmith.set_runtime_overrides(aio_to_thread=_untraced_inline_aio_to_thread)
    logger.debug("Installed inline LangSmith runtime override for untraced runs")
