"""Scoped QuickJS REPL sources for declarative subagent forks."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_quickjs._repl import _ThreadREPL


_fork_repl_sources: dict[str, list[_ThreadREPL]] = defaultdict(list)
_fork_repl_sources_lock = RLock()


def get_fork_repl_source(thread_id: object) -> _ThreadREPL | None:
    """Return the parent REPL registered for a nested task invocation."""
    if not isinstance(thread_id, str):
        return None
    with _fork_repl_sources_lock:
        sources = _fork_repl_sources.get(thread_id)
        return sources[-1] if sources else None


@contextmanager
def fork_repl_source(thread_id: object, source: _ThreadREPL) -> Iterator[None]:
    """Register a parent REPL while nested task invocations are active."""
    if not isinstance(thread_id, str):
        yield
        return
    with _fork_repl_sources_lock:
        _fork_repl_sources[thread_id].append(source)
    try:
        yield
    finally:
        with _fork_repl_sources_lock:
            sources = _fork_repl_sources.get(thread_id)
            if sources:
                if sources[-1] is source:
                    sources.pop()
                else:
                    sources.remove(source)
                if not sources:
                    del _fork_repl_sources[thread_id]
