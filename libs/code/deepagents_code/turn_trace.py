"""Retrospective metadata updates for the current turn's traced runs."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.runnables import RunnableConfig
    from langsmith.run_trees import RunTree

logger = logging.getLogger(__name__)
_MAX_THREADS = 128
_turns: OrderedDict[str, tuple[str, TurnTrace]] = OrderedDict()
_lock = Lock()


@dataclass
class TurnTrace:
    """Keep sibling invocation roots alive until the thread starts another turn."""

    roots: dict[str, RunTree] = field(default_factory=dict)

    def update(
        self, metadata: Mapping[str, object], *, remove: Sequence[str] = ()
    ) -> None:
        """Merge metadata without changing run payloads or active run timing."""
        pending = list(self.roots.values())
        while pending:
            run = pending.pop()
            pending.extend(run.child_runs)
            if run.extra.get("__disabled") or run.ls_client is None:
                continue
            try:
                merged = {k: v for k, v in run.metadata.items() if k not in remove}
                merged.update(metadata)
                run.extra = {**run.extra, "metadata": merged}
                if run.end_time is not None:
                    _patch_metadata(run)
            except Exception:
                logger.debug("Turn trace metadata update failed", exc_info=True)


def _patch_metadata(run: RunTree) -> None:
    if run.replicas:
        _patch_replica_metadata(run)
        return
    run.client.update_run(
        run.id,
        extra={**run.extra, "metadata": run.metadata.copy()},
        end_time=run.end_time,
        trace_id=run.trace_id,
        dotted_order=run.dotted_order,
        session_id=run.session_id,
        session_name=run.session_name,
        address=run.address,
    )


def _patch_replica_metadata(run: RunTree) -> None:
    from langsmith import run_trees

    for group in run._replica_groups():
        client, project, updates, primary, address = group.key
        replica = run._remap_for_project(
            project, updates, primary=primary, address=address
        )
        client.update_run(
            replica["id"],
            extra=replica["extra"],
            end_time=replica["end_time"],
            trace_id=replica["trace_id"],
            dotted_order=replica["dotted_order"],
            session_name=replica.get("session_name"),
            address=replica.get("address"),
            _replica_auths=[
                run_trees._extract_replica_auth(member) for member in group.members
            ],
        )


def _current_root(config: RunnableConfig) -> RunTree | None:
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import get_tracing_context

    if get_tracing_context().get("enabled") is False:
        return None
    callbacks = config.get("callbacks")
    parent_id = getattr(callbacks, "parent_run_id", None)
    for handler in getattr(callbacks, "handlers", ()):
        if isinstance(handler, LangChainTracer):
            run = handler.run_map.get(str(parent_id))
            if run is not None:
                while parent := handler.run_map.get(str(run.parent_run_id)):
                    if any(
                        parent.metadata.get(key) != config.get("metadata", {}).get(key)
                        for key in ("thread_id", "turn_id")
                    ):
                        break
                    run = parent
                return run
    return None


def capture_turn(config: RunnableConfig) -> TurnTrace | None:
    """Return current-turn roots, or `None` when no scoped trace is active."""
    root = _current_root(config)
    if root is None or root.extra.get("__disabled") or root.ls_client is None:
        return None
    metadata = config.get("metadata", {})
    thread = metadata.get("thread_id") or config.get("configurable", {}).get(
        "thread_id"
    )
    turn = metadata.get("turn_id")
    if (
        not isinstance(thread, str)
        or not thread
        or not isinstance(turn, str)
        or not turn
    ):
        return None
    with _lock:
        previous = _turns.get(thread)
        trace = previous[1] if previous and previous[0] == turn else TurnTrace()
        trace.roots[str(root.id)] = root
        _turns[thread] = (turn, trace)
        _turns.move_to_end(thread)
        while len(_turns) > _MAX_THREADS:
            _turns.popitem(last=False)
    return trace
