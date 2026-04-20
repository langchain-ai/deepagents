"""Project subagent runs as first-class child streams on a parent run.

`SubagentTransformer` is a thin overlay on top of `SubgraphTransformer`
— it does *not* run a second discovery or a second mini-mux pipeline.
When `SubgraphTransformer` fires a `started` event and creates a
`SubgraphRunStream` at some namespace, `SubagentTransformer` checks
that event's `graph_name` against the set of declared subagent
names; on a match, it wraps the existing `SubgraphRunStream` in a
`SubagentRunStream` and pushes that onto `run.subagents`.

Consequences of sharing the `SubgraphRunStream`'s mini-mux:

- Events under the subagent's namespace are dispatched *once* (into
  the shared mini-mux), not twice.
- `path`, `status`, `error`, `checkpoint`, and `output` are read from
  the wrapped handle — no separate tracking in this transformer.
- `finalize` / `fail` are not needed: `SubgraphTransformer` owns
  close/fail of the shared mini-mux.

The `_native = True` flag means `run.subagents` auto-binds as a
direct attribute. A subagent also shows up on `run.subgraphs` — that
projection is a superset, surfacing every nested subgraph (subagent
or otherwise) as an untyped handle.

The discovery signal is the `started` lifecycle event's `graph_name`,
not the namespace leaf segment. When the `task` tool invokes
`subagent.ainvoke(...)` from inside a tool node, langgraph namespaces
the child run as `("tools:<pregel_task_id>", ...)`, so the leaf is
`tools:…`, not the subagent's declared name. `graph_name`, however,
is populated by langgraph from the compiled subagent's `name`
attribute — which `deepagents` sets via
`create_agent(..., name=spec["name"])`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import BaseRunStream
from langgraph.stream.transformers import SubgraphRunStream, SubgraphTransformer

if TYPE_CHECKING:
    from langchain_protocol.protocol import CheckpointRef, LifecycleData
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent
    from langgraph.stream.transformers import SubgraphStatus


class SubagentRunStream(BaseRunStream):
    """Typed view of a single subagent execution.

    Wraps the `SubgraphRunStream` that `SubgraphTransformer` already
    built for this child namespace — shares its mini-mux, so
    projections (`.messages`, `.tool_calls`, `.middleware`,
    recursive `.subagents`, `.values`) come from the existing
    transformer pipeline without a second round of event dispatch.

    Events are driven by the parent run's pump and routed into this
    handle's mini-mux by `SubgraphTransformer` — a subagent does not
    own its own pump. Construction therefore bypasses
    `GraphRunStream.__init__` (which would wire a pump onto the
    mini-mux, shadowing the parent's) and calls
    `BaseRunStream.__init__` directly, seeding the pump-related
    fields so inherited methods that consult them (`output`, `abort`,
    `__exit__`) degrade to no-ops.

    Exposes `name` (the declared subagent name) as the typed state
    this layer adds. `path`, `status`, `error`, `checkpoint`,
    `trigger_call_id` delegate to the wrapped subgraph handle so
    subgraph-side state transitions — including terminal close
    driven by `SubgraphTransformer.finalize` / `fail` — are
    automatically visible here.
    """

    def __init__(self, subgraph: SubgraphRunStream, *, name: str) -> None:
        # Skip `GraphRunStream.__init__` to avoid calling
        # `mux.bind_pump` on the mini-mux — the parent's pump is
        # already bound via `make_child` pump inheritance.
        BaseRunStream.__init__(self, subgraph._mux)
        # Seed pump-related fields so inherited `output` / `abort` /
        # `__exit__` treat this stream as already-exhausted (no self
        # pump to drive). `_values_transformer._latest` still provides
        # the current snapshot because the parent's pump keeps it
        # fresh.
        self._graph_iter = None
        self._exhausted = True
        self._subgraph = subgraph
        self.name: str = name

    @property
    def output(self) -> dict[str, Any] | None:
        """Latest values snapshot from this subagent's mini-mux.

        Unlike root `GraphRunStream.output`, no pump is driven — the
        parent's pump keeps `_values_transformer._latest` fresh.
        """
        vt = self._values_transformer
        err = vt.error
        if err is not None:
            raise err
        return vt._latest

    @property
    def interrupted(self) -> bool:
        """True once an interrupt has flowed through this subagent's pipeline."""
        return self._values_transformer._interrupted

    @property
    def interrupts(self) -> list[Any]:
        """List of interrupt payloads seen by this subagent."""
        return list(self._values_transformer._interrupts)

    @property
    def path(self) -> tuple[str, ...]:
        return self._subgraph.path

    @property
    def trigger_call_id(self) -> str | None:
        return self._subgraph.trigger_call_id

    @property
    def status(self) -> SubgraphStatus:
        return self._subgraph.status

    @property
    def error(self) -> str | None:
        return self._subgraph.error

    @property
    def checkpoint(self) -> CheckpointRef | None:
        return self._subgraph.checkpoint


class SubagentTransformer(StreamTransformer):
    """Promote declared subagents into typed handles on `run.subagents`.

    Requires a `SubgraphTransformer` to be registered earlier in the
    mux's factory list (it is, via `GraphStreamer.builtin_factories`).
    This transformer does not own a mini-mux or a lifecycle state
    machine — it consults `SubgraphTransformer._by_ns` on each
    `started` event and promotes matching handles.

    `scope_exact = False` so the `started` event at the *child's*
    namespace (ns = scope + 1 segment) reaches this transformer's
    `process`. Events at deeper namespaces are ignored — we only
    need the top-of-subagent lifecycle boundary; state inside the
    subagent is handled by its mini-mux transformers.
    """

    _native: ClassVar[bool] = True
    scope_exact: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ("lifecycle",)

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        *,
        subagent_names: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(scope)
        self._names = subagent_names
        self._log: EventLog[SubagentRunStream] = EventLog()
        self._by_ns: dict[tuple[str, ...], SubagentRunStream] = {}
        self._subgraph_transformer: SubgraphTransformer | None = None

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        """Capture the sibling `SubgraphTransformer` we reuse handles from.

        Raises at registration time if `SubgraphTransformer` isn't
        present — failing loudly here is better than silently yielding
        zero subagent handles at runtime.
        """
        sub_t = mux.transformer_by_key("subgraphs")
        if not isinstance(sub_t, SubgraphTransformer):
            msg = (
                "SubagentTransformer requires a SubgraphTransformer to be "
                "registered earlier in the mux; none was found. Check that "
                "your streamer's builtin_factories includes SubgraphTransformer "
                "before SubagentTransformer."
            )
            raise TypeError(msg)
        self._subgraph_transformer = sub_t

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "lifecycle":
            return True

        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        # Direct children of this scope only — matches
        # SubgraphTransformer's own discovery depth.
        if len(ns) != depth + 1 or ns[:-1] != self.scope:
            return True

        data = cast("LifecycleData", event["params"]["data"])
        if data.get("event") != "started":
            return True

        graph_name = data.get("graph_name")
        if graph_name not in self._names or ns in self._by_ns:
            return True

        # SubgraphTransformer runs before us (factory order), so its
        # `_by_ns` already has the freshly-created handle for this
        # namespace. Reuse it — no second mini-mux, no duplicate
        # dispatch. `_on_register` guarantees `_subgraph_transformer`
        # is set before any event reaches `process`.
        sub_t = cast("SubgraphTransformer", self._subgraph_transformer)
        subgraph_handle = sub_t._by_ns.get(ns)
        if subgraph_handle is None:
            return True

        handle = SubagentRunStream(subgraph_handle, name=cast("str", graph_name))
        self._by_ns[ns] = handle
        self._log.push(handle)
        return True
