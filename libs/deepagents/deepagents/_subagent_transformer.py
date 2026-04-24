"""Project subagent runs as first-class child streams on a parent run.

`SubagentTransformer` performs three tightly-coupled jobs on events
flowing through the stream mux:

1. **Tool-call tracking.** Root-scope ``tools`` events from the
   parent's ``task`` tool calls are observed to build
   ``_pending_tool_calls``, keyed by ``tool_call_id``. Entries live
   from the originating ``tool-started`` until the subagent's
   ``lifecycle.started`` is correlated or the task errors out.

2. **Subagent handle promotion.** When a ``lifecycle.started`` fires
   at one level below this transformer's scope, we consult the
   sibling :class:`~langgraph.stream.transformers.SubgraphTransformer`'s
   ``_by_ns`` (populated earlier in the same dispatch because the
   subgraph transformer is registered first in the factory list).
   If the emitted ``graph_name`` matches a declared subagent name, we
   wrap the existing :class:`~langgraph.stream.transformers.SubgraphRunStream`
   in a :class:`SubagentRunStream` so ``run.subagents`` surfaces a
   typed handle with the standard projections
   (``.messages`` / ``.values`` / ``.subagents`` / ``.tool_calls``).

3. **Wire-level namespace rewrite.** Pregel schedules each subagent
   spawned via the ``task`` tool under a UUID-derived namespace like
   ``tools:<pregel_uuid>``, whereas wire subscribers filter on the
   client-visible ``tools:<tool_call_id>`` namespace. Rather than
   mutating ``checkpoint_ns`` in the subagent's config (which would
   interact with the Pregel scheduler and checkpoint layout) or
   fabricating synthetic protocol events, the transformer records a
   ``pregel_segment → tool_call_id`` mapping at the moment the first
   lifecycle event correlates to a pending ``task`` tool call, then
   rewrites the first segment of every downstream event's namespace
   in-place before the mux appends it to the main log. The
   SubgraphTransformer's ``_by_ns`` still keys off the pre-rewrite
   namespace (matching Pregel's checkpoint layout); only the
   wire-visible ``params.namespace`` is remapped.

Consequences of sharing the `SubgraphRunStream`'s mini-mux:

- Events under the subagent's namespace are dispatched *once* (into
  the shared mini-mux), not twice.
- `path`, `status`, `error`, `checkpoint`, and `cause` are read from
  the wrapped handle — no separate tracking in this transformer.
- `finalize` / `fail` are not needed: `SubgraphTransformer` owns
  close/fail of the shared mini-mux.

The `_native = True` flag means `run.subagents` auto-binds as a
direct attribute. A subagent also shows up on `run.subgraphs` — that
projection is a superset, surfacing every nested subgraph (subagent
or otherwise) as an untyped handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import BaseRunStream
from langgraph.stream.transformers import SubgraphRunStream, SubgraphTransformer

if TYPE_CHECKING:
    from langchain_protocol.protocol import (
        CheckpointRef,
        LifecycleCause,
        LifecycleData,
    )
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
    this layer adds. `path`, `status`, `error`, `checkpoint`, `cause`
    delegate to the wrapped subgraph handle so subgraph-side state
    transitions — including terminal close driven by
    `SubgraphTransformer.finalize` / `fail` — are automatically
    visible here.
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
    def cause(self) -> LifecycleCause | None:
        return self._subgraph.cause

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

    `scope_exact = False` so this transformer sees root-scope `tools`
    events (for tool-call tracking) *and* the promoted subagent's
    `lifecycle.started` at `scope + 1` namespace (for handle
    promotion) *and* every deeper event (for namespace rewriting).
    Events at deeper namespaces flow through `SubgraphTransformer`
    into the subagent's mini-mux where the standard `messages` /
    `values` transformers build projections.
    """

    _native: ClassVar[bool] = True
    scope_exact: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ("lifecycle", "tools")

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
        self._mux: StreamMux | None = None
        # Pending `task` tool calls awaiting a matching `lifecycle.started`.
        # Keyed by `tool_call_id`; value carries the originating input so
        # downstream consumers can correlate richer metadata than `cause`
        # alone conveys.
        #
        # Dict insertion order provides oldest-first pairing under
        # parallel fan-out of the same `subagent_type`, which matches
        # Pregel's scheduling order for the corresponding subgraph
        # spawns.
        self._pending_tool_calls: dict[str, dict[str, str]] = {}
        # Wire-level namespace rewrite map: Pregel-assigned first
        # segment (e.g. `"tools:<pregel_uuid>"`) → client-visible first
        # segment (e.g. `"tools:<tool_call_id>"`). Populated when a
        # `lifecycle.started` is correlated to a pending `task` tool
        # call by `graph_name` / `subagent_type`.
        self._ns_rewrite: dict[str, str] = {}

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        """Capture the sibling `SubgraphTransformer` and the enclosing mux.

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
        self._mux = mux

    def process(self, event: ProtocolEvent) -> bool:
        method = event["method"]

        # Root-scope `task` tool observations feed `_pending_tool_calls`
        # for later correlation. The event passes through unmodified.
        if method == "tools":
            self._track_task_tool_call(event)
            return True

        if method != "lifecycle":
            # Non-lifecycle events (values, messages, checkpoints,
            # input, tools at non-root ns) only need the wire-level
            # namespace rewrite applied.
            self._rewrite_ns_in_place(event)
            return True

        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)

        # Only depth + 1 lifecycle events are subagent-level; deeper
        # ones (nested subgraph / model / tool nodes) just need the
        # rewrite applied.
        if len(ns) != depth + 1 or ns[:-1] != self.scope:
            self._rewrite_ns_in_place(event)
            return True

        data = cast("LifecycleData", event["params"]["data"])
        if data.get("event") == "started":
            self._handle_started(ns, data)

        # Rewrite the ns on the way out so downstream wire consumers
        # see `tools:<tool_call_id>` even though internal state keys
        # off `tools:<pregel_uuid>`.
        self._rewrite_ns_in_place(event)
        return True

    # ─── Tool call tracking ───────────────────────────────────────────

    def _track_task_tool_call(self, event: ProtocolEvent) -> None:
        """Record / discard pending `task` tool calls observed on `tools`.

        Only `tool_name == "task"` events are tracked — other tool
        calls don't spawn subagents and would pollute the pending map.
        """
        ns = tuple(event["params"]["namespace"])
        # `tool-started` / `tool-finished` / `tool-error` fire at the
        # parent namespace, which is exactly `self.scope` for the
        # subagents we care about (direct children).
        if ns != self.scope:
            return
        data = cast("dict[str, Any]", event["params"]["data"])
        phase = data.get("event")
        if phase == "tool-started":
            if data.get("tool_name") != "task":
                return
            tcid = data.get("tool_call_id")
            raw_input = data.get("input")
            if not isinstance(tcid, str) or not isinstance(raw_input, dict):
                return
            self._pending_tool_calls[tcid] = {
                "subagent_type": str(raw_input.get("subagent_type") or ""),
                "description": str(raw_input.get("description") or ""),
            }
        elif phase in ("tool-finished", "tool-error"):
            tcid = data.get("tool_call_id")
            if isinstance(tcid, str):
                self._pending_tool_calls.pop(tcid, None)

    # ─── Started correlation + handle promotion ───────────────────────

    def _handle_started(
        self, ns: tuple[str, ...], data: LifecycleData
    ) -> None:
        """Correlate a subagent's `lifecycle.started` with a pending tool call.

        Three things happen here, in order:

        1. If ``data`` has no pre-populated ``cause`` and we have a
           pending ``task`` tool call whose ``subagent_type`` matches
           ``graph_name``, pair them. Dict insertion order gives
           oldest-first pairing under parallel fan-out, matching
           Pregel's scheduling order.
        2. If the pairing produces a ``tools:<tcid>`` that differs
           from the pre-rewrite ``tools:<pregel_seg>``, record a
           namespace rewrite entry so subsequent events under the
           same Pregel segment surface on the wire at the
           tool-call-id namespace.
        3. If ``graph_name`` is a declared subagent and
           :class:`SubgraphTransformer` has already created a handle
           at ``ns``, wrap that handle in a :class:`SubagentRunStream`
           and push it onto ``self._log``.
        """
        graph_name = data.get("graph_name")
        if not isinstance(graph_name, str):
            return

        if "cause" not in data:
            for tcid, info in self._pending_tool_calls.items():
                if info["subagent_type"] == graph_name:
                    cast("dict[str, Any]", data)["cause"] = {
                        "type": "toolCall",
                        "tool_call_id": tcid,
                    }
                    pregel_seg = ns[-1]
                    target_seg = f"tools:{tcid}"
                    if pregel_seg != target_seg:
                        self._ns_rewrite[pregel_seg] = target_seg
                    del self._pending_tool_calls[tcid]
                    break

        if graph_name not in self._names or ns in self._by_ns:
            return

        # SubgraphTransformer runs before us (factory order), so its
        # `_by_ns` already has the freshly-created handle for this
        # namespace. Reuse it — no second mini-mux, no duplicate
        # dispatch. `_on_register` guarantees `_subgraph_transformer`
        # is set before any event reaches `process`.
        sub_t = cast("SubgraphTransformer", self._subgraph_transformer)
        subgraph_handle = sub_t._by_ns.get(ns)
        if subgraph_handle is None:
            return

        # Propagate any cause we just augmented onto the SubgraphRunStream
        # instance too — it was constructed from the `data` payload
        # before we had a chance to pair a tool_call_id into it, so a
        # fresh read-back would return `None` without this sync.
        subgraph_cause = data.get("cause")
        if subgraph_cause is not None and subgraph_handle.cause is None:
            subgraph_handle.cause = subgraph_cause

        handle = SubagentRunStream(subgraph_handle, name=graph_name)
        self._by_ns[ns] = handle
        self._log.push(handle)

    # ─── Namespace rewriting ──────────────────────────────────────────

    def _rewrite_ns_in_place(self, event: ProtocolEvent) -> None:
        """Rewrite `event.params.namespace[0]` using `_ns_rewrite`.

        The namespace list is mutated in place so the mutation is
        visible to any downstream transformer *and* to the event
        that ultimately lands in the mux's main log. Only the first
        segment is rewritten — deeper segments belong to child
        subgraphs (``model:...``, ``tools:...``) whose naming is
        already stable.
        """
        if not self._ns_rewrite:
            return
        ns_list = event["params"]["namespace"]
        if not ns_list:
            return
        first = ns_list[0]
        rewritten = self._ns_rewrite.get(first)
        if rewritten is not None and rewritten != first:
            ns_list[0] = rewritten
