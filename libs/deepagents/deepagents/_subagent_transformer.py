"""Project subagent runs as first-class child streams on a parent run,
and synthesize wire-visible events for `task`-tool-invoked subagents.

`SubagentTransformer` does two jobs:

1. **In-process projection** — a thin overlay on top of
   `SubgraphTransformer`. It does *not* run a second discovery or a
   second mini-mux pipeline. When `SubgraphTransformer` fires a
   `started` event and creates a `SubgraphRunStream` at some
   namespace, `SubagentTransformer` checks that event's `graph_name`
   against the set of declared subagent names; on a match, it wraps
   the existing `SubgraphRunStream` in a `SubagentRunStream` and
   pushes that onto `run.subagents`.

2. **Synthetic wire-event emission** — for subagents invoked via the
   `task` tool, where `.ainvoke()` collects the subagent's final
   state synchronously and *no* internal pregel events propagate up
   to the parent mux. To keep those subagent runs visible on the
   wire — so remote clients subscribing via
   `/events?namespaces=[["tools:<tool_call_id>"]]` observe real
   protocol events and the namespace-subscription contract stays
   intact — this transformer fabricates a minimal but complete
   lifecycle + messages + values sequence at a stable
   `tools:<tool_call_id>` namespace. The `langgraph-api` layer stays
   product-agnostic; all deepagents-specific correlation lives here.

The synthesized event sequence on `tool-started`:

    lifecycle.started    { graph_name: <subagent_type>,
                           cause: { type: "toolCall",
                                    tool_call_id: <tcid> } }
    messages.message-start  { role: "human",
                              message_id: "subagent:<tcid>:human" }
    messages.content-block-start / -delta / -finish  (input.description)
    messages.message-finish
    values               { messages: [humanMsg] }

And, on `tool-finished` / `tool-error`:

    messages.message-start  { role: "ai",
                              message_id: "subagent:<tcid>:ai" }
    messages.content-block-start / -delta / -finish  (normalized output)
    messages.message-finish
    values                 { messages: [humanMsg, aiMsg] }
    lifecycle.completed    (or lifecycle.failed on tool-error)

Synthesized events are injected via `self._mux.push(...)`, which runs
them through the full transformer pipeline — so `SubgraphTransformer`
creates a real handle + child mini-mux for the synthetic namespace
(enabling `handle.messages` / `handle.output` projections), and this
transformer then promotes that handle to a `SubagentRunStream`.
Clients subscribing via `/events?namespaces=[["tools:<tcid>"]]` see
these events on the wire exactly like they would any other namespace.

Trade-off: the synthetic sequence is a fabrication, not a recording
of real subagent execution — there is no checkpoint timeline to time
travel against for these messages. If/when deepagents migrates the
`task` tool from `.ainvoke()` to real streaming, this synthesis can
be retired in favor of genuine per-subagent events reaching the
parent mux through the existing `SubgraphTransformer` path.

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
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import StreamTransformer
from langgraph.stream.run_stream import BaseRunStream
from langgraph.stream.transformers import SubgraphRunStream, SubgraphTransformer

if TYPE_CHECKING:
    from langchain_protocol.protocol import (
        CheckpointRef,
        LifecycleCause,
    )
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent
    from langgraph.stream.transformers import SubgraphStatus


@dataclass
class _SyntheticSubagentState:
    """Per-`task`-tool-call synthesis bookkeeping.

    `namespace` is the subagent's synthetic namespace
    (`scope + ("tools:<tool_call_id>",)`). `messages` accumulates
    the human + AI message pair that drive the synthetic `values`
    snapshots emitted alongside lifecycle transitions. `completed`
    flips True once the terminal `tool-finished` / `tool-error`
    has been processed so stragglers are silently ignored.
    """

    namespace: tuple[str, ...]
    messages: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False


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
    events (for synthetic emission) *and* the promoted subagent's
    `lifecycle.started` at `scope + 1` namespace (for handle
    promotion). Events at deeper namespaces flow through
    `SubgraphTransformer` into the subagent's mini-mux where the
    standard `messages` / `values` transformers build projections.
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
        # Synthesis bookkeeping, keyed by originating `tool_call_id`.
        # Entries live from `tool-started` through the terminal
        # `tool-finished` / `tool-error`; they're deleted once the
        # terminal synthetic events have been emitted.
        self._synthesized: dict[str, _SyntheticSubagentState] = {}

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        """Capture the sibling `SubgraphTransformer` and the enclosing mux.

        Raises at registration time if `SubgraphTransformer` isn't
        present — failing loudly here is better than silently yielding
        zero subagent handles at runtime. The mux reference is stored
        for re-entrant `push(...)` calls from `_emit_synthetic_*`.
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
        ns = tuple(event["params"]["namespace"])

        # Root-scope `task` tool observations drive synthetic emission.
        # The event is still kept on the wire (we return True) — only
        # the synthesis is a side effect.
        if method == "tools" and ns == self.scope:
            self._handle_task_tool_event(event)
            return True

        if method != "lifecycle":
            return True

        # Direct children of this scope only — matches
        # SubgraphTransformer's own discovery depth.
        depth = len(self.scope)
        if len(ns) != depth + 1 or ns[:-1] != self.scope:
            return True

        data = cast("dict[str, Any]", event["params"]["data"])
        if data.get("event") != "started":
            return True

        graph_name = data.get("graph_name")
        if not isinstance(graph_name, str):
            return True
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

        handle = SubagentRunStream(subgraph_handle, name=graph_name)
        self._by_ns[ns] = handle
        self._log.push(handle)
        return True

    # ─── Synthetic emission ───────────────────────────────────────────

    def _handle_task_tool_event(self, event: ProtocolEvent) -> None:
        """Branch on the root-scope `task` tool phase to drive synthesis.

        ``tool_name`` is only present on ``tool-started`` in the protocol
        (``tool-finished`` / ``tool-error`` omit it). We latch onto the
        ``task`` filter at start and then drive terminal synthesis purely
        from membership in ``self._synthesized`` — which is keyed by
        ``tool_call_id`` and only populated when a ``task`` tool-started
        has previously been observed. Non-``task`` tool calls never enter
        the map, so their terminal events naturally no-op.
        """
        data = cast("dict[str, Any]", event["params"]["data"])
        phase = data.get("event")
        tcid = data.get("tool_call_id")
        if not isinstance(tcid, str):
            return

        if phase == "tool-started":
            if data.get("tool_name") != "task":
                return
            raw_input = data.get("input")
            input_d: dict[str, Any] = (
                raw_input if isinstance(raw_input, dict) else {}
            )
            subagent_type = str(input_d.get("subagent_type") or "")
            description = str(input_d.get("description") or "")
            self._emit_synthetic_start(tcid, subagent_type, description)
        elif phase == "tool-finished":
            if tcid not in self._synthesized:
                return
            self._emit_synthetic_finish(tcid, data.get("output"), is_error=False)
        elif phase == "tool-error":
            if tcid not in self._synthesized:
                return
            self._emit_synthetic_finish(tcid, data.get("message"), is_error=True)

    def _emit_synthetic_start(
        self,
        tool_call_id: str,
        subagent_type: str,
        description: str,
    ) -> None:
        """Emit the full opening event sequence for a `task` tool call.

        Order (each pushed via `self._mux.push`, which re-enters the
        transformer pipeline):

            1. `lifecycle.started` — creates the child mini-mux via
               `SubgraphTransformer._on_started` and, because
               `graph_name` is the declared subagent name, gets
               promoted to a `SubagentRunStream` by this transformer
               on re-entry.
            2. `messages.message-start` (human) — opens a
               `ChatModelStream` in the child mini-mux's
               `MessagesTransformer`.
            3. `messages.content-block-start / -delta / -finish` —
               single text block carrying the prompt.
            4. `messages.message-finish`.
            5. `values` — snapshot with the one human message so
               `handle.output` has meaningful shape during the
               running phase.
        """
        if tool_call_id in self._synthesized:
            return
        synthetic_ns = (*self.scope, f"tools:{tool_call_id}")
        human_id = f"subagent:{tool_call_id}:human"
        human_msg: dict[str, Any] = {
            "type": "human",
            "content": description,
            "id": human_id,
        }
        self._synthesized[tool_call_id] = _SyntheticSubagentState(
            namespace=synthetic_ns,
            messages=[human_msg],
        )

        self._push_event(
            "lifecycle",
            synthetic_ns,
            {
                "event": "started",
                "graph_name": subagent_type,
                "cause": {"type": "toolCall", "tool_call_id": tool_call_id},
            },
        )
        self._emit_synthesized_message(
            synthetic_ns,
            role="human",
            message_id=human_id,
            content=description,
        )
        self._push_event(
            "values",
            synthetic_ns,
            {"messages": [dict(msg) for msg in self._synthesized[tool_call_id].messages]},
        )

    def _emit_synthetic_finish(
        self,
        tool_call_id: str,
        payload: Any,
        *,
        is_error: bool,
    ) -> None:
        """Emit the closing event sequence for a `task` tool call.

        On success, appends an AI message derived from the tool's
        `output` before emitting the terminal `lifecycle.completed`.
        On error, skips the AI message and emits
        `lifecycle.failed` with the error string.
        """
        state = self._synthesized.get(tool_call_id)
        if state is None or state.completed:
            return
        state.completed = True
        synthetic_ns = state.namespace

        if not is_error:
            ai_id = f"subagent:{tool_call_id}:ai"
            ai_content = _normalize_output_content(payload)
            state.messages.append(
                {"type": "ai", "content": ai_content, "id": ai_id}
            )
            self._emit_synthesized_message(
                synthetic_ns,
                role="ai",
                message_id=ai_id,
                content=ai_content,
            )
            self._push_event(
                "values",
                synthetic_ns,
                {"messages": [dict(msg) for msg in state.messages]},
            )
            self._push_event(
                "lifecycle",
                synthetic_ns,
                {"event": "completed"},
            )
        else:
            error_message = (
                str(payload) if payload not in (None, "") else "subagent failed"
            )
            self._push_event(
                "lifecycle",
                synthetic_ns,
                {"event": "failed", "error": error_message},
            )
        del self._synthesized[tool_call_id]

    def _emit_synthesized_message(
        self,
        namespace: tuple[str, ...],
        *,
        role: str,
        message_id: str,
        content: Any,
    ) -> None:
        """Emit a single-message `messages.*` sequence at the given namespace.

        Handles the two common content shapes: a plain string
        (single-text-block output) and a list of protocol content
        blocks (richer model output with reasoning / text mixed).
        Non-text / non-reasoning blocks pass through as-is on the
        content-block-* envelope.
        """
        self._push_messages_event(
            namespace,
            message_id,
            {"event": "message-start", "role": role, "message_id": message_id},
        )
        if isinstance(content, str):
            self._push_messages_event(
                namespace,
                message_id,
                {
                    "event": "content-block-start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if content:
                self._push_messages_event(
                    namespace,
                    message_id,
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "content_block": {"type": "text", "text": content},
                    },
                )
            self._push_messages_event(
                namespace,
                message_id,
                {
                    "event": "content-block-finish",
                    "index": 0,
                    "content_block": {"type": "text", "text": content},
                },
            )
        elif isinstance(content, list):
            for offset, raw_block in enumerate(content):
                block = _normalize_finalized_content_block(raw_block)
                if block is None:
                    continue
                index = (
                    block["index"]
                    if isinstance(block.get("index"), int)
                    else offset
                )
                btype = block.get("type")
                if btype == "text":
                    start_block: dict[str, Any] = {"type": "text", "text": ""}
                elif btype == "reasoning":
                    start_block = {"type": "reasoning", "reasoning": ""}
                else:
                    start_block = block
                self._push_messages_event(
                    namespace,
                    message_id,
                    {
                        "event": "content-block-start",
                        "index": index,
                        "content_block": start_block,
                    },
                )
                if btype == "text" and block.get("text"):
                    self._push_messages_event(
                        namespace,
                        message_id,
                        {
                            "event": "content-block-delta",
                            "index": index,
                            "content_block": {
                                "type": "text",
                                "text": block["text"],
                            },
                        },
                    )
                elif btype == "reasoning" and block.get("reasoning"):
                    self._push_messages_event(
                        namespace,
                        message_id,
                        {
                            "event": "content-block-delta",
                            "index": index,
                            "content_block": {
                                "type": "reasoning",
                                "reasoning": block["reasoning"],
                            },
                        },
                    )
                self._push_messages_event(
                    namespace,
                    message_id,
                    {
                        "event": "content-block-finish",
                        "index": index,
                        "content_block": block,
                    },
                )
        self._push_messages_event(
            namespace,
            message_id,
            {"event": "message-finish"},
        )

    def _push_event(
        self,
        method: str,
        namespace: tuple[str, ...],
        data: Any,
    ) -> None:
        """Wrap `data` in a ProtocolEvent envelope and re-enter the mux.

        Seq is assigned by `StreamMux.push` at append time; re-entrant
        dispatch during our own `process(...)` is an explicitly
        supported flow (see the note in `StreamMux.push`).
        """
        assert self._mux is not None, (
            "SubagentTransformer emitted before _on_register; "
            "transformer registration ordering is broken."
        )
        event: ProtocolEvent = {
            "type": "event",
            "method": method,
            "params": {
                "namespace": list(namespace),
                "timestamp": int(time.time() * 1000),
                "data": data,
            },
        }
        self._mux.push(event)

    def _push_messages_event(
        self,
        namespace: tuple[str, ...],
        message_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Push a synthetic `messages.*` event in the internal tuple shape.

        The core `MessagesTransformer` unpacks ``params["data"]`` as
        ``(payload, metadata)`` (see
        :class:`langgraph.stream.transformers.MessagesTransformer`), so
        synthetic messages events must use the same wire-internal
        shape. ``run_id`` is set to ``message_id`` so the transformer
        correlates ``message-start`` through ``message-finish`` on a
        single :class:`~langchain_core.language_models.chat_model_stream.ChatModelStream`.
        The ``langgraph_node`` field is stubbed as the subagent's
        namespace suffix so the resulting projection has a sensible
        ``node`` attribute.
        """
        node_name = namespace[-1] if namespace else None
        metadata: dict[str, Any] = {
            "run_id": message_id,
            "langgraph_node": node_name,
        }
        self._push_event("messages", namespace, (payload, metadata))


def _normalize_output_content(value: Any) -> Any:
    """Coerce a tool's `output` payload into `messages.*`-compatible content.

    The `task` tool returns a pregel ``Command`` value when its
    subagent completes. Stream-side, that surfaces on
    `tool-finished` as
    ``{"graph": None, "update": {"messages": [<tool msg>, ...]},
    "resume": None, "goto": [...]}``. The synthesized AI message's
    content is extracted from the *last* message in
    ``update.messages`` — that's the tool-role reply carrying the
    subagent's final text. Other shapes fall back as follows:

    - `None` → empty string (renders as an empty AI message).
    - `str` → itself (single text block).
    - `list` → passed through for per-block handling in
      `_emit_synthesized_message`.
    - `dict` with an ``update.messages`` list → recurse into the
      last message's ``content`` (Command-shaped tool output).
    - `dict` with a ``content`` key → recurse into ``content``
      (BaseMessage-like / artifact-like shapes).
    - Anything else → `str(value)` (defensive; keeps the wire
      string-typed).
    """
    if value is None:
        return ""
    if isinstance(value, (str, list)):
        return value
    if isinstance(value, dict):
        update = value.get("update")
        if isinstance(update, dict):
            messages = update.get("messages")
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    return _normalize_output_content(last)
        content = value.get("content")
        if content is not None:
            return _normalize_output_content(content)
        return str(value)
    return str(value)


def _normalize_finalized_content_block(block: Any) -> dict[str, Any] | None:
    """Coerce a raw content block into a canonical protocol shape.

    - `str` → `{type: "text", text: <str>}`.
    - `dict` with `type`:
        - `"text"` → `{type, text}` (optionally preserving `id` / `index`).
        - `"reasoning"` → `{type, reasoning}` (ditto).
        - any other `type` → passed through as-is (image / file / tool
          blocks travel whole; no partial synthesis for them).
    - Anything else → `None` (skipped — keeps bad inputs off the wire).
    """
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text":
        normalized: dict[str, Any] = {
            "type": "text",
            "text": str(block.get("text", "")),
        }
        if "id" in block:
            normalized["id"] = block["id"]
        if isinstance(block.get("index"), int):
            normalized["index"] = block["index"]
        return normalized
    if btype == "reasoning":
        normalized = {
            "type": "reasoning",
            "reasoning": str(block.get("reasoning", "")),
        }
        if "id" in block:
            normalized["id"] = block["id"]
        if isinstance(block.get("index"), int):
            normalized["index"] = block["index"]
        return normalized
    return block
