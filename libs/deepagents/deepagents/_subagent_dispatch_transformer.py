"""Wire-side custom-channel emitter for `task`-tool subagent dispatches.

Emits one structured payload per `task` tool dispatch onto a
``custom:subagent_dispatch`` wire channel so remote SDK consumers can
correlate lifecycle entries to subagent identity (`subagent_type`,
`description`) without walking the parent message's `tool_calls`.

Kept as a separate transformer from the in-process
`SubagentTransformer` (which exposes typed `run.subagents` handles to
local Python consumers) so the native transformer's projection stays
untouched. This one consumes the `tools` channel directly — gating on
`tool_name == "task"` — rather than the per-call Send envelope, so it
doesn't depend on `langchain.agents.create_agent`'s fan-out shape and
will keep working under future dispatch-layout changes as long as the
`task` tool's `tools` events look the same.

The emitted payload shape:

    {
        "tool_call_id": str,            # model-side id, joins to
                                         # `lifecycle.cause.tool_call_id`
        "subagent_type": str,           # from the `task` tool args
        "description": str | None,      # from the `task` tool args
        "namespace": list[str],         # ns the dispatching tool fired at
    }

Consumers (e.g. the JS SDK's `deriveSubagents`) join
`lifecycle.cause.tool_call_id` ↔ this payload's `tool_call_id` to
attach declared-subagent identity to a generic lifecycle entry.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream._types import StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent


class SubagentDispatchTransformer(StreamTransformer):
    """Emit a `custom:subagent_dispatch` event per `task` tool dispatch.

    Non-native (extension) transformer — the channel name
    ``subagent_dispatch`` is auto-prefixed to ``custom:subagent_dispatch``
    on the wire by the mux. Remote SDK clients subscribe via
    ``session.subscribe("custom:subagent_dispatch")``.
    """

    _native: ClassVar[bool] = False

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[dict[str, Any]] = StreamChannel("subagent_dispatch")

    def init(self) -> dict[str, Any]:
        return {"subagent_dispatch": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        record = self._extract_dispatch(event)
        if record is not None:
            self._channel.push(record)
        return True

    def _extract_dispatch(self, event: ProtocolEvent) -> dict[str, Any] | None:
        """Build a dispatch payload from a `task` `tool-started` event.

        Returns `None` when the event isn't a `task` dispatch we can
        surface (wrong method, wrong tool, missing ids, malformed
        input). All branches are silent — process() always passes the
        original event through.
        """
        if event.get("method") != "tools":
            return None
        params = event.get("params") or {}
        data = params.get("data") if isinstance(params, dict) else None
        if not isinstance(data, dict) or data.get("event") != "tool-started" or data.get("tool_name") != "task":
            return None
        tool_call_id = data.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None
        args = _parse_task_args(data.get("input"))
        if args is None:
            return None
        subagent_type = args.get("subagent_type")
        if not isinstance(subagent_type, str) or not subagent_type:
            return None
        record: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "subagent_type": subagent_type,
            "namespace": list(params.get("namespace") or ()),
        }
        description = args.get("description")
        if isinstance(description, str):
            record["description"] = description
        return record


def _parse_task_args(raw: object) -> dict[Any, Any] | None:
    """Return a parsed args dict from a `task` tool-started ``input``.

    Tolerates both shapes the wire may carry:

    - JSON-stringified args (langgraph's default tool-arg serialization)
    - Already-parsed dict (some upstream tool runners)

    Returns ``None`` for non-string/non-dict inputs and for strings that
    don't parse as JSON-objects.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
