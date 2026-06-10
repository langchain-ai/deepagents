"""QuickJS-local recreation of Deep Agents subagent specs."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deepagents.middleware.subagents import create_sub_agent
from langchain_core.messages import AIMessage, HumanMessage

try:
    from deepagents.middleware.subagents import SUBAGENT_SPECS_CONFIG_KEY
except ImportError:  # pragma: no cover - compatibility with older deepagents
    SUBAGENT_SPECS_CONFIG_KEY = "__deepagents_subagent_specs__"

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import Runnable, RunnableConfig

_VARIANT_TTL_S = 60.0
_VARIANT_MAX_ENTRIES = 64

_SCHEMA_MAX_BYTES = 4096
_SCHEMA_MAX_DEPTH = 5
_SCHEMA_MAX_PROPERTIES = 32

_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
}


def subagent_payload_from_configurable(
    configurable: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the configured top-level subagent specs payload, if present."""
    if not isinstance(configurable, dict):
        return None
    payload = configurable.get(SUBAGENT_SPECS_CONFIG_KEY)
    return payload if isinstance(payload, dict) else None


def _validate_response_schema(schema: dict[str, Any]) -> None:
    """Reject schemas that exceed size, depth, or property-count limits."""
    serialized = json.dumps(schema)
    if len(serialized) > _SCHEMA_MAX_BYTES:
        msg = (
            f"response_schema exceeds {_SCHEMA_MAX_BYTES}"
            f" byte limit ({len(serialized)} bytes)"
        )
        raise ValueError(msg)

    def _check(node: dict[str, Any], depth: int, prop_count: list[int]) -> None:
        if depth > _SCHEMA_MAX_DEPTH:
            msg = (
                "response_schema exceeds maximum nesting depth of"
                f" {_SCHEMA_MAX_DEPTH}"
            )
            raise ValueError(msg)
        props = node.get("properties")
        if isinstance(props, dict):
            prop_count[0] += len(props)
            if prop_count[0] > _SCHEMA_MAX_PROPERTIES:
                msg = (
                    "response_schema exceeds maximum of"
                    f" {_SCHEMA_MAX_PROPERTIES} properties"
                )
                raise ValueError(msg)
            for value in props.values():
                if isinstance(value, dict):
                    _check(value, depth + 1, prop_count)
        items = node.get("items")
        if isinstance(items, dict):
            _check(items, depth + 1, prop_count)

    _check(schema, 0, [0])


class VariantCache:
    """TTL cache for dynamically compiled subagent variants."""

    def __init__(
        self,
        ttl_s: float = _VARIANT_TTL_S,
        max_entries: int = _VARIANT_MAX_ENTRIES,
    ) -> None:
        self._entries: dict[str, tuple[Runnable, float]] = {}
        self._ttl_s = ttl_s
        self._max_entries = max_entries

    def get_or_create(self, key: str, factory: Callable[[], Runnable]) -> Runnable:
        """Return a cached runnable or create one via `factory` on cache miss."""
        self._sweep()
        entry = self._entries.get(key)
        if entry is not None:
            value, _ = entry
            self._entries[key] = (value, time.monotonic())
            return value
        if len(self._entries) >= self._max_entries:
            self._evict_lru()
        value = factory()
        self._entries[key] = (value, time.monotonic())
        return value

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, (_, accessed_at) in self._entries.items()
            if now - accessed_at > self._ttl_s
        ]
        for key in expired:
            del self._entries[key]

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        oldest_key = min(self._entries, key=lambda key: self._entries[key][1])
        del self._entries[oldest_key]


@dataclass
class _SubagentEntry:
    name: str
    description: str
    spec: dict[str, Any]
    state_schema: type | None
    runnable_backed: bool
    runnable: Runnable | None = None

    def base_runnable(self) -> Runnable:
        """Return the cached base runnable for this spec."""
        if self.runnable is not None:
            return self.runnable
        runnable = create_sub_agent(self.spec, state_schema=self.state_schema)
        self.runnable = runnable
        return runnable

    def variant_runnable(
        self,
        response_schema: dict[str, Any],
        cache: VariantCache,
    ) -> Runnable:
        """Return a schema-constrained runnable for this declarative spec."""
        if self.runnable_backed:
            msg = (
                "response_schema cannot be used with runnable-backed subagent "
                f'"{self.name}"; '
                "dynamic schemas require a declarative SubAgent spec."
            )
            raise ValueError(msg)
        cache_key = f"{self.name}::{json.dumps(response_schema, sort_keys=True)}"
        return cache.get_or_create(
            cache_key,
            lambda: create_sub_agent(
                {**self.spec, "response_format": response_schema},
                state_schema=self.state_schema,
            ),
        )


class QuickJSSubagentDispatcher:
    """Invoke subagents recreated from specs exposed by `create_deep_agent`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._private_state_keys = frozenset(payload.get("private_state_keys", ()))
        state_schema = payload.get("state_schema")
        self._entries: dict[str, _SubagentEntry] = {}
        self._variant_cache = VariantCache()

        for spec in payload.get("subagents", ()):
            name = spec["name"]
            description = spec["description"]
            if "runnable" in spec:
                runnable = spec["runnable"].with_config(
                    {
                        "metadata": {"lc_agent_name": name},
                        "run_name": name,
                    }
                )
                self._entries[name] = _SubagentEntry(
                    name=name,
                    description=description,
                    spec=spec,
                    state_schema=state_schema,
                    runnable_backed=True,
                    runnable=runnable,
                )
                continue
            self._entries[name] = _SubagentEntry(
                name=name,
                description=description,
                spec=spec,
                state_schema=state_schema,
                runnable_backed=False,
            )

    @property
    def subagent_descriptions(self) -> tuple[dict[str, str], ...]:
        """Configured subagent names and descriptions for prompt rendering."""
        return tuple(
            {"name": entry.name, "description": entry.description}
            for entry in self._entries.values()
        )

    async def ainvoke(
        self,
        *,
        description: str,
        subagent_type: str,
        response_schema: dict[str, Any] | None = None,
        runtime: Any = None,
    ) -> Any:
        """Invoke one configured subagent and return its extracted output."""
        entry = self._entries.get(subagent_type)
        if entry is None:
            available = ", ".join(self._entries)
            msg = f'Unknown subagent type "{subagent_type}". Available: {available}'
            raise ValueError(msg)

        if response_schema is not None:
            _validate_response_schema(response_schema)
            subagent = entry.variant_runnable(response_schema, self._variant_cache)
        else:
            subagent = entry.base_runnable()

        state = self._prepare_state(description, runtime)
        config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        result = await subagent.ainvoke(state, config)
        return _extract_output(result)

    def _prepare_state(self, description: str, runtime: Any) -> dict[str, Any]:
        parent = (
            runtime.state
            if runtime is not None and isinstance(runtime.state, dict)
            else {}
        )
        state = {
            key: value
            for key, value in parent.items()
            if key not in _EXCLUDED_STATE_KEYS and key not in self._private_state_keys
        }
        state["messages"] = [HumanMessage(content=description)]
        return state


def _native_structured_response(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _extract_output(result: Any) -> Any:
    if not isinstance(result, dict) or "messages" not in result:
        msg = (
            "Subagent must return a state containing a 'messages' key. "
            "Custom StateGraphs used as subagents should include 'messages' "
            "in their state schema."
        )
        raise ValueError(msg)

    structured = result.get("structured_response")
    if structured is not None:
        return _native_structured_response(structured)

    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            text = message.text.rstrip() if message.text else ""
            if text:
                return text
    return "Task completed"
