"""LangSmith secret redaction that reuses results across traced runs.

Each traced run carries the full message history, so the stock secret
anonymizer re-runs every rule over strings it already scanned on the previous
model call. Redaction is a pure function of each string, so this module caches
the masked result per input string and skips the regex passes on a repeat.

Imported lazily: it pulls in `langsmith.anonymizer`, which the tracing-off
startup path does not need.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from langsmith.anonymizer import (
    DEFAULT_SECRET_RULES,
    RuleNodeProcessor,
    StringNode,
    StringNodeProcessor,
    create_anonymizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langsmith.anonymizer import StringNodeRule

_MAX_DEPTH = 24
"""Traversal depth; matches the default of `create_secret_anonymizer`."""

_MAX_ENTRIES = 50_000
"""Most distinct strings the cache keeps before it evicts the oldest."""

_MAX_CHARS = 16 * 1024 * 1024
"""Most characters (keys plus changed values) the cache keeps alive."""


class CachingRuleProcessor(StringNodeProcessor):
    """Apply secret rules per string and remember the result in a bounded LRU.

    The output for a string is the same whether it comes from the cache or
    from the rules, so the cache cannot let a secret through. Strings too large
    for the character budget are masked each time and not cached.
    """

    def __init__(
        self,
        rules: list[StringNodeRule],
        *,
        max_entries: int = _MAX_ENTRIES,
        max_chars: int = _MAX_CHARS,
    ) -> None:
        """Build the processor.

        Args:
            rules: Redaction rules, applied in order to every string.
            max_entries: Upper bound on cached strings.
            max_chars: Upper bound on characters the cache holds.
        """
        self._rules = RuleNodeProcessor(rules)
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._chars = 0
        # The tracer can call the anonymizer from its background thread as
        # well as from the event loop.
        self._lock = threading.Lock()

    def mask_nodes(self, nodes: list[StringNode]) -> list[StringNode]:
        """Return the nodes whose value changes after masking.

        Args:
            nodes: String nodes extracted from a traced payload.

        Returns:
            Replacement nodes, one for each string that the rules changed.
        """
        # One lock round-trip per payload, not per string: a long history has
        # tens of thousands of string nodes.
        with self._lock:
            hits = [self._lookup(node["value"]) for node in nodes]
        result: list[StringNode] = []
        for node, hit in zip(nodes, hits, strict=True):
            value = node["value"]
            masked = self._apply_rules(value) if hit is None else hit
            if masked != value:
                result.append(StringNode(value=masked, path=node["path"]))
        return result

    def _lookup(self, value: str) -> str | None:
        """Return the cached mask for `value` and mark it recent; caller locks."""
        masked = self._cache.get(value)
        if masked is not None:
            self._cache.move_to_end(value)
        return masked

    def _apply_rules(self, value: str) -> str:
        """Run every rule over `value` and cache the result.

        Returns:
            The masked string, or `value` itself when no rule matched.
        """
        changed = self._rules.mask_nodes([StringNode(value=value, path=[])])
        masked = changed[0]["value"] if changed else value
        self._store(value, masked)
        return masked

    def _store(self, value: str, masked: str) -> None:
        """Cache `masked` for `value` and evict the oldest entries past bounds."""
        cost = _cost(value, masked)
        if cost > self._max_chars:
            return
        with self._lock:
            if value in self._cache:
                return
            self._cache[value] = masked
            self._chars += cost
            while len(self._cache) > self._max_entries or self._chars > self._max_chars:
                old_value, old_masked = self._cache.popitem(last=False)
                self._chars -= _cost(old_value, old_masked)


def _cost(value: str, masked: str) -> int:
    """Return the characters a cache entry keeps alive.

    An unchanged string stores the key object as its own value, so it counts
    once.
    """
    return len(value) + (0 if masked is value else len(masked))


def create_cached_secret_anonymizer() -> Callable[[Any], Any]:
    """Build a secret anonymizer that matches `create_secret_anonymizer`.

    It uses the same default rules and depth, and adds a per-string cache so
    unchanged history strings are not scanned again on each traced run.

    Returns:
        An anonymizer to pass to `langsmith.Client(anonymizer=...)`.
    """
    processor = CachingRuleProcessor(list(DEFAULT_SECRET_RULES))
    return create_anonymizer(processor, max_depth=_MAX_DEPTH)
