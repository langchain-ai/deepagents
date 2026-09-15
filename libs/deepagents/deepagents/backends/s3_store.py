"""S3-compatible implementation of LangGraph's `BaseStore`."""

import asyncio
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, TypeAlias, cast
from urllib.parse import quote, unquote

from langgraph.store.base import BaseStore, GetOp, Item, ListNamespacesOp, Op, PutOp, Result, SearchItem, SearchOp

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class S3Client(Protocol):
    """Subset of the boto3 S3 client used by `S3Store`."""

    def delete_object(self, **kwargs: Any) -> Mapping[str, Any]:
        """Delete an object."""
        ...

    def get_object(self, **kwargs: Any) -> Mapping[str, Any]:
        """Get an object."""
        ...

    def list_objects_v2(self, **kwargs: Any) -> Mapping[str, Any]:
        """List objects."""
        ...

    def put_object(self, **kwargs: Any) -> Mapping[str, Any]:
        """Put an object."""
        ...


class S3Store(BaseStore):
    """LangGraph store backed by any S3-compatible object-storage service.

    Args:
        client: A boto3-compatible synchronous S3 client.
        bucket: Bucket containing store objects.
        prefix: Optional object-key prefix reserved for the store.
    """

    def __init__(self, *, client: S3Client, bucket: str, prefix: str = "") -> None:
        """Initialize an S3-backed store."""
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _object_key(self, namespace: tuple[str, ...], key: str) -> str:
        components = [*(quote(part, safe="") for part in namespace), quote(key, safe="") + ".json"]
        path = "/".join(components)
        return f"{self._prefix}/{path}" if self._prefix else path

    def _decode_key(self, object_key: str) -> tuple[tuple[str, ...], str] | None:
        relative = object_key[len(self._prefix) + 1 :] if self._prefix else object_key
        parts = relative.split("/")
        if len(parts) < 2:  # noqa: PLR2004  # A store path requires a namespace and key.
            return None
        if not parts[-1].endswith(".json"):
            return None
        return tuple(unquote(part) for part in parts[:-1]), unquote(parts[-1][:-5])

    def _load(self, namespace: tuple[str, ...], key: str) -> Item | None:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._object_key(namespace, key))
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            if error.get("Code") in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        body = response["Body"].read()
        document = json.loads(body)
        return Item(
            namespace=namespace,
            key=key,
            value=document["value"],
            created_at=datetime.fromisoformat(document["created_at"]),
            updated_at=datetime.fromisoformat(document["updated_at"]),
        )

    def _put(self, op: PutOp) -> None:
        object_key = self._object_key(op.namespace, op.key)
        if op.value is None:
            self._client.delete_object(Bucket=self._bucket, Key=object_key)
            return
        existing = self._load(op.namespace, op.key)
        now = datetime.now(UTC)
        document = {
            "value": op.value,
            "created_at": (existing.created_at if existing else now).isoformat(),
            "updated_at": now.isoformat(),
        }
        self._client.put_object(
            Bucket=self._bucket,
            Key=object_key,
            Body=json.dumps(document, separators=(",", ":")).encode(),
            ContentType="application/json",
        )

    def _list_keys(self) -> list[str]:
        prefix = f"{self._prefix}/" if self._prefix else ""
        token: str | None = None
        keys: list[str] = []
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            response = self._client.list_objects_v2(**kwargs)
            keys.extend(entry["Key"] for entry in response.get("Contents", []) if isinstance(entry.get("Key"), str))
            if not response.get("IsTruncated"):
                return keys
            token = cast("str", response["NextContinuationToken"])

    @classmethod
    def _matches_filter(cls, value: JsonValue, filter_value: JsonValue) -> bool:
        if not isinstance(filter_value, dict):
            return value == filter_value
        filters = cast("dict[str, JsonValue]", filter_value)
        if any(key.startswith("$") for key in filters):
            return all(cls._apply_operator(value, operator, operand) for operator, operand in filters.items())
        if not isinstance(value, dict):
            return False
        document = cast("dict[str, JsonValue]", value)
        return all(cls._matches_filter(document.get(key), expected) for key, expected in filters.items())

    @staticmethod
    def _apply_operator(value: JsonValue, operator: str, operand: JsonValue) -> bool:
        if operator == "$eq":
            return value == operand
        if operator == "$ne":
            return value != operand
        comparisons = {"$gt": float.__gt__, "$gte": float.__ge__, "$lt": float.__lt__, "$lte": float.__le__}
        if operator not in comparisons:
            msg = f"Unsupported filter operator: {operator}"
            raise ValueError(msg)
        if not isinstance(value, (int, float, str)) or not isinstance(operand, (int, float, str)):
            return False
        return comparisons[operator](float(value), float(operand))

    def _search(self, op: SearchOp) -> list[SearchItem]:
        if op.query is not None:
            msg = "S3Store does not support semantic search."
            raise NotImplementedError(msg)
        items: list[SearchItem] = []
        for object_key in self._list_keys():
            decoded = self._decode_key(object_key)
            if decoded is None or decoded[0][: len(op.namespace_prefix)] != op.namespace_prefix:
                continue
            item = self._load(*decoded)
            if item is not None and self._matches_filter(item.value, op.filter):
                items.append(
                    SearchItem(namespace=item.namespace, key=item.key, value=item.value, created_at=item.created_at, updated_at=item.updated_at)
                )
        items.sort(key=lambda item: (item.namespace, item.key))
        return items[op.offset : op.offset + op.limit]

    @staticmethod
    def _matches_namespace(namespace: tuple[str, ...], op: ListNamespacesOp) -> bool:
        for condition in op.match_conditions or ():
            path = condition.path
            if len(namespace) < len(path):
                return False
            compared = namespace[: len(path)] if condition.match_type == "prefix" else namespace[-len(path) :]
            if any(expected not in {"*", actual} for actual, expected in zip(compared, path, strict=True)):
                return False
        return True

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()
        for object_key in self._list_keys():
            decoded = self._decode_key(object_key)
            if decoded is None or not self._matches_namespace(decoded[0], op):
                continue
            namespace = decoded[0][: op.max_depth] if op.max_depth is not None else decoded[0]
            namespaces.add(namespace)
        return sorted(namespaces)[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute store operations against S3."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._load(op.namespace, op.key))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                msg = f"Unsupported store operation: {type(op).__name__}"
                raise TypeError(msg)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute store operations without blocking the event loop."""
        return await asyncio.to_thread(self.batch, list(ops))
