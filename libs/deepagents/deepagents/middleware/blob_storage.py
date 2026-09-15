"""Middleware for moving binary tool results out of graph checkpoints."""

import base64
import hashlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage
from langgraph.store.base import BaseStore
from langgraph.types import Command

_BLOB_REF = "blob_ref"
_DEFAULT_NAMESPACE = ("deepagents", "blobs")


class BlobStorageMiddleware(AgentMiddleware):
    """Store binary `read_file` results outside checkpointed message history.

    Binary content is stored by SHA-256 digest in a LangGraph `BaseStore`. State
    retains only a stable reference, while model requests are rehydrated from the
    store without reading the source file again.

    Args:
        store: Store to use outside a graph execution. If omitted, the store from
            the active LangGraph runtime is used.
        namespace: Namespace used for content-addressed blobs.
    """

    def __init__(self, *, store: BaseStore | None = None, namespace: tuple[str, ...] = _DEFAULT_NAMESPACE) -> None:
        """Initialize blob storage middleware."""
        self._store = store
        self._namespace = namespace

    def _resolve_store(self, runtime_store: BaseStore | None) -> BaseStore:
        store = self._store or runtime_store
        if store is None:
            msg = "BlobStorageMiddleware requires a LangGraph store or an explicit `store`."
            raise RuntimeError(msg)
        return store

    def _offload_block(self, block: dict[str, Any], store: BaseStore) -> dict[str, Any]:
        payload = block.get("base64")
        if not isinstance(payload, str):
            return block
        try:
            raw = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            msg = "Binary `read_file` content is not valid base64."
            raise ValueError(msg) from exc
        digest = hashlib.sha256(raw).hexdigest()
        if store.get(self._namespace, digest) is None:
            store.put(self._namespace, digest, {"base64": payload}, index=False)
        return {key: value for key, value in block.items() if key != "base64"} | {_BLOB_REF: digest}

    async def _aoffload_block(self, block: dict[str, Any], store: BaseStore) -> dict[str, Any]:
        payload = block.get("base64")
        if not isinstance(payload, str):
            return block
        try:
            raw = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            msg = "Binary `read_file` content is not valid base64."
            raise ValueError(msg) from exc
        digest = hashlib.sha256(raw).hexdigest()
        if await store.aget(self._namespace, digest) is None:
            await store.aput(self._namespace, digest, {"base64": payload}, index=False)
        return {key: value for key, value in block.items() if key != "base64"} | {_BLOB_REF: digest}

    def _offload_message(self, message: AnyMessage, store: BaseStore) -> AnyMessage:
        if not isinstance(message.content, list):
            return message
        content = [self._offload_block(block, store) if isinstance(block, dict) else block for block in message.content]
        return message if content == message.content else message.model_copy(update={"content": content})

    async def _aoffload_message(self, message: AnyMessage, store: BaseStore) -> AnyMessage:
        if not isinstance(message.content, list):
            return message
        content = [await self._aoffload_block(block, store) if isinstance(block, dict) else block for block in message.content]
        return message if content == message.content else message.model_copy(update={"content": content})

    def _offload_result(self, result: ToolMessage | Command[Any], store: BaseStore) -> ToolMessage | Command[Any]:
        if isinstance(result, ToolMessage):
            message = self._offload_message(result, store)
            return ToolMessage.model_validate(message.model_dump())
        if not isinstance(result.update, dict) or not isinstance(result.update.get("messages"), Sequence):
            return result
        messages = [self._offload_message(message, store) if isinstance(message, BaseMessage) else message for message in result.update["messages"]]
        return Command(graph=result.graph, update={**result.update, "messages": messages}, resume=result.resume, goto=result.goto)

    async def _aoffload_result(self, result: ToolMessage | Command[Any], store: BaseStore) -> ToolMessage | Command[Any]:
        if isinstance(result, ToolMessage):
            message = await self._aoffload_message(result, store)
            return ToolMessage.model_validate(message.model_dump())
        if not isinstance(result.update, dict) or not isinstance(result.update.get("messages"), Sequence):
            return result
        messages = [
            await self._aoffload_message(message, store) if isinstance(message, BaseMessage) else message for message in result.update["messages"]
        ]
        return Command(graph=result.graph, update={**result.update, "messages": messages}, resume=result.resume, goto=result.goto)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Offload successful binary `read_file` results."""
        result = handler(request)
        if request.tool_call["name"] != "read_file":
            return result
        store = self._resolve_store(request.runtime.store)
        return self._offload_result(result, store)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Offload successful binary `read_file` results asynchronously."""
        result = await handler(request)
        if request.tool_call["name"] != "read_file":
            return result
        store = self._resolve_store(request.runtime.store)
        return await self._aoffload_result(result, store)

    def _hydrate_block(self, block: dict[str, Any], store: BaseStore) -> dict[str, Any]:
        ref = block.get(_BLOB_REF)
        if not isinstance(ref, str):
            return block
        item = store.get(self._namespace, ref)
        if item is None or not isinstance(item.value.get("base64"), str):
            msg = f"Blob {ref!r} is missing from the configured store."
            raise RuntimeError(msg)
        return {key: value for key, value in block.items() if key != _BLOB_REF} | {"base64": item.value["base64"]}

    async def _ahydrate_block(self, block: dict[str, Any], store: BaseStore) -> dict[str, Any]:
        ref = block.get(_BLOB_REF)
        if not isinstance(ref, str):
            return block
        item = await store.aget(self._namespace, ref)
        if item is None or not isinstance(item.value.get("base64"), str):
            msg = f"Blob {ref!r} is missing from the configured store."
            raise RuntimeError(msg)
        return {key: value for key, value in block.items() if key != _BLOB_REF} | {"base64": item.value["base64"]}

    def _hydrate_message(self, message: AnyMessage, store: BaseStore) -> AnyMessage:
        if not isinstance(message.content, list):
            return message
        content = [self._hydrate_block(block, store) if isinstance(block, dict) else block for block in message.content]
        return message if content == message.content else message.model_copy(update={"content": content})

    async def _ahydrate_message(self, message: AnyMessage, store: BaseStore) -> AnyMessage:
        if not isinstance(message.content, list):
            return message
        content = [await self._ahydrate_block(block, store) if isinstance(block, dict) else block for block in message.content]
        return message if content == message.content else message.model_copy(update={"content": content})

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Hydrate blob references only in the provider-facing request."""
        store = self._resolve_store(request.runtime.store)
        messages = [self._hydrate_message(message, store) for message in request.messages]
        return handler(request.override(messages=messages))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Hydrate blob references only in the provider-facing request asynchronously."""
        store = self._resolve_store(request.runtime.store)
        messages = [await self._ahydrate_message(message, store) for message in request.messages]
        return await handler(request.override(messages=messages))
