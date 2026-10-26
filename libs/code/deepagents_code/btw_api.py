"""Side-question generation and durable usage without conversation writes."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from functools import cache
from typing import TYPE_CHECKING, cast, override

from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.btw_cost import answer_with_cost, load_cost
from deepagents_code.workspace import (
    WorkspaceConflictError,
    _database_path,
    require_thread_workspace,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from starlette.requests import Request
    from starlette.types import Receive, Scope, Send

    from deepagents_code.cost_tracking import CostBreakdown, CostState

logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000
_MAX_HISTORY_LENGTH = 128_000


def _parse_history(raw: object) -> list[tuple[str, str]]:
    """Validate completed text exchanges without accepting message roles.

    Returns:
        Question/answer pairs for the side conversation.

    Raises:
        TypeError: If history is not a list.
        ValueError: If an exchange is malformed or history exceeds the limit.
    """
    history: list[tuple[str, str]] = []
    if not isinstance(raw, list):
        msg = "History must be a list of question/answer pairs."
        raise TypeError(msg)
    for pair in raw:
        match pair:
            case [str(question), str(answer)] if question.strip() and answer.strip():
                history.append((question, answer))
            case _:
                msg = "History must contain nonempty question/answer text pairs."
                raise ValueError(msg)
    if (
        sum(len(question) + len(answer) for question, answer in history)
        > _MAX_HISTORY_LENGTH
    ):
        msg = "Side conversation is too long. Press Ctrl+X in /btw to clear it."
        raise ValueError(msg)
    return history


async def _wait_for_disconnect(request: Request) -> None:
    """Listen after the request body has been fully consumed."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _answer_while_connected[T](
    request: Request, answer: Coroutine[object, object, T]
) -> T | None:
    """Keep generation scoped to this HTTP connection.

    Returns:
        The answer, or `None` if the client disconnected.
    """
    generation = asyncio.create_task(answer)
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            (generation, disconnect), return_when=asyncio.FIRST_COMPLETED
        )
        if disconnect in done:
            disconnect.result()
            return None
        return generation.result()
    finally:
        generation.cancel()
        disconnect.cancel()
        await asyncio.gather(generation, disconnect, return_exceptions=True)


class _BtwStreamingResponse(Response):
    """Send fragments directly, keeping generation scoped to the connection."""

    def __init__(
        self,
        request: Request,
        answer: Callable[
            [Callable[[str], Awaitable[None]]],
            Coroutine[object, object, tuple[str, CostBreakdown | None]],
        ],
    ) -> None:
        super().__init__(
            media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )
        del self.headers["content-length"]
        self._request = request
        self._answer = answer

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {"type": "http.response.start", "status": 200, "headers": self.raw_headers}
        )
        await _answer_while_connected(self._request, self._send_answer(send))
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _send_answer(self, send: Send) -> bool:
        async def emit(event: str, data: object) -> None:
            body = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            try:
                await send(
                    {"type": "http.response.body", "body": body, "more_body": True}
                )
            except OSError as exc:
                raise ClientDisconnect from exc

        async def on_text(text: str) -> None:
            await emit("text", text)

        try:
            async with asyncio.timeout(120):
                text, cost = await self._answer(on_text)
        except TimeoutError:
            await emit("error", {"detail": "Side question timed out. Try again."})
        except ClientDisconnect:
            raise  # A closed transport cannot receive an error event.
        except Exception, SystemExit:
            logger.exception("Side question failed")
            await emit(
                "error",
                {"detail": "Side question failed on the server; see the server log."},
            )
        else:
            await emit("complete", {"text": text, "cost": cost})
        return True


async def btw(request: Request) -> Response:
    """Answer without starting or updating a graph run.

    Returns:
        Answer text or a safe error response.
    """
    from langchain_core.messages import convert_to_messages

    from deepagents_code.offload_api import _thread_client, get_server_runtime

    try:
        payload = await request.json()
        if (
            not isinstance(payload, dict)
            or not {"question", "workspace"} <= payload.keys()
            or payload.keys() - {"question", "workspace", "history"}
        ):
            return JSONResponse(
                {"detail": "Expected question and workspace."}, status_code=422
            )
        question = payload["question"]
        if (
            not isinstance(question, str)
            or not 0 < len(question.strip()) <= _MAX_QUESTION_LENGTH
        ):
            return JSONResponse(
                {"detail": "Question must contain 1 to 16000 characters."},
                status_code=422,
            )
        history = _parse_history(payload.get("history", []))
        thread_id = request.path_params["thread_id"]
        binding = await require_thread_workspace(thread_id, payload["workspace"])
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except WorkspaceConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    try:
        async with asyncio.timeout(120):
            server = await get_server_runtime(binding)
            operation = getattr(server.backend, BTW_OPERATION_ATTR, None)
            if not isinstance(operation, BtwOperation):
                return JSONResponse(
                    {"detail": "This server does not support /btw."}, status_code=503
                )
            client = _thread_client()
            snapshot = await client.threads.get_state(thread_id)
            # Accounting must recognize prior AI messages, including legacy
            # responses without saved costs. Keep the checkpoint untouched.
            state = dict(snapshot.get("values") or {})
            state["messages"] = convert_to_messages(state.get("messages", []))
            if "text/event-stream" in request.headers.get("accept", ""):
                return _BtwStreamingResponse(
                    request,
                    lambda on_text: answer_with_cost(
                        operation.answer(
                            thread_id,
                            state,
                            question.strip(),
                            history=history,
                            on_text=on_text,
                        ),
                        thread_id=thread_id,
                        state=cast("CostState", state),
                    ),
                )
            result = await _answer_while_connected(
                request,
                answer_with_cost(
                    operation.answer(
                        thread_id, state, question.strip(), history=history
                    ),
                    thread_id=thread_id,
                    state=cast("CostState", state),
                ),
            )
        if result is None:
            return JSONResponse({"detail": "Client disconnected."}, status_code=499)
        text, cost = result
        response: dict[str, str | CostBreakdown] = {"text": text}
        if cost is not None:
            response["cost"] = cost
        return JSONResponse(response)
    except TimeoutError:
        return JSONResponse(
            {"detail": "Side question timed out. Try again."}, status_code=504
        )
    except Exception, SystemExit:
        logger.exception("Side question failed")
        return JSONResponse(
            {"detail": "Side question failed on the server; see the server log."},
            status_code=500,
        )


async def btw_cost(request: Request) -> JSONResponse:
    """Read persisted side-question usage independently of main-task state.

    Returns:
        The cumulative side-question breakdown, or `None` before any usage.
    """
    cost = await asyncio.to_thread(load_cost, request.path_params["thread_id"])
    return JSONResponse({"cost": cost})


_USAGE_CHANNELS = (
    "_session_cost_usd",
    "_session_cost_breakdown",
    "_last_model_request_at",
    "_last_cache_write",
    "_last_cache_use",
    "_last_cache_model_spec",
    "_last_cache_params",
    "_last_cache_endpoint",
    "_model_spec",
)
"""Checkpoint channels the client reads to settle cost and prompt-cache state."""


def _read_usage_channels(thread_id: str) -> dict[str, object]:
    """Read usage channels from the latest root checkpoint row.

    The row stores these channels inline. Delta channels such as `messages`
    are not replayed, so the cost does not grow with the thread.

    Returns:
        The usage channels present in the checkpoint; empty without one.
    """
    path = _database_path()
    if not path.exists():
        return {}
    with closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=0.05)
    ) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
        ).fetchone():
            return {}
        row = conn.execute(
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = '' "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    if row is None or not row[0] or not row[1]:
        return {}
    values = _serializer().loads_typed((row[0], row[1])).get("channel_values", {})
    return {key: values[key] for key in _USAGE_CHANNELS if key in values}


@cache
def _serializer() -> JsonPlusSerializer:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer()


def _load_usage(thread_id: str) -> dict[str, object]:
    return {"cost": load_cost(thread_id), "checkpoint": _read_usage_channels(thread_id)}


async def thread_usage(request: Request) -> JSONResponse:
    """Read cost and prompt-cache state without the message history.

    The client settles its display after every turn. Full thread state
    rebuilds and sends every message, which is megabytes on a long thread.
    This route sends only the channels the client reads, plus side spend.

    Returns:
        Side-question spend under `cost` and checkpoint usage channels under
            `checkpoint`.
    """
    usage = await asyncio.to_thread(_load_usage, request.path_params["thread_id"])
    return JSONResponse(usage)
