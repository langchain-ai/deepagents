"""Side-question generation and durable usage without conversation writes."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import sqlite3
from typing import TYPE_CHECKING, cast, override

from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.btw_cost import answer_with_cost, load_cost
from deepagents_code.btw_history import append_exchange, clear_history, load_history
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from starlette.requests import Request
    from starlette.types import Receive, Scope, Send

    from deepagents_code.cost_tracking import CostBreakdown, CostState

logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000
_MAX_HISTORY_LENGTH = 128_000
_UNREADABLE_HISTORY = (
    "Stored /btw history could not be read. Press Ctrl+X in /btw to clear it."
)


async def _remember(
    answer: Awaitable[tuple[str, CostBreakdown | None]],
    *,
    thread_id: str,
    epoch: int,
    question: str,
) -> tuple[str, CostBreakdown | None, bool]:
    """Store the exchange only after its answer completes within the request.

    A storage failure does not fail the answer, but the caller must tell the
    client so the user knows follow-ups will not see it.

    Returns:
        The completed answer, its cost, and whether storage failed.
    """
    text, cost = await answer
    try:
        await asyncio.to_thread(
            functools.partial(append_exchange, thread_id, question, text, epoch=epoch)
        )
    except (sqlite3.Error, ValueError):
        logger.warning("Could not save the side exchange", exc_info=True)
        return text, cost, True
    return text, cost, False


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
            Coroutine[object, object, tuple[str, CostBreakdown | None, bool]],
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
                text, cost, save_failed = await self._answer(on_text)
        except TimeoutError:
            await emit("error", {"detail": "Side question timed out. Try again."})
        except ClientDisconnect:
            raise  # A closed transport cannot receive an error event.
        except (Exception, SystemExit):
            logger.exception("Side question failed")
            await emit(
                "error",
                {"detail": "Side question failed on the server; see the server log."},
            )
        else:
            complete: dict[str, object] = {"text": text, "cost": cost}
            if save_failed:
                complete["save_failed"] = True
            await emit("complete", complete)
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
        if not isinstance(payload, dict) or payload.keys() != {"question", "workspace"}:
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
        thread_id = request.path_params["thread_id"]
        binding = await require_thread_workspace(thread_id, payload["workspace"])
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except WorkspaceConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    question = question.strip()
    try:
        history, epoch = await asyncio.to_thread(load_history, thread_id)
    except ValueError:
        logger.exception("Could not read side history")
        return JSONResponse({"detail": _UNREADABLE_HISTORY}, status_code=500)
    try:
        if sum(len(q) + len(a) for q, a in history) > _MAX_HISTORY_LENGTH:
            return JSONResponse(
                {
                    "detail": "Side conversation is too long. "
                    "Press Ctrl+X in /btw to clear it."
                },
                status_code=422,
            )
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

            def answer(
                on_text: Callable[[str], Awaitable[None]] | None = None,
            ) -> Coroutine[object, object, tuple[str, CostBreakdown | None, bool]]:
                return _remember(
                    answer_with_cost(
                        operation.answer(
                            thread_id,
                            state,
                            question,
                            history=history,
                            on_text=on_text,
                        ),
                        thread_id=thread_id,
                        state=cast("CostState", state),
                    ),
                    thread_id=thread_id,
                    epoch=epoch,
                    question=question,
                )

            if "text/event-stream" in request.headers.get("accept", ""):
                return _BtwStreamingResponse(request, answer)
            result = await _answer_while_connected(request, answer())
        if result is None:
            return JSONResponse({"detail": "Client disconnected."}, status_code=499)
        text, cost, save_failed = result
        response: dict[str, str | CostBreakdown | bool] = {"text": text}
        if cost is not None:
            response["cost"] = cost
        if save_failed:
            response["save_failed"] = True
        return JSONResponse(response)
    except TimeoutError:
        return JSONResponse(
            {"detail": "Side question timed out. Try again."}, status_code=504
        )
    except (Exception, SystemExit):
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


async def _bound_thread(request: Request) -> str | JSONResponse:
    """Confirm the caller's workspace owns the thread before history access.

    Returns:
        The thread ID, or an error response when the binding does not match.
    """
    thread_id = request.path_params["thread_id"]
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or payload.keys() != {"workspace"}:
            return JSONResponse({"detail": "Expected workspace."}, status_code=422)
        await require_thread_workspace(thread_id, payload["workspace"])
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except WorkspaceConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    return thread_id


async def btw_history(request: Request) -> JSONResponse:
    """Read the stored side exchanges so a client can show them on open.

    Returns:
        Ordered question/answer pairs for the thread, or a safe error response.
    """
    thread_id = await _bound_thread(request)
    if isinstance(thread_id, JSONResponse):
        return thread_id
    try:
        history, _ = await asyncio.to_thread(load_history, thread_id)
    except ValueError:
        logger.exception("Could not read side history")
        return JSONResponse({"detail": _UNREADABLE_HISTORY}, status_code=500)
    return JSONResponse({"history": history})


async def btw_clear(request: Request) -> JSONResponse:
    """Empty the side history so answers still in progress are not stored.

    Returns:
        The now-empty history, or a safe error response.
    """
    thread_id = await _bound_thread(request)
    if isinstance(thread_id, JSONResponse):
        return thread_id
    await asyncio.to_thread(clear_history, thread_id)
    return JSONResponse({"history": []})
