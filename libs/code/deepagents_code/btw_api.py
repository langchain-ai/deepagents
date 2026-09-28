"""Side-question generation and durable usage without conversation writes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, cast, override

from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.btw_cost import answer_with_cost, load_cost
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

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
        except (Exception, SystemExit):
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
