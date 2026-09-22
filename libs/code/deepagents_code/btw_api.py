"""Side-question generation and durable usage without conversation writes."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, cast

from starlette.responses import JSONResponse

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.btw_cost import answer_with_cost, load_cost
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from starlette.requests import Request

    from deepagents_code.cost_tracking import CostBreakdown, CostState

logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000


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


async def btw(request: Request) -> JSONResponse:
    """Answer without starting or updating a graph run.

    Returns:
        Answer text or a safe error response.
    """
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
            state = snapshot.get("values") or {}
            result = await _answer_while_connected(
                request,
                answer_with_cost(
                    operation.answer(thread_id, state, question.strip()),
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
    """Read the side-question subtotal without accessing or changing a graph.

    Returns:
        Persisted usage, or `null` if the thread has no side-question charges.
    """
    cost = await asyncio.to_thread(load_cost, request.path_params["thread_id"])
    return JSONResponse({"cost": cost})
