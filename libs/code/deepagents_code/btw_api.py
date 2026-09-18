"""Read-only HTTP boundary for side questions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000


async def btw(request: Request) -> JSONResponse:
    """Answer without starting or updating a graph run.

    Returns:
        Answer text or a safe error response.
    """
    from deepagents_code.offload_api import _thread_client, get_server_runtime

    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) != {"question", "workspace"}:
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
            snapshot = await _thread_client().threads.get_state(thread_id)
            state = snapshot.get("values") or {}
            text = await operation.answer(thread_id, state, question.strip())
        return JSONResponse({"text": text})
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
