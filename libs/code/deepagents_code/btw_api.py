"""Read-only HTTP boundary for side questions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, cast

from starlette.responses import JSONResponse

from deepagents_code.btw import BTW_OPERATION_ATTR, BtwOperation
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from starlette.requests import Request

logger = logging.getLogger(__name__)
_MAX_QUESTION_LENGTH = 16_000
_GENERATION_PARAMS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "reasoning_effort",
        "reasoning",
        "thinking",
        "thinking_level",
        "thinking_config",
        "output_config",
        "effort",
        "verbosity",
        "stop",
        "stop_sequences",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    }
)
"""Only generation options may cross the HTTP boundary.

Provider configuration, credentials, endpoints, and arbitrary constructor kwargs
remain server-owned. In particular, `model_kwargs` and `extra_body` cannot bypass
this allowlist. `create_model` enforces the server's model policy on resolution.
"""


def _model_selection(
    payload: dict[str, object],
) -> tuple[str | None, dict[str, object]]:
    """Validate selection without accepting provider connection settings.

    Returns:
        The requested model and generation overrides.

    Raises:
        ValueError: If the selection has invalid or unsupported fields.
    """
    model = payload.get("model")
    params = payload.get("model_params", {})
    if model is not None and (not isinstance(model, str) or not model.strip()):
        msg = "model must be a non-empty string or null."
        raise ValueError(msg)
    if not isinstance(params, dict) or params.keys() - _GENERATION_PARAMS:
        msg = "model_params must contain only supported generation options."
        raise ValueError(msg)
    if params and not model:
        msg = "model is required with model_params."
        raise ValueError(msg)
    # The allowlist above narrows the decoded JSON object's keys to strings.
    return model, cast("dict[str, object]", params)


async def _wait_for_disconnect(request: Request) -> None:
    """Listen after the request body has been fully consumed."""
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _answer_while_connected(
    request: Request, answer: Coroutine[object, object, str]
) -> str | None:
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
        if (
            not isinstance(payload, dict)
            or not {"question", "workspace"} <= payload.keys()
            or payload.keys() - {"question", "workspace", "model", "model_params"}
        ):
            return JSONResponse(
                {"detail": "Expected question and workspace."}, status_code=422
            )
        model, params = _model_selection(payload)
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
            text = await _answer_while_connected(
                request,
                operation.answer(
                    thread_id,
                    state,
                    question.strip(),
                    model_spec=model,
                    model_params=params,
                ),
            )
        if text is None:
            return JSONResponse({"detail": "Client disconnected."}, status_code=499)
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
