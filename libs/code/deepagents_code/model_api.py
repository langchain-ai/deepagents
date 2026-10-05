"""Launch metadata and workspace-bound model resolution in the server process."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

from deepagents_code.model_config import ModelConfigError
from deepagents_code.model_metadata import ModelMetadata
from deepagents_code.workspace import WorkspaceConflictError, require_thread_workspace

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)


async def startup_model_metadata(_request: Request) -> JSONResponse:
    """Read launch metadata without binding or validating a conversation thread.

    Args:
        _request: The startup metadata request.

    Returns:
        Cached launch model properties or an availability error.
    """
    from deepagents_code.server_graph import get_server_runtime

    try:
        runtime = await get_server_runtime()
        if runtime.model_metadata is not None:
            return JSONResponse(runtime.model_metadata.to_payload())
    except (Exception, SystemExit):
        logger.exception("Server startup model metadata is unavailable")
    return JSONResponse(
        {"detail": "Startup model metadata is unavailable. Restart the server."},
        status_code=503,
    )


async def model_metadata(request: Request) -> JSONResponse:
    """Resolve metadata without committing a model switch or running inference.

    Returns:
        Model properties or a validation/availability error.
    """
    from deepagents_code.server_graph import (
        _resolve_bound_workspace_config,
        _workspace_runtime,
    )

    try:
        body = await request.json()
        if not isinstance(body, dict) or body.keys() - {
            "workspace",
            "model_spec",
            "extra_kwargs",
            "purpose",
        }:
            return JSONResponse(
                {"detail": "Invalid model metadata request."}, status_code=422
            )
        spec = body.get("model_spec")
        params = body.get("extra_kwargs")
        purpose = body.get("purpose", "main")
        if (
            not isinstance(purpose, str)
            or purpose not in {"main", "auxiliary"}
            or (purpose == "auxiliary" and (spec is None or params is not None))
        ):
            return JSONResponse(
                {"detail": "Invalid model resolution purpose or parameters."},
                status_code=422,
            )
        if (spec is not None and (not isinstance(spec, str) or not spec)) or (
            params is not None and not isinstance(params, dict)
        ):
            return JSONResponse(
                {"detail": "Invalid model specification or parameters."},
                status_code=422,
            )
        binding = await require_thread_workspace(
            request.path_params["thread_id"], body.get("workspace")
        )
        if spec is None and params is not None:
            return JSONResponse(
                {"detail": "Model parameters require a model specification."},
                status_code=422,
            )
        runtime = await _workspace_runtime(binding)
        if spec is None:
            metadata = runtime.model_metadata
            if metadata is None:
                return JSONResponse(
                    {"detail": "Model metadata is unavailable."}, status_code=503
                )
        else:
            if runtime.model_environment is None:
                return JSONResponse(
                    {"detail": "Model resolution environment is unavailable."},
                    status_code=503,
                )
            config = await _resolve_bound_workspace_config(binding)

            def resolve() -> ModelMetadata:
                from deepagents_code.config import (
                    create_model,
                    use_environment,
                )

                with use_environment(runtime.model_environment):
                    result = create_model(
                        spec,
                        extra_kwargs=params,
                        profile_overrides=(
                            config.profile_overrides if purpose == "main" else None
                        ),
                        cli_max_retries=config.cli_max_retries,
                    )
                profile = result.model.profile
                structured_output = (
                    profile.get("structured_output") if profile else None
                )
                return ModelMetadata(
                    result.model_name,
                    result.provider,
                    result.context_limit,
                    result.unsupported_modalities,
                    structured_output=structured_output,
                )

            metadata = await asyncio.to_thread(resolve)
        return JSONResponse(metadata.to_payload())
    except WorkspaceConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    except (ModelConfigError, TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except (Exception, SystemExit):
        logger.exception("Server model metadata resolution failed")
        return JSONResponse(
            {
                "detail": (
                    "The server could not resolve model metadata. "
                    "Check the server log and retry."
                )
            },
            status_code=503,
        )
