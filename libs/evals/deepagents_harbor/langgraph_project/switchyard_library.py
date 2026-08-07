"""Build in-process Switchyard routing for Harbor's bare Deep Agent."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from typing import Any

    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel

_CONFIG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_RUNTIME_DIR = Path(__file__).resolve().parent / ".local_deps" / "switchyard-runtime"
_SESSION_HEADER = "x-switchyard-session-id"


class _RunnableAlgorithm(Protocol):
    async def run(
        self,
        request: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run one buffered Switchyard request."""


class _Algorithms(Protocol):
    def passthrough(self, target: object) -> _RunnableAlgorithm:
        """Construct passthrough routing."""

    def llm_escalation(
        self,
        judge_target: object,
        efficient_target: object,
        capable_target: object,
        *,
        confirmations: int,
        recent_turn_window: int,
        window_message_chars: int,
        max_output_tokens: int,
    ) -> _RunnableAlgorithm:
        """Construct escalation routing."""


@dataclass(frozen=True, slots=True)
class _Bindings:
    llm_target: Callable[[str, object], object]
    client: Callable[[BaseChatModel], object]
    middleware: Callable[[_RunnableAlgorithm], AgentMiddleware[Any, Any, Any]]
    algorithms: _Algorithms


@dataclass(frozen=True, slots=True)
class SwitchyardComponents:
    """Model and middleware needed to route one Deep Agent through Switchyard.

    Attributes:
        model: Efficient or passthrough target used as Deep Agents' required base model.
        middleware: Middleware that replaces every model call with the routed model.
    """

    model: BaseChatModel
    middleware: AgentMiddleware[Any, Any, Any]


class _SessionAlgorithm:
    """Attach Harbor's stable task session id to every native algorithm call."""

    def __init__(self, algorithm: _RunnableAlgorithm, session_id: str) -> None:
        self._algorithm = algorithm
        self._headers = {_SESSION_HEADER: session_id}

    async def run(
        self,
        request: Mapping[str, object],
        headers: Mapping[str, str] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Run with the load-bearing session header regardless of caller headers."""
        if headers:
            msg = "Switchyard session wrapper does not accept caller-provided headers"
            raise ValueError(msg)
        return await self._algorithm.run(request, headers=self._headers)


def _activate_runtime(runtime_dir: Path) -> None:
    if not runtime_dir.is_dir():
        msg = f"Switchyard runtime artifact is missing: {runtime_dir}"
        raise RuntimeError(msg)
    runtime = str(runtime_dir)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)


def _load_bindings(runtime_dir: Path) -> _Bindings:
    _activate_runtime(runtime_dir)
    try:
        from langchain_nvidia_switchyard import (  # noqa: PLC0415  # ty: ignore[unresolved-import]  # artifact path activated above
            LangChainLlmClient,
            SwitchyardRoutingMiddleware,
        )
        from switchyard.libsy import (  # noqa: PLC0415  # ty: ignore[unresolved-import]  # artifact path activated above
            LlmTarget,
            algorithms,
        )
    except ImportError as exc:
        msg = f"Switchyard runtime artifact could not be imported from {runtime_dir}"
        raise RuntimeError(msg) from exc
    return _Bindings(
        llm_target=LlmTarget,
        client=LangChainLlmClient,
        middleware=SwitchyardRoutingMiddleware,
        algorithms=cast("_Algorithms", algorithms),
    )


def _table(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"Switchyard config {path} must be a table"
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"Switchyard config {path} must be a non-empty string"
        raise ValueError(msg)
    return value


def _positive_int(value: object, path: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"Switchyard config {path} must be a positive integer"
        raise ValueError(msg)
    return value


def _load_config(config_name: str, runtime_dir: Path) -> Mapping[str, object]:
    if not _CONFIG_NAME.fullmatch(config_name):
        msg = f"Invalid Switchyard config name: {config_name!r}"
        raise ValueError(msg)
    path = runtime_dir / "routes" / f"routes-{config_name}.toml"
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except FileNotFoundError as exc:
        msg = f"Switchyard route config is missing: {path}"
        raise RuntimeError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"Switchyard route config is invalid: {path}: {exc}"
        raise ValueError(msg) from exc


def _target_model(
    config: Mapping[str, object],
    target_name: str,
) -> BaseChatModel:
    targets = _table(config.get("targets"), "targets")
    target = _table(targets.get(target_name), f"targets.{target_name}")
    client_name = _string(target.get("llm_client"), f"targets.{target_name}.llm_client")
    clients = _table(config.get("llm_clients"), "llm_clients")
    client = _table(clients.get(client_name), f"llm_clients.{client_name}")

    model_id = _string(target.get("id"), f"targets.{target_name}.id")
    base_url = _string(client.get("base_url"), f"llm_clients.{client_name}.base_url")
    key_env = _string(client.get("api_key_env"), f"llm_clients.{client_name}.api_key_env")
    api_key = os.environ.get(key_env)
    if not api_key:
        msg = f"Switchyard target {target_name!r} requires environment variable {key_env}"
        raise RuntimeError(msg)

    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    extra_body = target.get("extra_body")
    if extra_body is not None:
        kwargs["extra_body"] = dict(_table(extra_body, f"targets.{target_name}.extra_body"))
    model_kwargs = target.get("model_kwargs")
    if model_kwargs is not None:
        kwargs["model_kwargs"] = dict(
            _table(model_kwargs, f"targets.{target_name}.model_kwargs")
        )

    client_format = _string(client.get("format"), f"llm_clients.{client_name}.format")
    if client_format == "openai_chat":
        kwargs["use_responses_api"] = False
        model_spec = f"openai:{model_id}"
    elif client_format == "anthropic_messages":
        model_spec = f"anthropic:{model_id}"
    else:
        msg = f"Switchyard target {target_name!r} uses unsupported client format {client_format!r}"
        raise ValueError(msg)
    return init_chat_model(model_spec, **kwargs)


def _switchyard_route(config: Mapping[str, object]) -> Mapping[str, object]:
    routes = _table(config.get("routes"), "routes")
    return _table(routes.get("switchyard"), "routes.switchyard")


def _passthrough_components(
    config: Mapping[str, object],
    route: Mapping[str, object],
    bindings: _Bindings,
    session_id: str,
) -> SwitchyardComponents:
    target_name = _string(route.get("target"), "routes.switchyard.target")
    model = _target_model(config, target_name)
    target = bindings.llm_target(target_name, bindings.client(model))
    algorithm = bindings.algorithms.passthrough(target)
    routed = cast("_RunnableAlgorithm", _SessionAlgorithm(algorithm, session_id))
    return SwitchyardComponents(model=model, middleware=bindings.middleware(routed))


def _escalation_components(
    config: Mapping[str, object],
    route: Mapping[str, object],
    bindings: _Bindings,
    session_id: str,
) -> SwitchyardComponents:
    judge_name = _string(route.get("classifier_target"), "routes.switchyard.classifier_target")
    weak_name = _string(route.get("weak_target"), "routes.switchyard.weak_target")
    strong_name = _string(route.get("strong_target"), "routes.switchyard.strong_target")
    judge_model = _target_model(config, judge_name)
    weak_model = _target_model(config, weak_name)
    strong_model = _target_model(config, strong_name)

    escalation = _table(route.get("escalation", {}), "routes.switchyard.escalation")
    algorithm = bindings.algorithms.llm_escalation(
        bindings.llm_target(judge_name, bindings.client(judge_model)),
        bindings.llm_target(weak_name, bindings.client(weak_model)),
        bindings.llm_target(strong_name, bindings.client(strong_model)),
        confirmations=_positive_int(
            escalation.get("confirmations"),
            "routes.switchyard.escalation.confirmations",
            2,
        ),
        recent_turn_window=_positive_int(
            escalation.get("recent_turn_window"),
            "routes.switchyard.escalation.recent_turn_window",
            28,
        ),
        window_message_chars=_positive_int(
            escalation.get("window_message_chars"),
            "routes.switchyard.escalation.window_message_chars",
            500,
        ),
        max_output_tokens=_positive_int(
            escalation.get("max_output_tokens"),
            "routes.switchyard.escalation.max_output_tokens",
            4096,
        ),
    )
    routed = cast("_RunnableAlgorithm", _SessionAlgorithm(algorithm, session_id))
    return SwitchyardComponents(model=weak_model, middleware=bindings.middleware(routed))


def build_switchyard_components(
    config_name: str,
    session_id: str,
    *,
    runtime_dir: Path | None = None,
) -> SwitchyardComponents:
    """Build provider models plus in-process Switchyard routing middleware.

    Args:
        config_name: Route arm name matching `routes-<name>.toml` in the runtime artifact.
        session_id: Stable Harbor task id used for escalation affinity.
        runtime_dir: Override for the assembled native runtime, primarily for tests.

    Returns:
        The base model and routing middleware for `create_deep_agent`.

    Raises:
        RuntimeError: If the runtime, route, provider credentials, or imports are missing.
        TypeError: If a route field has the wrong container type.
        ValueError: If the route has an unsupported or invalid shape.
    """
    if not session_id:
        msg = "HARBOR_SESSION_ID is required for a Switchyard library run"
        raise RuntimeError(msg)
    resolved_runtime = _DEFAULT_RUNTIME_DIR if runtime_dir is None else runtime_dir
    config = _load_config(config_name, resolved_runtime)
    bindings = _load_bindings(resolved_runtime)
    route = _switchyard_route(config)
    route_type = _string(route.get("type"), "routes.switchyard.type")
    if route_type == "passthrough":
        return _passthrough_components(config, route, bindings, session_id)
    if route_type == "llm_classifier" and route.get("mode") == "escalation":
        return _escalation_components(config, route, bindings, session_id)
    msg = "Switchyard library mode supports only passthrough and llm_classifier escalation routes"
    raise ValueError(msg)
