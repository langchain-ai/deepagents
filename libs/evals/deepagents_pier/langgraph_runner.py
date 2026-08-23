from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import inspect
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Expected {config_path} to contain a JSON object")
    return config


def _graph_path(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and isinstance(spec.get("path"), str):
        return spec["path"]
    raise TypeError("Graph spec must be a string or object with a string 'path' field")


def _select_graph(config: dict[str, Any], graph_name: str | None) -> tuple[str, str]:
    graphs = config.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise ValueError("langgraph.json must define at least one graph")

    if graph_name is None:
        if len(graphs) != 1:
            available = ", ".join(sorted(str(name) for name in graphs))
            raise ValueError(
                "langgraph.json defines multiple graphs; pass --graph. "
                f"Available graphs: {available}"
            )
        graph_name = next(iter(graphs))  # ty: ignore[invalid-assignment]

    if graph_name not in graphs:
        available = ", ".join(sorted(str(name) for name in graphs))
        raise ValueError(f"Unknown graph '{graph_name}'. Available graphs: {available}")

    return graph_name, _graph_path(graphs[graph_name])


def _is_node_graph(path_spec: str) -> bool:
    graph_file = path_spec.split(":", maxsplit=1)[0]
    return Path(graph_file).suffix in {".ts", ".mts", ".cts", ".js", ".mjs", ".cjs"}


def _load_graph(project_dir: Path, path_spec: str) -> Any:
    if ":" not in path_spec:
        raise ValueError(
            f"Invalid graph path '{path_spec}'. Expected './path/to/file.py:attribute'"
        )
    if _is_node_graph(path_spec):
        raise ValueError("The Harbor langgraph agent supports Python graphs only")

    module_path_raw, attribute_path = path_spec.split(":", maxsplit=1)
    module_path = (project_dir / module_path_raw).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Graph module not found: {module_path}")

    sys.path.insert(0, str(project_dir))
    module_name = f"_harbor_langgraph_{module_path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import graph module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    value: Any = module
    for part in attribute_path.split("."):
        value = getattr(value, part)
    return value


def _message_to_jsonable(value: Any) -> dict[str, Any] | None:
    if not hasattr(value, "type") or not hasattr(value, "content"):
        return None
    data: dict[str, Any] = {
        "type": getattr(value, "type"),
        "content": _to_jsonable(getattr(value, "content")),
    }
    tool_calls = getattr(value, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = _to_jsonable(tool_calls)
    usage_metadata = getattr(value, "usage_metadata", None)
    if usage_metadata:
        data["usage_metadata"] = _to_jsonable(usage_metadata)
    return data


def _to_jsonable(value: Any) -> Any:
    message = _message_to_jsonable(value)
    if message is not None:
        return message
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_text(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        for message in reversed(value["messages"]):
            if hasattr(message, "content"):
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
                return repr(content)
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if content is not None:
                    return repr(content)
    return json.dumps(_to_jsonable(value), indent=2)


def _aggregate_usage(result: Any) -> dict[str, int]:
    """Sum token usage across the result's messages (best effort, always ints)."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
    }
    messages = result.get("messages") if isinstance(result, dict) else None
    for message in messages or []:
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["output_tokens"] += int(usage.get("output_tokens") or 0)
            totals["total_tokens"] += int(usage.get("total_tokens") or 0)
            details = usage.get("input_token_details")
            if isinstance(details, dict):
                totals["cache_read_tokens"] += int(details.get("cache_read") or 0)
    return totals


def _load_json_arg(value: str | None, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be a JSON object")
    return parsed


@contextlib.asynccontextmanager
async def _resolved_graph(loaded: Any, config: dict[str, Any]) -> AsyncIterator[Any]:
    """Yield an invokable graph from a loaded ``langgraph.json`` target.

    Supports two shapes:

    - a pre-compiled graph (exposes ``invoke``/``ainvoke``): used as-is.
    - a factory callable that builds the graph from ``config`` (the LangGraph
      deployment convention, e.g. ``def make_graph(config): ...``). The factory may
      be sync or async and may return the graph directly or a sync/async context
      manager. Building the graph here, from ``config``, lets the model be chosen at
      runtime via ``config["configurable"]["model"]`` instead of being baked in at
      import time — the same pattern LangGraph platform / agent builder use.
    """
    if hasattr(loaded, "ainvoke") or hasattr(loaded, "invoke"):
        yield loaded
        return
    if not callable(loaded):
        raise TypeError(
            "Graph target must be a compiled graph or a factory callable that returns one"
        )

    try:
        takes_arg = len(inspect.signature(loaded).parameters) >= 1
    except (TypeError, ValueError):
        takes_arg = True
    produced = loaded(config) if takes_arg else loaded()

    if hasattr(produced, "__aenter__"):
        async with produced as graph:
            yield graph
    elif hasattr(produced, "__enter__"):
        with produced as graph:
            yield graph
    elif inspect.isawaitable(produced):
        yield await produced
    else:
        yield produced


async def _ainvoke(
    graph: Any, input_value: dict[str, Any], config: dict[str, Any]
) -> Any:
    if hasattr(graph, "ainvoke"):
        return await graph.ainvoke(input_value, config=config)
    if hasattr(graph, "invoke"):
        return graph.invoke(input_value, config=config)
    raise TypeError("Selected graph must expose invoke() or ainvoke()")


def _invoke_config(configurable: dict[str, Any], graph_name: str) -> dict[str, Any]:
    return {
        "configurable": configurable,
        "metadata": {
            "ls_runner": "harbor",
            "harbor_agent": "langgraph",
            "langgraph_graph": graph_name,
        },
    }


def _parent_tracing_context() -> Any:
    """Nest this rollout under a harbor-provided LangSmith parent run.

    When harbor passes a parent trace context (the ``langsmith-trace`` /
    ``baggage`` distributed-tracing headers) via ``HARBOR_LANGSMITH_PARENT`` /
    ``HARBOR_LANGSMITH_BAGGAGE``, the graph's trace is attached under that parent
    so the agent trajectory shows up inside the harbor experiment run rather than
    as a disconnected trace. Returns a no-op context when no parent is set.
    """
    parent = os.environ.get("HARBOR_LANGSMITH_PARENT")
    if not parent:
        return contextlib.nullcontext()
    try:
        # langsmith is provided by the project's venv inside the environment, not by
        # harbor itself, so this import is intentionally lazy and may be absent.
        from langsmith.run_helpers import tracing_context
    except ImportError:
        print(
            "HARBOR_LANGSMITH_PARENT is set but langsmith is not installed; "
            "rollout will not nest under the harbor experiment run",
            file=sys.stderr,
        )
        return contextlib.nullcontext()
    headers = {"langsmith-trace": parent}
    baggage = os.environ.get("HARBOR_LANGSMITH_BAGGAGE")
    if baggage:
        headers["baggage"] = baggage
    return tracing_context(parent=headers)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", default="langgraph.json")
    parser.add_argument("--graph")
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path")
    parser.add_argument("--model")
    parser.add_argument("--model-kwargs-json")
    parser.add_argument("--configurable-json")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    config_path = (project_dir / args.config).resolve()
    instruction = Path(args.instruction_file).read_text()

    langgraph_config = _load_config(config_path)
    graph_name, path_spec = _select_graph(langgraph_config, args.graph)
    graph = _load_graph(project_dir, path_spec)

    model_kwargs = _load_json_arg(args.model_kwargs_json, label="model_kwargs_json")
    configurable = _load_json_arg(args.configurable_json, label="configurable_json")
    configurable.setdefault(
        "thread_id", os.environ.get("HARBOR_SESSION_ID") or str(uuid.uuid4())
    )
    if args.model:
        configurable.setdefault("model", args.model)
    if model_kwargs:
        configurable.setdefault("model_kwargs", model_kwargs)

    invoke_config = _invoke_config(configurable, graph_name)
    async with _resolved_graph(graph, invoke_config) as resolved:
        with _parent_tracing_context():
            result = await _ainvoke(
                resolved,
                {"messages": [{"role": "user", "content": instruction}]},
                invoke_config,
            )

    jsonable_result = _to_jsonable(result)
    Path(args.result_path).write_text(json.dumps(jsonable_result, indent=2))
    Path(args.output_path).write_text(_extract_text(result))

    if args.summary_path:
        summary = {
            "answer_written": _extract_text(result),
            "usage": _aggregate_usage(result),
        }
        Path(args.summary_path).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
