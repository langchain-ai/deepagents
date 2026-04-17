"""Compile ``SubAgent``/``CompiledSubAgent`` specs into a name→Runnable map.

Reuses the same shape that :class:`deepagents.middleware.subagents.SubAgentMiddleware`
accepts so a user can configure one list of specs and feed them to both
the ``task`` tool and the REPL's ``swarm()`` global.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain_core.runnables import Runnable


def compile_subagents(
    subagents: Sequence[SubAgent | CompiledSubAgent],
) -> tuple[dict[str, Runnable], list[dict[str, str]]]:
    """Compile a list of subagent specs into a name→Runnable map and descriptions.

    Mirrors ``SubAgentMiddleware._get_subagents`` so callers can pass the
    same list of specs to both middlewares and get consistent behaviour.
    Descriptions are returned alongside the graph map so the caller can
    render them into the system prompt without re-introspecting the specs.
    """
    # Local imports: langchain / deepagents import chains are heavyweight,
    # and this module is imported at REPL middleware construction time
    # which should stay cheap when swarm is not configured.
    from langchain.agents import create_agent  # noqa: PLC0415
    from langchain.agents.middleware import HumanInTheLoopMiddleware  # noqa: PLC0415

    graphs: dict[str, Runnable] = {}
    descriptions: list[dict[str, str]] = []

    for spec in subagents:
        if "runnable" in spec:
            compiled = cast("CompiledSubAgent", spec)
            graphs[compiled["name"]] = compiled["runnable"]
            descriptions.append(
                {"name": compiled["name"], "description": compiled["description"]}
            )
            continue

        sub = cast("SubAgent", spec)
        if "model" not in sub:
            msg = f"SubAgent '{sub['name']}' must specify 'model'"
            raise ValueError(msg)
        if "tools" not in sub:
            msg = f"SubAgent '{sub['name']}' must specify 'tools'"
            raise ValueError(msg)

        from deepagents._models import resolve_model  # noqa: PLC0415

        model = resolve_model(sub["model"])
        middleware = list(sub.get("middleware", []))
        interrupt_on = sub.get("interrupt_on")
        if interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        graphs[sub["name"]] = create_agent(
            model,
            system_prompt=sub["system_prompt"],
            tools=sub["tools"],
            middleware=middleware,
            name=sub["name"],
            response_format=sub.get("response_format"),
        )
        descriptions.append({"name": sub["name"], "description": sub["description"]})

    return graphs, descriptions


def describe_subagent_graphs(graphs: Mapping[str, object]) -> str:
    """Render the available-subagent list for the system prompt."""
    return ", ".join(graphs.keys())
