"""Harbor agent that runs the Deep Agents LangGraph project."""

from typing import Any

from harbor.agents.installed.langgraph import LangGraph


class DeepAgentsLangGraph(LangGraph):
    """Harbor's `langgraph` agent, plus the task's `skills_dir`.

    Harbor hands every agent the task's `[environment] skills_dir`, but its
    `langgraph` agent does not forward it. The runner's only channel into the
    graph is `configurable`, so pass it there for the graph factory to load.
    """

    capabilities = LangGraph.capabilities.model_copy(update={"skills": True})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the agent and forward `skills_dir` into `configurable`."""
        super().__init__(*args, **kwargs)
        if self.skills_dir:
            self.configurable.setdefault("skills_dir", self.skills_dir)
