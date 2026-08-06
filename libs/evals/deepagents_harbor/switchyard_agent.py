"""LangGraph Harbor agent that removes setup egress before evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from harbor.agents.installed.langgraph import LangGraph

from deepagents_harbor.switchyard_environment import SwitchyardLangSmithEnvironment

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


class SwitchyardLangGraph(LangGraph):
    """Install with temporary egress, then run only through Switchyard."""

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Install the agent and fail closed while removing setup egress.

        Args:
            environment: Switchyard-enabled LangSmith environment for this trial.

        Raises:
            TypeError: If used without `SwitchyardLangSmithEnvironment`.
        """
        if not isinstance(environment, SwitchyardLangSmithEnvironment):
            msg = "SwitchyardLangGraph requires SwitchyardLangSmithEnvironment"
            raise TypeError(msg)
        try:
            await super().setup(environment)
        finally:
            await environment.isolate_main_after_setup()
