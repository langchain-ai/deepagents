"""LangGraph Harbor agent that keeps provider credentials in Switchyard."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from harbor.agents.installed.langgraph import LangGraph

from deepagents_harbor.switchyard_environment import (
    SwitchyardDockerEnvironment,
    SwitchyardLangSmithEnvironment,
)

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext


class SwitchyardLangGraph(LangGraph):
    """Run through a sidecar and remove its provider access before verification."""

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Install the agent and apply provider-specific network isolation.

        Args:
            environment: Switchyard-enabled Docker or LangSmith environment.

        Raises:
            TypeError: If used without a Switchyard-enabled environment.
        """
        if isinstance(environment, SwitchyardDockerEnvironment):
            await super().setup(environment)
            return
        if not isinstance(environment, SwitchyardLangSmithEnvironment):
            msg = "SwitchyardLangGraph requires a Switchyard Harbor environment"
            raise TypeError(msg)
        try:
            await super().setup(environment)
        finally:
            await environment.isolate_main_after_setup()

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run through Switchyard and close the Docker sidecar afterward.

        Args:
            instruction: Harbor task instruction.
            environment: Switchyard-enabled Harbor environment for this trial.
            context: Agent context populated by the LangGraph runner.

        Raises:
            TypeError: If used without a Switchyard-enabled environment.
        """
        if not isinstance(
            environment,
            SwitchyardDockerEnvironment | SwitchyardLangSmithEnvironment,
        ):
            msg = "SwitchyardLangGraph requires a Switchyard Harbor environment"
            raise TypeError(msg)
        try:
            await super().run(instruction, environment, context)
        finally:
            if isinstance(environment, SwitchyardDockerEnvironment):
                await environment.capture_and_stop_switchyard()
