"""Simple extension template for quickly experimenting with custom agents."""

from __future__ import annotations

from pathlib import Path

from .deepagent_harbor import DeepAgentHarbor

#Change this to what you want
CUSTOM_SYSTEM_PROMPT = """You are a Harbor DeepAgent specialized for custom experiments.

Adjust this prompt, override methods, or replace the class entirely to suit
your use case. The default behavior inherits DeepAgentHarbor's tooling stack.
"""


class CustomDeepAgent(DeepAgentHarbor):
    """Starter agent that forwards to DeepAgentHarbor with a tweakable prompt.

    Replace the prompt, override `run`, or layer in additional initialization
    without having to touch the stock DeepAgentHarbor implementation. Point
    Harbor configs at `harbor_deepagents.agents.custom_agent:CustomDeepAgent`
    once you're ready to iterate on your own behavior.
    """

    def __init__(
        self,
        logs_dir: Path,
        *,
        system_prompt: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            system_prompt=system_prompt or CUSTOM_SYSTEM_PROMPT,
            **kwargs,
        )
