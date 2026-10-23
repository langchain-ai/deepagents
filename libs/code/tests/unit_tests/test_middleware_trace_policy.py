"""Trace policy guard for the middleware that `create_cli_agent` installs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from deepagents_code.agent import create_cli_agent

if TYPE_CHECKING:
    from pathlib import Path


def _installed_middleware(tmp_path: Path) -> list[AgentMiddleware]:
    """Return the main and subagent middleware of a default interactive agent."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}
    with patch("deepagents_code.agent.create_deep_agent") as create:
        create_cli_agent(model=model, assistant_id="test", cwd=tmp_path)
    kwargs = create.call_args.kwargs
    middleware = list(kwargs["middleware"])
    for spec in kwargs.get("subagents") or []:
        middleware.extend(spec.get("middleware", []))
    return middleware


def test_dcode_middleware_omits_hook_inputs_from_traces(tmp_path: Path) -> None:
    """Every dcode middleware declares a trace policy.

    Without one, hook spans trace the full request, including the whole message
    history, on every model call.
    """
    middleware = [
        item
        for item in _installed_middleware(tmp_path)
        if type(item).__module__.startswith("deepagents_code.")
    ]

    assert middleware
    missing = [
        type(item).__qualname__
        for item in middleware
        if type(item).trace_policy is AgentMiddleware.trace_policy
    ]
    assert missing == []
