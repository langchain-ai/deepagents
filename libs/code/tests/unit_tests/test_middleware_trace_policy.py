"""Trace-policy contracts for coding-agent middleware."""

import pytest
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code._glm_5p2_profile import _GlmTerminalStallRecovery
from deepagents_code.agent import AsyncApprovalHITLMiddleware, ShellAllowListMiddleware
from deepagents_code.ask_user import AskUserMiddleware
from deepagents_code.auto_mode import AutoModeHITLMiddleware, HeadlessMCPGuardMiddleware
from deepagents_code.configurable_model import ConfigurableModelMiddleware
from deepagents_code.cost_tracking import CostTrackingMiddleware
from deepagents_code.goal_rubric import (
    GoalCriteriaMiddleware,
    _ContextToolCallBudgetMiddleware,
    _CriteriaContextBudgetMiddleware,
    _GoalContextFallbackMiddleware,
    _RepositoryToolBudgetMiddleware,
    _WebSearchBudgetMiddleware,
)
from deepagents_code.goal_tools import GoalToolsMiddleware
from deepagents_code.hooks.server_middleware import ServerHooksMiddleware
from deepagents_code.local_context import LocalContextMiddleware
from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware
from deepagents_code.resume_state import ResumeStateMiddleware


@pytest.mark.parametrize(
    "middleware",
    [
        _GlmTerminalStallRecovery,
        ShellAllowListMiddleware,
        AsyncApprovalHITLMiddleware,
        AskUserMiddleware,
        AutoModeHITLMiddleware,
        HeadlessMCPGuardMiddleware,
        ConfigurableModelMiddleware,
        CostTrackingMiddleware,
        _GoalContextFallbackMiddleware,
        _CriteriaContextBudgetMiddleware,
        _ContextToolCallBudgetMiddleware,
        _RepositoryToolBudgetMiddleware,
        _WebSearchBudgetMiddleware,
        GoalCriteriaMiddleware,
        GoalToolsMiddleware,
        ServerHooksMiddleware,
        LocalContextMiddleware,
        ManagedMemoryGuardMiddleware,
        ResumeStateMiddleware,
    ],
)
def test_middleware_omits_trace_inputs(middleware: type[AgentMiddleware]) -> None:
    """Coding-agent middleware must not duplicate state in trace inputs."""
    assert middleware.trace_policy is not None
    assert middleware.trace_policy.process_inputs is not None
    assert middleware.trace_policy.process_inputs({"messages": ["secret"]}) == {}
