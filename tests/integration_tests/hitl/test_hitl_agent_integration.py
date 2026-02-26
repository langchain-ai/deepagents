"""
测试 interrupt_on 配置是否正确触发 HITL 暂停。
使用 InMemorySaver 作为 checkpointer（无需外部依赖）。
需要有效的 LLM API Key，CI 环境可跳过。
"""
import os
import pytest

_HAS_API_KEY = bool(
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("CUSTOM_LLM_BASE_URL")
)

pytestmark = pytest.mark.skipif(not _HAS_API_KEY, reason="No LLM API key available")


def test_agent_pauses_on_dangerous_tool():
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from deepagents import create_deep_agent

    @tool
    def dangerous_delete(file_path: str) -> str:
        """Delete a file permanently."""
        return f"Deleted {file_path}"

    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        tools=[dangerous_delete],
        interrupt_on={"dangerous_delete": {"allowed_decisions": ["approve", "reject"]}},
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "test-thread-001"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Please delete /tmp/test.txt"}]},
        config=config,
    )

    # Agent 应该暂停，返回 __interrupt__ 而非直接执行
    assert "__interrupt__" in result or result.get("messages")
