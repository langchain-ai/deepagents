import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents.graph import create_deep_agent

from ..utils import assert_all_deepagent_qualities, get_soccer_scores, get_weather, sample_tool

SAMPLE_TOOL_CONFIG = {
    "sample_tool": True,
    "get_weather": False,
    "get_soccer_scores": {"allowed_decisions": ["approve", "reject"]},
}


class TestHITL:
    def test_hitl_agent(self):
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[sample_tool, get_weather, get_soccer_scores], interrupt_on=SAMPLE_TOOL_CONFIG, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Call the sample tool, get the weather in New York and get scores for the latest soccer games in parallel",
                    }
                ]
            },
            config=config,
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any([tool_call["name"] == "sample_tool" for tool_call in tool_calls])
        assert any([tool_call["name"] == "get_weather" for tool_call in tool_calls])
        assert any([tool_call["name"] == "get_soccer_scores" for tool_call in tool_calls])

        assert result["__interrupt__"] is not None
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        assert len(interrupts) == 2
        assert any([action_request["name"] == "sample_tool" for action_request in action_requests])
        assert any([action_request["name"] == "get_soccer_scores" for action_request in action_requests])
        review_configs = interrupts["review_configs"]
        assert any(
            [
                review_config["action_name"] == "sample_tool" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
                for review_config in review_configs
            ]
        )
        assert any(
            [
                review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "reject"]
                for review_config in review_configs
            ]
        )

        result2 = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)
        tool_results = [msg for msg in result2.get("messages", []) if msg.type == "tool"]
        assert any([tool_result.name == "sample_tool" for tool_result in tool_results])
        assert any([tool_result.name == "get_weather" for tool_result in tool_results])
        assert any([tool_result.name == "get_soccer_scores" for tool_result in tool_results])
        assert "__interrupt__" not in result2

    def test_subagent_with_hitl(self):
        checkpointer = MemorySaver()
        agent = create_deep_agent(tools=[sample_tool, get_weather, get_soccer_scores], interrupt_on=SAMPLE_TOOL_CONFIG, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the task tool to kick off the general-purpose subagent. Tell it to call the sample tool, get the weather in New York and get scores for the latest soccer games in parallel",
                    }
                ]
            },
            config=config,
        )
        assert result["__interrupt__"] is not None
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        assert len(interrupts) == 2
        assert any([action_request["name"] == "sample_tool" for action_request in action_requests])
        assert any([action_request["name"] == "get_soccer_scores" for action_request in action_requests])
        review_configs = interrupts["review_configs"]
        assert any(
            [
                review_config["action_name"] == "sample_tool" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
                for review_config in review_configs
            ]
        )
        assert any(
            [
                review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "reject"]
                for review_config in review_configs
            ]
        )
        result2 = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)
        assert "__interrupt__" not in result2

    def test_subagent_with_custom_interrupt_on(self):
        checkpointer = MemorySaver()
        agent = create_deep_agent(
            tools=[sample_tool, get_weather, get_soccer_scores],
            interrupt_on=SAMPLE_TOOL_CONFIG,
            checkpointer=checkpointer,
            subagents=[
                {
                    "name": "task_handler",
                    "description": "A subagent that can handle all sorts of tasks",
                    "system_prompt": "You are a task handler. You can handle all sorts of tasks.",
                    "tools": [sample_tool, get_weather, get_soccer_scores],
                    "interrupt_on": {"sample_tool": False, "get_weather": True, "get_soccer_scores": True},
                },
            ],
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the task tool to kick off the task_handler subagent. Tell it to call the sample tool, get the weather in New York and get scores for the latest soccer games in parallel",
                    }
                ]
            },
            config=config,
        )
        assert result["__interrupt__"] is not None
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        assert len(interrupts) == 2
        assert any([action_request["name"] == "get_weather" for action_request in action_requests])
        assert any([action_request["name"] == "get_soccer_scores" for action_request in action_requests])
        review_configs = interrupts["review_configs"]
        assert any(
            [
                review_config["action_name"] == "get_weather" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
                for review_config in review_configs
            ]
        )
        assert any(
            [
                review_config["action_name"] == "get_soccer_scores" and review_config["allowed_decisions"] == ["approve", "edit", "reject"]
                for review_config in review_configs
            ]
        )
        result2 = agent.invoke(Command(resume={"decisions": [{"type": "approve"}, {"type": "approve"}]}), config=config)
        assert "__interrupt__" not in result2
