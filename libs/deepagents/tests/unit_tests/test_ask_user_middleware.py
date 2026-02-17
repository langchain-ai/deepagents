"""Tests for AskUserMiddleware functionality."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents.middleware.ask_user import (
    AskUserMiddleware,
    AskUserRequest,
    AskUserResponse,
    Question,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


class TestAskUserMiddleware:
    """Tests for AskUserMiddleware behavior."""

    def test_middleware_adds_ask_user_tool(self) -> None:
        """Verify the middleware exposes an ask_user tool."""
        middleware = AskUserMiddleware()
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "ask_user"

    def test_middleware_tool_schema(self) -> None:
        """Verify the ask_user tool has correct schema."""
        middleware = AskUserMiddleware()
        tool = middleware.tools[0]
        schema = tool.tool_call_schema.model_json_schema()
        assert "questions" in schema.get("properties", {})

    def test_ask_user_request_type(self) -> None:
        """Verify AskUserRequest can be constructed correctly."""
        req = AskUserRequest(
            type="ask_user",
            questions=[
                Question(question="What color?", type="text"),
                Question(
                    question="Pick one:",
                    type="multiple_choice",
                    choices=[{"value": "A"}, {"value": "B"}],
                ),
            ],
            tool_call_id="test_id",
        )
        assert req["type"] == "ask_user"
        assert len(req["questions"]) == 2
        assert req["questions"][0]["type"] == "text"
        assert req["questions"][1]["type"] == "multiple_choice"
        assert len(req["questions"][1]["choices"]) == 2

    def test_ask_user_response_type(self) -> None:
        """Verify AskUserResponse can be constructed correctly."""
        resp = AskUserResponse(answers=["blue", "A"])
        assert resp["answers"] == ["blue", "A"]

    def test_middleware_system_prompt_injection(self) -> None:
        """Verify the middleware injects the system prompt."""
        middleware = AskUserMiddleware()

        fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_agent(
            model=fake_model,
            middleware=[middleware],
        )
        result = agent.invoke({"messages": [HumanMessage(content="hi")]})
        assert any(msg.type == "ai" and msg.content == "Hello" for msg in result["messages"])

    def test_custom_system_prompt(self) -> None:
        """Verify custom system prompt is used."""
        custom_prompt = "Custom ask user instructions"
        middleware = AskUserMiddleware(system_prompt=custom_prompt)
        assert middleware.system_prompt == custom_prompt

    def test_custom_tool_description(self) -> None:
        """Verify custom tool description is used."""
        custom_desc = "Custom tool description"
        middleware = AskUserMiddleware(tool_description=custom_desc)
        assert middleware.tool_description == custom_desc

    def test_ask_user_interrupt_flow(self) -> None:
        """Test the full interrupt/resume flow with ask_user."""
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="Let me ask you something.",
                        tool_calls=[
                            {
                                "name": "ask_user",
                                "args": {
                                    "questions": [
                                        {
                                            "question": "What is your name?",
                                            "type": "text",
                                        }
                                    ]
                                },
                                "id": "call_ask_1",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="Thanks for telling me your name!"),
                ]
            )
        )

        checkpointer = InMemorySaver()
        agent = create_agent(
            model=fake_model,
            middleware=[AskUserMiddleware()],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "test-ask-user-1"}}

        result = agent.invoke(
            {"messages": [HumanMessage(content="Ask me something")]},
            config=config,
        )

        assert "__interrupt__" in result
        interrupts = result["__interrupt__"]
        assert len(interrupts) == 1
        interrupt_value = interrupts[0].value
        assert interrupt_value["type"] == "ask_user"
        assert len(interrupt_value["questions"]) == 1
        assert interrupt_value["questions"][0]["question"] == "What is your name?"
        assert interrupt_value["tool_call_id"] == "call_ask_1"

        result2 = agent.invoke(
            Command(resume={"answers": ["Alice"]}),
            config=config,
        )

        assert "__interrupt__" not in result2
        tool_messages = [msg for msg in result2["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "Alice" in tool_messages[0].content
        assert "What is your name?" in tool_messages[0].content

    def test_ask_user_multiple_questions(self) -> None:
        """Test interrupt/resume with multiple questions."""
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ask_user",
                                "args": {
                                    "questions": [
                                        {
                                            "question": "Name?",
                                            "type": "text",
                                        },
                                        {
                                            "question": "Color?",
                                            "type": "multiple_choice",
                                            "choices": [
                                                {"value": "red"},
                                                {"value": "blue"},
                                            ],
                                        },
                                    ]
                                },
                                "id": "call_ask_2",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="Got it!"),
                ]
            )
        )

        checkpointer = InMemorySaver()
        agent = create_agent(
            model=fake_model,
            middleware=[AskUserMiddleware()],
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "test-ask-user-2"}}

        result = agent.invoke(
            {"messages": [HumanMessage(content="ask me")]},
            config=config,
        )

        assert "__interrupt__" in result
        interrupt_value = result["__interrupt__"][0].value
        assert len(interrupt_value["questions"]) == 2

        result2 = agent.invoke(
            Command(resume={"answers": ["Bob", "blue"]}),
            config=config,
        )

        assert "__interrupt__" not in result2
        tool_messages = [msg for msg in result2["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "Bob" in tool_messages[0].content
        assert "blue" in tool_messages[0].content

    def test_ask_user_with_create_deep_agent(self) -> None:
        """Test that enable_ask_user adds the tool via create_deep_agent."""
        from deepagents import create_deep_agent  # noqa: PLC0415

        fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_deep_agent(
            model=fake_model,
            enable_ask_user=True,
        )

        tool_names = set()
        for node_data in agent.get_graph().nodes.values():
            if hasattr(node_data, "data") and hasattr(node_data.data, "tools_by_name"):
                tool_names.update(node_data.data.tools_by_name.keys())

        assert "ask_user" in tool_names

    def test_ask_user_not_present_by_default(self) -> None:
        """Test that ask_user is NOT present when enable_ask_user=False."""
        from deepagents import create_deep_agent  # noqa: PLC0415

        fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        agent = create_deep_agent(
            model=fake_model,
            enable_ask_user=False,
        )

        tool_names = set()
        for node_data in agent.get_graph().nodes.values():
            if hasattr(node_data, "data") and hasattr(node_data.data, "tools_by_name"):
                tool_names.update(node_data.data.tools_by_name.keys())

        assert "ask_user" not in tool_names
