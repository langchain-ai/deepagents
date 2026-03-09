from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents_codex.chat_models import ChatCodexOAuth, _convert_message


class TestConvertMessage:
    def test_system(self) -> None:
        msg = SystemMessage(content="You are helpful")
        result = _convert_message(msg)
        assert result == {"role": "system", "content": "You are helpful"}

    def test_human(self) -> None:
        msg = HumanMessage(content="Hi")
        result = _convert_message(msg)
        assert result == {"role": "user", "content": "Hi"}

    def test_ai(self) -> None:
        msg = AIMessage(content="Hello!")
        result = _convert_message(msg)
        assert result == {"role": "assistant", "content": "Hello!"}

    def test_ai_with_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search", "args": {"q": "test"}}
            ],
        )
        result = _convert_message(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"


class TestChatCodexOAuth:
    def test_llm_type(self) -> None:
        model = ChatCodexOAuth()
        assert model._llm_type == "codex-oauth"

    def test_identifying_params(self) -> None:
        model = ChatCodexOAuth(model="gpt-4o-mini")
        assert model._identifying_params == {"model": "gpt-4o-mini"}

    @patch.object(ChatCodexOAuth, "_get_client")
    def test_generate_non_streaming(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.chat_completions.return_value = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
        }
        mock_get_client.return_value = mock_client

        model = ChatCodexOAuth(streaming=False)
        result = model._generate([HumanMessage(content="Hello")])
        assert result.generations[0].message.content == "Hi!"

    def test_bind_tools(self) -> None:
        model = ChatCodexOAuth()
        tool_def = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        bound = model.bind_tools([tool_def])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
