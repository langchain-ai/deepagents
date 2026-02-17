"""Tests for ask_user tool integration in the CLI."""

from deepagents_cli.tool_display import format_tool_display


class TestAskUserToolDisplay:
    """Tests for ask_user formatting in tool_display."""

    def test_format_single_question(self) -> None:
        result = format_tool_display(
            "ask_user",
            {
                "questions": [
                    {"question": "What is your name?", "type": "text"},
                ]
            },
        )
        assert "ask_user" in result
        assert "1 question" in result

    def test_format_multiple_questions(self) -> None:
        result = format_tool_display(
            "ask_user",
            {
                "questions": [
                    {"question": "Name?", "type": "text"},
                    {
                        "question": "Color?",
                        "type": "multiple_choice",
                        "choices": [{"value": "red"}, {"value": "blue"}],
                    },
                ]
            },
        )
        assert "ask_user" in result
        assert "2 questions" in result

    def test_format_empty_questions(self) -> None:
        result = format_tool_display("ask_user", {"questions": []})
        assert "ask_user" in result
        assert "0 questions" in result

    def test_format_no_questions_key(self) -> None:
        result = format_tool_display("ask_user", {})
        assert "ask_user" in result
