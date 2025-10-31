"""Tests for token counting utilities."""

from unittest.mock import Mock

from deepagents_cli.token_utils import calculate_baseline_tokens, get_memory_system_prompt


def test_get_memory_system_prompt_returns_string():
    """Test that get_memory_system_prompt returns a formatted string."""
    prompt = get_memory_system_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # Should contain the memory path
    assert "/memories/" in prompt


def test_get_memory_system_prompt_formatting():
    """Test that get_memory_system_prompt properly formats the template."""
    prompt = get_memory_system_prompt()

    # The prompt should be formatted with the memory path
    # Check that there are no unformatted template placeholders
    assert "{memory_path}" not in prompt


def test_calculate_baseline_tokens_with_agent_md(tmp_path):
    """Test calculate_baseline_tokens with an agent.md file."""
    # Create a temporary agent directory with agent.md
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    agent_md = agent_dir / "agent.md"
    agent_md.write_text("# Test Agent\n\nThis is a test agent.")

    # Create a mock model with token counting
    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.return_value = 100

    system_prompt = "You are a helpful assistant."

    tokens = calculate_baseline_tokens(mock_model, agent_dir, system_prompt)

    # Should return the token count from the model
    assert tokens == 100
    # Should have called the model's token counting method
    mock_model.get_num_tokens_from_messages.assert_called_once()


def test_calculate_baseline_tokens_without_agent_md(tmp_path):
    """Test calculate_baseline_tokens without an agent.md file."""
    # Create a temporary agent directory without agent.md
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    # Create a mock model
    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.return_value = 50

    system_prompt = "You are a helpful assistant."

    tokens = calculate_baseline_tokens(mock_model, agent_dir, system_prompt)

    # Should still return a token count
    assert tokens == 50
    mock_model.get_num_tokens_from_messages.assert_called_once()


def test_calculate_baseline_tokens_includes_system_components(tmp_path):
    """Test that calculate_baseline_tokens includes all system prompt components."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    agent_md = agent_dir / "agent.md"
    agent_md.write_text("Test memory")

    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.return_value = 200

    system_prompt = "Base prompt"

    calculate_baseline_tokens(mock_model, agent_dir, system_prompt)

    # Check that the messages passed to the model include all components
    call_args = mock_model.get_num_tokens_from_messages.call_args[0][0]
    assert len(call_args) == 1  # Should be one SystemMessage
    message_content = call_args[0].content

    # Should include agent memory
    assert "<agent_memory>" in message_content
    assert "Test memory" in message_content

    # Should include base prompt
    assert "Base prompt" in message_content

    # Should include memory system prompt
    assert "/memories/" in message_content


def test_calculate_baseline_tokens_handles_exception(tmp_path):
    """Test that calculate_baseline_tokens handles exceptions gracefully."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    # Create a mock model that raises an exception
    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.side_effect = Exception("Token counting failed")

    system_prompt = "Test prompt"

    # Should return 0 on exception, not raise
    tokens = calculate_baseline_tokens(mock_model, agent_dir, system_prompt)
    assert tokens == 0


def test_calculate_baseline_tokens_with_empty_agent_md(tmp_path):
    """Test calculate_baseline_tokens with an empty agent.md file."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    agent_md = agent_dir / "agent.md"
    agent_md.write_text("")  # Empty file

    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.return_value = 75

    system_prompt = "Prompt"

    tokens = calculate_baseline_tokens(mock_model, agent_dir, system_prompt)

    # Should still work with empty agent.md
    assert tokens == 75


def test_calculate_baseline_tokens_creates_system_message(tmp_path):
    """Test that calculate_baseline_tokens creates a proper SystemMessage."""
    from langchain_core.messages import SystemMessage

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    mock_model = Mock()
    mock_model.get_num_tokens_from_messages.return_value = 100

    calculate_baseline_tokens(mock_model, agent_dir, "Test")

    # Verify SystemMessage was created
    call_args = mock_model.get_num_tokens_from_messages.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
