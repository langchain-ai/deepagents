"""Test command type allowlist functionality for execute tool."""

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP
from deepagents_acp.utils import extract_command_types
from tests.chat_model import GenericFakeChatModel


class TestExtractCommandTypes:
    """Test the extract_command_types function for multiple commands."""

    def test_simple_non_sensitive_command(self):
        """Test extracting command types from simple non-sensitive commands."""
        assert extract_command_types("ls -la") == ["ls"]
        assert extract_command_types("pwd") == ["pwd"]
        assert extract_command_types("cat file.txt") == ["cat"]

    def test_npm_commands_with_subcommands(self):
        """Test that npm commands include their subcommands."""
        assert extract_command_types("npm install") == ["npm install"]
        assert extract_command_types("npm test") == ["npm test"]
        assert extract_command_types("npm run build") == ["npm run build"]
        assert extract_command_types("npm start") == ["npm start"]

    def test_python_with_module_flag(self):
        """Test that python -m commands include the full module name."""
        assert extract_command_types("python -m pytest tests/") == ["python -m pytest"]
        assert extract_command_types("python3 -m pip install package") == ["python3 -m pip"]
        assert extract_command_types("python -m venv .venv") == ["python -m venv"]

    def test_python_with_code_flag(self):
        """Test that python -c commands include only the flag, not the code."""
        assert extract_command_types("python -c 'print(1)'") == ["python -c"]
        assert extract_command_types('python3 -c "import os"') == ["python3 -c"]
        # Different code should result in the same signature
        assert extract_command_types("python -c 'print(1)'") == extract_command_types(
            "python -c 'malicious code'"
        )

    def test_python_script_execution(self):
        """Test that python script.py is just 'python' without sensitive flags."""
        assert extract_command_types("python script.py") == ["python"]
        assert extract_command_types("python3 my_script.py --arg value") == ["python3"]

    def test_node_commands(self):
        """Test that node commands with -e or -p include only the flag, not the code."""
        assert extract_command_types("node -e 'console.log(1)'") == ["node -e"]
        assert extract_command_types("node -p 'process.version'") == ["node -p"]
        assert extract_command_types("node script.js") == ["node"]
        # Different code should result in the same signature
        assert extract_command_types("node -e 'console.log(1)'") == extract_command_types(
            "node -e 'malicious code'"
        )

    def test_npx_with_package(self):
        """Test that npx commands include the package name."""
        assert extract_command_types("npx jest") == ["npx jest"]
        assert extract_command_types("npx prettier --write .") == ["npx prettier"]

    def test_yarn_commands(self):
        """Test that yarn commands include their subcommands."""
        assert extract_command_types("yarn install") == ["yarn install"]
        assert extract_command_types("yarn test") == ["yarn test"]
        assert extract_command_types("yarn run build") == ["yarn run build"]

    def test_uv_commands(self):
        """Test that uv commands include their subcommands and targets."""
        assert extract_command_types("uv run pytest") == ["uv run pytest"]
        assert extract_command_types("uv run python script.py") == ["uv run python"]
        assert extract_command_types("uv pip install package") == ["uv pip"]
        assert extract_command_types("uv add requests") == ["uv add"]
        assert extract_command_types("uv sync") == ["uv sync"]

    def test_command_with_and_operator(self):
        """Test extracting command types from commands with && operator."""
        assert extract_command_types("cd /path && npm install") == ["cd", "npm install"]
        assert extract_command_types("cd /path && python -m pytest tests/") == [
            "cd",
            "python -m pytest",
        ]
        assert extract_command_types("mkdir dir && cd dir && npm test") == [
            "mkdir",
            "cd",
            "npm test",
        ]

    def test_command_with_pipes_and_and_operator(self):
        """Test extracting command types from commands with both pipes and &&."""
        # All commands in a pipeline are extracted from each && segment
        assert extract_command_types("ls -la | grep foo && cat file.txt") == ["ls", "grep", "cat"]
        assert extract_command_types("cd dir && ls | wc -l") == ["cd", "ls", "wc"]

    def test_empty_command(self):
        """Test extracting command types from empty string."""
        assert extract_command_types("") == []
        assert extract_command_types("   ") == []

    def test_command_with_trailing_and_operator(self):
        """Test extracting command types when && has trailing/leading spaces."""
        assert extract_command_types("cd /path  &&  npm install") == ["cd", "npm install"]
        assert extract_command_types("cd /path&& npm install") == ["cd", "npm install"]

    def test_duplicate_commands_preserved(self):
        """Test that duplicate command types are preserved."""
        assert extract_command_types("npm install && npm test && npm run build") == [
            "npm install",
            "npm test",
            "npm run build",
        ]

    def test_complex_real_world_command(self):
        """Test extracting command types from real-world complex command."""
        cmd = "cd /Users/jacoblee/langchain/deepagents/libs/acp && python -m pytest tests/test_agent.py -v"  # noqa: E501
        assert extract_command_types(cmd) == ["cd", "python -m pytest"]

    def test_security_python_different_modules(self):
        """Test that different python modules are treated as different command types."""
        # These should be different to prevent over-permissioning
        assert extract_command_types("python -m pytest") != extract_command_types("python -m pip")
        assert extract_command_types("python -m pytest") == ["python -m pytest"]
        assert extract_command_types("python -m pip install") == ["python -m pip"]
        assert extract_command_types("python -c 'code'") == ["python -c"]

    def test_security_npm_different_subcommands(self):
        """Test that different npm subcommands are treated as different command types."""
        # These should be different to prevent over-permissioning
        assert extract_command_types("npm install") != extract_command_types("npm test")
        assert extract_command_types("npm install") == ["npm install"]
        assert extract_command_types("npm test") == ["npm test"]


class TestCommandTypeAllowlist:
    """Test command type allowlist tracking."""

    def test_allowed_command_types_initialized(self):
        """Test that allowed command types dict is initialized."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)
        assert hasattr(server, "_allowed_command_types")
        assert isinstance(server._allowed_command_types, dict)
        assert len(server._allowed_command_types) == 0

    def test_can_add_allowed_command_type(self):
        """Test that command types can be added to allowlist."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)
        session_id = "test_session"

        # Initialize the set for this session
        server._allowed_command_types[session_id] = set()

        # Add some command types (with their full signatures for sensitive commands)
        server._allowed_command_types[session_id].add(("execute", "npm install"))
        server._allowed_command_types[session_id].add(("execute", "python -m pytest"))

        # Verify they're in the set
        assert ("execute", "npm install") in server._allowed_command_types[session_id]
        assert ("execute", "python -m pytest") in server._allowed_command_types[session_id]
        assert ("execute", "ls") not in server._allowed_command_types[session_id]
        # Verify that approving "npm install" doesn't approve "npm test"
        assert ("execute", "npm test") not in server._allowed_command_types[session_id]
        # Verify that approving "python -m pytest" doesn't approve "python -m pip"
        assert ("execute", "python -m pip") not in server._allowed_command_types[session_id]

    def test_command_types_are_session_specific(self):
        """Test that allowed command types are tracked per session."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)

        # Add command types for session 1
        session1_id = "session_1"
        server._allowed_command_types[session1_id] = {
            ("execute", "npm install"),
            ("execute", "python -m pytest"),
        }

        # Add different command types for session 2
        session2_id = "session_2"
        server._allowed_command_types[session2_id] = {("execute", "ls"), ("execute", "cat")}

        # Verify each session has its own set
        assert ("execute", "npm install") in server._allowed_command_types[session1_id]
        assert ("execute", "npm install") not in server._allowed_command_types[session2_id]
        assert ("execute", "ls") in server._allowed_command_types[session2_id]
        assert ("execute", "ls") not in server._allowed_command_types[session1_id]

    def test_multiple_command_types_in_single_command(self):
        """Test that commands with && require all command types to be allowed."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)

        session_id = "test_session"

        # Only allow 'cd' commands
        server._allowed_command_types[session_id] = {("execute", "cd")}

        # Verify that a command with both 'cd' and 'python' requires both to be allowed
        cmd1 = "cd /path && python script.py"
        types1 = extract_command_types(cmd1)
        assert types1 == ["cd", "python"]

        # Only 'cd' is allowed, so not all command types are allowed
        all_allowed = all(
            ("execute", cmd_type) in server._allowed_command_types[session_id]
            for cmd_type in types1
        )
        assert not all_allowed

        # Now allow 'python' as well
        server._allowed_command_types[session_id].add(("execute", "python"))

        # Now all command types should be allowed
        all_allowed = all(
            ("execute", cmd_type) in server._allowed_command_types[session_id]
            for cmd_type in types1
        )
        assert all_allowed

    def test_security_python_pytest_vs_pip(self):
        """Test that approving 'python -m pytest' doesn't auto-approve 'python -m pip'."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)

        session_id = "test_session"
        server._allowed_command_types[session_id] = {("execute", "python -m pytest")}

        # Commands with python -m pytest should be allowed
        cmd_pytest = "python -m pytest tests/"
        types_pytest = extract_command_types(cmd_pytest)
        assert types_pytest == ["python -m pytest"]
        assert all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_pytest
        )

        # Commands with python -m pip should NOT be allowed
        cmd_pip = "python -m pip install malicious-package"
        types_pip = extract_command_types(cmd_pip)
        assert types_pip == ["python -m pip"]
        assert not all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_pip
        )

        # Commands with python -c should NOT be allowed
        cmd_code = "python -c 'import os; os.system(\"rm -rf /\")'"
        types_code = extract_command_types(cmd_code)
        assert types_code == ["python -c"]
        assert not all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_code
        )

    def test_security_npm_install_vs_run(self):
        """Test that approving 'npm install' doesn't auto-approve 'npm run'."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)

        session_id = "test_session"
        server._allowed_command_types[session_id] = {("execute", "npm install")}

        # npm install should be allowed
        cmd_install = "npm install"
        types_install = extract_command_types(cmd_install)
        assert types_install == ["npm install"]
        assert all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_install
        )

        # npm run should NOT be allowed
        cmd_run = "npm run arbitrary-script"
        types_run = extract_command_types(cmd_run)
        assert types_run == ["npm run arbitrary-script"]
        assert not all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_run
        )

    def test_security_uv_run_pytest_vs_python(self):
        """Test that approving 'uv run pytest' doesn't auto-approve 'uv run python'."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))
        graph = create_deep_agent(model=model, checkpointer=MemorySaver())
        server = AgentServerACP(agent=graph)

        session_id = "test_session"
        server._allowed_command_types[session_id] = {("execute", "uv run pytest")}

        # uv run pytest should be allowed
        cmd_pytest = "uv run pytest tests/"
        types_pytest = extract_command_types(cmd_pytest)
        assert types_pytest == ["uv run pytest"]
        assert all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_pytest
        )

        # uv run python should NOT be allowed
        cmd_python = "uv run python script.py"
        types_python = extract_command_types(cmd_python)
        assert types_python == ["uv run python"]
        assert not all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_python
        )

        # uv pip should NOT be allowed
        cmd_pip = "uv pip install package"
        types_pip = extract_command_types(cmd_pip)
        assert types_pip == ["uv pip"]
        assert not all(
            ("execute", ct) in server._allowed_command_types[session_id] for ct in types_pip
        )
