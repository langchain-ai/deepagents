"""Comprehensive tests for custom slash commands feature.

Tests cover:
- Frontmatter parsing (YAML extraction)
- Command discovery (user and project level)
- Argument substitution ($ARGUMENTS, $1, $2, etc.)
- Shell command injection (!`command`)
- File inclusion (@filepath)
- Command execution and metadata
- Help formatting
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from deepagents_cli.slash_commands import (
    CommandExecutionResult,
    SlashCommand,
    _split_args,
    discover_commands_in_directory,
    execute_command,
    execute_shell_injections,
    format_commands_for_help,
    parse_allowed_tools,
    parse_command_file,
    parse_frontmatter,
    process_template,
    resolve_file_inclusions,
    substitute_arguments,
)


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_no_frontmatter(self):
        """Content without frontmatter returns empty dict and full content."""
        content = "This is just regular content\nwith multiple lines."
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_simple_frontmatter(self):
        """Basic frontmatter with description."""
        content = """---
description: Test command
---

This is the body."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {"description": "Test command"}
        assert body == "This is the body."

    def test_full_frontmatter(self):
        """Frontmatter with all supported fields."""
        content = """---
description: Deploy to production
allowed-tools: Bash(git:*), Read, Grep
argument-hint: "[environment] [version]"
model: claude-3-5-haiku-20241022
disable-model-invocation: true
---

Deploy $1 with version $2"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["description"] == "Deploy to production"
        assert frontmatter["allowed-tools"] == "Bash(git:*), Read, Grep"
        assert frontmatter["argument-hint"] == "[environment] [version]"
        assert frontmatter["model"] == "claude-3-5-haiku-20241022"
        assert frontmatter["disable-model-invocation"] is True
        assert body == "Deploy $1 with version $2"

    def test_missing_end_delimiter(self):
        """Missing end delimiter treats content as body."""
        content = """---
description: Incomplete
This is the body."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert "---" in body

    def test_invalid_yaml(self):
        """Invalid YAML returns empty dict."""
        content = """---
invalid: yaml: content:
  - broken
---

Body here."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == "Body here."

    def test_empty_frontmatter(self):
        """Empty frontmatter block."""
        content = """---
---

Just body."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == "Just body."


class TestParseAllowedTools:
    """Tests for parsing allowed-tools field."""

    def test_none_value(self):
        """None returns empty list."""
        assert parse_allowed_tools(None) == []

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert parse_allowed_tools("") == []

    def test_simple_list_string(self):
        """Comma-separated tools."""
        result = parse_allowed_tools("Read, Grep, Write")
        assert result == ["Read", "Grep", "Write"]

    def test_tools_with_patterns(self):
        """Tools with parenthesized patterns."""
        result = parse_allowed_tools("Bash(git add:*), Bash(git status:*), Read")
        assert result == ["Bash(git add:*)", "Bash(git status:*)", "Read"]

    def test_nested_parentheses(self):
        """Complex patterns with nested content."""
        result = parse_allowed_tools("Bash(cmd(arg)), Tool")
        assert result == ["Bash(cmd(arg))", "Tool"]

    def test_list_input(self):
        """List input passes through."""
        result = parse_allowed_tools(["Bash(git:*)", "Read"])
        assert result == ["Bash(git:*)", "Read"]


class TestParseCommandFile:
    """Tests for parsing command markdown files."""

    def test_basic_command_file(self):
        """Parse a basic command file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""---
description: Run tests
---

Run the test suite for $ARGUMENTS""")
            f.flush()
            path = Path(f.name)

        try:
            cmd = parse_command_file(path, "user")
            assert cmd.name == path.stem.lower()
            assert cmd.description == "Run tests"
            assert cmd.template == "Run the test suite for $ARGUMENTS"
            assert cmd.source == "user"
            assert cmd.namespace is None
        finally:
            path.unlink()

    def test_command_with_namespace(self):
        """Command file with namespace."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Deploy command")
            f.flush()
            path = Path(f.name)

        try:
            cmd = parse_command_file(path, "project", namespace="ci")
            assert cmd.source == "project"
            assert cmd.namespace == "ci"
            assert cmd.display_source == "project:ci"
        finally:
            path.unlink()

    def test_command_all_fields(self):
        """Command with all frontmatter fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""---
description: Full command
allowed-tools: Bash(git:*), Read
argument-hint: "[file]"
model: claude-3-5-haiku
disable-model-invocation: true
---

Process @$1""")
            f.flush()
            path = Path(f.name)

        try:
            cmd = parse_command_file(path, "user")
            assert cmd.description == "Full command"
            assert cmd.allowed_tools == ["Bash(git:*)", "Read"]
            assert cmd.argument_hint == "[file]"
            assert cmd.model == "claude-3-5-haiku"
            assert cmd.disable_model_invocation is True
        finally:
            path.unlink()


class TestDiscoverCommands:
    """Tests for command discovery."""

    def test_discover_empty_directory(self):
        """Empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            commands = discover_commands_in_directory(Path(tmpdir), "user")
            assert commands == {}

    def test_discover_nonexistent_directory(self):
        """Nonexistent directory returns empty dict."""
        commands = discover_commands_in_directory(Path("/nonexistent/path"), "user")
        assert commands == {}

    def test_discover_single_command(self):
        """Discover a single command file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_file = Path(tmpdir) / "deploy.md"
            cmd_file.write_text("""---
description: Deploy app
---

Deploy to production""")

            commands = discover_commands_in_directory(Path(tmpdir), "project")
            assert "deploy" in commands
            assert commands["deploy"].description == "Deploy app"
            assert commands["deploy"].source == "project"

    def test_discover_multiple_commands(self):
        """Discover multiple command files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("Run tests")
            (Path(tmpdir) / "build.md").write_text("Build project")
            (Path(tmpdir) / "deploy.md").write_text("Deploy app")

            commands = discover_commands_in_directory(Path(tmpdir), "user")
            assert len(commands) == 3
            assert "test" in commands
            assert "build" in commands
            assert "deploy" in commands

    def test_discover_with_subdirectories(self):
        """Discover commands in subdirectories with namespaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            ci_dir = Path(tmpdir) / "ci"
            ci_dir.mkdir()
            (ci_dir / "test.md").write_text("CI test command")

            commands = discover_commands_in_directory(Path(tmpdir), "project")
            assert "test" in commands
            assert commands["test"].namespace == "ci"
            assert commands["test"].display_source == "project:ci"

    def test_discover_user_and_project_commands(self):
        """Project commands override user commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate user and project directories
            user_dir = Path(tmpdir) / "user" / "commands"
            user_dir.mkdir(parents=True)
            (user_dir / "deploy.md").write_text("""---
description: User deploy
---

User deploy command""")

            project_dir = Path(tmpdir) / "project" / "commands"
            project_dir.mkdir(parents=True)
            (project_dir / "deploy.md").write_text("""---
description: Project deploy
---

Project deploy command""")

            # Mock the paths
            with patch("deepagents_cli.slash_commands.Path") as mock_path:
                mock_path.home.return_value = Path(tmpdir) / "user"

                # Manually test the override behavior
                user_commands = discover_commands_in_directory(user_dir, "user")
                project_commands = discover_commands_in_directory(project_dir, "project")

                # Simulate merging with project override
                all_commands = {**user_commands, **project_commands}
                assert all_commands["deploy"].source == "project"
                assert all_commands["deploy"].description == "Project deploy"


class TestSubstituteArguments:
    """Tests for argument substitution."""

    def test_arguments_placeholder(self):
        """$ARGUMENTS replaced with full args string."""
        template = "Process $ARGUMENTS"
        result = substitute_arguments(template, "file1.txt file2.txt")
        assert result == "Process file1.txt file2.txt"

    def test_positional_arguments(self):
        """$1, $2, etc. replaced with positional args."""
        template = "Copy $1 to $2"
        result = substitute_arguments(template, "source.txt dest.txt")
        assert result == "Copy source.txt to dest.txt"

    def test_mixed_placeholders(self):
        """Mix of $ARGUMENTS and positional."""
        template = "First: $1, All: $ARGUMENTS"
        result = substitute_arguments(template, "one two three")
        assert result == "First: one, All: one two three"

    def test_empty_arguments(self):
        """Empty arguments string."""
        template = "Run with $ARGUMENTS"
        result = substitute_arguments(template, "")
        assert result == "Run with "

    def test_missing_positional(self):
        """Missing positional arguments become empty."""
        template = "Args: $1, $2, $3"
        result = substitute_arguments(template, "only-one")
        assert result == "Args: only-one, , "

    def test_quoted_arguments(self):
        """Quoted strings preserved as single argument."""
        template = "File: $1, Message: $2"
        result = substitute_arguments(template, 'file.txt "hello world"')
        assert result == "File: file.txt, Message: hello world"

    def test_double_digit_positional(self):
        """$10 doesn't interfere with $1."""
        template = "$1 and $10"
        args = "a b c d e f g h i j"
        result = substitute_arguments(template, args)
        assert result == "a and j"


class TestSplitArgs:
    """Tests for argument splitting."""

    def test_simple_split(self):
        """Simple space-separated args."""
        assert _split_args("one two three") == ["one", "two", "three"]

    def test_quoted_string(self):
        """Quoted strings stay together."""
        assert _split_args('one "two three" four') == ["one", "two three", "four"]

    def test_single_quotes(self):
        """Single quotes work too."""
        assert _split_args("one 'two three' four") == ["one", "two three", "four"]

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert _split_args("") == []
        assert _split_args("   ") == []

    def test_multiple_spaces(self):
        """Multiple spaces handled correctly."""
        assert _split_args("one    two") == ["one", "two"]


class TestShellInjections:
    """Tests for shell command execution."""

    def test_simple_shell_command(self):
        """Execute simple shell command."""
        template = "Output: !`echo hello`"
        result = execute_shell_injections(template)
        assert result == "Output: hello"

    def test_multiple_shell_commands(self):
        """Multiple shell commands in template."""
        template = "First: !`echo one` Second: !`echo two`"
        result = execute_shell_injections(template)
        assert result == "First: one Second: two"

    def test_no_shell_commands(self):
        """Template without shell commands unchanged."""
        template = "No commands here"
        result = execute_shell_injections(template)
        assert result == "No commands here"

    def test_shell_command_with_cwd(self):
        """Shell command respects working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = "Dir: !`pwd`"
            result = execute_shell_injections(template, cwd=Path(tmpdir))
            assert tmpdir in result

    def test_shell_command_error(self):
        """Failed shell command returns error message."""
        template = "!`nonexistent_command_xyz`"
        result = execute_shell_injections(template)
        assert "[Error" in result or "not found" in result.lower()

    def test_shell_command_timeout(self):
        """Timed out command returns timeout message."""
        # This test would be slow, so we'll mock instead
        with patch("deepagents_cli.slash_commands.subprocess.run") as mock_run:
            import subprocess

            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            template = "!`sleep 100`"
            result = execute_shell_injections(template)
            assert "timed out" in result.lower()


class TestFileInclusions:
    """Tests for file content inclusion."""

    def test_simple_file_inclusion(self):
        """Include file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("File content here")

            template = f"Content: @{test_file}"
            result = resolve_file_inclusions(template)
            assert result == "Content: File content here"

    def test_relative_file_inclusion(self):
        """Include file with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "data.txt"
            test_file.write_text("Relative content")

            template = "@data.txt"
            result = resolve_file_inclusions(template, cwd=Path(tmpdir))
            assert result == "Relative content"

    def test_multiple_file_inclusions(self):
        """Multiple file inclusions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("AAA")
            (Path(tmpdir) / "b.txt").write_text("BBB")

            template = "@a.txt and @b.txt"
            result = resolve_file_inclusions(template, cwd=Path(tmpdir))
            assert result == "AAA and BBB"

    def test_nonexistent_file(self):
        """Nonexistent file returns error message."""
        template = "@nonexistent_file_xyz.txt"
        result = resolve_file_inclusions(template)
        assert "[File not found" in result

    def test_no_file_inclusions(self):
        """Template without file refs unchanged."""
        template = "No files @mentioned but email@example.com is not a file"
        # The pattern @[\w./\-_]+ won't match email@example.com fully
        result = resolve_file_inclusions(template)
        # Just check it doesn't crash
        assert "No files" in result


class TestProcessTemplate:
    """Tests for full template processing."""

    def test_full_processing_pipeline(self):
        """All substitutions applied in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.txt"
            test_file.write_text("CONFIG_VALUE")

            template = """Args: $ARGUMENTS
Shell: !`echo shell_output`
File: @config.txt"""

            result = process_template(
                template, "my args", cwd=Path(tmpdir), execute_shell=True, resolve_files=True
            )

            assert "Args: my args" in result
            assert "Shell: shell_output" in result
            assert "File: CONFIG_VALUE" in result

    def test_disable_shell_execution(self):
        """Shell execution can be disabled."""
        template = "!`echo hello`"
        result = process_template(template, "", execute_shell=False)
        assert result == "!`echo hello`"

    def test_disable_file_resolution(self):
        """File resolution can be disabled."""
        template = "@some/file.txt"
        result = process_template(template, "", resolve_files=False)
        assert result == "@some/file.txt"


class TestExecuteCommand:
    """Tests for command execution."""

    def test_execute_simple_command(self):
        """Execute a simple command."""
        cmd = SlashCommand(
            name="test",
            template="Run tests for $ARGUMENTS",
            path=Path("/test.md"),
            source="user",
            description="Run tests",
        )

        result = execute_command(cmd, "my-module")
        assert isinstance(result, CommandExecutionResult)
        assert result.prompt == "Run tests for my-module"
        assert result.command == cmd

    def test_execute_command_with_metadata(self):
        """Command metadata passed through."""
        cmd = SlashCommand(
            name="deploy",
            template="Deploy",
            path=Path("/deploy.md"),
            source="project",
            model="claude-3-5-haiku",
            allowed_tools=["Bash(git:*)"],
        )

        result = execute_command(cmd, "")
        assert result.model == "claude-3-5-haiku"
        assert result.allowed_tools == ["Bash(git:*)"]


class TestFormatCommandsForHelp:
    """Tests for help formatting."""

    def test_format_empty_commands(self):
        """Empty dict returns empty list."""
        result = format_commands_for_help({})
        assert result == []

    def test_format_single_command(self):
        """Format single command."""
        commands = {
            "deploy": SlashCommand(
                name="deploy",
                template="Deploy",
                path=Path("/deploy.md"),
                source="project",
                description="Deploy to production",
            )
        }

        result = format_commands_for_help(commands)
        assert len(result) == 1
        assert result[0] == ("/deploy", "Deploy to production", "project")

    def test_format_command_with_hint(self):
        """Command with argument hint."""
        commands = {
            "review": SlashCommand(
                name="review",
                template="Review PR",
                path=Path("/review.md"),
                source="user",
                description="Review PR",
                argument_hint="[pr-number]",
            )
        }

        result = format_commands_for_help(commands)
        assert result[0][0] == "/review [pr-number]"

    def test_format_sorted_by_name(self):
        """Commands sorted alphabetically."""
        commands = {
            "zebra": SlashCommand(
                name="zebra", template="Z", path=Path("/z.md"), source="user"
            ),
            "alpha": SlashCommand(
                name="alpha", template="A", path=Path("/a.md"), source="user"
            ),
        }

        result = format_commands_for_help(commands)
        assert result[0][0] == "/alpha"
        assert result[1][0] == "/zebra"

    def test_format_no_description(self):
        """Command without description shows placeholder."""
        commands = {
            "cmd": SlashCommand(
                name="cmd", template="Do something", path=Path("/cmd.md"), source="user"
            )
        }

        result = format_commands_for_help(commands)
        assert result[0][1] == "(no description)"


class TestSlashCommandDataclass:
    """Tests for SlashCommand dataclass."""

    def test_display_source_no_namespace(self):
        """Display source without namespace."""
        cmd = SlashCommand(
            name="test", template="Test", path=Path("/test.md"), source="user"
        )
        assert cmd.display_source == "user"

    def test_display_source_with_namespace(self):
        """Display source with namespace."""
        cmd = SlashCommand(
            name="test",
            template="Test",
            path=Path("/test.md"),
            source="project",
            namespace="ci",
        )
        assert cmd.display_source == "project:ci"

    def test_default_values(self):
        """Default values set correctly."""
        cmd = SlashCommand(
            name="test", template="Test", path=Path("/test.md"), source="user"
        )
        assert cmd.namespace is None
        assert cmd.description is None
        assert cmd.allowed_tools == []
        assert cmd.argument_hint is None
        assert cmd.model is None
        assert cmd.disable_model_invocation is False


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_command_workflow(self):
        """Test complete workflow from discovery to execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create command file
            commands_dir = Path(tmpdir) / "commands"
            commands_dir.mkdir()

            cmd_file = commands_dir / "greet.md"
            cmd_file.write_text("""---
description: Greet someone
argument-hint: [name]
---

Hello $1! Welcome to the system.
Current time: !`date +%H:%M`""")

            # Discover commands
            commands = discover_commands_in_directory(commands_dir, "user")
            assert "greet" in commands

            # Execute command
            result = execute_command(commands["greet"], "Alice")
            assert "Hello Alice!" in result.prompt
            # Time should be injected
            assert ":" in result.prompt  # Colon from time format

    def test_project_overrides_user(self):
        """Project commands override user commands with same name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create user commands
            user_dir = Path(tmpdir) / "user"
            user_dir.mkdir()
            (user_dir / "deploy.md").write_text("""---
description: User deploy
---

User version""")

            # Create project commands
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            (project_dir / "deploy.md").write_text("""---
description: Project deploy
---

Project version""")

            # Discover both
            user_cmds = discover_commands_in_directory(user_dir, "user")
            project_cmds = discover_commands_in_directory(project_dir, "project")

            # Merge (project overrides)
            all_cmds = {**user_cmds, **project_cmds}

            assert all_cmds["deploy"].source == "project"
            result = execute_command(all_cmds["deploy"], "")
            assert "Project version" in result.prompt
