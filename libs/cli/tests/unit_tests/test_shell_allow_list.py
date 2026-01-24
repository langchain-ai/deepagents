"""Tests for shell command allow-list functionality."""
# ruff: noqa: ANN001

import pytest

from deepagents_cli.config import is_shell_command_allowed


@pytest.fixture
def basic_allow_list():
    """Basic allow-list with common read-only commands."""
    return ["ls", "cat", "grep"]


@pytest.fixture
def extended_allow_list():
    """Extended allow-list with common read-only commands."""
    return ["ls", "cat", "grep", "wc", "pwd", "echo", "head", "tail", "find", "sort"]


@pytest.fixture
def semicolon_allow_list():
    """Allow-list for semicolon-separated command tests."""
    return ["ls", "cat", "pwd"]


@pytest.fixture
def quoted_allow_list():
    """Allow-list for quoted argument tests."""
    return ["echo", "grep"]


@pytest.fixture
def pipeline_allow_list():
    """Allow-list for complex pipeline tests."""
    return ["cat", "grep", "wc", "sort"]


@pytest.mark.parametrize(
    "command",
    ["ls", "cat file.txt", "rm -rf /"],
)
def test_empty_allow_list_rejects_all(command):
    """Test that empty allow-list rejects all commands."""
    assert not is_shell_command_allowed(command, [])


@pytest.mark.parametrize(
    "command",
    ["ls", "cat file.txt"],
)
def test_none_allow_list_rejects_all(command):
    """Test that None allow-list rejects all commands."""
    assert not is_shell_command_allowed(command, None)


@pytest.mark.parametrize(
    "command",
    ["ls", "cat", "grep"],
)
def test_simple_command_allowed(command, basic_allow_list):
    """Test simple commands that are in the allow-list."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["rm", "mv", "chmod"],
)
def test_simple_command_not_allowed(command, basic_allow_list):
    """Test simple commands that are not in the allow-list."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat file.txt",
        "grep 'pattern' file.txt",
        "ls -la /tmp/test",
    ],
)
def test_command_with_arguments_allowed(command, basic_allow_list):
    """Test commands with arguments that are in the allow-list."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["rm -rf /tmp", "mv file1 file2"],
)
def test_command_with_arguments_not_allowed(command, basic_allow_list):
    """Test commands with arguments that are not in the allow-list."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    ("command", "allow_list"),
    [
        ("ls | grep test", ["ls", "cat", "grep", "wc"]),
        ("cat file.txt | grep pattern", ["ls", "cat", "grep", "wc"]),
        ("ls -la | wc -l", ["ls", "cat", "grep", "wc"]),
        ("cat file | grep foo | wc", ["ls", "cat", "grep", "wc"]),
    ],
)
def test_piped_commands_all_allowed(command, allow_list):
    """Test piped commands where all parts are in the allow-list."""
    assert is_shell_command_allowed(command, allow_list)


@pytest.mark.parametrize(
    "command",
    [
        "ls | wc -l",  # wc not in allow-list
        "cat file.txt | sort",  # sort not in allow-list
        "grep pattern file | rm",  # rm not in allow-list
    ],
)
def test_piped_commands_some_not_allowed(command, basic_allow_list):
    """Test piped commands where some parts are not in the allow-list."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["ls; pwd", "pwd; ls -la"],
)
def test_semicolon_separated_commands_all_allowed(command, semicolon_allow_list):
    """Test semicolon-separated commands where all are in the allow-list."""
    assert is_shell_command_allowed(command, semicolon_allow_list)


@pytest.mark.parametrize(
    "command",
    ["ls; rm file", "pwd; mv file1 file2"],
)
def test_semicolon_separated_commands_some_not_allowed(command, semicolon_allow_list):
    """Test semicolon-separated commands where some are not in the allow-list."""
    assert not is_shell_command_allowed(command, semicolon_allow_list)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls && cat file", True),
        ("ls && rm file", False),
    ],
)
def test_and_operator_commands(command, expected, basic_allow_list):
    """Test commands with && operator."""
    assert is_shell_command_allowed(command, basic_allow_list) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls || cat file", True),
        ("ls || rm file", False),
    ],
)
def test_or_operator_commands(command, expected, basic_allow_list):
    """Test commands with || operator."""
    assert is_shell_command_allowed(command, basic_allow_list) == expected


@pytest.mark.parametrize(
    "command",
    [
        'echo "hello world"',
        "grep 'pattern' file.txt",
        'echo "test" | grep "te"',
    ],
)
def test_quoted_arguments(command, quoted_allow_list):
    """Test commands with quoted arguments."""
    assert is_shell_command_allowed(command, quoted_allow_list)


@pytest.mark.parametrize(
    "command",
    ["  ls  ", "ls   -la", "ls | cat"],
)
def test_whitespace_handling(command, basic_allow_list):
    """Test proper handling of whitespace."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ['ls "unclosed', "cat 'missing quote"],
)
def test_malformed_commands_rejected(command, basic_allow_list):
    """Test that malformed commands are rejected for safety."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["", "   "],
)
def test_empty_command_rejected(command, basic_allow_list):
    """Test that empty commands are rejected."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls", True),
        ("LS", False),
        ("Cat", False),
    ],
)
def test_case_sensitivity(command, expected, basic_allow_list):
    """Test that command matching is case-sensitive."""
    assert is_shell_command_allowed(command, basic_allow_list) == expected


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat file.txt",
        "grep pattern file",
        "pwd",
        "echo 'hello'",
        "head -n 10 file",
        "tail -f log.txt",
        "find . -name '*.py'",
        "wc -l file",
    ],
)
def test_common_read_only_commands(command, extended_allow_list):
    """Test common read-only commands are allowed."""
    assert is_shell_command_allowed(command, extended_allow_list)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "mv file /dev/null",
        "chmod 777 file",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda",
    ],
)
def test_dangerous_commands_not_allowed(command, basic_allow_list):
    """Test that dangerous commands are not allowed by default."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat file.txt | grep error | sort | wc -l", True),
        ("cat file.txt | grep error | rm | wc -l", False),
    ],
)
def test_complex_pipeline(command, expected, pipeline_allow_list):
    """Test complex pipelines with multiple operators."""
    assert is_shell_command_allowed(command, pipeline_allow_list) == expected


class TestShellInjectionPrevention:
    """Tests for shell injection attack prevention.

    These tests verify that dangerous shell patterns are blocked BEFORE parsing,
    preventing attackers from bypassing the allow-list through shell metacharacters.
    """

    @pytest.fixture
    def injection_allow_list(self):
        """Allow-list for injection tests."""
        return ["ls", "cat", "grep", "echo"]

    @pytest.mark.parametrize(
        "command",
        [
            "ls $(rm -rf /)",
            "ls $(cat /etc/shadow)",
            'echo "$(whoami)"',
            "cat $(echo /etc/passwd)",
        ],
    )
    def test_command_substitution_blocked(self, command, injection_allow_list):
        """Command substitution $(...) must be blocked even in arguments."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls `rm -rf /`",
            "ls `cat /etc/shadow`",
            "echo `whoami`",
            "cat `echo /etc/passwd`",
        ],
    )
    def test_backtick_substitution_blocked(self, command, injection_allow_list):
        """Backtick command substitution must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls\nrm -rf /",
            "cat file\nwhoami",
            "echo hello\n/bin/sh",
        ],
    )
    def test_newline_injection_blocked(self, command, injection_allow_list):
        """Newline characters must be blocked (command injection)."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls\rrm -rf /",
            "cat file\rwhoami",
        ],
    )
    def test_carriage_return_injection_blocked(self, command, injection_allow_list):
        """Carriage return characters must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls\trm",
            "cat\t/etc/passwd",
        ],
    )
    def test_tab_injection_blocked(self, command, injection_allow_list):
        """Tab characters must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat <(rm -rf /)",
            "cat <(whoami)",
            "grep pattern <(cat /etc/shadow)",
        ],
    )
    def test_process_substitution_input_blocked(self, command, injection_allow_list):
        """Process substitution <(...) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat >(rm -rf /)",
            "echo test >(cat)",
        ],
    )
    def test_process_substitution_output_blocked(self, command, injection_allow_list):
        """Process substitution >(...) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<< 'rm -rf /'",
            "grep pattern <<< 'test'",
        ],
    )
    def test_here_string_blocked(self, command, injection_allow_list):
        """Here-strings (<<<) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat << EOF\nmalicious\nEOF",
            "cat <<EOF",
        ],
    )
    def test_here_doc_blocked(self, command, injection_allow_list):
        """Here-documents (<<) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello > /tmp/test",
            "ls > output.txt",
            "cat file > /etc/passwd",
        ],
    )
    def test_output_redirect_blocked(self, command, injection_allow_list):
        """Output redirection (>) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello >> /tmp/test",
            "ls >> output.txt",
        ],
    )
    def test_append_redirect_blocked(self, command, injection_allow_list):
        """Append redirection (>>) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat < /etc/passwd",
            "grep pattern < input.txt",
        ],
    )
    def test_input_redirect_blocked(self, command, injection_allow_list):
        """Input redirection (<) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "echo ${PATH}",
            "cat ${HOME}/.bashrc",
            "ls ${PWD}",
        ],
    )
    def test_brace_variable_expansion_blocked(self, command, injection_allow_list):
        """Brace variable expansion ${...} must be blocked (can contain commands)."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "grep pattern file",
            "echo hello world",
            "ls | grep test",
            "cat file | grep pattern",
        ],
    )
    def test_safe_commands_still_allowed(self, command, injection_allow_list):
        """Verify that safe commands without dangerous patterns are still allowed."""
        assert is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "/bin/rm -rf /",
            "./malicious",
            "../../../bin/sh",
        ],
    )
    def test_path_bypass_blocked(self, command, injection_allow_list):
        """Commands with paths (not in allow-list) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)
