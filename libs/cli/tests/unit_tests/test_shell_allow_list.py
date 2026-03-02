"""Tests for shell command allow-list functionality."""

import pytest

from deepagents_cli.config import contains_dangerous_patterns, is_shell_command_allowed


@pytest.fixture
def basic_allow_list() -> list[str]:
    """Basic allow-list with common read-only commands."""
    return ["ls", "cat", "grep"]


@pytest.fixture
def extended_allow_list() -> list[str]:
    """Extended allow-list with common read-only commands."""
    return ["ls", "cat", "grep", "wc", "pwd", "echo", "head", "tail", "find", "sort"]


@pytest.fixture
def semicolon_allow_list() -> list[str]:
    """Allow-list for semicolon-separated command tests."""
    return ["ls", "cat", "pwd"]


@pytest.fixture
def quoted_allow_list() -> list[str]:
    """Allow-list for quoted argument tests."""
    return ["echo", "grep"]


@pytest.fixture
def pipeline_allow_list() -> list[str]:
    """Allow-list for complex pipeline tests."""
    return ["cat", "grep", "wc", "sort"]


@pytest.mark.parametrize(
    "command",
    ["ls", "cat file.txt", "rm -rf /"],
)
def test_empty_allow_list_rejects_all(command: str) -> None:
    """Test that empty allow-list rejects all commands."""
    assert not is_shell_command_allowed(command, [])


@pytest.mark.parametrize(
    "command",
    ["ls", "cat file.txt"],
)
def test_none_allow_list_rejects_all(command: str) -> None:
    """Test that None allow-list rejects all commands."""
    assert not is_shell_command_allowed(command, None)


@pytest.mark.parametrize(
    "command",
    ["ls", "cat", "grep"],
)
def test_simple_command_allowed(command: str, basic_allow_list: list[str]) -> None:
    """Test simple commands that are in the allow-list."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["rm", "mv", "chmod"],
)
def test_simple_command_not_allowed(command: str, basic_allow_list: list[str]) -> None:
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
def test_command_with_arguments_allowed(
    command: str, basic_allow_list: list[str]
) -> None:
    """Test commands with arguments that are in the allow-list."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["rm -rf /tmp", "mv file1 file2"],
)
def test_command_with_arguments_not_allowed(
    command: str, basic_allow_list: list[str]
) -> None:
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
def test_piped_commands_all_allowed(command: str, allow_list: list[str]) -> None:
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
def test_piped_commands_some_not_allowed(
    command: str, basic_allow_list: list[str]
) -> None:
    """Test piped commands where some parts are not in the allow-list."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["ls; pwd", "pwd; ls -la"],
)
def test_semicolon_separated_commands_all_allowed(
    command: str, semicolon_allow_list: list[str]
) -> None:
    """Test semicolon-separated commands where all are in the allow-list."""
    assert is_shell_command_allowed(command, semicolon_allow_list)


@pytest.mark.parametrize(
    "command",
    ["ls; rm file", "pwd; mv file1 file2"],
)
def test_semicolon_separated_commands_some_not_allowed(
    command: str, semicolon_allow_list: list[str]
) -> None:
    """Test semicolon-separated commands where some are not in the allow-list."""
    assert not is_shell_command_allowed(command, semicolon_allow_list)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls && cat file", True),
        ("ls && rm file", False),
        ("ls -la && grep pattern file && cat output.txt", True),
        ("ls && cat file && grep test", True),
        ("cat a.txt && cat b.txt && cat c.txt", True),
        ("ls && rm -rf /", False),
    ],
)
def test_and_operator_commands(
    command: str, *, expected: bool, basic_allow_list: list[str]
) -> None:
    """Test commands with && operator (commonly used by Claude for chaining)."""
    assert is_shell_command_allowed(command, basic_allow_list) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("ls || cat file", True),
        ("ls || rm file", False),
    ],
)
def test_or_operator_commands(
    command: str, *, expected: bool, basic_allow_list: list[str]
) -> None:
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
def test_quoted_arguments(command: str, quoted_allow_list: list[str]) -> None:
    """Test commands with quoted arguments."""
    assert is_shell_command_allowed(command, quoted_allow_list)


@pytest.mark.parametrize(
    "command",
    ["  ls  ", "ls   -la", "ls | cat"],
)
def test_whitespace_handling(command: str, basic_allow_list: list[str]) -> None:
    """Test proper handling of whitespace."""
    assert is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ['ls "unclosed', "cat 'missing quote"],
)
def test_malformed_commands_rejected(command: str, basic_allow_list: list[str]) -> None:
    """Test that malformed commands are rejected for safety."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    "command",
    ["", "   "],
)
def test_empty_command_rejected(command: str, basic_allow_list: list[str]) -> None:
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
def test_case_sensitivity(
    command: str, *, expected: bool, basic_allow_list: list[str]
) -> None:
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
def test_common_read_only_commands(
    command: str, extended_allow_list: list[str]
) -> None:
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
def test_dangerous_commands_not_allowed(
    command: str, basic_allow_list: list[str]
) -> None:
    """Test that dangerous commands are not allowed by default."""
    assert not is_shell_command_allowed(command, basic_allow_list)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat file.txt | grep error | sort | wc -l", True),
        ("cat file.txt | grep error | rm | wc -l", False),
    ],
)
def test_complex_pipeline(
    command: str, *, expected: bool, pipeline_allow_list: list[str]
) -> None:
    """Test complex pipelines with multiple operators."""
    assert is_shell_command_allowed(command, pipeline_allow_list) == expected


class TestShellInjectionPrevention:
    """Tests for shell injection attack prevention."""

    @pytest.fixture
    def injection_allow_list(self) -> list[str]:
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
    def test_command_substitution_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_backtick_substitution_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_newline_injection_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Newline characters must be blocked (command injection)."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls\rrm -rf /",
            "cat file\rwhoami",
        ],
    )
    def test_carriage_return_injection_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Carriage return characters must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls\trm",
            "cat\t/etc/passwd",
        ],
    )
    def test_tab_injection_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_process_substitution_input_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Process substitution <(...) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat >(rm -rf /)",
            "echo test >(cat)",
        ],
    )
    def test_process_substitution_output_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Process substitution >(...) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat <<< 'rm -rf /'",
            "grep pattern <<< 'test'",
        ],
    )
    def test_here_string_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Here-strings (<<<) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat << EOF\nmalicious\nEOF",
            "cat <<EOF",
        ],
    )
    def test_here_doc_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_output_redirect_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Output redirection (>) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello >> /tmp/test",
            "ls >> output.txt",
        ],
    )
    def test_append_redirect_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Append redirection (>>) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "cat < /etc/passwd",
            "grep pattern < input.txt",
        ],
    )
    def test_input_redirect_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_brace_variable_expansion_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_safe_commands_still_allowed(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
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
    def test_path_bypass_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """Commands with paths (not in allow-list) must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            r"cat $'\050whoami\051'",
            r"cat $'\044\050whoami\051'",
            r"echo $'\140whoami\140'",
            r"cat $'\x24\x28whoami\x29'",
            r"echo $'\076/tmp/evil'",
            r"cat $'\074/etc/passwd'",
        ],
    )
    def test_ansi_c_quoting_blocked(
        self, command: str, injection_allow_list: list[str]
    ) -> None:
        """ANSI-C quoting $'...' with escape sequences must be blocked."""
        assert not is_shell_command_allowed(command, injection_allow_list)


class TestBackgroundOperatorBlocking:
    """Tests for standalone & (background operator) detection."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls &",
            "cat file.txt & echo done",
            "sleep 10 &",
        ],
    )
    def test_background_operator_blocked(
        self, command: str, basic_allow_list: list[str]
    ) -> None:
        """Standalone & (background execution) must be blocked."""
        assert not is_shell_command_allowed(command, basic_allow_list)

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("ls && cat file", True),
            ("ls && grep test file", True),
        ],
    )
    def test_double_ampersand_still_works(
        self, command: str, *, expected: bool, basic_allow_list: list[str]
    ) -> None:
        """Double && (AND operator) should still be allowed."""
        assert is_shell_command_allowed(command, basic_allow_list) == expected


class TestBareVariableBlocking:
    """Tests for bare $VARIABLE expansion detection."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls $HOME",
            "cat $PATH",
            "echo $USER",
            "grep $IFS file.txt",
        ],
    )
    def test_bare_variable_blocked(
        self, command: str, basic_allow_list: list[str]
    ) -> None:
        """Bare $VARIABLE expansion must be blocked to prevent info leaks."""
        assert not is_shell_command_allowed(command, basic_allow_list)

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "grep pattern file",
        ],
    )
    def test_commands_without_variables_still_work(
        self, command: str, basic_allow_list: list[str]
    ) -> None:
        """Commands without variable expansion should still be allowed."""
        assert is_shell_command_allowed(command, basic_allow_list)


class TestContainsDangerousPatterns:
    """Direct tests for contains_dangerous_patterns()."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "grep pattern file",
            "wc -l file.txt",
            "head -n 10 file.txt",
            "echo hello world",
        ],
    )
    def test_safe_commands_pass(self, command: str) -> None:
        """Safe commands should not be flagged as dangerous."""
        assert not contains_dangerous_patterns(command)

    @pytest.mark.parametrize(
        ("command", "description"),
        [
            ("ls $(whoami)", "command substitution with $()"),
            ("cat `whoami`", "backtick command substitution"),
            ("echo $'\\x41'", "ANSI-C quoting"),
            ("ls\nrm -rf /", "newline injection"),
            ("ls\rrm -rf /", "carriage return injection"),
            ("ls\trm", "tab injection"),
            ("cat <(ls)", "process substitution (input)"),
            ("cat >(ls)", "process substitution (output)"),
            ("cat <<EOF", "here-doc"),
            ("cat <<<word", "here-string"),
            ("ls >> file", "append redirect"),
            ("ls > file", "output redirect"),
            ("cat < file", "input redirect"),
            ("echo ${PATH}", "variable expansion with braces"),
            ("ls $HOME", "bare variable expansion"),
            ("ls &", "background operator"),
            ("sleep 10 &", "trailing background operator"),
        ],
    )
    def test_dangerous_patterns_detected(self, command: str, description: str) -> None:
        """Dangerous shell patterns should be detected."""
        assert contains_dangerous_patterns(command), f"Failed to detect: {description}"

    def test_double_ampersand_not_flagged(self) -> None:
        """Double && should not be flagged as dangerous (it's a safe operator)."""
        assert not contains_dangerous_patterns("ls && cat file")

    def test_empty_command(self) -> None:
        """Empty command should not be flagged."""
        assert not contains_dangerous_patterns("")

    def test_pattern_in_middle_of_command(self) -> None:
        """Dangerous patterns embedded within arguments should be detected."""
        assert contains_dangerous_patterns("echo foo$(bar)baz")

    def test_multiple_dangerous_patterns(self) -> None:
        """Commands with multiple dangerous patterns should be detected."""
        assert contains_dangerous_patterns("cat $(cmd) > /tmp/out")


class TestFindExecLimitation:
    """Document that `find -exec` is NOT caught by `contains_dangerous_patterns`.

    `find -exec` enables arbitrary command execution but uses a flag rather than
    a shell metacharacter, so the substring-based pattern check cannot detect it.
    This is why `find` is excluded from `RECOMMENDED_SAFE_SHELL_COMMANDS`. Users
    who add `find` to a custom allow-list accept this risk.
    """

    def test_find_exec_not_caught_by_dangerous_patterns(self) -> None:
        """Find -exec bypasses dangerous-pattern detection (known limitation)."""
        # The `-exec` flag is not a shell metacharacter, so the pattern
        # checker cannot detect it.
        assert not contains_dangerous_patterns("find . -exec rm {} +")

    def test_find_exec_plus_allowed_when_find_in_allow_list(self) -> None:
        """Find -exec ... + is allowed if find is in the custom allow-list.

        This is a known limitation: the `+` variant of `-exec` does not
        trigger any operator or dangerous-pattern check, so the allow-list
        gate is the only protection.
        """
        assert is_shell_command_allowed("find . -exec rm {} +", ["find"])

    def test_find_exec_rejected_when_find_not_in_allow_list(self) -> None:
        """Find -exec is rejected when find is not in the allow-list."""
        assert not is_shell_command_allowed("find . -exec rm {} +", ["ls", "cat"])

    def test_find_delete_not_caught_by_dangerous_patterns(self) -> None:
        """Find -delete bypasses dangerous-pattern detection (known limitation)."""
        assert not contains_dangerous_patterns("find . -name '*.tmp' -delete")


class TestGrepRegexLimitation:
    """Document that grep patterns with `$` followed by a letter are blocked.

    The bare-variable check (`$VAR`) cannot distinguish regex anchors inside
    quoted strings from actual shell variable expansion, so it errs on the
    side of caution. Users needing such patterns must use interactive mode.
    """

    def test_grep_dollar_anchor_blocked_known_limitation(self) -> None:
        """Grep regex with $ followed by a letter is blocked (known limitation)."""
        assert not is_shell_command_allowed("grep 'pattern$A' file", ["grep"])
