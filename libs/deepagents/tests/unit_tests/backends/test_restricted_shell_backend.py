from deepagents.backends.restricted_shell import RestrictedShellBackend


def test_restricted_shell_allowed_command():
    backend = RestrictedShellBackend(allowed_commands=["echo"])
    result = backend.execute("echo 'hello'")
    assert result.exit_code == 0
    assert "hello" in result.output


def test_restricted_shell_blocked_command():
    backend = RestrictedShellBackend(allowed_commands=["echo"])
    result = backend.execute("ls -la")
    assert result.exit_code == 1
    assert "Security Error" in result.output
    assert "Command 'ls' is not in the allowed whitelist" in result.output


def test_restricted_shell_metacharacters_blocked():
    backend = RestrictedShellBackend(allowed_commands=["echo"])
    # Pipe blocked
    result = backend.execute("echo hello | cat")
    assert result.exit_code == 1
    assert "Security Error: Shell metacharacters are not allowed" in result.output

    # Semicolon blocked
    result = backend.execute("echo hello; ls")
    assert result.exit_code == 1
    assert "Security Error" in result.output


def test_restricted_shell_metacharacters_allowed():
    backend = RestrictedShellBackend(allowed_commands=["echo", "cat"], allow_metacharacters=True)
    result = backend.execute("echo 'hello' | cat")
    assert result.exit_code == 0
    assert "hello" in result.output


def test_restricted_shell_empty_whitelist_blocks_all_commands():
    # If allowed_commands is empty/None, it blocks everything.
    backend = RestrictedShellBackend(allowed_commands=[])
    result = backend.execute("echo hello")
    assert "not in the allowed whitelist" in result.output


def test_restricted_shell_unrestricted_whitelist_if_none():
    # If we want it to be unrestricted by default, we'd need to change init.
    # But RestrictedShellBackend should probably REQUIRE a whitelist or be very explicit.
    backend = RestrictedShellBackend(allowed_commands=None)
    result = backend.execute("echo hello")
    assert "not in the allowed whitelist" in result.output


def test_restricted_shell_newline_bypass_blocked():
    backend = RestrictedShellBackend(allowed_commands=["echo"])
    result = backend.execute("echo ok\ncat /etc/passwd")
    assert result.exit_code == 1
    assert "Security Error: Shell metacharacters" in result.output
