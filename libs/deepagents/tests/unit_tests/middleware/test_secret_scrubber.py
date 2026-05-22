"""Tests for the execute-tool secret scrubber.

The scrubber is the tool-boundary defense-in-depth that backs the
`<secret_handling>` system-prompt block: even when the model ignores
the prompt and pastes a literal API key into `command`, we want the
execute tool to refuse rather than persist the secret in tool-call
args.

Note: realistic secret-token shapes are assembled at runtime from
fragments so the source file itself does not contain anything that
matches GitHub push-protection's secret-scanning patterns.
"""

from __future__ import annotations

import pytest

from deepagents.middleware._secret_scrubber import (
    SecretInCommandError,
    scan_command_for_secrets,
)

# Assemble realistic-shape fake secrets at runtime from fragments. Each
# resulting token matches the scrubber's regex but never appears as a
# contiguous literal in this source file, so push-protection's secret
# scanner doesn't flag the test fixtures.
_HEX32 = "abcdef0123456789abcdef0123456789"
_HEX10 = "0123456789"
_FAKE_LANGSMITH_PT = "lsv2_" + "pt_" + _HEX32 + "_" + _HEX10
_FAKE_LANGSMITH_SK = "lsv2_" + "sk_" + _HEX32 + "_" + _HEX10
_FAKE_OPENAI = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwx" + _HEX10
_FAKE_ANTHROPIC = "sk-" + "ant-" + "api03-" + "abcdefghijklmnopqrstuvwxyz" + _HEX10


class TestRejectsRealisticSecrets:
    """High-confidence patterns must trip on realistic key shapes."""

    def test_rejects_langsmith_personal_token_as_env_prefix(self) -> None:
        command = f"LANGSMITH_API_KEY={_FAKE_LANGSMITH_PT} langsmith project list"
        with pytest.raises(SecretInCommandError) as excinfo:
            scan_command_for_secrets(command)
        assert excinfo.value.secret_name == "LangSmith API key"
        # The error message must steer the model toward the fix.
        msg = str(excinfo.value)
        assert "env" in msg.lower()
        assert "KEY=value" in msg

    def test_rejects_langsmith_service_key(self) -> None:
        command = f"LANGSMITH_API_KEY={_FAKE_LANGSMITH_SK} langsmith ls"
        with pytest.raises(SecretInCommandError) as excinfo:
            scan_command_for_secrets(command)
        assert excinfo.value.secret_name == "LangSmith API key"

    def test_rejects_openai_api_key(self) -> None:
        command = f"OPENAI_API_KEY={_FAKE_OPENAI} python run.py"
        with pytest.raises(SecretInCommandError) as excinfo:
            scan_command_for_secrets(command)
        assert excinfo.value.secret_name == "OpenAI API key"

    def test_rejects_anthropic_api_key(self) -> None:
        command = f"ANTHROPIC_API_KEY={_FAKE_ANTHROPIC} claude --help"
        with pytest.raises(SecretInCommandError) as excinfo:
            scan_command_for_secrets(command)
        assert excinfo.value.secret_name == "Anthropic API key"

    def test_rejects_secret_as_flag_argument(self) -> None:
        command = f"langsmith --api-key {_FAKE_LANGSMITH_PT} project list"
        with pytest.raises(SecretInCommandError):
            scan_command_for_secrets(command)


class TestAllowsCleanCommands:
    """Commands without literal secrets must pass through unchanged."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "pytest /foo/bar/tests",
            "langsmith project list",  # relies on env var
            "echo $LANGSMITH_API_KEY",  # references by name, not value
            "OPENAI_API_KEY=$ALT_KEY python run.py",
            "git diff HEAD~1 HEAD",
            "",  # empty string
        ],
    )
    def test_clean_command_does_not_raise(self, command: str) -> None:
        # Should return None without raising.
        assert scan_command_for_secrets(command) is None

    def test_documentation_string_with_sk_prefix_does_not_match(self) -> None:
        # `sk-` alone (no >=20 chars of entropy) should not trip the
        # OpenAI pattern. Without this, every README snippet referencing
        # the prefix would break legitimate calls.
        command = "echo 'OpenAI keys start with sk-'"
        assert scan_command_for_secrets(command) is None


class TestErrorMessageGuidance:
    """The raised error must steer the model toward the env-var fix."""

    def test_error_mentions_env_var_reference_pattern(self) -> None:
        command = f"LANGSMITH_API_KEY={_FAKE_LANGSMITH_PT} langsmith ls"
        with pytest.raises(SecretInCommandError) as excinfo:
            scan_command_for_secrets(command)
        msg = str(excinfo.value)
        assert "$ALT_KEY" in msg or "env-var name" in msg
