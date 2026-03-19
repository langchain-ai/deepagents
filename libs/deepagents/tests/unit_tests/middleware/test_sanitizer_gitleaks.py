import json
from unittest.mock import MagicMock, patch

import pytest

from deepagents.middleware.sanitizer_gitleaks import GitleaksSanitizerProvider


@pytest.fixture
def provider():
    with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value="/usr/bin/gitleaks"):
        return GitleaksSanitizerProvider()


def _mock_run(findings: list[dict], returncode: int = 1) -> MagicMock:
    """Create a mock subprocess.run result with JSON stdout."""
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = json.dumps(findings) if findings else ""
    return mock


class TestGitleaksSanitize:
    def test_redacts_secret(self, provider):
        findings = [{"RuleID": "github-pat", "Match": "ghp_xxx", "Secret": "ghp_xxx"}]
        content = "token = ghp_xxx"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = _mock_run(findings)
            result = provider.sanitize(content)
        assert "ghp_" not in result["content"]
        assert "<REDACTED:github-pat>" in result["content"]
        assert len(result["findings"]) == 1

    def test_passes_content_via_stdin(self, provider):
        """Verify gitleaks is called with stdin subcommand and content piped in."""
        content = "some content to scan"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = _mock_run([], returncode=0)
            provider.sanitize(content)
            call_args = mock_run.call_args
            assert "stdin" in call_args[0][0]  # subcommand is 'stdin'
            assert call_args[1]["input"] == content  # content piped via input=

    def test_clean_output_no_findings(self, provider):
        content = "nothing sensitive"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = _mock_run([], returncode=0)
            result = provider.sanitize(content)
        assert result["content"] == content
        assert result["findings"] == []

    def test_graceful_when_binary_missing(self):
        with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value=None):
            provider = GitleaksSanitizerProvider()
        result = provider.sanitize("SECRET123")
        assert result["content"] == "SECRET123"
        assert result["findings"] == []

    def test_graceful_on_gitleaks_error(self, provider):
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = _mock_run([], returncode=2)
            result = provider.sanitize("some output")
        assert result["content"] == "some output"
        assert result["findings"] == []

    def test_multiple_findings_deduplicated(self, provider):
        findings = [
            {"RuleID": "github-pat", "Match": "ghp_abc", "Secret": "ghp_abc"},
            {"RuleID": "github-pat", "Match": "ghp_abc", "Secret": "ghp_abc"},
            {"RuleID": "private-key", "Match": "-----BEGIN RSA PRIVATE KEY-----", "Secret": "-----BEGIN RSA PRIVATE KEY-----"},
        ]
        content = "token=ghp_abc key=-----BEGIN RSA PRIVATE KEY-----"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = _mock_run(findings)
            result = provider.sanitize(content)
        assert len(result["findings"]) == 2  # deduplicated


class TestGitleaksAsanitize:
    @pytest.mark.asyncio
    async def test_async_graceful_when_binary_missing(self):
        with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value=None):
            provider = GitleaksSanitizerProvider()
        result = await provider.asanitize("SECRET123")
        assert result["content"] == "SECRET123"
        assert result["findings"] == []
