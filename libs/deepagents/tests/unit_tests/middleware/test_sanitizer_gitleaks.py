from unittest.mock import MagicMock, patch
import pytest
from deepagents.middleware.sanitizer_gitleaks import GitleaksSanitizerProvider

@pytest.fixture
def provider():
    with patch("deepagents.middleware.sanitizer_gitleaks.shutil.which", return_value="/usr/bin/gitleaks"):
        return GitleaksSanitizerProvider()

class TestGitleaksSanitize:
    def test_redacts_github_pat(self, provider):
        findings = [{"RuleID": "github-pat", "Match": "ghp_xxx", "Secret": "ghp_xxx"}]
        content = "token = ghp_xxx"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run, \
             patch("deepagents.middleware.sanitizer_gitleaks._read_report", return_value=findings):
            mock_run.return_value = MagicMock(returncode=1)
            result = provider.sanitize(content)
        assert "ghp_" not in result["content"]
        assert "<REDACTED:github-pat>" in result["content"]
        assert len(result["findings"]) == 1

    def test_clean_output_no_findings(self, provider):
        content = "nothing sensitive"
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
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
            mock_run.return_value = MagicMock(returncode=2)
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
        with patch("deepagents.middleware.sanitizer_gitleaks.subprocess.run") as mock_run, \
             patch("deepagents.middleware.sanitizer_gitleaks._read_report", return_value=findings):
            mock_run.return_value = MagicMock(returncode=1)
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
