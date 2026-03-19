"""Gitleaks-based sanitizer provider for secret detection and redaction."""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from deepagents.middleware.sanitizer import SanitizeFinding, SanitizeResult

logger = logging.getLogger(__name__)


def _read_report(report_path: Path) -> list[dict]:
    """Read and parse a gitleaks JSON report file."""
    try:
        text = report_path.read_text()
        if not text.strip():
            return []
        return json.loads(text)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _process_findings(content: str, raw_findings: list[dict]) -> SanitizeResult:
    """Deduplicate findings, perform replacements, return result."""
    seen: dict[str, str] = {}  # secret -> rule_id
    for finding in raw_findings:
        secret = finding.get("Secret", "")
        rule_id = finding.get("RuleID", "unknown")
        if secret and secret not in seen:
            seen[secret] = rule_id

    findings: list[SanitizeFinding] = []
    for secret, rule_id in seen.items():
        redacted_as = f"<REDACTED:{rule_id}>"
        content = content.replace(secret, redacted_as)
        findings.append(SanitizeFinding(rule_id=rule_id, redacted_as=redacted_as))

    return SanitizeResult(content=content, findings=findings)


class GitleaksSanitizerProvider:
    """Sanitizer provider that uses gitleaks for secret detection."""

    def __init__(self) -> None:
        self._binary = shutil.which("gitleaks")
        if self._binary is None:
            logger.warning("gitleaks binary not found in PATH — sanitizer will be a no-op")

    @property
    def name(self) -> str:
        return "gitleaks"

    def _run_gitleaks(self, content: str) -> SanitizeResult:
        if self._binary is None:
            return SanitizeResult(content=content, findings=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.txt"
            report_path = Path(tmpdir) / "report.json"
            source_path.write_text(content)

            result = subprocess.run(
                [self._binary, "detect", "--no-git",
                 "--report-format", "json",
                 "--report-path", str(report_path),
                 "--source", str(source_path)],
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0:
                return SanitizeResult(content=content, findings=[])
            if result.returncode != 1:
                logger.warning("gitleaks exited with code %d — skipping", result.returncode)
                return SanitizeResult(content=content, findings=[])

            raw_findings = _read_report(report_path)
            if not raw_findings:
                return SanitizeResult(content=content, findings=[])
            return _process_findings(content, raw_findings)

    def sanitize(self, content: str) -> SanitizeResult:
        return self._run_gitleaks(content)

    async def asanitize(self, content: str) -> SanitizeResult:
        if self._binary is None:
            return SanitizeResult(content=content, findings=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.txt"
            report_path = Path(tmpdir) / "report.json"
            source_path.write_text(content)

            proc = await asyncio.create_subprocess_exec(
                self._binary, "detect", "--no-git",
                "--report-format", "json",
                "--report-path", str(report_path),
                "--source", str(source_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)

            if proc.returncode == 0:
                return SanitizeResult(content=content, findings=[])
            if proc.returncode != 1:
                logger.warning("gitleaks exited with code %d — skipping", proc.returncode)
                return SanitizeResult(content=content, findings=[])

            raw_findings = _read_report(report_path)
            if not raw_findings:
                return SanitizeResult(content=content, findings=[])
            return _process_findings(content, raw_findings)
