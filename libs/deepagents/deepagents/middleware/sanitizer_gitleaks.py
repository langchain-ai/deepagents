"""Gitleaks-based sanitizer provider for secret detection and redaction.

Uses ``gitleaks stdin`` to pipe content directly and read the JSON report
from stdout — no temp files needed.
"""

import asyncio
import json
import logging
import shutil
import subprocess

from deepagents.middleware.sanitizer import SanitizeFinding, SanitizeResult

logger = logging.getLogger(__name__)


def _parse_report(stdout: str) -> list[dict]:
    """Parse gitleaks JSON report from stdout."""
    try:
        if not stdout.strip():
            return []
        return json.loads(stdout)
    except json.JSONDecodeError:
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
    """Sanitizer provider that uses gitleaks for secret detection.

    Pipes content to ``gitleaks stdin`` and reads the JSON report from stdout.
    No temp files are created.
    """

    def __init__(self) -> None:
        self._binary = shutil.which("gitleaks")
        if self._binary is None:
            msg = (
                "gitleaks binary not found in PATH. "
                "Install gitleaks (https://github.com/gitleaks/gitleaks) or remove --sanitizer flag."
            )
            raise FileNotFoundError(msg)

    @property
    def name(self) -> str:
        return "gitleaks"

    def sanitize(self, content: str) -> SanitizeResult:
        if self._binary is None:
            return SanitizeResult(content=content, findings=[])

        result = subprocess.run(
            [self._binary, "stdin", "--report-format", "json", "--report-path", "/dev/stdout"],
            input=content,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Exit 0 = clean, exit 1 = findings, other = error
        if result.returncode == 0:
            return SanitizeResult(content=content, findings=[])
        if result.returncode != 1:
            logger.warning("gitleaks exited with code %d — skipping", result.returncode)
            return SanitizeResult(content=content, findings=[])

        raw_findings = _parse_report(result.stdout)
        if not raw_findings:
            return SanitizeResult(content=content, findings=[])
        return _process_findings(content, raw_findings)

    async def asanitize(self, content: str) -> SanitizeResult:
        if self._binary is None:
            return SanitizeResult(content=content, findings=[])

        proc = await asyncio.create_subprocess_exec(
            self._binary, "stdin", "--report-format", "json", "--report-path", "/dev/stdout",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(content.encode()), timeout=10)

        if proc.returncode == 0:
            return SanitizeResult(content=content, findings=[])
        if proc.returncode != 1:
            logger.warning("gitleaks exited with code %d — skipping", proc.returncode)
            return SanitizeResult(content=content, findings=[])

        raw_findings = _parse_report(stdout_bytes.decode())
        if not raw_findings:
            return SanitizeResult(content=content, findings=[])
        return _process_findings(content, raw_findings)
