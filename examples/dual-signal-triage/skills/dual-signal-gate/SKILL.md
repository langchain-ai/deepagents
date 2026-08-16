---
name: dual-signal-gate
description: Enforce SnortML dual-signal triage — ML score ≠ signature TP; never auto-contain ML-only highs.
---

# Dual-signal remediation gate

When triaging IDS/SIEM alerts:

1. Call `recommend_action` before any remediation tool.
2. **ML-only high (GID 411, no SID):** `escalate` or `corroborate_alert`. Never `contain_host`.
3. **Signature or signature+ML:** `contain_host` may be proposed; human-in-the-loop interrupt is required.
4. **Low ML / low Splunk notables:** `accept_alert` or T1 triage.

Never equate an ML probability with a classic signature true positive.
