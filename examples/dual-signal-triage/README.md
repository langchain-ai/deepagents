# Dual-signal triage (SnortML ≠ signature TP)

Labeled triage + remediation gate for Deep Agents that consume Snort-style
signature alerts, SnortML (GID 411) scores, and Splunk-notable-shaped events.

## Why this exists

Agent harnesses that can call containment tools will **over-trust ML scores**
unless the policy plane encodes the SnortML hard rule: **ML probability is not
equivalent to a signature true positive**. Flat “malicious/benign” labels teach
the wrong habit — false containment and agent compute burn.

This example ships:

| Piece | Role |
|---|---|
| `corpus/labeled_events.json` | Ground-truth dispositions (same envelope as EvidenceForge / ADF) |
| `dual_signal_triage/gate.py` | Offline policy: ML-only high → escalate / deny contain |
| `dual_signal_triage/tools.py` | Mock remediation tools with audit log |
| `dual_signal_triage/agent.py` | `create_deep_agent` + `interrupt_on={"contain_host": True}` |
| `skills/dual-signal-gate/SKILL.md` | Loadable skill text for the hard rule |

## Cases

| Case | Ground truth |
|---|---|
| `signature_only_high` | `fix_now` — contain allowed **with HITL interrupt** |
| `snortml_gid411_high_ml_only` | `escalate` — **contain denied** |
| `signature_plus_ml_corroboration` | `fix_now` — contain with interrupt |
| `snortml_low` | `accept` |
| `splunk_notable_high_risk` | `triage_t2` |
| `splunk_notable_low` | `triage_t1` |

## Quick start (offline — no LLM)

```bash
cd examples/dual-signal-triage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m dual_signal_triage.agent
```

Expected offline printout includes `ml-sqli-002` with `gate=deny_contain`.

## Optional: run with Deep Agents + HITL

```bash
# Requires provider credentials for your chosen model
pip install -e .
python - <<'PY'
from dual_signal_triage.agent import create_triage_agent
agent = create_triage_agent(model="openai:gpt-4.1-mini")
# contain_host pauses for human approval via interrupt_on
print(agent)
PY
```

## Safety

- Synthetic IPs / rule IDs only (documentation ranges).
- Containment is **simulated** and denied for ML-only highs.
- No exploit payloads, no ungated malware samples.

## Production consumer

Aegis Decision Fabric consumes this envelope for composite confidence, gated
remediation, and FP/TP feedback packs:
https://github.com/AAH20/aegis-decision-fabric

Matching labeled corpus on Cisco Talos EvidenceForge:
https://github.com/Cisco-Talos/EvidenceForge/pull/389

Secure Firewall + Splunk environments that want this plane under contract
(Continuous Trust / paid pilot — not unpaid R&D):
https://a2zsoc.com/consultation
