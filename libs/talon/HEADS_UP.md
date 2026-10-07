# Experimental heads-up observer

Enable a read-only second look at interactive Talon tasks:

```bash
DEEPAGENTS_TALON_HEADS_UP=true AGENT_MODEL=<provider>:<model-id> uv run deepagents-talon
```

Use your normal channel configuration and model credentials. From the repository
root, use `uv run --directory libs/talon deepagents-talon`. Set the variable to
`false` (or unset it) and restart to disable the feature. It is off by default.

Try a research, reporting, email-drafting, or planning task with several constraints.
The observer looks for consequential overlooked requirements, unsupported conclusions,
misleading success reports, and material tradeoffs. It normally stays silent. When
it finds something, Talon sends a short “Heads up” with an excerpt from the relevant
user, assistant, or tool message to the same chat. The final answer is unchanged;
notices do not enter the agent's history or cause automatic remediation.

Checks run every six main-model calls and before a nonempty final answer. There
are at most three checks and two notices per invocation; the last check is reserved
for the final answer. Periodic checks run concurrently, one at a time. Final checks
cancel any outstanding periodic review and can delay the answer by up to ten seconds.
Cancellation discards unfinished reviews. Scheduled runs, background-result follow-ups,
subagents, and invocations without a channel callback are excluded.

## Cost and privacy

The observer uses the same model selected for the main chat, with no tools, on a
separate request. It sends a bounded snapshot of visible conversation text and tool
arguments/results to that provider. System messages, reasoning blocks, and attachments
are not included. Recent messages take priority, with at most 2,000 characters per
message and 24,000 bytes of serialized evidence. This is not a full conversation fork
and does not preserve the main request's prompt-cache prefix. Additional model charges
apply; do not assume these calls are cached or free. Normal tracing configuration
also applies to observer requests.

Known labeled secrets and bearer tokens are redacted using Talon's existing helper.
This is best-effort redaction, **not guaranteed removal of credentials or personal data**.
Evidence excerpts and summaries are visible to everyone in the originating chat.
Enable only for conversations whose data may be sent to the configured provider and
whose participants may see those notices.

## Prototype limitations

An exact excerpt must exist in the supplied evidence before a notice is sent. That
checks the reference, not whether the model's interpretation is correct. Truncated
context may hide important evidence. The observer can share the main model's blind
spots, produce false alarms, or overlook genuine issues; it is advisory, not an
approval gate or security boundary.

Duplicate evidence is suppressed within one invocation. Cross-turn dismissal,
“already knew” feedback, and adaptive notification frequency are not implemented.
Evaluate genuinely new useful findings, false alarms, notification burden, latency,
and cost before considering default enablement. The prompt and implementation are
original; no code or extracted prompt from the unlicensed Pi extension is included.
