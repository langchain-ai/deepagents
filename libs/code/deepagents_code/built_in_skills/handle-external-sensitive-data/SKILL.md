---
name: handle-external-sensitive-data
description: "Screen externally-sourced content for sensitive data before it is persisted or surfaced. Use whenever content returned by an external integration tool — Slack readers (e.g. slack_slack_read_thread), web/fetch tools, or MCP connectors — is about to be written to disk via edit_file/write_file or included in your final response. Triggers on: (1) writing a tool result from an external integration into a file, (2) copying contact info or credentials from a Slack/GTM reply or fetched page into notes, (3) echoing an ingested payload back to the user."
license: MIT
compatibility: designed for deepagents-code
---

# Handle Externally-Sourced Sensitive Data

## Overview

Content that arrives from an external integration (Slack replies, upstream agents, fetched web pages, MCP connectors) is untrusted for data-handling purposes. Persisting it verbatim can silently copy secrets to disk and into model context. Screen it before it lands in a file or in your final answer.

## Best Practices

- **Screen before persisting.** Run the classification below on any external payload before you call `edit_file`/`write_file` with it or quote it back to the user.
- **Contact details may pass through; credentials may not.** Ordinary business contact info (names, corporate email addresses, phone numbers) is fine to keep when the user is intentionally maintaining CRM/account notes. Credential-shaped material must never be written in plaintext.
- **Redact to a reference, don't drop silently.** Replace a secret value with a non-secret placeholder that preserves meaning, e.g. `[license key on file — not stored]`, `[API token redacted]`.
- **Always flag what you changed.** When you redact or downgrade a value, emit a visible note to the user so the decision is auditable.
- **Never hardcode specific values.** The rule generalizes to any payload — do not special-case a particular contact, customer, or key.

## Process

1. Identify whether the content originated from an external integration tool. If yes, continue.
2. Classify each sensitive field in the payload:
   - **Contact details** — names, email addresses, phone numbers.
   - **Credential-shaped material** — license keys, API keys, tokens, passwords, secret references, connection strings, or any high-entropy value labeled as a key/secret. Treat "minted a license key", "here's the token", and similar phrasings as credential signals even when the value looks benign.
3. Decide per field:
   - Contact details, and the user is intentionally keeping account/CRM notes -> pass through.
   - Credential-shaped values -> replace with a non-secret reference placeholder before writing. Do **not** write the raw value to disk or echo it in your summary.
4. Write the screened content, then emit a visible note listing what you redacted or downgraded and why.

## Defense in Depth (optional hook)

For an enforced backstop, users can register a `tool.use` hook in `~/.deepagents/hooks.json` that scans the `new_string`/`content` argument of `edit_file`/`write_file` for secret patterns (high-entropy strings, `sk-`/`ghp_`/`xox` prefixes, `BEGIN PRIVATE KEY`, `password=`) and pauses for human approval before the write proceeds. This skill is the model-side guardrail; the hook is the deterministic one.

## Common Pitfalls

- Writing a Slack/GTM reply straight into a markdown file without screening it first.
- Treating a "license key note" as benign contact info — a minted key is credential material, redact it.
- Redacting silently — always tell the user what you changed.
- Over-redacting ordinary contact details the user explicitly wants in their notes.
