---
name: slack-delegation-outcome
description: "Verify the outcome of a task delegated to another agent over Slack before using its result. Use whenever you send a request to another agent (e.g. the GTM Agent) with slack_slack_send_message and then read the reply thread with slack_slack_read_thread, or otherwise await a reply from a delegated agent over Slack. Trigger when a thread read may contain only your own outbound message, when polling a thread for a reply, or before presenting transcript contents, Salesforce/CRM status, or investor/funding facts that depended on a delegated lookup."
license: MIT
compatibility: designed for deepagents-code
---

# Slack Delegation Outcome Check

## Overview

`slack_slack_read_thread` returns `is_error: false` even when the delegated agent has not replied — the result can contain only the outbound message you just sent. A non-error result is NOT evidence that the delegation completed. Do not treat the echoed request, or any data that depended on the delegated lookup, as verified until a genuine reply has arrived.

## Best Practices

- **A thread with only your outbound message means "no reply yet."** If the read returns only the parent message you sent (or its content is byte-identical to the previous poll), the delegated agent has not responded. Keep waiting or re-poll; do not proceed as if data was returned.
- **Bound your polling.** After a small, fixed number of polls (e.g. 3–5) with no new reply, stop. Do not loop indefinitely and do not silently give up.
- **Report the delegation outcome explicitly.** The final message must state whether the delegation completed. If it did not, say so plainly: "the GTM Agent did not respond in time / the delegation did not complete." Never let the missing reply be framed as a system limitation ("the transcript isn't accessible to me") — the accurate cause is an incomplete delegation.
- **Never present unretrieved data as verified.** Anything that depended on the delegated lookup — transcript contents, Salesforce email/CRM status, investor or funding facts — must be omitted or clearly marked as unconfirmed and sourced from general knowledge, not from the requested system.

## Process

1. After `slack_slack_send_message`, note the outbound message text/timestamp so you can recognize it in the thread.
2. Read the thread with `slack_slack_read_thread`. Ignore `is_error`; instead inspect the messages.
3. Determine reply state:
   - Only the outbound parent present, or content identical to the previous poll -> **no reply yet.**
   - A new message from the delegated agent with usable content -> **reply received.**
4. If no reply yet and polls remain, wait and re-poll (step 2). If the poll budget is exhausted, treat the delegation as **not completed.**
5. Compose the answer:
   - Reply received: use the returned data and state that the delegation completed.
   - Not completed: state that the delegated agent did not respond, and either omit dependent facts or mark them clearly as unconfirmed general knowledge.

## Common Pitfalls

- Treating `is_error: false` as proof the reply arrived. It only means the read call itself succeeded.
- Presenting substitute knowledge (funding rounds, CRM status, transcript quotes) as if it came from the delegated system.
- Blaming tool access ("I can't access the transcript") when the real cause is that the delegation never returned.
- Polling forever, or stopping after one poll without reporting the no-response outcome.
