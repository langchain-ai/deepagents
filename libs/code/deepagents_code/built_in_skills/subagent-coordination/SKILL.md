---
name: subagent-coordination
description: "Protocol for waiting on an unresponsive sub-agent or external collaborator that you poll through a thread- or status-read tool (e.g. a Slack read-thread tool). Use whenever you are 'waiting for', 'polling', or 'checking if X replied' on a thread or resource you do not control, so you cap redundant polls and escalate instead of spinning."
license: MIT
compatibility: designed for deepagents-code
---

# Sub-agent Coordination

## Overview
When you hand work to a sub-agent or external collaborator and then wait on a
thread you do not control (for example, reading a Slack thread to see if they
replied), it is easy to poll the same resource over and over. Each read of an
unchanged thread re-ingests the **entire** thread into the model context for
**zero** new information — at full token cost. Repeated polling of a stalled
thread has driven single turns to hundreds of thousands or millions of tokens
with no forward progress. Poll deliberately, detect stalls, and escalate.

## Best Practices
- **Cap identical polls at 2–3.** Do not read the same thread/resource with the
  same arguments more than 2–3 consecutive times.
- **Compare each poll to the previous one.** If the returned content is
  unchanged from the last read, treat it as zero new information and stop
  polling immediately — do not re-read.
- **Escalate, don't spin.** Once the cap is hit or the content is stable, ask
  the developer how to proceed. Never poll forever.
- **Understand the cost.** Re-reading a static thread copies the whole thread
  back into context every time. Unchanged content means you paid full token
  cost to learn nothing.

## Process
1. Poll the thread once and record the returned content.
2. Before each subsequent poll, decide whether it is worth it. Do not exceed
   2–3 consecutive reads with the same arguments.
3. After each poll, compare the content to the previous read. If it is
   unchanged, stop — the collaborator has not responded yet, and another read
   will not change that.
4. When the cap is reached or the content is stable, stop polling and escalate
   to the developer with a concise message that names the stalled sub-agent or
   collaborator and asks how to proceed, e.g. "Sub-agent `gtm` has not
   responded after 3 checks of its Slack thread — how do you want to proceed?"
5. Wait for the developer's direction instead of continuing to read.

## Common Pitfalls
- **Retry-forever loops.** Never keep re-reading a thread hoping it changes.
  Cap the polls and escalate.
- **Ignoring unchanged content.** Two identical reads in a row is your signal to
  stop, not to try again.
- **Silent spinning.** Do not burn turns polling without telling the developer
  the collaborator is stalled; surface the stall early and clearly.
