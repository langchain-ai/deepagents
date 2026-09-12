"""Slack channel for the agentic shopper.

Trigger Server provisions and hosts the Slack app, so this project supplies
only display fields — there is no signing secret or bot token to manage
(`requiredEnv` is empty).

ACCESS SCOPE — READ THIS
------------------------
This surface has no user allowlist. Anyone in the installed workspace who can
DM or mention the bot drives THIS agent, which spends from a single
agent-owned Link wallet (`identity.py` resolves every caller to the same
principal). The Link app approval still gates every charge, so nobody can
spend silently — but they can trigger purchase attempts and send approval
prompts to the wallet owner's phone.

Two knobs below reduce the blast radius:
  - trigger_on_all_messages=False -> only explicit @mentions and DMs, so the
    bot ignores ordinary channel chatter it happens to see.
  - allow_bot_triggers=False -> other bots cannot drive it, which also rules
    out automation loops.

To make this genuinely single-user, install the app into a Slack workspace
where you are the only member, and do not invite the bot to shared channels.
A per-caller wallet is the real fix and needs Supabase identity plus
user-owned connections — see `identity.py`.
"""

from __future__ import annotations

from managed_deepagents import channels

channel = channels.slack(
    name="agentic-shopper",
    description="Finds products, drives a real browser to checkout, and pays with Link.",
    # Only respond when addressed; ignore ambient channel traffic.
    trigger_on_all_messages=False,
    # Never let another bot initiate spending.
    allow_bot_triggers=False,
)
