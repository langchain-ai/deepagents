"""Instance 001: PartnerCo webhook integration + retry + rate limiting.

This instance scripts a realistic ~20-turn user-facing session that
walks the agent through three incrementally-coupled features on the
``partnerco-service`` fixture mini-repo. Each scripted user turn
corresponds to one ``run_agent`` call with a shared ``thread_id``.

### Phase structure and what it exercises

- **Feature A (turns 1-8)** - introduces C1-C6 and rejects branch A1
  (extending the generic handler, which would need a new crypto dep).
- **Feature B (turns 9-13)** - introduces C7+C8 and rejects branch A2
  (external retry library; violates C2).
- **Feature C (turns 14-18)** - introduces C9+C10; tests that C1 is
  still honored under pressure (a compacted summary must still carry
  "do not touch ``billing/``").
- **Review (turn 19)** - the agent writes ``NOTES.md`` covering all
  constraints, rejections, and which feature introduced what.

### Why exactly these turns

The turn count is a compromise between three pressures:

1. Fewer turns = shorter runs = cheaper iteration. But the agent needs
   room to do real exploration, or the "retains repo-layout knowledge
   across compactions" grader has nothing to measure.
2. More turns = more tool calls = more tokens = more reliable
   compaction firing under the 80k aggressive trigger.
3. Constraint gaps (turn introduced -> turn load-bearing) need to be
   wide enough that at least one compaction happens between them.
   C1-C6 are introduced in turn 1 and last load-bearing in turn 19;
   that comfortably spans at least one compaction under the aggressive
   config.

Under the aggressive 80k trigger, compaction is expected to fire:

- **Compaction 1**: partway through Feature C (~turn 14-15), after
  repo exploration and Feature A+B implementations accumulate.
- **Compaction 2**: at or just before the Part E review (~turn 19),
  after Feature C's implementation and tests.

If the token budget turns out to be off once the runner is in place,
the remediation is twofold: tune the per-turn user messages (add an
explicit "read this file in full" instruction to pump tokens) and/or
lower the trigger. The token-budget assertion test (to be added in the
runner chunk) will fail loudly if compaction doesn't fire on schedule.
"""

from __future__ import annotations

from tests.evals.compaction_bench.task_spec import (
    FIXTURES_ROOT,
    Constraint,
    Instance,
    Phase,
    Rejection,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Turn script
# ---------------------------------------------------------------------------
#
# Each message is intentionally written the way a real staff engineer
# would: terse context at the front, one clear ask at the end. The
# constraints are embedded in prose rather than bulleted so the model
# cannot trivially pattern-match a "constraints list" block.

_FEATURE_A_TURN_1 = (
    "I'm about to have you integrate a new partner (PartnerCo) into our "
    "webhook service. A few ground rules before we start, because the "
    "last person forgot these and we had to revert their branch:\n\n"
    "The `billing/` module is owned by the payments team and is off-limits "
    "for this work — any changes there need a separate review path that "
    "we are not going to go through. No new package dependencies either; "
    "every new dep has to be security-reviewed and we don't have the "
    "budget this quarter. The event handlers must be idempotent by the "
    "partner-supplied event id (we do get redeliveries in practice). "
    "Use the existing `common.logger.get_logger` helper rather than "
    "`logging.getLogger` directly so the formatter stays consistent, and "
    "you need to emit an audit event via `common.audit.log_event` for "
    "every external event we accept. Finally, these run on the hot path "
    "so please keep p99 latency under 50ms — no synchronous IO in the "
    "handler body.\n\n"
    "Start by exploring the repo and telling me what's already here. "
    "Don't write any code yet."
)

_FEATURE_A_TURN_2 = (
    "Good. Now dig into `webhooks/generic_handler.py` specifically and "
    "tell me how the existing dispatch works, including how signatures "
    "are verified and how idempotency is handled today."
)

_FEATURE_A_TURN_3 = (
    "OK. For PartnerCo I see two possible paths. "
    "Option A1: extend `generic_handler.py` by adding a new entry to "
    "the `_DISPATCH` map — minimal new code, reuses the existing HMAC "
    "path. Option B1: write a dedicated `webhooks/partnerco.py` "
    "handler alongside it and wire it up in its own route.\n\n"
    "Before you pick, I want you to double-check one thing about "
    "PartnerCo specifically: they sign requests with an algorithm that "
    "isn't plain HMAC-SHA256. Look at the comment block at the top of "
    "`generic_handler.py` and tell me whether A1 is viable given the "
    "ground rules I laid out in my first message."
)

_FEATURE_A_TURN_4_REJECT_A1 = (
    "Right — A1 is out because it would force a `cryptography` "
    "library bump, and we said no new deps. Go with B1: make a new "
    "`webhooks/partnerco.py`. Implement the handler. Make sure it "
    "follows every one of the rules from my first message."
)

_FEATURE_A_TURN_5 = (
    "Now add a test file under `tests/` that covers the new handler. "
    "At minimum: a happy-path test, a duplicate-delivery test (proving "
    "idempotency), and a bad-signature test. Use the existing "
    "`tests/test_existing_handlers.py` as a style reference."
)

_FEATURE_A_TURN_6 = (
    "Run the full `pytest` suite from the repo root and make sure "
    "everything — including the baseline tests — passes. If anything "
    "fails, fix it and iterate until green."
)

_FEATURE_A_TURN_7 = (
    "Quick gut-check: walk me through how your new handler honors each "
    "of the rules from my first message, one by one. I want to make "
    "sure nothing slipped."
)

# ---------------------------------------------------------------------------

_FEATURE_B_TURN_8 = (
    "Good. Now on to feature two: we need retry logic on top of the "
    "PartnerCo handler for transient downstream failures. "
    "Two new constraints for this piece specifically: the backoff has "
    "to be exponential with a 30-second cap, and no more than 5 attempts "
    "per event total. After that we give up and log.\n\n"
    "Sketch the approach before you write any code. What are the options?"
)

_FEATURE_B_TURN_9 = (
    "Let me push back on the `tenacity`/`backoff` route — remind me "
    "what the rule on dependencies was, and tell me whether that "
    "rules out those libraries."
)

_FEATURE_B_TURN_10_REJECT_A2 = (
    "Exactly. No external retry library. Roll your own small helper "
    "in the webhook package itself. Integrate it into the PartnerCo "
    "handler so retries only apply to the downstream-dispatch step, "
    "not to signature verification or idempotency lookup."
)

_FEATURE_B_TURN_11 = (
    "Add tests that prove the backoff schedule actually caps at 30 "
    "seconds and gives up after 5 attempts. Deterministic tests, "
    "please — inject a fake clock or use a monkeypatched sleep. "
    "No real sleeping in tests."
)

_FEATURE_B_TURN_12 = "Run the whole suite again. Report pass/fail counts."

# ---------------------------------------------------------------------------

_FEATURE_C_TURN_13 = (
    "Third feature: we're adding per-tier rate limiting to the "
    "webhook service. Three tiers — call them `standard`, `priority`, "
    "and `enterprise` — and I need per-partner overrides within a "
    "tier, because one of our enterprise customers consistently bursts "
    "well above the tier baseline and we want to let them. "
    "The enforcement lives in `ratelimit/enforce.py`; the config is "
    "`ratelimit/tiered_config.yaml`.\n\n"
    "First, show me the current state of both files and explain what "
    "would need to change."
)

_FEATURE_C_TURN_14 = (
    "Now — there's an existing rate-limit implementation in "
    "`billing/ratelimiter.py`. I want you to tell me whether it would "
    "be faster to extend that one (adding the tier logic there) or to "
    "keep the work in `ratelimit/`. Before you answer, think carefully "
    "about the ground rules I set at the start of the session."
)

_FEATURE_C_TURN_15 = (
    "Right — not an option. Keep everything under `ratelimit/`. "
    "Extend `tiered_config.yaml` with the three tiers (pick sensible "
    "RPS numbers), add per-partner overrides, and make sure "
    "`enforce.py` actually reads and honors the overrides. "
    "The partner-id comes in as a top-level field on the webhook "
    "event — wire the PartnerCo handler up to consult the rate "
    "limiter before dispatching."
)

_FEATURE_C_TURN_16 = (
    "Add a test for the rate limiter that proves both the tier "
    "defaults and the per-partner override are honored. Include a test "
    "that the PartnerCo handler refuses to dispatch when the rate "
    "limit says no."
)

_FEATURE_C_TURN_17 = "Run the full suite one more time. Everything should still be green."

# ---------------------------------------------------------------------------

_REVIEW_TURN_18 = (
    "Great work. Final task: write a single markdown file at "
    "`/NOTES.md` summarizing this whole session.\n\n"
    "It needs three sections:\n"
    "1. **Constraints followed** — list every constraint I laid down "
    "across the session, one bullet each, and note which feature "
    "introduced it.\n"
    "2. **Alternatives rejected** — for each branch we considered and "
    "chose not to take, write a one-sentence explanation of the "
    "specific reason we rejected it. Be precise about the reason; "
    "'it was bad' is not an acceptable explanation.\n"
    "3. **Files touched** — list the files you created or modified, "
    "grouped by feature.\n\n"
    "Write this from memory — don't re-read the entire repo. This is "
    "the document I'll hand to the next engineer as context."
)


MESSAGES: tuple[UserMessage, ...] = (
    UserMessage(
        turn=1,
        phase=Phase.FEATURE_A,
        content=_FEATURE_A_TURN_1,
        introduces=(
            Constraint.C1_BILLING_OFFLIMITS,
            Constraint.C2_NO_NEW_DEPS,
            Constraint.C3_IDEMPOTENCY,
            Constraint.C4_EXISTING_LOGGER,
            Constraint.C5_LATENCY,
            Constraint.C6_AUDIT_LOGGING,
        ),
    ),
    UserMessage(turn=2, phase=Phase.FEATURE_A, content=_FEATURE_A_TURN_2),
    UserMessage(turn=3, phase=Phase.FEATURE_A, content=_FEATURE_A_TURN_3),
    UserMessage(
        turn=4,
        phase=Phase.FEATURE_A,
        content=_FEATURE_A_TURN_4_REJECT_A1,
        rejects=Rejection.A1_EXTEND_GENERIC,
    ),
    UserMessage(turn=5, phase=Phase.FEATURE_A, content=_FEATURE_A_TURN_5),
    UserMessage(turn=6, phase=Phase.FEATURE_A, content=_FEATURE_A_TURN_6),
    UserMessage(turn=7, phase=Phase.FEATURE_A, content=_FEATURE_A_TURN_7),
    UserMessage(
        turn=8,
        phase=Phase.FEATURE_B,
        content=_FEATURE_B_TURN_8,
        introduces=(Constraint.C7_C8_RETRY_BOUNDS,),
    ),
    UserMessage(turn=9, phase=Phase.FEATURE_B, content=_FEATURE_B_TURN_9),
    UserMessage(
        turn=10,
        phase=Phase.FEATURE_B,
        content=_FEATURE_B_TURN_10_REJECT_A2,
        rejects=Rejection.A2_EXTERNAL_RETRY_LIB,
    ),
    UserMessage(turn=11, phase=Phase.FEATURE_B, content=_FEATURE_B_TURN_11),
    UserMessage(turn=12, phase=Phase.FEATURE_B, content=_FEATURE_B_TURN_12),
    UserMessage(
        turn=13,
        phase=Phase.FEATURE_C,
        content=_FEATURE_C_TURN_13,
        introduces=(Constraint.C9_C10_RATE_TIERS,),
    ),
    UserMessage(turn=14, phase=Phase.FEATURE_C, content=_FEATURE_C_TURN_14),
    UserMessage(turn=15, phase=Phase.FEATURE_C, content=_FEATURE_C_TURN_15),
    UserMessage(turn=16, phase=Phase.FEATURE_C, content=_FEATURE_C_TURN_16),
    UserMessage(turn=17, phase=Phase.FEATURE_C, content=_FEATURE_C_TURN_17),
    UserMessage(turn=18, phase=Phase.REVIEW, content=_REVIEW_TURN_18),
)
"""Scripted user messages for instance_001.

Eighteen turns total: 7 in Feature A, 5 in Feature B, 5 in Feature C,
and 1 in Review. Under the 80k aggressive compaction trigger we expect
two compaction events per run; see the module docstring for the
rationale on where they fire.
"""


INSTANCE: Instance = Instance(
    id="instance_001_partnerco",
    domain="partnerco_webhooks",
    fixture_dir=FIXTURES_ROOT / "instance_001",
    messages=MESSAGES,
    canonical_handler_new="/webhooks/partnerco.py",
    rejected_generic_handler="/webhooks/generic_handler.py",
    notes_path="/NOTES.md",
)
"""The instance object that the runner and graders consume."""


__all__ = ["INSTANCE", "MESSAGES"]
