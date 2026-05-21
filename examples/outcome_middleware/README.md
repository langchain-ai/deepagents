# OutcomeMiddleware example

`OutcomeMiddleware` lets a caller declare *what done looks like* via a markdown
rubric, then has the agent self-iterate (with critic-style feedback injected
between turns) until either the rubric is met or a hard iteration cap is
reached. It generalizes Claude Code's `/goal` and Anthropic Managed Agents'
`user.define_outcome` into a deepagents middleware.

The rubric is supplied **per invocation** via the `rubric` state field, so a
single agent definition can serve many different outcomes. When no rubric is
supplied the middleware is a no-op, so it's safe to install unconditionally.

## What this example shows

`outcome_demo.py` walks through the **four terminal outcomes** the middleware
exposes, running each scenario end-to-end against a scripted main chat model
and a stubbed grader. Each run asserts the final `outcome_status` matches
expectation, so the demo doubles as a smoke test.

| # | Scenario | Demonstrates |
|---|---|---|
| 1 | `satisfied` first try | Grader signs off on the first attempt; one model call, one grader call, no loop. |
| 2 | `needs_revision` → `satisfied` | Grader rejects iteration 0 with per-criterion gaps; middleware injects feedback as a tagged synthetic `HumanMessage` (`name="outcome_grader"`); agent revises on iteration 1; grader accepts. |
| 3 | `max_iterations_reached` | Grader rejects every attempt; with `max_iterations=2`, the second rejection terminates the loop instead of looping again. |
| 4 | Grader exception → `failed` | Grader sub-agent raises before producing a verdict; middleware records a `failed` evaluation with the exception text and terminates cleanly (no host-process crash). |

All scenarios are run sequentially. Everything is mocked with fake chat models
so the example runs offline.

## Running

```bash
cd examples/outcome_middleware
python outcome_demo.py
```

Expected output:

```
--- satisfied first try ---
Agent's first draft satisfies every criterion; grader signs off immediately. One model call, one grader call.
  iter 0: satisfied -- all criteria met on the first attempt
  final outcome_status: satisfied

--- needs_revision then satisfied ---
First draft misses the rubric; grader returns needs_revision with per-criterion gaps; middleware injects feedback as a tagged HumanMessage and loops; second draft satisfies.
  iter 0: needs_revision -- the haiku does not mention the sea
  iter 1: satisfied -- all criteria met after revision
  final outcome_status: satisfied

--- max_iterations_reached ---
Every grader pass returns needs_revision. With max_iterations=2, the second rejection terminates with max_iterations_reached instead of looping again.
  iter 0: needs_revision -- still missing the sea reference
  iter 1: needs_revision -- still missing the sea reference
  final outcome_status: max_iterations_reached

--- grader exception -> failed ---
Grader raises before producing a verdict; middleware records a failed evaluation with the exception text and terminates cleanly.
  iter 0: failed -- Grader raised RuntimeError: grader sub-agent unavailable
  final outcome_status: failed

All 4 scenarios completed successfully.
```

## Using your own models

To run against real models, replace the `_ScriptedChatModel` and the
`_stub_grader` call with whatever you'd normally pass to `create_deep_agent`
and `OutcomeMiddleware`:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from deepagents import OutcomeMiddleware, create_deep_agent

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-opus-4-7"),
    middleware=[
        OutcomeMiddleware(
            evaluator_model="anthropic:claude-haiku-4-5",
            max_iterations=5,
        ),
    ],
)
result = agent.invoke({
    "messages": [HumanMessage("write a haiku about the sea")],
    "rubric": "- exactly 3 lines\n- mentions the sea\n- syllable pattern 5/7/5",
})
print(result["outcome_status"])          # 'satisfied' | 'max_iterations_reached' | ...
print(result["outcome_evaluations"][-1]) # final per-criterion verdict
```

## Other patterns

- **Pure-LLM grader (default).** `OutcomeMiddleware(max_iterations=5)`. The
  grader reasons from the transcript alone.
- **Script-based grader.** Pass tools the grader may call before deciding:
  `OutcomeMiddleware(grader_tools=[run_shell, read_file])`. The grader is
  itself a `create_agent(..., response_format=GraderResponse)`, so it can
  interleave verification calls (e.g. running `pytest`, checking a file
  exists) with its final structured verdict.
- **Chained outcomes on a checkpointed thread.** Run one outcome to
  completion, then `.invoke` again with a *new* `rubric` on the same
  `thread_id` to start a fresh outcome with conversation history intact.
  Each outcome attempt gets a fresh `outcome_id` and evaluations are
  appended (not replaced) so the full history is preserved on state.
- **Tagged synthetic messages.** The revision message the middleware
  injects between turns is a `HumanMessage` with
  `name="outcome_grader"` and
  `additional_kwargs={"lc_source": "outcome_grader"}` so downstream
  consumers (evals, UIs, observability) can distinguish grader-injected
  turns from real user input. The constant is exported as
  `deepagents.OUTCOME_GRADER_MESSAGE_SOURCE`.

See `libs/deepagents/deepagents/middleware/outcomes.py` for the full API.
