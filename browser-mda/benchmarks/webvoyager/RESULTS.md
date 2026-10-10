# Baseline results

15-task stratified sample (1 per site, seed 7) against the deployed
`browser-agent`. Steps effectively uncapped; 900s wall-clock bound.

| | SUCCESS | BLOCKED | FAILURE | cut off |
| --- | --- | --- | --- | --- |
| **current** | **12/15 (80%)** | 1 | 1 | 1 timeout |
| earlier run, steps capped at 60 | 8/15 (53%) | 1 | 2 | 4 |

The gap between those two rows is almost entirely the step cap, not agent
quality. All four tasks the cap cut off succeed without it:

| task | capped at 60 | uncapped |
| --- | --- | --- |
| ArXiv--41 | cut off | SUCCESS at 100 steps |
| Booking--4 | cut off | SUCCESS at 140 steps |
| Google Flights--3 | cut off | SUCCESS at 66 steps |
| Google Map--32 | cut off | SUCCESS at 54 steps |

This is why the runner does not ship a low step ceiling. A ceiling ends a
slow-but-converging run and a stuck run identically, and the result is recorded
as a failure either way.

## Cost

| | steps | seconds |
| --- | --- | --- |
| median | 22 | 192 |
| mean | 40 | — |
| max | 140 (Booking) | 600 |

Read-shaped tasks are cheap and stable across runs: Apple 8 steps, Amazon 12,
GitHub 14, Coursera 16. Interactive multi-step tasks are 3-6x that. Booking's
140 steps and ArXiv's 100 are real work -- both completed -- but they set the
practical wall-clock budget.

## Open problems

- **Final answers narrate process.** Most replies open with "I now have all the
  information I need. Let me analyze it:" before answering. `instructions.md`
  asks for the answer first; the model ignores it. The one remaining FAILURE
  (BBC News) contains the correct tournament and the correct count, and most
  likely grades as a failure for this reason -- so true quality is plausibly
  13/15. Needs a structural fix (response format or a final-answer step), not
  more prose.
- **One regression.** Google Search answered in 20 steps/81s under the capped
  run and timed out at 906s here, having wandered onto a third-party site. Step
  counts on open-ended search tasks have high variance.
- **Judge variance.** Single-sample grading has flipped individual tasks between
  identical result sets. Do not read a 1-2 task difference as signal.
