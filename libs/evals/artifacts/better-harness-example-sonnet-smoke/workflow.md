# Better Harness Example

- Created: 2026-04-07T22:42:02+00:00
- Model: `claude-sonnet-4-6`
- Output dir: `artifacts/better-harness-example-sonnet-smoke`

## Split

| Group | Cases |
| --- | --- |
| Optimization | `tool_indirect_email_report, followup_vague_send_report` |
| Holdout | `tool_direct_slack_dm` |
| Acceptance | `tests/evals/test_tool_selection.py::test_direct_request_slack_dm, tests/evals/test_followup_quality.py::test_followup_question_quality[claude-sonnet-4-6-vague_send_report]` |

## Hill Climb

| Split | Baseline | Final |
| --- | --- | --- |
| Optimization | `0/2` | `1/2` |
| Holdout | `1/1` | `1/1` |

- Selected modules: `clarify_semantics_without_reasking`

## Acceptance

| Variant | Tool Use | Conversation | Combined |
| --- | --- | --- | --- |
| Baseline | `1/1` | `0/1` | `1/2` |
| Optimized | `1/1` | `1/1` | `2/2` |
