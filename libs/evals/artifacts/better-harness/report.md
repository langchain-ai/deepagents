# Better Harness Report

- Created: 2026-04-03T07:24:34+00:00
- Model: `claude-sonnet-4-6`
- Selected modules: `baseline`

## Score Summary

| Split | Baseline | Final |
| --- | --- | --- |
| Optimization | 0/4 | 0/4 |
| Holdout | 2/4 | 2/4 |

## Iterations

- Iteration 1: accepted `none` from baseline 0/4
  - `assume_default_email_scope` -> 0/4 (did not improve optimization score)
  - `customer_support_domain_probe` -> 0/4 (did not improve optimization score)
  - `automation_semantics_over_schedule` -> 0/4 (did not improve optimization score)
  - `minimum_necessary_followups` -> 0/4 (did not improve optimization score)
  - `avoid_capability_preambles` -> 0/4 (did not improve optimization score)
  - `act_on_explicit_send_requests` -> 0/4 (did not improve optimization score)

## Final Holdout Failures

- `followup_vague_monitor_system`: TypeError: `model` must be a string (e.g., 'openai:gpt-4o'), got ChatAnthropic. If you've already constructed a chat model object, use it directly instead of passing it to init_chat_model().
- `followup_vague_customer_support`: TypeError: `model` must be a string (e.g., 'openai:gpt-4o'), got ChatAnthropic. If you've already constructed a chat model object, use it directly instead of passing it to init_chat_model().
