# Better Harness Report

- Created: 2026-04-07T22:41:12+00:00
- Model: `claude-sonnet-4-6`
- Selected modules: `clarify_semantics_without_reasking`

## Score Summary

| Split | Baseline | Final |
| --- | --- | --- |
| Optimization | 0/2 | 1/2 |
| Holdout | 1/1 | 1/1 |

## Iterations

- Iteration 1: accepted `clarify_semantics_without_reasking` from baseline 0/2
  - `clarify_semantics_without_reasking` -> 1/2 (best non-regressing improvement)
  - `direct_send_with_reasonable_defaults` -> 1/2 (improved optimization score)
  - `completion_confirmations_include_identifiers` -> 0/2 (did not improve optimization score)
  - `summary_defaults_and_delivery` -> 0/2 (did not improve optimization score)
  - `bounded_research_then_deliver` -> 0/2 (did not improve optimization score)
  - `customer_support_domain_probe` -> 0/2 (did not improve optimization score)

## Final Holdout Failures

- None
