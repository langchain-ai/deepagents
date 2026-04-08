# Better Harness Report

- Created: 2026-04-07T06:12:52+00:00
- Model: `claude-sonnet-4-6`
- Selected modules: `clarify_semantics_without_reasking, summary_defaults_and_delivery`

## Score Summary

| Split | Baseline | Final |
| --- | --- | --- |
| Optimization | 1/5 | 4/5 |
| Holdout | 2/4 | 2/4 |

## Iterations

- Iteration 1: accepted `clarify_semantics_without_reasking` from baseline 1/5
  - `clarify_semantics_without_reasking` -> 3/5 (best non-regressing improvement)
  - `summary_defaults_and_delivery` -> 2/5 (improved optimization score)
  - `direct_send_with_reasonable_defaults` -> 2/5 (improved optimization score)
  - `customer_support_domain_probe` -> 1/5 (did not improve optimization score)
  - `completion_confirmations_include_identifiers` -> 1/5 (did not improve optimization score)
  - `bounded_research_then_deliver` -> 1/5 (did not improve optimization score)
- Iteration 2: accepted `summary_defaults_and_delivery` from baseline 3/5
  - `summary_defaults_and_delivery` -> 4/5 (best non-regressing improvement)
  - `bounded_research_then_deliver` -> 3/5 (did not improve optimization score)
  - `direct_send_with_reasonable_defaults` -> 3/5 (did not improve optimization score)
  - `completion_confirmations_include_identifiers` -> 3/5 (did not improve optimization score)
  - `customer_support_domain_probe` -> 3/5 (did not improve optimization score)

## Final Holdout Failures

- `tool_chain_search_then_email`: GraphRecursionError: Recursion limit of 12 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
- `followup_vague_customer_support`: 1/3 criteria failed — Criteria 2: The agent asks about the current setup and the type of help needed, but does not ask about the domain or product to understand what kinds of customer questions to expect. There is no question about the nature of the business, the product/service being offered, or the subject matter of the customer inquiries.
