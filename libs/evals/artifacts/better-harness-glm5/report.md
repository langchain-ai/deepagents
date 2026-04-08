# Better Harness Report

- Created: 2026-04-07T06:21:10+00:00
- Model: `baseten:zai-org/GLM-5`
- Selected modules: `summary_defaults_and_delivery, clarify_semantics_without_reasking`

## Score Summary

| Split | Baseline | Final |
| --- | --- | --- |
| Optimization | 0/5 | 5/5 |
| Holdout | 2/4 | 2/4 |

## Iterations

- Iteration 1: accepted `summary_defaults_and_delivery` from baseline 0/5
  - `summary_defaults_and_delivery` -> 4/5 (best non-regressing improvement)
  - `completion_confirmations_include_identifiers` -> 2/5 (improved optimization score)
  - `direct_send_with_reasonable_defaults` -> 1/5 (improved optimization score)
  - `bounded_research_then_deliver` -> 1/5 (improved optimization score)
  - `clarify_semantics_without_reasking` -> 1/5 (improved optimization score)
  - `customer_support_domain_probe` -> 0/5 (did not improve optimization score)
- Iteration 2: accepted `clarify_semantics_without_reasking` from baseline 4/5
  - `clarify_semantics_without_reasking` -> 5/5 (best non-regressing improvement)
  - `customer_support_domain_probe` -> 3/5 (regressed existing optimization pass)
  - `bounded_research_then_deliver` -> 3/5 (regressed existing optimization pass)
  - `direct_send_with_reasonable_defaults` -> 3/5 (regressed existing optimization pass)
  - `completion_confirmations_include_identifiers` -> 1/5 (regressed existing optimization pass)

## Final Holdout Failures

- `tool_chain_search_then_email`: GraphRecursionError: Recursion limit of 12 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
For troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
- `followup_vague_customer_support`: 1/3 criteria failed — Criteria 2: The agent asks about where customer questions are received, what kind of help is needed, and where data is stored, but never asks about the domain or product (e.g., what industry the user is in, what their product/service is, or what kinds of questions customers typically ask). This information is important to understand the nature of the questions to expect.
