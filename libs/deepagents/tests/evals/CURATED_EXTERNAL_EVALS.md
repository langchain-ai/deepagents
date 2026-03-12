# Curated External Eval Pack

This pack freezes 50 offline evals for the Deep Agents pytest/LangSmith harness.

## Source Breakdown

- `bfcl`: 20 cases
- `gorillabench`: 12 cases
- `hotpotqa`: 12 cases
- `toolbench`: 6 cases

## Notes

- ToolBench uses the official repository example queries because the repo does not vendor the full dataset in-tree.
- BFCL and APIBench come from the official Gorilla repository.
- HotpotQA comes from the official `hotpotqa/hotpot_qa` `distractor` validation split, and only the selected documents needed for each case are frozen into the JSON fixture.
- The generated JSON is what tests consume in CI, so the GitHub Actions workflow stays offline and deterministic.
