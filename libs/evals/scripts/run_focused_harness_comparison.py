"""Run focused baseline-vs-improved harness comparisons for one or more models."""

from deepagents_evals.better_harness.focused_comparison import (
    build_parser,
    run_focused_comparison,
)

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_focused_comparison(
        models=args.models,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
    print((args.output_dir / "comparison.md").read_text())
