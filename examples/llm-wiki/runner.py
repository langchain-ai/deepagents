"""CLI entrypoint for the LLM wiki example."""

from __future__ import annotations

from collections.abc import Sequence

from helpers import WikiError, parse_config, run


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LLM wiki CLI."""
    try:
        config = parse_config(argv)
        run_result = run(config)
    except WikiError as exc:
        print(f"error: {exc}")  # noqa: T201
        return 1

    if run_result.answer:
        print(run_result.answer)  # noqa: T201
    if run_result.hub_url:
        print(f"Context Hub: {run_result.hub_url}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
