"""CLI entrypoint for the topic wiki runner example."""

from __future__ import annotations

from collections.abc import Sequence

from topic_wiki_helpers import TopicWikiError, parse_config, run


def main(argv: Sequence[str] | None = None) -> int:
    """Run the topic wiki runner CLI."""
    try:
        config = parse_config(argv)
        run_result = run(config)
    except TopicWikiError as exc:
        print(f"error: {exc}")  # noqa: T201
        return 1

    if run_result.answer:
        print(run_result.answer)  # noqa: T201
    if run_result.hub_url:
        print(f"Context Hub: {run_result.hub_url}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
