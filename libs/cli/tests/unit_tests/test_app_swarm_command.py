"""Unit tests for `/swarm` command parsing."""

from deepagents_cli.app import _parse_swarm_command


def test_parse_swarm_default_num_parallel() -> None:
    """Should parse the basic swarm invocation."""
    parsed, error = _parse_swarm_command("/swarm tasks.jsonl")

    assert error is None
    assert parsed is not None
    assert parsed.enrich_mode is False
    assert parsed.file_path == "tasks.jsonl"
    assert parsed.num_parallel == 10


def test_parse_swarm_num_parallel_flag() -> None:
    """Should parse the preferred --num-parallel flag."""
    parsed, error = _parse_swarm_command("/swarm tasks.jsonl --num-parallel 6")

    assert error is None
    assert parsed is not None
    assert parsed.num_parallel == 6


def test_parse_swarm_concurrency_compat_flag() -> None:
    """Should keep backward compatibility with --concurrency."""
    parsed, error = _parse_swarm_command("/swarm tasks.jsonl --concurrency 4")

    assert error is None
    assert parsed is not None
    assert parsed.num_parallel == 4


def test_parse_swarm_enrich_mode() -> None:
    """Should parse enrich-specific options."""
    parsed, error = _parse_swarm_command(
        "/swarm --enrich companies.csv --num-parallel 3 --output enriched.csv --id-column ticker"
    )

    assert error is None
    assert parsed is not None
    assert parsed.enrich_mode is True
    assert parsed.file_path == "companies.csv"
    assert parsed.num_parallel == 3
    assert parsed.output_path == "enriched.csv"
    assert parsed.id_column == "ticker"


def test_parse_swarm_missing_file_path() -> None:
    """Should surface a clear error for missing file path."""
    parsed, error = _parse_swarm_command("/swarm --enrich")

    assert parsed is None
    assert error == "Error: No file path provided"
