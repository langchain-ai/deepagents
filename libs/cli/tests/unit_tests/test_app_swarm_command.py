"""Unit tests for `/swarm` command parsing."""

from deepagents_cli.app import _parse_swarm_command


def test_parse_swarm_default_num_parallel() -> None:
    """Should parse the basic swarm invocation."""
    parsed, error = _parse_swarm_command("/swarm tasks.jsonl")

    assert error is None
    assert parsed is not None
    assert parsed.file_path == "tasks.jsonl"
    assert parsed.num_parallel == 10


def test_parse_swarm_num_parallel_flag() -> None:
    """Should parse the preferred --num-parallel flag."""
    parsed, error = _parse_swarm_command("/swarm tasks.jsonl --num-parallel 6")

    assert error is None
    assert parsed is not None
    assert parsed.num_parallel == 6


def test_parse_swarm_output_dir() -> None:
    """Should parse output directory override."""
    parsed, error = _parse_swarm_command(
        "/swarm tasks.jsonl --num-parallel 3 --output-dir ./out"
    )

    assert error is None
    assert parsed is not None
    assert parsed.output_dir == "./out"


def test_parse_swarm_rejects_removed_enrich_mode() -> None:
    """Should return a clear error when --enrich is provided."""
    parsed, error = _parse_swarm_command("/swarm --enrich companies.csv")

    assert parsed is None
    assert error is not None
    assert "--enrich was removed" in error


def test_parse_swarm_missing_file_path() -> None:
    """Should surface a clear error for missing file path."""
    parsed, error = _parse_swarm_command("/swarm")

    assert parsed is None
    assert error is not None
    assert "Usage:" in error
