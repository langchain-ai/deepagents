"""Tests for check_extras_sync main() checks and multi-path run() aggregation."""

from pathlib import Path

from check_extras_sync import main, run


def _write(path: Path, *, required: str, optional: str | None = None) -> Path:
    """Write a minimal pyproject.toml with the given required/optional deps."""
    body = f'[project]\nname = "x"\nversion = "0.1.0"\ndependencies = [{required}]\n'
    if optional is not None:
        body += f"[project.optional-dependencies]\nall = [{optional}]\n"
    path.write_text(body)
    return path


def test_main_in_sync(tmp_path) -> None:
    """An extra whose version matches the required dep passes."""
    pyproject = _write(
        tmp_path / "pyproject.toml", required='"httpx>=1.0"', optional='"httpx>=1.0"'
    )
    assert main(pyproject) == 0


def test_main_mismatch(tmp_path) -> None:
    """An extra whose version drifts from the required dep fails."""
    pyproject = _write(
        tmp_path / "pyproject.toml", required='"httpx>=1.0"', optional='"httpx>=2.0"'
    )
    assert main(pyproject) == 1


def test_main_no_optional_dependencies(tmp_path) -> None:
    """A package without optional extras passes trivially."""
    pyproject = _write(tmp_path / "pyproject.toml", required='"httpx>=1.0"')
    assert main(pyproject) == 0


def test_run_all_paths_in_sync(tmp_path) -> None:
    """`run` returns 0 when every path is in sync."""
    a = _write(tmp_path / "a.toml", required='"httpx>=1.0"', optional='"httpx>=1.0"')
    b = _write(tmp_path / "b.toml", required='"rich>=15"')
    assert run([str(a), str(b)]) == 0


def test_run_aggregates_failure_without_short_circuit(tmp_path) -> None:
    """`run` returns 1 if any path fails, checking every path.

    The good path is first so a buggy implementation that returns the first
    path's result (rather than aggregating) would wrongly return 0.
    """
    good = _write(
        tmp_path / "good.toml", required='"httpx>=1.0"', optional='"httpx>=1.0"'
    )
    bad = _write(
        tmp_path / "bad.toml", required='"httpx>=1.0"', optional='"httpx>=2.0"'
    )
    assert run([str(good), str(bad)]) == 1


def test_run_defaults_to_pyproject(tmp_path, monkeypatch) -> None:
    """With no args, `run` checks ./pyproject.toml in the working directory."""
    _write(
        tmp_path / "pyproject.toml", required='"httpx>=1.0"', optional='"httpx>=1.0"'
    )
    monkeypatch.chdir(tmp_path)
    assert run([]) == 0
