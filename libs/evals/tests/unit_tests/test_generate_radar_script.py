from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parents[2]
_SCRIPT = _EVALS_DIR / "scripts" / "generate_radar.py"


def _run_generate_radar(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=_EVALS_DIR,
        timeout=30,
    )


def test_empty_summary_clears_stale_outputs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("[]", encoding="utf-8")

    output = tmp_path / "charts" / "radar.png"
    output.parent.mkdir(parents=True)
    output.write_text("stale aggregate", encoding="utf-8")

    individual_dir = tmp_path / "individual"
    individual_dir.mkdir()
    stale_png = individual_dir / "stale-model.png"
    stale_png.write_text("stale png", encoding="utf-8")
    keep_txt = individual_dir / "notes.txt"
    keep_txt.write_text("keep me", encoding="utf-8")

    result = _run_generate_radar(
        "--summary",
        str(summary),
        "--output",
        str(output),
        "--individual-dir",
        str(individual_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "skipped: no results to plot" in result.stdout
    assert not output.exists()
    assert not stale_png.exists()
    assert keep_txt.exists()


def test_too_few_categories_clears_stale_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            [
                {
                    "model": "openai:gpt-5.4",
                    "scores": {"file_operations": 0.8, "memory": 0.9},
                }
            ]
        ),
        encoding="utf-8",
    )

    output = tmp_path / "charts" / "radar.png"
    output.parent.mkdir(parents=True)
    output.write_text("stale aggregate", encoding="utf-8")

    individual_dir = tmp_path / "individual"
    individual_dir.mkdir()
    stale_png = individual_dir / "stale-model.png"
    stale_png.write_text("stale png", encoding="utf-8")

    result = _run_generate_radar(
        "--results",
        str(results),
        "--output",
        str(output),
        "--individual-dir",
        str(individual_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "skipped: radar chart needs >= 3 categories, got 2" in result.stdout
    assert not output.exists()
    assert not stale_png.exists()
