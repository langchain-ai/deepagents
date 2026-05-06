r"""Generate a composite radar chart by overlaying multiple GitHub Actions eval runs.

Each `📊 Evals` run uploads an `evals-summary` artifact whose payload is a
JSON array of model results. To compare results across separate dispatches
(e.g. a bake-off where each model was run as its own workflow dispatch), this
script downloads each run's `evals-summary`, flattens them into one array, and
invokes `generate_radar.py` so every entry becomes a separate trace on a single
radar.

Requires the `gh` CLI (authenticated against the target repo). The script
shells out to `gh run download` so SSO / token handling is identical to the
manual procedure.

Usage:
    python scripts/composite_radar.py 25403850424 25403883357 25403894412 \\
        -o /tmp/composite-radar.png \\
        --title "Composite — open-weights bake-off"

    # Custom repo (defaults to langchain-ai/deepagents):
    python scripts/composite_radar.py 123 456 --repo my-org/my-fork -o out.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parents[1]
"""Root of the evals package (libs/evals/)."""

_DEFAULT_REPO = "langchain-ai/deepagents"
"""Repo to pull artifacts from when `--repo` is not provided."""

_ARTIFACT_NAME = "evals-summary"
"""Artifact name uploaded by the `📋 Aggregate evals` job."""

_SUMMARY_FILENAME = "evals_summary.json"
"""Filename inside the `evals-summary` artifact."""


def _download_summary(run_id: str, repo: str, dest: Path) -> Path:
    """Download `evals-summary` from a single GHA run via `gh run download`.

    Args:
        run_id: GitHub Actions run ID.
        repo: Owner/name slug, e.g. `langchain-ai/deepagents`.
        dest: Directory to extract the artifact into. Created if absent.

    Returns:
        Path to the extracted `evals_summary.json`.

    Raises:
        FileNotFoundError: If the artifact extracted but the expected JSON file
            is missing.
        subprocess.CalledProcessError: If `gh run download` fails (run does not
            exist, artifact not present, auth failure, etc.).
    """
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gh", "run", "download", run_id, "-R", repo, "-n", _ARTIFACT_NAME, "-D", str(dest)],
        check=True,
    )
    summary = dest / _SUMMARY_FILENAME
    if not summary.is_file():
        msg = f"{_ARTIFACT_NAME} for run {run_id} did not contain {_SUMMARY_FILENAME}"
        raise FileNotFoundError(msg)
    return summary


def _concat_summaries(summaries: list[Path]) -> list[dict]:
    """Flatten a list of per-run summary arrays into a single array.

    Args:
        summaries: Paths to per-run `evals_summary.json` files.

    Returns:
        Combined list of model-result dicts in the same order as `summaries`.

    Raises:
        TypeError: If any summary file is not a JSON array.
    """
    combined: list[dict] = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            msg = f"{path} is not a JSON array (got {type(data).__name__})"
            raise TypeError(msg)
        combined.extend(data)
    return combined


def _render(combined_path: Path, output: Path, title: str, individual_dir: Path | None) -> None:
    """Invoke `generate_radar.py` with the merged summary.

    Args:
        combined_path: Path to the merged summary JSON.
        output: Output PNG path; the dark variant is derived by the radar script.
        title: Chart title.
        individual_dir: If provided, also emit per-model PNGs into this dir.

    Raises:
        subprocess.CalledProcessError: If radar generation fails.
    """
    cmd = [
        sys.executable,
        str(_EVALS_DIR / "scripts" / "generate_radar.py"),
        "--summary",
        str(combined_path),
        "-o",
        str(output),
        "--title",
        title,
    ]
    if individual_dir is not None:
        cmd.extend(["--individual-dir", str(individual_dir)])
    subprocess.run(cmd, check=True)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Overlay eval results from multiple GHA runs onto one radar chart.",
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="One or more GitHub Actions run IDs whose `evals-summary` artifacts will be merged.",
    )
    parser.add_argument(
        "--repo",
        default=_DEFAULT_REPO,
        help=f"GitHub repo slug to pull artifacts from (default: {_DEFAULT_REPO}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("charts/composite-radar.png"),
        help="Output chart path (default: charts/composite-radar.png).",
    )
    parser.add_argument(
        "--title",
        default="Composite eval results",
        help="Chart title.",
    )
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=None,
        help="Optional directory for per-model radar PNGs.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Where to stage downloaded artifacts. A temp dir is used and "
        "cleaned up automatically when omitted.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="If `--workdir` is set, do not delete it on exit (useful for debugging).",
    )

    args = parser.parse_args()

    if shutil.which("gh") is None:
        print("error: `gh` CLI not found on PATH", file=sys.stderr)
        sys.exit(1)

    explicit_workdir = args.workdir is not None
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="composite-radar-"))
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        summaries: list[Path] = []
        for run_id in args.run_ids:
            try:
                summaries.append(_download_summary(run_id, args.repo, workdir / run_id))
            except subprocess.CalledProcessError as exc:
                print(
                    f"error: failed to download {_ARTIFACT_NAME} from run {run_id}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                sys.exit(1)

        try:
            combined = _concat_summaries(summaries)
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"error: could not merge summaries: {exc}", file=sys.stderr)
            sys.exit(1)

        if not combined:
            print(
                "error: merged summary is empty — none of the runs produced "
                "model results to chart.",
                file=sys.stderr,
            )
            sys.exit(1)

        combined_path = workdir / "combined_summary.json"
        combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"merged {len(combined)} model entries from {len(summaries)} run(s)")

        try:
            _render(combined_path, args.output, args.title, args.individual_dir)
        except subprocess.CalledProcessError as exc:
            print(f"error: radar generation failed: {exc}", file=sys.stderr)
            sys.exit(1)
    finally:
        if not explicit_workdir or not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
