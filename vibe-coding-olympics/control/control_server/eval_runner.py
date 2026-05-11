"""Invoke the Vibe Coding Olympics judge for one site and parse its result.

The judge lives in a separate `uv` workspace at
`vibe-coding-olympics/eval/` because it pulls Playwright, openevals, and
LangSmith — heavyweight deps we do not want in the controller's runtime.

We shell out via `uv run --project <eval_dir> python judge.py ...`. The
judge writes a `round-{n}-{name}.json` file we read back. On any
subprocess, parse, or schema failure we fall back to randomized axis
scores so the OBS composite always has numbers to render.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_SUBPROCESS_TIMEOUT_SECS = 180.0
"""Hard cap on a single-site eval. 6 vision-LLM calls + Playwright is normally well under this."""

LLM_AXES: tuple[str, ...] = (
    "color",
    "typography",
    "layout",
    "content_completeness",
    "creativity",
    "interpretation_quality",
    "accessibility",
)


def _default_eval_project_dir() -> Path:
    """Return the path to the bundled `eval/` workspace."""
    return Path(__file__).resolve().parent.parent.parent / "eval"


def _eval_project_dir() -> Path:
    """Return the eval workspace directory, honoring `VIBE_EVAL_DIR`."""
    raw = os.environ.get("VIBE_EVAL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_eval_project_dir()


@dataclass(slots=True)
class EvalResult:
    """Per-site evaluation outcome, suitable for JSON serialization."""

    site_name: str
    url: str
    prompt: str
    round_num: int
    axes: dict[str, float | None] = field(default_factory=dict)
    overall: float | None = None
    fallback: bool = False
    fallback_reason: str | None = None


def _random_axes() -> dict[str, float | None]:
    """Return a synthetic axis-score dict so the composite always renders."""
    return {axis: round(random.uniform(0.45, 0.85), 3) for axis in LLM_AXES}  # noqa: S311  # not cryptographic


def _sanitize_axes(raw: dict[str, Any]) -> dict[str, float | None]:
    """Coerce judge JSON axis values to `float | None`, dropping unknown axes."""
    out: dict[str, float | None] = {}
    for axis in LLM_AXES:
        value = raw.get(axis)
        if value is None:
            out[axis] = None
            continue
        try:
            out[axis] = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Judge returned non-numeric value for axis %r: %r", axis, value
            )
            out[axis] = None
    return out


def aggregate(axes: dict[str, float | None]) -> float:
    """Default-weighted mean of present axes, in `[0, 1]`."""
    weights = {
        "color": 0.10,
        "typography": 0.10,
        "layout": 0.20,
        "content_completeness": 0.20,
        "creativity": 0.15,
        "interpretation_quality": 0.15,
        "accessibility": 0.10,
    }
    total = sum(weights[axis] for axis in weights if axis in axes)
    if total == 0:
        return 0.0
    weighted = sum(
        (axes[axis] or 0.0) * weights[axis] for axis in weights if axis in axes
    )
    return weighted / total


def _safe_site_filename(name: str) -> str:
    """Match `eval.judge.write_json`'s file-naming convention."""
    return name.replace(" ", "_")


async def _spawn_judge(
    *,
    url: str,
    name: str,
    prompt: str,
    round_num: int,
    work_dir: Path,
) -> tuple[int, bytes, bytes]:
    """Run `judge.py` in a subprocess and return `(returncode, stdout, stderr)`."""
    project_dir = _eval_project_dir()
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        msg = "uv binary not found on PATH; cannot invoke the judge"
        raise FileNotFoundError(msg)

    cmd = [
        uv_bin,
        "run",
        "--project",
        str(project_dir),
        "python",
        str(project_dir / "judge.py"),
        "--url",
        url,
        "--name",
        name,
        "--prompt",
        prompt,
        "--round",
        str(round_num),
        "--out",
        str(work_dir),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(project_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=EVAL_SUBPROCESS_TIMEOUT_SECS
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        msg = f"judge subprocess exceeded {EVAL_SUBPROCESS_TIMEOUT_SECS}s"
        raise TimeoutError(msg) from None
    return proc.returncode or 0, stdout, stderr


async def run_eval(
    *,
    url: str,
    site_name: str,
    prompt: str,
    round_num: int,
    work_dir: Path,
) -> EvalResult:
    """Evaluate one site and return scores + an aggregated overall in `[0, 1]`.

    On any failure (timeout, non-zero exit, missing or malformed JSON
    result) the returned `EvalResult` has `fallback=True` and `axes` set
    to a randomized payload so the controller can still post numbers to
    OBS. The caller is expected to log the fallback loudly.

    Args:
        url: Site URL the judge should fetch.
        site_name: Display name passed to the judge; also keys the JSON
            output filename.
        prompt: Contestant's original prompt.
        round_num: Round index used for filenames and LangSmith.
        work_dir: Directory the judge should write `round-N-name.json`
            into. The runner reads the file back from here.

    Returns:
        An `EvalResult` for this site. Never raises.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    result = EvalResult(
        site_name=site_name,
        url=url,
        prompt=prompt,
        round_num=round_num,
    )

    try:
        rc, stdout, stderr = await _spawn_judge(
            url=url,
            name=site_name,
            prompt=prompt,
            round_num=round_num,
            work_dir=work_dir,
        )
    except (FileNotFoundError, TimeoutError) as exc:
        logger.warning("Judge subprocess failed for %s: %s", site_name, exc)
        result.fallback = True
        result.fallback_reason = str(exc)
        result.axes = _random_axes()
        result.overall = aggregate(result.axes)
        return result

    if rc != 0:
        tail = stderr[-2048:].decode("utf-8", errors="replace") if stderr else ""
        logger.warning("Judge exited %d for %s. stderr tail: %s", rc, site_name, tail)
        result.fallback = True
        result.fallback_reason = f"judge exit {rc}"
        result.axes = _random_axes()
        result.overall = aggregate(result.axes)
        return result

    result_path = work_dir / f"round-{round_num}-{_safe_site_filename(site_name)}.json"
    if not result_path.exists():
        logger.warning(
            "Judge JSON missing at %s; stdout: %s", result_path, stdout[-512:]
        )
        result.fallback = True
        result.fallback_reason = "judge output JSON missing"
        result.axes = _random_axes()
        result.overall = aggregate(result.axes)
        return result

    try:
        raw = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read judge JSON at %s: %s", result_path, exc)
        result.fallback = True
        result.fallback_reason = "judge output JSON unreadable"
        result.axes = _random_axes()
        result.overall = aggregate(result.axes)
        return result

    axes_raw = raw.get("axes") if isinstance(raw, dict) else None
    if not isinstance(axes_raw, dict):
        logger.warning("Judge JSON missing `axes` mapping in %s", result_path)
        result.fallback = True
        result.fallback_reason = "judge output JSON missing axes"
        result.axes = _random_axes()
        result.overall = aggregate(result.axes)
        return result

    result.axes = _sanitize_axes(axes_raw)
    result.overall = aggregate(result.axes)
    return result


def to_obs_score(overall: float) -> float:
    """Scale an aggregated `[0, 1]` result to the OBS scoreboard's 0..10 range."""
    return round(max(0.0, min(1.0, overall)) * 10.0, 2)


def _ensure_eval_importable() -> None:
    """Best-effort sanity check that the bundled eval workspace exists."""
    project_dir = _eval_project_dir()
    if not (project_dir / "judge.py").exists():
        print(
            f"[eval_runner] warning: no judge.py at {project_dir}; "
            "auto-eval will fall back to random scores.",
            file=sys.stderr,
        )


_ensure_eval_importable()
