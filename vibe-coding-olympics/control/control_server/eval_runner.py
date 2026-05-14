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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_SUBPROCESS_TIMEOUT_SECS = 15.0
"""Hard cap on a single-site eval before live scoring falls back."""

LLM_AXES: tuple[str, ...] = (
    "color",
    "typography",
    "layout",
    "content_completeness",
    "creativity",
    "interpretation_quality",
    "accessibility",
)

_AXIS_WEIGHTS: dict[str, float] = {
    "color": 0.10,
    "typography": 0.10,
    "layout": 0.20,
    "content_completeness": 0.20,
    "creativity": 0.15,
    "interpretation_quality": 0.15,
    "accessibility": 0.10,
}

_importability_check_done = False


def _default_eval_project_dir() -> Path:
    """Return the path to the bundled `eval/` workspace."""
    return Path(__file__).resolve().parent.parent.parent / "eval"


def _eval_project_dir() -> Path:
    """Return the eval workspace directory, honoring `VIBE_EVAL_DIR`."""
    raw = os.environ.get("VIBE_EVAL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_eval_project_dir()


def _random_axis_score() -> float:
    """Return a whole-number display score on the internal `[0, 1]` scale."""
    return random.randint(8, 9) / 10.0  # noqa: S311  # not cryptographic


def _random_axes() -> dict[str, float | None]:
    """Return a synthetic axis-score dict so the composite always renders."""
    return {axis: _random_axis_score() for axis in LLM_AXES}


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Per-site evaluation outcome, suitable for JSON serialization.

    Frozen so `run_eval`'s many fallback branches must each construct a
    complete, valid result up-front instead of mutating a half-built
    instance. Use `EvalResult.success(...)` or `EvalResult.fallback_for(...)`
    rather than the raw constructor when possible.
    """

    site_name: str
    url: str
    prompt: str
    round_num: int
    axes: dict[str, float | None] = field(default_factory=dict)
    overall: float | None = None
    fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        """Enforce shape invariants the run_eval consumers rely on."""
        if self.fallback and not self.fallback_reason:
            msg = "fallback EvalResult requires a non-empty fallback_reason"
            raise ValueError(msg)
        if self.overall is not None and not 0.0 <= self.overall <= 1.0:
            msg = f"overall must be in [0, 1], got {self.overall}"
            raise ValueError(msg)
        unknown = set(self.axes) - set(LLM_AXES)
        if unknown:
            msg = f"unknown axes in EvalResult: {sorted(unknown)}"
            raise ValueError(msg)

    @classmethod
    def success(
        cls,
        *,
        site_name: str,
        url: str,
        prompt: str,
        round_num: int,
        axes: dict[str, float | None],
    ) -> EvalResult:
        """Build a non-fallback result with `overall` derived from `axes`."""
        return cls(
            site_name=site_name,
            url=url,
            prompt=prompt,
            round_num=round_num,
            axes=axes,
            overall=aggregate(axes),
        )

    @classmethod
    def fallback_for(
        cls,
        *,
        site_name: str,
        url: str,
        prompt: str,
        round_num: int,
        reason: str,
    ) -> EvalResult:
        """Build a fallback result with randomized axes and a stamped reason."""
        axes = _random_axes()
        return cls(
            site_name=site_name,
            url=url,
            prompt=prompt,
            round_num=round_num,
            axes=axes,
            overall=aggregate(axes),
            fallback=True,
            fallback_reason=reason,
        )


def _sanitize_axes(raw: dict[str, Any]) -> dict[str, float | None]:
    """Coerce judge JSON axis values, defaulting missing values to random ints."""
    out: dict[str, float | None] = {}
    for axis in LLM_AXES:
        value = raw.get(axis)
        if value is None:
            out[axis] = _random_axis_score()
            continue
        try:
            score = float(value)
            if score > 1.0:
                score /= 10.0
            out[axis] = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            logger.warning(
                "Judge returned non-numeric value for axis %r: %r", axis, value
            )
            out[axis] = _random_axis_score()
    return out


def _has_llm_signal(axes: dict[str, float | None]) -> bool:
    """Return whether at least one non-accessibility axis has a score."""
    return any(
        axis != "accessibility" and value is not None
        for axis, value in axes.items()
    )


def aggregate(axes: dict[str, float | None]) -> float:
    """Weighted mean of present axes, in `[0, 1]`.

    Axes whose value is `None` are excluded from both the numerator and
    the denominator so a missing axis does not silently penalize the
    overall score. Returns `0.0` when no axis has a numeric value.
    """
    total = 0.0
    weighted = 0.0
    for axis, weight in _AXIS_WEIGHTS.items():
        value = axes.get(axis)
        if value is None:
            continue
        total += weight
        weighted += value * weight
    if total == 0.0:
        return 0.0
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


def _check_eval_importable_once() -> None:
    """Warn (once) if the bundled eval workspace is missing.

    Deferred from import time so the warning routes through the
    operator's logging configuration and respects late changes to
    `VIBE_EVAL_DIR`.
    """
    global _importability_check_done
    if _importability_check_done:
        return
    _importability_check_done = True
    project_dir = _eval_project_dir()
    if not (project_dir / "judge.py").exists():
        logger.warning(
            "No judge.py at %s; auto-eval will fall back to random scores.",
            project_dir,
        )


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
    _check_eval_importable_once()
    work_dir.mkdir(parents=True, exist_ok=True)
    fallback_kwargs = {
        "site_name": site_name,
        "url": url,
        "prompt": prompt,
        "round_num": round_num,
    }

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
        return EvalResult.fallback_for(**fallback_kwargs, reason=str(exc))

    if rc != 0:
        tail = stderr[-2048:].decode("utf-8", errors="replace") if stderr else ""
        logger.warning("Judge exited %d for %s. stderr tail: %s", rc, site_name, tail)
        return EvalResult.fallback_for(**fallback_kwargs, reason=f"judge exit {rc}")

    result_path = work_dir / f"round-{round_num}-{_safe_site_filename(site_name)}.json"
    if not result_path.exists():
        logger.warning(
            "Judge JSON missing at %s; stdout: %s", result_path, stdout[-512:]
        )
        return EvalResult.fallback_for(
            **fallback_kwargs, reason="judge output JSON missing"
        )

    try:
        raw = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read judge JSON at %s: %s", result_path, exc)
        return EvalResult.fallback_for(
            **fallback_kwargs, reason="judge output JSON unreadable"
        )

    axes_raw = raw.get("axes") if isinstance(raw, dict) else None
    if not isinstance(axes_raw, dict):
        logger.warning("Judge JSON missing `axes` mapping in %s", result_path)
        return EvalResult.fallback_for(
            **fallback_kwargs, reason="judge output JSON missing axes"
        )

    sanitized = _sanitize_axes(axes_raw)
    if not _has_llm_signal(sanitized):
        logger.warning("Judge JSON has no usable LLM axes in %s", result_path)
        return EvalResult.fallback_for(
            **fallback_kwargs, reason="judge output missing LLM scores"
        )
    return EvalResult.success(**fallback_kwargs, axes=sanitized)


def to_obs_score(overall: float) -> float:
    """Scale an aggregated `[0, 1]` result to the OBS scoreboard's 0..10 range."""
    return round(max(0.0, min(1.0, overall)) * 10.0, 2)
