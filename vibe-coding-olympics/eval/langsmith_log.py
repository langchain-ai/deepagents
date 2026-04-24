"""Best-effort LangSmith logging for single-site evaluations.

Swallows all LangSmith errors — scoring succeeds locally even when
LangSmith is unreachable or not configured.
"""

from __future__ import annotations

import sys
from itertools import starmap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def log_to_langsmith(
    *,
    site_name: str,
    prompt: str,
    html: str,
    screenshot_b64: str,
    axes: dict[str, float | None],
    round_num: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log one site's per-axis scores to LangSmith as an experiment.

    Each axis becomes its own evaluator-result so the experiment view can
    be sorted/filtered by axis. Axes with score `None` are skipped.

    Args:
        site_name: Display name of the site being evaluated.
        prompt: The original prompt shown to the contestant.
        html: HTML source of the rendered page.
        screenshot_b64: Base64-encoded PNG screenshot.
        axes: Per-axis scores in [0.0, 1.0]; `None` values are skipped.
        round_num: Optional round number for experiment naming and metadata.
        metadata: Optional extra metadata to attach to the run.
    """
    try:
        from langsmith import evaluate as ls_evaluate
    except ImportError:
        return

    scored_axes = {k: v for k, v in axes.items() if v is not None}
    if not scored_axes:
        return

    try:

        def _target(inputs: dict[str, Any]) -> dict[str, Any]:
            return {
                "html": inputs["html"],
                "screenshot_b64": inputs["screenshot_b64"],
            }

        def _make_evaluator(key: str, score: float) -> Callable[..., dict[str, Any]]:
            def _eval(
                run: Any,  # noqa: ARG001
                example: Any,  # noqa: ARG001
            ) -> dict[str, Any]:
                return {"key": key, "score": score}

            return _eval

        evaluators = list(starmap(_make_evaluator, scored_axes.items()))

        prefix = (
            f"round-{round_num}-{site_name}"
            if round_num is not None
            else f"{site_name}"
        )
        base_metadata: dict[str, Any] = {
            "event": "interrupt-2026",
            "player": site_name,
            "prompt": prompt,
        }
        if round_num is not None:
            base_metadata["round"] = round_num
        if metadata:
            base_metadata.update(metadata)

        ls_evaluate(
            target=_target,
            data=[
                {
                    "inputs": {
                        "prompt": prompt,
                        "html": html,
                        "screenshot_b64": screenshot_b64,
                    },
                }
            ],
            evaluators=evaluators,
            experiment_prefix=prefix,
            metadata=base_metadata,
        )
        print(f"  LangSmith: logged {site_name}")
    except Exception as exc:
        msg = f"LangSmith: failed for {site_name}: {exc}"
        print(f"  {msg}", file=sys.stderr)
