"""Evaluation utilities for MemoryAgentBench integration.

Adapted from https://github.com/HUST-AI-HYZ/MemoryAgentBench
(ICLR 2026: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions)

Only the subset needed for Conflict Resolution and Test-Time Learning splits is
included here. Normalized substring matching is implemented by
`substring_match()`, which compares `normalize_answer(ground_truth)` against
`normalize_answer(prediction)`.
"""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize text for evaluation.

    Lowercases, strips punctuation, removes articles, and collapses whitespace.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text.
    """
    result = text.lower()
    result = "".join(ch for ch in result if ch not in string.punctuation)
    result = re.sub(r"\b(a|an|the)\b", " ", result)
    return " ".join(result.split())


def substring_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized ground truth is a substring of normalized prediction.

    This helper defines the normalized substring match used by the
    MemoryAgentBench scorer.

    Args:
        prediction: The predicted text.
        ground_truth: The ground truth text.

    Returns:
        Whether the ground truth is contained in the prediction.
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def substring_match_any(
    prediction: str,
    ground_truths: str | list[str] | list[list[str]],
) -> bool:
    """Check substring match against any acceptable ground truth.

    This is the entry point used by `_score_predictions()` in
    `test_memory_agent_bench.py`.

    Args:
        prediction: The predicted text.
        ground_truths: One or more acceptable ground truth answers.

    Returns:
        `True` if at least one ground truth is a substring of the prediction.
    """
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        gt_list = [gt for sub in ground_truths for gt in sub]
    else:
        gt_list = list(ground_truths)

    return any(substring_match(prediction, gt) for gt in gt_list)
