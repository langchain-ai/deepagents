"""Evaluation utilities for MemoryAgentBench integration.

Adapted from https://github.com/HUST-AI-HYZ/MemoryAgentBench
(ICLR 2026: Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions)

Only the subset needed for Conflict Resolution and Test-Time Learning splits is
included here.
"""

from __future__ import annotations

import re
import string
from collections import Counter


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


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Token-level F1 between prediction and ground truth.

    Args:
        prediction: The predicted text.
        ground_truth: The ground truth text.

    Returns:
        Tuple of (f1, precision, recall).
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    special = {"yes", "no", "noanswer"}
    if (norm_pred in special or norm_gt in special) and norm_pred != norm_gt:
        return (0.0, 0.0, 0.0)

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return (0.0, 0.0, 0.0)

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check normalized exact match.

    Args:
        prediction: The predicted text.
        ground_truth: The ground truth text.

    Returns:
        Whether the normalized texts match exactly.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized ground truth is a substring of normalized prediction.

    Args:
        prediction: The predicted text.
        ground_truth: The ground truth text.

    Returns:
        Whether the ground truth is contained in the prediction.
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def max_over_ground_truths(
    metric_fn: object,
    prediction: str,
    ground_truths: str | list[str] | list[list[str]],
) -> float:
    """Compute the max of `metric_fn` over all ground truths.

    Args:
        metric_fn: A callable `(prediction, single_gt) -> float | bool`.
        prediction: The predicted text.
        ground_truths: One or more acceptable ground truth answers.

    Returns:
        Maximum score across all ground truths.
    """
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        gt_list = [gt for sub in ground_truths for gt in sub]
    else:
        gt_list = list(ground_truths)

    return max(float(metric_fn(prediction, gt)) for gt in gt_list)


def calculate_metrics(
    prediction: str,
    ground_truths: str | list[str] | list[list[str]],
) -> dict[str, float]:
    """Compute standard QA metrics for a single prediction.

    Args:
        prediction: The predicted text.
        ground_truths: One or more acceptable ground truth answers.

    Returns:
        Dict with `exact_match`, `f1`, and `substring_exact_match` scores.
    """
    return {
        "exact_match": max_over_ground_truths(exact_match, prediction, ground_truths),
        "f1": max_over_ground_truths(lambda p, g: f1_score(p, g)[0], prediction, ground_truths),
        "substring_exact_match": max_over_ground_truths(substring_match, prediction, ground_truths),
    }
