"""Request-admission rate limit enforcement.

Loads the tier/partner configuration from ``tiered_config.yaml`` and
exposes a single ``allow`` function that returns whether a request
should be admitted. The feature work (Feature C) is expected to
extend the YAML and keep the enforcement surface in this file small.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from common.logger import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent / "tiered_config.yaml"

_history: dict[str, list[float]] = defaultdict(list)


def _load_config() -> dict[str, Any]:
    """Load the tiered rate-limit configuration.

    Returns:
        Parsed YAML as a mapping.
    """
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_limit(config: dict[str, Any], partner: str, tier: str) -> int:
    """Resolve the effective per-second limit for a partner/tier pair.

    Args:
        config: Parsed YAML config.
        partner: Partner identifier.
        tier: Tier name (e.g. ``"standard"``).

    Returns:
        Requests-per-second cap; a per-partner override takes precedence
        over the tier default.
    """
    overrides = (config.get("overrides") or {}).get(partner) or {}
    if tier in overrides:
        return int(overrides[tier])
    tiers = config.get("tiers") or {}
    return int(tiers.get(tier, 1))


def allow(partner: str, tier: str) -> bool:
    """Decide whether a request from ``partner`` on ``tier`` should be admitted.

    Args:
        partner: Partner identifier.
        tier: Tier name.

    Returns:
        Whether the request is within the per-second cap.
    """
    config = _load_config()
    limit = _resolve_limit(config, partner, tier)

    now = time.time()
    window = _history[partner]
    window[:] = [ts for ts in window if now - ts < 1.0]
    if len(window) >= limit:
        logger.info("rate-limited partner=%s tier=%s", partner, tier)
        return False
    window.append(now)
    return True
