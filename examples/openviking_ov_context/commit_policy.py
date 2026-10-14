"""07: decide whether to commit and submit eligible sessions to OpenViking.

The caller owns write confirmation, session serialization, run counting, and
session metrics. This module owns the capacity/Run decision and OV commit call;
it does not estimate tokens or manage Phase 2 tracking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
import tomllib


class OVCommitClient(Protocol):
    async def commit_session(
        self,
        session_id: str,
        *,
        keep_recent_count: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CommitConfig:
    token_budget: int
    commit_token_threshold_ratio: Decimal | float | int
    completed_run_threshold: int
    keep_recent_count: int

    def __post_init__(self) -> None:
        for name in ("token_budget", "completed_run_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.keep_recent_count, bool)
            or not isinstance(self.keep_recent_count, int)
            or self.keep_recent_count < 0
            or self.keep_recent_count > 10_000
        ):
            raise ValueError("keep_recent_count must be an integer from 0 to 10000")
        ratio_value = self.commit_token_threshold_ratio
        if isinstance(ratio_value, bool) or not isinstance(
            ratio_value, (Decimal, float, int)
        ):
            raise ValueError("commit_token_threshold_ratio must be in (0, 0.7]")
        ratio = Decimal(str(ratio_value))
        if not ratio.is_finite() or not (Decimal(0) < ratio <= Decimal("0.7")):
            raise ValueError("commit_token_threshold_ratio must be in (0, 0.7]")
        object.__setattr__(self, "commit_token_threshold_ratio", ratio)


def load_commit_config(path: str | Path) -> CommitConfig:
    """Load the four first-version commit settings from a TOML [commit] table."""
    with Path(path).open("rb") as config_file:
        document = tomllib.load(config_file)
    raw = document.get("commit")
    if not isinstance(raw, dict):
        raise ValueError("config must contain a [commit] table")
    common = {
        "completed_run_threshold",
        "keep_recent_count",
    }
    snake_keys = {"token_budget", "commit_token_threshold_ratio"}
    legacy_keys = {"tokenBudget", "commitTokenThresholdRatio"}
    if set(raw) == common | snake_keys:
        budget_key, ratio_key = "token_budget", "commit_token_threshold_ratio"
    elif set(raw) == common | legacy_keys:
        budget_key, ratio_key = "tokenBudget", "commitTokenThresholdRatio"
    else:
        raise ValueError(
            "invalid [commit] keys: use either token_budget and "
            "commit_token_threshold_ratio, or the legacy camelCase pair, "
            "along with completed_run_threshold and keep_recent_count"
        )
    ratio = raw[ratio_key]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError(f"{ratio_key} must be numeric")
    return CommitConfig(
        token_budget=raw[budget_key],
        commit_token_threshold_ratio=Decimal(str(ratio)),
        completed_run_threshold=raw["completed_run_threshold"],
        keep_recent_count=raw["keep_recent_count"],
    )


@dataclass(frozen=True, slots=True)
class CommitResult:
    status: str  # skipped | archived | not_archived | unknown | failed
    reason: str | None = None  # pending_capacity | completed_runs
    archived: bool | None = None
    task_id: str | None = None
    archive_uri: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CommitDecision:
    """A threshold decision result."""

    should_commit: bool
    reason: str | None = None
    keep_recent_count: int | None = None


class CommitPolicy:
    def __init__(self, config: CommitConfig, ov_client: OVCommitClient) -> None:
        self.config = config
        self.ov_client = ov_client

    async def commit(
        self, session_id: str, pending_tokens: int, runs: int
    ) -> CommitResult:
        """Skip when thresholds miss; otherwise commit through the configured OV client."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        for name, value in (("pending_tokens", pending_tokens), ("runs", runs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        threshold = self.config.token_budget * self.config.commit_token_threshold_ratio
        pending_hit = Decimal(pending_tokens) >= threshold
        runs_hit = runs >= self.config.completed_run_threshold and pending_tokens > 0
        if not (pending_hit or runs_hit):
            return CommitResult(status="skipped")
        reason = "pending_capacity" if pending_hit else "completed_runs"
        response = await self.ov_client.commit_session(
            session_id,
            keep_recent_count=self.config.keep_recent_count,
        )

        if not isinstance(response, Mapping) or not isinstance(
            response.get("archived"), bool
        ):
            return CommitResult(
                status="failed",
                reason=reason,
                detail="OV commit response must contain a boolean 'archived' field",
            )
        archived = response["archived"]
        return CommitResult(
            status="archived" if archived else "not_archived",
            reason=reason,
            archived=archived,
            task_id=response.get("task_id"),
            archive_uri=response.get("archive_uri"),
        )
