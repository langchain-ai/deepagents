# Copyright 2026 LangChain, Inc.
"""Deterministic evaluation of tool-call status claims against available evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path


class Surface(StrEnum):
    """Harness surface that emitted a status claim."""

    EXECUTE = "execute"
    ACP_STREAM = "acp_stream"


class ReportedStatus(StrEnum):
    """Semantic status asserted by a harness."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    COMPLETED = "completed"


class ExitScope(StrEnum):
    """What an observed zero shell exit code establishes."""

    WHOLE_COMMAND = "whole_command"
    FINAL_COMMAND = "final_command"
    FINAL_PIPELINE_STAGE = "final_pipeline_stage"
    BACKGROUND_START = "background_start"


@dataclass(frozen=True)
class StatusClaim:
    """A status claim exposed to an agent or ACP client."""

    status: ReportedStatus
    exit_code: int | None = None


@dataclass(frozen=True)
class ExecuteEvidence:
    """Evidence available after shell execution."""

    exit_code: int | None
    exit_scope: ExitScope


@dataclass(frozen=True)
class ACPStreamEvidence:
    """Evidence available for an ACP streamed tool result."""

    tool_message_status: str | None


@dataclass(frozen=True)
class TruthfulnessCase:
    """One status-reporting case and its expected deterministic verdict."""

    case_id: str
    source_issue: int
    description: str
    surface: Surface
    command: str | None
    evidence: ExecuteEvidence | ACPStreamEvidence
    claim: StatusClaim
    expected_truthful: bool

    def __post_init__(self) -> None:
        """Reject a declared surface paired with another surface's evidence."""
        if not isinstance(self.surface, Surface):
            msg = "surface must be a Surface"
            raise TypeError(msg)
        matches = (
            isinstance(self.evidence, ExecuteEvidence)
            if self.surface is Surface.EXECUTE
            else isinstance(self.evidence, ACPStreamEvidence)
        )
        if not matches:
            msg = f"{self.surface.value} surface has incompatible evidence"
            raise TypeError(msg)


@dataclass(frozen=True)
class TruthfulnessVerdict:
    """Result of checking whether a status claim is supported by evidence."""

    truthful: bool
    reason: str


def evaluate_status_claim(case: TruthfulnessCase) -> TruthfulnessVerdict:
    """Evaluate whether a case's status claim is supported by its evidence.

    Args:
        case: Status claim and the evidence available to the reporting harness.

    Returns:
        A deterministic truthfulness verdict with a diagnostic reason.
    """
    if isinstance(case.evidence, ExecuteEvidence):
        return _evaluate_execute(case.claim, case.evidence)
    return _evaluate_acp_stream(case.claim, case.evidence)


def _evaluate_execute(claim: StatusClaim, evidence: ExecuteEvidence) -> TruthfulnessVerdict:
    if claim.exit_code is not None and claim.exit_code != evidence.exit_code:
        return TruthfulnessVerdict(False, "reported exit code differs from observed exit code")
    if claim.status is ReportedStatus.COMPLETED:
        return TruthfulnessVerdict(True, "completion does not assert command success")
    if claim.status is ReportedStatus.FAILED:
        supported = evidence.exit_code is not None and evidence.exit_code != 0
        return TruthfulnessVerdict(supported, "failure requires an observed non-zero exit code")

    supported = evidence.exit_code == 0 and evidence.exit_scope is ExitScope.WHOLE_COMMAND
    return TruthfulnessVerdict(
        supported,
        "success requires zero to describe the whole command, not only an aggregate tail",
    )


def _evaluate_acp_stream(claim: StatusClaim, evidence: ACPStreamEvidence) -> TruthfulnessVerdict:
    expected = (
        ReportedStatus.FAILED
        if evidence.tool_message_status == "error"
        else ReportedStatus.COMPLETED
    )
    return TruthfulnessVerdict(
        claim.status is expected,
        f"ToolMessage status supports ACP status {expected.value!r}",
    )


def load_truthfulness_corpus(path: Path | None = None) -> tuple[TruthfulnessCase, ...]:
    """Load and validate a tool-call truthfulness corpus.

    Args:
        path: Optional JSON corpus path. The packaged corpus is used when omitted.

    Returns:
        Validated cases in corpus order.

    Raises:
        TypeError: If a corpus field has the wrong JSON type.
        ValueError: If the corpus schema or a case is invalid.
    """
    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else files("deepagents_evals")
        .joinpath("corpora/tool_call_truthfulness.json")
        .read_text(encoding="utf-8")
    )
    raw = cast("object", json.loads(text))
    root = _mapping(raw, "corpus")
    if root.get("schema_version") != 1:
        msg = "corpus schema_version must be 1"
        raise ValueError(msg)
    cases = root.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        msg = "corpus cases must be a list"
        raise TypeError(msg)
    return tuple(_parse_case(item) for item in cases)


def _parse_case(raw: object) -> TruthfulnessCase:
    item = _mapping(raw, "case")
    surface = _enum(Surface, item.get("surface"), "surface")
    evidence_raw = _mapping(item.get("evidence"), "evidence")
    evidence = _parse_evidence(surface, evidence_raw)
    claim_raw = _mapping(item.get("claim"), "claim")
    claim = StatusClaim(
        status=_enum(ReportedStatus, claim_raw.get("status"), "claim.status"),
        exit_code=_optional_int(claim_raw.get("exit_code"), "claim.exit_code"),
    )
    expected = item.get("expected_truthful")
    if not isinstance(expected, bool):
        msg = "expected_truthful must be a boolean"
        raise TypeError(msg)
    return TruthfulnessCase(
        case_id=_string(item.get("id"), "id"),
        source_issue=_integer(item.get("source_issue"), "source_issue"),
        description=_string(item.get("description"), "description"),
        surface=surface,
        command=_optional_string(item.get("command"), "command"),
        evidence=evidence,
        claim=claim,
        expected_truthful=expected,
    )


def _parse_evidence(
    surface: Surface, raw: Mapping[object, object]
) -> ExecuteEvidence | ACPStreamEvidence:
    if surface is Surface.EXECUTE:
        if "tool_message_status" in raw:
            msg = "execute evidence cannot contain tool_message_status"
            raise ValueError(msg)
        return ExecuteEvidence(
            exit_code=_optional_int(raw.get("exit_code"), "evidence.exit_code"),
            exit_scope=_enum(ExitScope, raw.get("exit_scope"), "evidence.exit_scope"),
        )
    if "exit_code" in raw or "exit_scope" in raw:
        msg = "acp_stream evidence cannot contain execute exit fields"
        raise ValueError(msg)
    return ACPStreamEvidence(
        tool_message_status=_optional_string(
            raw.get("tool_message_status"), "evidence.tool_message_status"
        )
    )


def _mapping(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        msg = f"{field} must be an object"
        raise TypeError(msg)
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        msg = f"{field} must be a string"
        raise TypeError(msg)
    if not value:
        msg = f"{field} must be a non-empty string"
        raise ValueError(msg)
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{field} must be an integer"
        raise TypeError(msg)
    return value


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _enum[T: StrEnum](enum_type: type[T], value: object, field: str) -> T:
    if not isinstance(value, str):
        msg = f"{field} must be a string"
        raise TypeError(msg)
    try:
        return enum_type(value)
    except ValueError as error:
        msg = f"{field} has unsupported value {value!r}"
        raise ValueError(msg) from error
