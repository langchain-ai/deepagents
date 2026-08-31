# Copyright 2026 LangChain, Inc.
from __future__ import annotations

import json

import pytest

from deepagents_evals.tool_call_truthfulness import (
    ACPStreamEvidence,
    ExecuteEvidence,
    ExitScope,
    ReportedStatus,
    StatusClaim,
    Surface,
    TruthfulnessCase,
    evaluate_status_claim,
    load_truthfulness_corpus,
)


def test_packaged_corpus_matches_all_expected_verdicts() -> None:
    cases = load_truthfulness_corpus()

    assert cases
    assert {case.source_issue for case in cases} == {5938, 5955}
    for case in cases:
        verdict = evaluate_status_claim(case)
        assert verdict.truthful is case.expected_truthful, f"{case.case_id}: {verdict.reason}"


def test_corpus_covers_required_status_semantics() -> None:
    cases = load_truthfulness_corpus()

    execute_scopes = {
        case.evidence.exit_scope for case in cases if isinstance(case.evidence, ExecuteEvidence)
    }
    assert execute_scopes == set(ExitScope)
    assert any(isinstance(case.evidence, ACPStreamEvidence) for case in cases)
    assert any("2>&1" in (case.command or "") for case in cases)
    assert any("&>" in (case.command or "") for case in cases)
    assert any(">&2" in (case.command or "") for case in cases)


def test_neutral_completion_preserves_aggregate_exit_without_claiming_success() -> None:
    evidence = ExecuteEvidence(exit_code=0, exit_scope=ExitScope.FINAL_PIPELINE_STAGE)
    completed = TruthfulnessCase(
        case_id="pipeline-completed",
        source_issue=5955,
        description="pipeline aggregate",
        command="check | tail",
        surface=Surface.EXECUTE,
        evidence=evidence,
        claim=StatusClaim(ReportedStatus.COMPLETED, exit_code=0),
        expected_truthful=True,
    )
    succeeded = TruthfulnessCase(
        case_id="pipeline-succeeded",
        source_issue=5955,
        description="pipeline aggregate",
        command="check | tail",
        surface=Surface.EXECUTE,
        evidence=evidence,
        claim=StatusClaim(ReportedStatus.SUCCEEDED, exit_code=0),
        expected_truthful=False,
    )

    assert evaluate_status_claim(completed).truthful
    assert not evaluate_status_claim(succeeded).truthful


def test_acp_stream_uses_tool_message_status_as_evidence() -> None:
    case = TruthfulnessCase(
        case_id="acp-error",
        source_issue=5938,
        description="streamed failure",
        command=None,
        surface=Surface.ACP_STREAM,
        evidence=ACPStreamEvidence(tool_message_status="error"),
        claim=StatusClaim(ReportedStatus.COMPLETED),
        expected_truthful=False,
    )

    verdict = evaluate_status_claim(case)

    assert not verdict.truthful
    assert "failed" in verdict.reason


def test_case_rejects_evidence_from_another_surface() -> None:
    with pytest.raises(TypeError, match="incompatible evidence"):
        TruthfulnessCase(
            case_id="mismatched",
            source_issue=5938,
            description="declared ACP with execute evidence",
            surface=Surface.ACP_STREAM,
            command=None,
            evidence=ExecuteEvidence(exit_code=0, exit_scope=ExitScope.WHOLE_COMMAND),
            claim=StatusClaim(ReportedStatus.COMPLETED),
            expected_truthful=False,
        )


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        ({"schema_version": 2, "cases": []}, ValueError, "schema_version"),
        (
            {"schema_version": 1, "cases": "not-a-list"},
            TypeError,
            "cases must be a list",
        ),
        (
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "bad",
                        "source_issue": 5955,
                        "description": "bad scope",
                        "surface": "execute",
                        "command": "check | tail",
                        "evidence": {"exit_code": 0, "exit_scope": "guessed"},
                        "claim": {"status": "succeeded"},
                        "expected_truthful": False,
                    }
                ],
            },
            ValueError,
            "evidence.exit_scope",
        ),
        (
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "mismatched",
                        "source_issue": 5938,
                        "description": "ACP surface with execute evidence",
                        "surface": "acp_stream",
                        "evidence": {"exit_code": 0, "exit_scope": "whole_command"},
                        "claim": {"status": "completed"},
                        "expected_truthful": False,
                    }
                ],
            },
            ValueError,
            "cannot contain execute exit fields",
        ),
    ],
)
def test_loader_rejects_invalid_corpus(
    tmp_path, payload: object, error_type: type[Exception], match: str
) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error_type, match=match):
        load_truthfulness_corpus(corpus)
