"""Verify Jev wire translation and the router's typed response contract."""

import json
from functools import partial

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai.decisions import DecisionRequest, Level, Score

from deepagents_evals.routing.decisions import make_decisions


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [200, 403])
async def test_jev_score_round_trip(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, status: int
) -> None:
    monkeypatch.setenv("LC_GATEWAY_KEY", "lsv2_pt_example")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.smith.langchain.com/typesafe/v1/systemone"
        body = json.loads(request.content)
        assert body["model"] == "jev-1.14.0"
        assert "Check my balance" in json.dumps(body["state"])
        assert body["questions"]["min_tier"] == {
            "type": "score",
            "instructions": "How complex is this request?",
            "criteria": ["Simple lookup", "Complex planning"],
        }
        if status == 403:
            return httpx.Response(403, json={"error": {"message": "Access denied"}})
        return httpx.Response(
            200,
            json={
                "model": "jev-1.14.0",
                "answers": {
                    "min_tier": {
                        "type": "score",
                        "score": 0.25,
                        "confidence": 0.5,
                        "legend": {"0": "Simple lookup", "1": "Complex planning"},
                        "probabilities": {"0": 0.75, "1": 0.25},
                    }
                },
                "usage": {"input_tokens": 300, "output_tokens": 19},
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        openai, "OpenAI", partial(openai.OpenAI, http_client=httpx.Client(transport=transport))
    )
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        partial(openai.AsyncOpenAI, http_client=httpx.AsyncClient(transport=transport)),
    )
    decisions = make_decisions("jev-1.14.0")
    request: DecisionRequest = {
        "input": [HumanMessage(content="Check my balance")],
        "questions": {
            "min_tier": Score(
                instructions="How complex is this request?",
                levels=[Level(label="cheap", description="Simple lookup"), "Complex planning"],
            )
        },
    }
    if status == 403:
        with pytest.raises(openai.PermissionDeniedError):
            await decisions.ainvoke(request) if asynchronous else decisions.invoke(request)
        return
    result = await decisions.ainvoke(request) if asynchronous else decisions.invoke(request)

    assert result.scores["min_tier"].probabilities == {0: 0.75, 1: 0.25}
    assert result.usage.input_tokens == 300
