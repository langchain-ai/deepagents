"""Adapt Jev's native Score API to the router's typed decision interface."""

from functools import partial

import openai
from langchain_core.load.dump import dumpd
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai.decisions import (
    DecisionRequest,
    DecisionResponse,
    Level,
    OpenAIDecisions,
    Score,
)

from deepagents_evals.routing.config import GATEWAY_OPENAI_URL, load_gateway_key

_JEV_URL = "https://gateway.smith.langchain.com/typesafe/v1"


def _payload(model: str, request: DecisionRequest) -> dict[str, object]:
    questions: dict[str, object] = {}
    for name, question in request["questions"].items():
        if not isinstance(question, Score):
            msg = "The routing Jev adapter supports Score questions only."
            raise TypeError(msg)
        questions[name] = {
            "type": "score",
            "instructions": question.instructions,
            "criteria": [
                level.description or level.label if isinstance(level, Level) else level
                for level in question.levels
            ],
        }
    return {"model": model, "state": dumpd(request["input"]), "questions": questions}


def _invoke(model: str, request: DecisionRequest) -> DecisionResponse:
    with openai.OpenAI(api_key=load_gateway_key(), base_url=_JEV_URL) as client:
        data = client.post("/systemone", body=_payload(model, request), cast_to=object)
    return DecisionResponse.model_validate(data)


async def _ainvoke(model: str, request: DecisionRequest) -> DecisionResponse:
    async with openai.AsyncOpenAI(api_key=load_gateway_key(), base_url=_JEV_URL) as client:
        data = await client.post("/systemone", body=_payload(model, request), cast_to=object)
    return DecisionResponse.model_validate(data)


def make_decisions(model: str) -> Runnable[DecisionRequest, DecisionResponse]:
    """Build a gateway client for Jev or an OpenAI decision model.

    Args:
        model: Native provider model ID; Jev IDs start with `jev-`.

    Returns:
        A sync and async runnable returning typed scores and usage.
    """
    if not model.startswith("jev-"):
        return OpenAIDecisions(model=model, api_key=load_gateway_key, base_url=GATEWAY_OPENAI_URL)
    return RunnableLambda(partial(_invoke, model), afunc=partial(_ainvoke, model)).with_config(
        metadata={"ls_provider": "typesafe", "ls_model_name": model, "ls_model_type": "chat"}
    )
