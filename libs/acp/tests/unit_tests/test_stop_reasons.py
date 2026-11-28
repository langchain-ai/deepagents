"""Provider stop reasons map to ACP stop reasons."""

import pytest
from langchain_core.messages import AIMessage

from deepagents_acp.server import _stop_reason


@pytest.mark.parametrize(
    ("metadata", "kwargs", "expected"),
    [
        ({"stop_reason": "end_turn"}, {}, None),
        ({"stop_reason": "max_tokens"}, {}, "max_tokens"),
        ({"stop_reason": "refusal"}, {}, "refusal"),
        ({"finish_reason": "length"}, {}, "max_tokens"),
        ({"finish_reason": "content_filter"}, {}, "refusal"),
        ({"incomplete_details": {"reason": "max_output_tokens"}}, {}, "max_tokens"),
        ({"finish_reason": "SAFETY"}, {}, "refusal"),
        ({"finish_reason": "STOP"}, {}, None),
        ({}, {"refusal": "I can't help with that."}, "refusal"),
    ],
)
def test_stop_reason(metadata: dict, kwargs: dict, expected: str | None) -> None:
    message = AIMessage(content="", response_metadata=metadata, additional_kwargs=kwargs)

    assert _stop_reason(message) == expected


def test_responses_refusal_block_is_a_refusal() -> None:
    message = AIMessage(content=[{"type": "refusal", "refusal": "I can't help with that."}])

    assert _stop_reason(message) == "refusal"
