from unittest.mock import Mock

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from deepagents_code.fork_prompt import ForkPromptMiddleware


def _request(system_prompt: str, state: dict) -> ModelRequest:
    return ModelRequest(
        model=Mock(),
        messages=[],
        system_message=SystemMessage(content=system_prompt),
        state=state,
    )


def test_fork_prompt_uses_complete_parent_prefix() -> None:
    middleware = ForkPromptMiddleware()
    parent_prompt = "parent\n\n## Shell paths vs. virtual paths\nUse the shell."
    parent = middleware.wrap_model_call(
        _request(parent_prompt, {}), lambda _: ModelResponse(result=[])
    )
    parent_state = parent.command.update

    seen: list[ModelRequest] = []
    role_prompt = "You are the general-purpose subagent."
    fork = middleware.wrap_model_call(
        _request(
            role_prompt,
            {
                "_deepagents_forked_context": True,
                **parent_state,
            },
        ),
        lambda request: seen.append(request) or ModelResponse(result=[]),
    )

    assert seen[0].system_prompt == f"{parent_prompt}\n\n{role_prompt}"
    assert seen[0].system_prompt.startswith(parent_prompt)
    assert seen[0].system_prompt.index(role_prompt) > len(parent_prompt)
    assert fork.result == []
