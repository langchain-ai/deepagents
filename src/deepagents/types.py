from typing import NotRequired, Union, Any
from typing_extensions import TypedDict
from langchain_core.language_models import LanguageModelLike
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import Runnable


class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    # Optional per-subagent model: can be either a model instance OR dict settings
    model: NotRequired[Union[LanguageModelLike, dict[str, Any]]]
    middleware: NotRequired[list[AgentMiddleware]]


class CustomSubAgent(TypedDict):
    name: str
    description: str
    graph: Runnable