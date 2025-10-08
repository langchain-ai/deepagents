from langchain.agents.middleware import AgentState
from typing import NotRequired, Annotated
from typing import Literal
from typing_extensions import TypedDict


class Todo(TypedDict):
    """Todo to track."""

    content: str
    status: Literal["pending", "in_progress", "completed"]



def file_reducer(l, r):
    if not isinstance(r, dict):
        return l
    if not isinstance(l, dict):
        l = {}
    return {**l, **r}


class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


class PlanningState(AgentState):
    todos: NotRequired[list[Todo]]


class FilesystemState(AgentState):
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
