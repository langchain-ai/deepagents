from langgraph.graph import MessagesState
from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired


class Todo(TypedDict):
    """Todo to track."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(l, r):
    if l is None:
        return r
    elif r is None:
        return l
    else:
        # Combine lists and remove duplicates while preserving order
        combined = l.copy()
        for item in r:
            if item not in combined:
                combined.append(item)
        return combined


class DeepAgentState(MessagesState):
    todos: NotRequired[list[Todo]]


class PlanningState(MessagesState):
    todos: NotRequired[list[Todo]]


class FilesystemState(MessagesState):
    files: Annotated[NotRequired[list[str]], file_reducer]
    has_started: NotRequired[bool]