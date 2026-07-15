"""Private state used to gate browser access."""

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from pydantic import StrictBool


class BrowserState(AgentState):
    """Agent state containing the private browser activation flag.

    Browser tools are available only when `_browser_enabled` is exactly `True`.
    The strict boolean type prevents truthy strings and integers from enabling access.
    """

    _browser_enabled: NotRequired[Annotated[StrictBool, PrivateStateAttr]]
