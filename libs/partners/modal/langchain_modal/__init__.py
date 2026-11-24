"""Modal integration for LangChain and Deep Agents."""

from langchain_modal._version import __version__
from langchain_modal.chat_models import ChatModal
from langchain_modal.sandbox import ModalSandbox

__all__ = [
    "ChatModal",
    "ModalSandbox",
    "__version__",
]
