import time

from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_filesystem_middleware_creation_is_fast() -> None:
    """Measure the cost of repeated `FilesystemMiddleware` setup."""
    start = time.time()
    for _ in range(100):
        FilesystemMiddleware()
    end = time.time()
    assert end - start < 0.01


def test_create_deep_agent_with_fake_chat_model_is_fast() -> None:
    """Measure `create_deep_agent()` setup with a fake chat model."""
    start = time.time()
    model = GenericFakeChatModel(messages=iter([]))
    for _ in range(10):
        agent = create_deep_agent(model=model)
    end = time.time()
    assert agent is not None
    delta = end - start
    assert delta < 0.3
