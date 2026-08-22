import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"
