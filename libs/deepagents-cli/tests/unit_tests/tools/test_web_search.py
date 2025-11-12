import pytest
from parallel.types.beta.web_search_result import WebSearchResult

from deepagents_cli import tools


@pytest.fixture
def mock_tavily_client():
    original = tools.tavily_client
    yield
    tools.tavily_client = original


@pytest.fixture
def mock_parallel_client():
    original = tools.parallel_client
    yield
    tools.parallel_client = original


def test_tavily_search_success(mock_tavily_client):
    expected = {
        "results": [
            {"title": "Python", "url": "https://python.org", "content": "Guide", "score": 0.9}
        ],
        "query": "python",
    }

    class MockClient:
        def search(self, *args, **kwargs):
            return expected

    tools.tavily_client = MockClient()
    result = tools.tavily_search("python")

    assert result == expected


def test_tavily_search_no_client(mock_tavily_client):
    tools.tavily_client = None
    result = tools.tavily_search("python")

    assert "error" in result


def test_tavily_search_error(mock_tavily_client):
    class MockClient:
        def search(self, *args, **kwargs):
            raise Exception("API error")

    tools.tavily_client = MockClient()
    result = tools.tavily_search("python")

    assert "error" in result


def test_parallel_search_success(mock_parallel_client):
    class MockResult:
        results = [WebSearchResult(url="https://python.org", title="Python", excerpts=["Guide"])]
        search_id = "123"

    class MockBeta:
        def search(self, *args, **kwargs):
            return MockResult()

    class MockClient:
        beta = MockBeta()

    tools.parallel_client = MockClient()
    result = tools.parallel_search(["python"])

    assert result.search_id == "123"
    assert len(result.results) == 1


def test_parallel_search_with_objective(mock_parallel_client):
    class MockResult:
        results = []
        search_id = "456"

    class MockBeta:
        def search(self, objective, search_queries, **kwargs):
            assert objective == "Learn Python"
            assert search_queries == ["python"]
            return MockResult()

    class MockClient:
        beta = MockBeta()

    tools.parallel_client = MockClient()
    result = tools.parallel_search(["python"], objective="Learn Python")

    assert result.search_id == "456"


def test_parallel_search_no_client(mock_parallel_client):
    tools.parallel_client = None
    result = tools.parallel_search(["python"])

    assert "error" in result


def test_parallel_search_error(mock_parallel_client):
    class MockBeta:
        def search(self, *args, **kwargs):
            raise Exception("API error")

    class MockClient:
        beta = MockBeta()

    tools.parallel_client = MockClient()
    result = tools.parallel_search(["python"])

    assert "error" in result


def test_get_web_search_tool_calls_parallel_when_tavily_not_available(
    mock_tavily_client, mock_parallel_client
):
    class MockResult:
        results = [WebSearchResult(url="https://python.org", title="Python", excerpts=["Guide"])]
        search_id = "789"

    class MockBeta:
        def search(self, objective, search_queries, **kwargs):
            return MockResult()

    class MockClient:
        beta = MockBeta()

    tools.tavily_client = None
    tools.parallel_client = MockClient()
    search_tool = tools.get_web_search_tool()
    result = search_tool(["python"])

    assert result.search_id == "789"
    assert len(result.results) == 1
