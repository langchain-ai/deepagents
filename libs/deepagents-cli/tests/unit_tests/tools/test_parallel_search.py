import pytest
from parallel.types.beta.web_search_result import WebSearchResult

from deepagents_cli import tools


@pytest.fixture
def mock_parallel_client():
    original = tools.parallel_client
    yield
    tools.parallel_client = original


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
    result = tools.parallel_search("Learn Python", ["python"])

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
    result = tools.parallel_search("Learn Python", ["python"])

    assert result.search_id == "456"


def test_parallel_search_no_client(mock_parallel_client):
    tools.parallel_client = None
    result = tools.parallel_search("Learn Python", ["python"])

    assert "error" in result


def test_parallel_search_error(mock_parallel_client):
    class MockBeta:
        def search(self, *args, **kwargs):
            raise Exception("API error")

    class MockClient:
        beta = MockBeta()

    tools.parallel_client = MockClient()
    result = tools.parallel_search("Learn Python", ["python"])

    assert "error" in result
