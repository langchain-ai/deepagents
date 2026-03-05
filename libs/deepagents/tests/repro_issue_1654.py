from deepagents.backends.composite import _route_for_path
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend


class MockRuntime:
    def __init__(self) -> None:
        self.state = {"messages": [], "files": {}}
        self.store = None
        self.tool_call_id = "test"
        self.config = {}


def test_route_prefix_no_trailing_slash_boundary() -> None:
    rt = MockRuntime()
    default = StateBackend(rt)
    store = StoreBackend(rt)

    # Route configured WITHOUT trailing slash
    # In current code, routes are usually provided with trailing slash,
    # but the issue says if it's not, it's a bug.
    sorted_routes = [("/abcd", store)]

    # This path should NOT match the route "/abcd"
    backend, stripped_key, _ = _route_for_path(default=default, sorted_routes=sorted_routes, path="/abcde/file.txt")

    assert backend is default, "Should use default backend for /abcde/file.txt when route is /abcd"
    assert stripped_key == "/abcde/file.txt"


def test_route_prefix_with_trailing_slash_boundary() -> None:
    rt = MockRuntime()
    default = StateBackend(rt)
    store = StoreBackend(rt)

    sorted_routes = [("/abcd/", store)]

    # This already works because startswith("/abcd/") doesn't match "/abcde/file.txt"
    backend, _, _ = _route_for_path(default=default, sorted_routes=sorted_routes, path="/abcde/file.txt")

    assert backend is default
