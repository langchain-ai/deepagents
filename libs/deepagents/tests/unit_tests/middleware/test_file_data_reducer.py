"""Unit tests for _file_data_reducer list-type handling.

Regression tests for https://github.com/langchain-ai/deepagents/issues/731
"""

from deepagents.middleware.filesystem import FileData, _file_data_reducer


def _make_file_data(content: str) -> FileData:
    return FileData(
        content=content,
        encoding="utf-8",
        created_at="2026-01-01T00:00:00Z",
        modified_at="2026-01-01T00:00:00Z",
    )


def test_reducer_handles_left_as_list_of_dicts() -> None:
    """When left is a list of dicts (from parallel channel updates), merge them."""
    left = [
        {"/a.txt": _make_file_data("a")},
        {"/b.txt": _make_file_data("b")},
    ]
    right = {"/c.txt": _make_file_data("c")}
    result = _file_data_reducer(left, right)  # type: ignore[arg-type]
    assert set(result.keys()) == {"/a.txt", "/b.txt", "/c.txt"}


def test_reducer_handles_right_as_list_of_dicts() -> None:
    """When right is a list of dicts, merge them before applying updates."""
    left = {"/a.txt": _make_file_data("a")}
    right = [
        {"/b.txt": _make_file_data("b")},
        {"/c.txt": _make_file_data("c")},
    ]
    result = _file_data_reducer(left, right)  # type: ignore[arg-type]
    assert set(result.keys()) == {"/a.txt", "/b.txt", "/c.txt"}


def test_reducer_handles_empty_list_as_none() -> None:
    """An empty list for left should behave like None (initialization)."""
    left: list = []  # type: ignore[type-arg]
    right = {"/a.txt": _make_file_data("a")}
    result = _file_data_reducer(left, right)  # type: ignore[arg-type]
    assert result == {"/a.txt": _make_file_data("a")}


def test_reducer_handles_both_as_lists() -> None:
    """Both left and right as lists should work."""
    left = [{"/a.txt": _make_file_data("a")}]
    right = [{"/b.txt": _make_file_data("b")}]
    result = _file_data_reducer(left, right)  # type: ignore[arg-type]
    assert set(result.keys()) == {"/a.txt", "/b.txt"}


def test_reducer_list_last_write_wins() -> None:
    """When a list contains duplicate keys, last write wins."""
    left = [
        {"/a.txt": _make_file_data("old")},
        {"/a.txt": _make_file_data("new")},
    ]
    right = {"/b.txt": _make_file_data("b")}
    result = _file_data_reducer(left, right)  # type: ignore[arg-type]
    assert result["/a.txt"]["content"] == "new"


def test_reducer_normal_dict_inputs_unchanged() -> None:
    """Normal dict inputs still work as before."""
    left = {"/a.txt": _make_file_data("a")}
    right = {"/a.txt": _make_file_data("updated"), "/b.txt": _make_file_data("b")}
    result = _file_data_reducer(left, right)
    assert result["/a.txt"]["content"] == "updated"
    assert result["/b.txt"]["content"] == "b"


def test_reducer_deletion_still_works() -> None:
    """None values in right still trigger deletion."""
    left = {"/a.txt": _make_file_data("a"), "/b.txt": _make_file_data("b")}
    right = {"/a.txt": None}
    result = _file_data_reducer(left, right)
    assert "/a.txt" not in result
    assert "/b.txt" in result
