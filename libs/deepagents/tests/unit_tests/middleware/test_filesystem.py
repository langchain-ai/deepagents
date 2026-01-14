from deepagents.middleware.filesystem import (
    FileData,
    _file_data_reducer,
)


def test_file_data_reducer_handles_list():
    """Verify _file_data_reducer can handle a list of updates."""
    # Simulate initial state where left is a list of dicts
    left = [
        {"/file1.txt": FileData(content=["one"], created_at="", modified_at="")},
        {"/file2.txt": FileData(content=["two"], created_at="", modified_at="")},
    ]
    right = {"/file3.txt": FileData(content=["three"], created_at="", modified_at="")}

    # The reducer should merge the list and the new dict
    result = _file_data_reducer(left, right)

    assert result == {
        "/file1.txt": FileData(content=["one"], created_at="", modified_at=""),
        "/file2.txt": FileData(content=["two"], created_at="", modified_at=""),
        "/file3.txt": FileData(content=["three"], created_at="", modified_at=""),
    }

    # Also test with deletions within the list-based state
    right_with_deletion = {
        "/file1.txt": None,
        "/file4.txt": FileData(content=["four"], created_at="", modified_at=""),
    }
    result_with_deletion = _file_data_reducer(left, right_with_deletion)
    assert result_with_deletion == {
        "/file2.txt": FileData(content=["two"], created_at="", modified_at=""),
        "/file4.txt": FileData(content=["four"], created_at="", modified_at=""),
    }
