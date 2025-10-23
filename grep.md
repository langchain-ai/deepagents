def _glob_search_files(
    files: dict[str, FileData],
    pattern: str,
    path: str = "/",
) -> str:
    r"""Search files dict for paths matching glob pattern.
    Args:
        files: Dictionary of file paths to FileData.
        pattern: Glob pattern (e.g., "*.py", "**/*.ts").
        path: Base path to search from.
    Returns:
        Newline-separated file paths, sorted by modification time (most recent first).
        Returns "No files found" if no matches.
    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py" (sorted by modified_at)
        ```
    """
    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No files found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    matches = []
    for file_path, file_data in filtered.items():
        relative = file_path[len(normalized_path) :].lstrip("/")
        if not relative:
            # If relative is empty, file_path == normalized_path, check if pattern matches basename
            relative = file_path.split("/")[-1]

        if wcglob.globmatch(relative, pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _grep_search_files(
    files: dict[str, FileData],
    pattern: str,
    path: str = "/",
    include: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """Search file contents for regex pattern.
    Args:
        files: Dictionary of file paths to FileData.
        pattern: Regex pattern to search for.
        path: Base path to search from.
        include: Optional glob pattern to filter files (e.g., "*.py").
        output_mode: Output format - "files_with_matches", "content", or "count".
    Returns:
        Formatted search results. Returns "No matches found" if no results.
    Example:
        ```python
        files = {"/file.py": FileData(content=["import os", "print('hi')"], ...)}
        _grep_search_files(files, "import", "/")
        # Returns: "/file.py" (with output_mode="files_with_matches")
        ```
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No matches found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if include:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, include, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)