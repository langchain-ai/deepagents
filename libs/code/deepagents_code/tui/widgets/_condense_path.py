"""Width-aware condensation of paths for single-line display."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from rich.cells import cell_len
from textual.content import Content

from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def condense_path(path: str, width: int, prefix: str = "") -> str:
    """Condense `path` so it renders within `width` terminal cells.

    Directory components are replaced by a single ellipsis, growing outward
    from the middle of the component list, because the tail of a path — the
    directory or file the user is actually looking at — is the part worth
    keeping. The last component is only end-truncated once every directory
    component has already been dropped.

    Args:
        path: Path to display, already home-substituted where wanted.
        width: Cells available for `prefix` and the path together.
        prefix: Text rendered immediately before the path; it is returned as
            part of the result and counted against `width`.

    Returns:
        `prefix` followed by as much of the path as fits in `width`.
    """
    return _condense(path, width, prefix, get_glyphs().ellipsis)


@lru_cache(maxsize=512)
def _condense(path: str, width: int, prefix: str, ellipsis: str) -> str:
    """Condense `path` for one specific ellipsis glyph.

    The glyph is an explicit argument so that the cache is keyed by it and a
    charset switch cannot serve a result built from the other glyph set.

    Returns:
        `prefix` followed by the condensed path.
    """
    budget = width - cell_len(prefix)
    if cell_len(path) <= budget:
        return prefix + path
    components = [part for part in path.split("/") if part]
    if not components:
        return prefix + _truncate(path, budget, ellipsis)
    root = "/" if path.startswith("/") else ""
    for candidate in _candidates(root, components, ellipsis):
        if cell_len(candidate) <= budget:
            return prefix + candidate
    return prefix + _truncate(components[-1], budget, ellipsis)


def _candidates(root: str, components: Sequence[str], ellipsis: str) -> Iterator[str]:
    """Yield progressively shorter renderings of a component list.

    Args:
        root: `'/'` for an absolute path, otherwise the empty string.
        components: Path components, with the displayed one last.
        ellipsis: Glyph standing in for the dropped components.

    Yields:
        The path with a growing run of middle directory components collapsed
            into one ellipsis, and finally the bare last component.
    """
    *directories, last = components
    for dropped in range(1, len(directories) + 1):
        start = (len(directories) - dropped) // 2
        kept = [*directories[:start], ellipsis, *directories[start + dropped :], last]
        yield root + "/".join(kept)
    yield last


def _truncate(text: str, width: int, ellipsis: str) -> str:
    """End-truncate `text` to `width` cells, marking the cut with `ellipsis`.

    Returns:
        The leading cells of `text` followed by `ellipsis`, or as much of
            `ellipsis` alone as fits when there is no room for both.
    """
    if width <= 0:
        return ""
    marker = cell_len(ellipsis)
    if width <= marker:
        return Content(ellipsis).truncate(width).plain
    # `truncate` pads with a space when the cut splits a double-width cell.
    return Content(text).truncate(width - marker).plain.rstrip() + ellipsis
