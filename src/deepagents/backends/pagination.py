from typing import Generic, TypedDict, TypeVar

T = TypeVar("T")


class PaginationCursor(TypedDict):
    """Pagination cursor for listing sandboxes."""

    next_cursor: str | None
    """Cursor for the next page of results.

    None OR empty string if there are no more results.

    string to be interpreted as an opaque token.
    """
    has_more: bool
    """Whether there are more results to fetch."""


class PageResults(TypedDict, Generic[T]):
    """Page results for listing sandboxes."""

    items: list[T]
    """List of sandbox IDs."""
    cursor: PaginationCursor
    """Pagination cursor."""
