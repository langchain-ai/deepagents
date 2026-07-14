"""Tests for shared inline-prompt primitives."""

from __future__ import annotations

import asyncio

from deepagents_code.tui.widgets._inline_prompt import InlinePromptCompletion


class TestInlinePromptCompletion:
    """Resolve-at-most-once semantics, independent of call ordering."""

    async def test_resolve_delivers_to_future_set_first(self) -> None:
        """The common path: the future is wired before the result arrives."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert completion.resolve("done") is True
        assert completion.resolved is True
        assert await future == "done"

    async def test_resolve_before_set_future_still_delivers(self) -> None:
        """A result recorded before the future is wired must not be stranded."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()

        assert completion.resolve("done") is True
        assert completion.resolved is True

        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert await future == "done"

    async def test_second_resolve_is_ignored(self) -> None:
        """Only the first terminal result wins; later ones return `False`."""
        completion: InlinePromptCompletion[str] = InlinePromptCompletion()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        completion.set_future(future)

        assert completion.resolve("first") is True
        assert completion.resolve("second") is False
        assert await future == "first"
