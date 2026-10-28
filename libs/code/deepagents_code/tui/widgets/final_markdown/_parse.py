"""Parse markdown into the block tree Textual's `Markdown` would mount."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from textual.content import Content, Span
from textual.style import Style

if TYPE_CHECKING:
    from markdown_it.token import Token

LISTS = frozenset({"bullet_list", "ordered_list"})

_TABLE_PARTS = frozenset({"thead", "tbody", "tr", "th", "td"})


@dataclass(slots=True)
class Node:
    """A parsed markdown block."""

    kind: str
    inline: Content = field(default_factory=Content)
    children: list[Node] = field(default_factory=list)
    code: str = ""
    lexer: str = ""
    number: str = ""
    nested: bool = False
    headers: list[Content] = field(default_factory=list)
    rows: list[list[Content]] = field(default_factory=list)


def _token_to_content(token: Token) -> Content:
    """Convert an inline token to `Content`.

    Mirrors `MarkdownBlock._token_to_content` from Textual 8.2.8, including
    its quirks (inline HTML is dropped), so text and link spans match the
    streamed `Markdown`.

    Args:
        token: A markdown-it `inline` token.

    Returns:
        The styled inline content.
    """
    text: list[str] = []
    spans: list[Span] = []
    stack: list[tuple[Style | str, int]] = []
    position = 0

    def add(value: str) -> None:
        nonlocal position
        text.append(value)
        position += len(value)

    def close() -> None:
        style, start = stack.pop()
        spans.append(Span(start, position, style))

    for child in token.children or ():
        kind = child.type
        if kind == "text":
            add(re.sub(r"\s+", " ", child.content))
        elif kind == "hardbreak":
            add("\n")
        elif kind == "softbreak":
            add(" ")
        elif kind == "code_inline":
            stack.append((".code_inline", position))
            add(child.content)
            close()
        elif kind in {"em_open", "strong_open", "s_open"}:
            stack.append((f".{kind.removesuffix('_open')}", position))
        elif kind in {"link_open", "image"}:
            href = child.attrs.get("href" if kind == "link_open" else "src", "")
            stack.append((Style.from_meta({"@click": f"link({href!r})"}), position))
            if kind == "image":
                _add_image(child, add)
                close()
        elif kind.endswith("_close"):
            close()
    return Content("".join(text), spans=spans)


def _add_image(token: Token, add: Any) -> None:  # noqa: ANN401  # local closure
    """Add an image's marker, alt text, and caption like `Markdown` does."""
    add("🖼  ")
    if alt := token.attrs.get("alt", ""):
        add(f"({alt})")
    for child in token.children or ():
        add(child.content)


def parse(source: str) -> list[Node]:
    """Parse markdown into the block tree Textual's `Markdown` would mount.

    Returns:
        Top-level blocks. HTML blocks are dropped, as `Markdown` drops them.
    """
    from markdown_it import MarkdownIt

    root = Node("root")
    stack = [root]
    for token in MarkdownIt("gfm-like").parse(source):
        kind = token.type
        if kind in {"fence", "code_block"}:
            fence = Node("fence", code=token.content.rstrip(), lexer=token.info)
            stack[-1].children.append(fence)
        elif kind == "hr":
            stack[-1].children.append(Node("hr"))
        elif kind == "inline":
            _add_inline(stack[-1], _token_to_content(token))
        elif kind.endswith("_open"):
            stack.append(_open_node(token, stack[-1]))
        elif kind.endswith("_close"):
            stack.pop()
    _mark_nested_lists(root.children)
    return root.children


def _open_node(token: Token, parent: Node) -> Node:
    """Create the node for an `*_open` token.

    Table sections, rows, and cells share the enclosing table's header and
    row lists instead of becoming blocks of their own.

    Returns:
        The node to push on the parse stack.
    """
    kind = token.type.removesuffix("_open")
    if kind in _TABLE_PARTS:
        node = Node(kind, headers=parent.headers, rows=parent.rows)
        if kind == "tr":
            node.rows.append([])
        return node
    node = Node(token.tag if kind == "heading" else kind, number=token.info)
    parent.children.append(node)
    return node


def _add_inline(node: Node, content: Content) -> None:
    """Attach inline content to a block or table cell."""
    if node.kind == "th":
        node.headers.append(content)
    elif node.kind == "td":
        node.rows[-1].append(content)
    else:
        node.inline = content


def _mark_nested_lists(nodes: list[Node], *, in_list: bool = False) -> None:
    """Flag lists inside other lists, which Textual's CSS gives no margin."""
    for node in nodes:
        is_list = node.kind in LISTS
        node.nested = is_list and in_list
        _mark_nested_lists(node.children, in_list=in_list or is_list)
