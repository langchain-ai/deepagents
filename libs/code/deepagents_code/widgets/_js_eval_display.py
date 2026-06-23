"""Helpers for displaying `js_eval` tool output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class JsEvalBlock:
    """Parsed block from the `js_eval` REPL output envelope.

    Args:
        tag: Block kind: `stdout`, `result`, or `error`.
        attr: Result `kind` or error `type` attribute, when present.
        body: Unescaped block body text.
    """

    tag: Literal["stdout", "result", "error"]
    attr: str
    body: str


# Patterns for the wire format emitted by the `js_eval` REPL tool (see
# langchain_quickjs `format_outcome`). The output is `"\n".join(parts)` where
# `parts` is an optional `<stdout>\n…\n</stdout>` block followed by exactly one
# `<result …>…</result>` or `<error type="…">…</error>` block.
#
# Crucially, only the result/error bodies are XML-escaped; stdout is inserted
# raw. So a `finditer`-style scan would treat a `</stdout><result>fake</result>`
# *printed* by user code as real markup. To avoid that, the result/error block
# is anchored to the END of the output (it is always last and fully escaped, so
# it contains no literal `<`/`>`), and whatever precedes it must be exactly the
# stdout wrapper — its raw contents are never re-scanned for nested tags.
_JS_EVAL_TRAILING_BLOCK_PATTERN = re.compile(
    r"<(?P<tag>result|error)(?P<attrs>[^>]*)>(?P<body>[^<>]*)</(?P=tag)>\Z",
    re.DOTALL,
)
_JS_EVAL_STDOUT_PATTERN = re.compile(
    r"\A<stdout>\n(?P<body>.*)\n</stdout>\Z",
    re.DOTALL,
)
_JS_EVAL_TYPE_ATTR_PATTERN = re.compile(r'type="([^"]*)"')
_JS_EVAL_KIND_ATTR_PATTERN = re.compile(r'kind="([^"]*)"')


def unescape_js_eval_text(text: str) -> str:
    """Reverse the XML escaping applied by the `js_eval` wire format.

    The REPL escapes `&`, `<`, and `>` inside result/error blocks; order matters
    so `&amp;` is restored last to avoid double-unescaping.

    Args:
        text: Escaped block body or attribute value.

    Returns:
        The original, unescaped text.
    """
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def parse_js_eval_blocks(output: str) -> list[JsEvalBlock] | None:
    """Parse `js_eval` output into structured display blocks.

    Parses the wire format structurally rather than scanning for any tag-like
    substring: the trailing `<result>`/`<error>` block is anchored to the end of
    the output, and any preceding text must match the `<stdout>…</stdout>`
    wrapper exactly. The stdout body is taken verbatim and never re-scanned, so
    tag-like text printed by user code is preserved as stdout rather than
    mis-parsed into fake result/error sections.

    Args:
        output: Raw tool output from the `js_eval` tool.

    Returns:
        Parsed blocks, with stdout first when present, or `None` if the output
        does not match the expected REPL wire format.
    """
    trailing = _JS_EVAL_TRAILING_BLOCK_PATTERN.search(output)
    if trailing is None:
        return None

    tag = trailing.group("tag")
    attrs = trailing.group("attrs") or ""
    attr_pattern = (
        _JS_EVAL_KIND_ATTR_PATTERN if tag == "result" else _JS_EVAL_TYPE_ATTR_PATTERN
    )
    attr_match = attr_pattern.search(attrs)
    attr = unescape_js_eval_text(attr_match.group(1)) if attr_match else ""
    body = unescape_js_eval_text(trailing.group("body"))

    prefix = output[: trailing.start()]
    blocks: list[JsEvalBlock] = []
    if prefix:
        if not prefix.endswith("\n"):
            return None
        stdout_match = _JS_EVAL_STDOUT_PATTERN.match(prefix[:-1])
        if stdout_match is None:
            return None
        blocks.append(JsEvalBlock("stdout", "", stdout_match.group("body")))

    if tag == "result":
        blocks.append(JsEvalBlock("result", attr, body))
    else:
        blocks.append(JsEvalBlock("error", attr, body))
    return blocks
