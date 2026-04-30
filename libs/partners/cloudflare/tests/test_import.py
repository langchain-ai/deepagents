from __future__ import annotations

import langchain_cloudflare


def test_import_cloudflare() -> None:
    assert langchain_cloudflare is not None
