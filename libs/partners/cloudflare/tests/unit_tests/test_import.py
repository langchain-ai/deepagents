from __future__ import annotations

import langchain_cloudflare
from langchain_cloudflare.sandbox import CloudflareSandbox


def test_import_cloudflare() -> None:
    assert langchain_cloudflare is not None


def test_cloudflare_sandbox_class_exists() -> None:
    assert CloudflareSandbox is not None
