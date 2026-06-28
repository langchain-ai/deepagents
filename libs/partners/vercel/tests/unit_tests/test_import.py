from __future__ import annotations

from deepagents.backends.sandbox import BaseSandbox

import langchain_vercel_sandbox
from langchain_vercel_sandbox import VercelSandbox


def test_import_vercel_sandbox() -> None:
    assert langchain_vercel_sandbox is not None
    assert issubclass(VercelSandbox, BaseSandbox)
