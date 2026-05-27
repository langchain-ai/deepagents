from __future__ import annotations

import langchain_cubesandbox
from langchain_cubesandbox import CubeSandbox


def test_import_cubesandbox() -> None:
    assert langchain_cubesandbox is not None
    assert CubeSandbox is not None
