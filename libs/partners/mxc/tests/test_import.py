from __future__ import annotations

import langchain_mxc


def test_import_mxc() -> None:
    assert langchain_mxc is not None
    assert langchain_mxc.MXCSandbox is not None
    assert langchain_mxc.__version__
