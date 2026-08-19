from __future__ import annotations

import langchain_docker
from langchain_docker import DockerError, DockerSandbox


def test_import_docker() -> None:
    assert langchain_docker is not None
    assert DockerSandbox is langchain_docker.DockerSandbox
    assert issubclass(DockerError, RuntimeError)
