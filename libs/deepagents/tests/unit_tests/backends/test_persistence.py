"""Tests for `persists_to_filesystem`."""

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
    persists_to_filesystem,
)


def test_filesystem_backend_persists():
    assert persists_to_filesystem(FilesystemBackend()) is True


def test_local_shell_backend_persists():
    assert persists_to_filesystem(LocalShellBackend()) is True


def test_state_backend_does_not_persist():
    assert persists_to_filesystem(StateBackend()) is False


def test_store_backend_does_not_persist():
    assert persists_to_filesystem(StoreBackend()) is False


def test_composite_all_persistent():
    backend = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/cache/": LocalShellBackend()},
    )
    assert persists_to_filesystem(backend) is True


def test_composite_default_ephemeral():
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/cache/": FilesystemBackend()},
    )
    assert persists_to_filesystem(backend) is False


def test_composite_route_ephemeral():
    backend = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/memories/": StateBackend()},
    )
    assert persists_to_filesystem(backend) is False


def test_composite_all_ephemeral():
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend()},
    )
    assert persists_to_filesystem(backend) is False


def test_composite_nested_persistent():
    inner = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/cache/": LocalShellBackend()},
    )
    backend = CompositeBackend(
        default=inner,
        routes={"/data/": FilesystemBackend()},
    )
    assert persists_to_filesystem(backend) is True


def test_composite_nested_ephemeral():
    inner = CompositeBackend(
        default=FilesystemBackend(),
        routes={"/memories/": StateBackend()},
    )
    backend = CompositeBackend(
        default=inner,
        routes={"/data/": FilesystemBackend()},
    )
    assert persists_to_filesystem(backend) is False


def test_factory_does_not_persist():
    assert persists_to_filesystem(lambda _rt: FilesystemBackend()) is False
