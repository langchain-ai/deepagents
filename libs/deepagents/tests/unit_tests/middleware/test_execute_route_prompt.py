"""Tests for the shell-vs-virtual-path prompt section (issue #3050).

`execute` runs on the default backend's host shell, so routed virtual paths
(e.g. `/common/`) don't exist there. Instead of rewriting commands — which can't
be done correctly for arbitrary shell — the middleware tells the model how to
translate each route's virtual prefix to its host path so it forms the correct
command itself.

A route gets a host mapping only when its files live on the same filesystem the
default's shell runs in: a `LocalShellBackend` default (local shell) paired with
a `FilesystemBackend` route (local disk). A remote/sandbox default runs its shell
elsewhere, so local filesystem routes are not reachable and must be classified as
shell-inaccessible. These tests cover that matrix.
"""

from pathlib import Path

from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import _route_host_path_prompt

_NO_HOST_HEADING = "Virtual mounts without a host path mapping"


def _store() -> StoreBackend:
    return StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("ns",))


def _local_shell() -> LocalShellBackend:
    """A local-shell default whose shell shares the local filesystem with routes."""
    return LocalShellBackend(virtual_mode=True)


class _RemoteSandbox(SandboxBackendProtocol, StoreBackend):
    """A sandbox-capable default that is NOT a LocalShellBackend (e.g. remote).

    Its shell runs in a separate filesystem, so local filesystem routes are not
    reachable from it.
    """

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    @property
    def id(self) -> str:
        return "remote_sandbox"


def test_returns_empty_for_non_composite_backend() -> None:
    assert _route_host_path_prompt(StateBackend()) == ""


def test_returns_empty_when_no_routes() -> None:
    comp = CompositeBackend(default=_local_shell(), routes={})
    assert _route_host_path_prompt(comp) == ""


def test_maps_virtual_route_to_host_path(tmp_path: Path) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert "## Shell paths vs. virtual paths" in prompt
    # The mount is listed under "Host path mappings" with its resolved host path,
    # plus a nested-path example so the substitution is unambiguous (G3/G4).
    assert f"- `/common/` -> `{route.cwd}/`" in prompt
    assert f"`/common/dir/x.py` -> `{route.cwd}/dir/x.py`" in prompt


def test_routes_without_host_path_marked_inaccessible() -> None:
    comp = CompositeBackend(default=_local_shell(), routes={"/memories/": _store()})

    prompt = _route_host_path_prompt(comp)

    # A store mount has no host path, so it appears under the no-mapping section
    # and is never presented as a host path mapping — even with a local default.
    assert _NO_HOST_HEADING in prompt
    assert "`/memories/`" in prompt
    assert " -> " not in prompt


def test_non_virtual_filesystem_route_maps_to_root(tmp_path: Path) -> None:
    # Non-virtual routes strip the prefix and use the remaining absolute path
    # as-is on the host (root_dir ignored), so the prefix maps to the filesystem
    # root `/`.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert "- `/common/` -> `/`" in prompt
    assert "`/common/dir/x.py` -> `/dir/x.py`" in prompt


def test_non_trailing_route_prefix_renders_with_slash(tmp_path: Path) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(default=_local_shell(), routes={"/data": route})

    prompt = _route_host_path_prompt(comp)

    assert f"- `/data/` -> `{route.cwd}/`" in prompt
    assert "`/data/dir/x.py`" in prompt
    assert "/datadir" not in prompt


def test_non_virtual_route_not_mapped_under_remote_sandbox(tmp_path: Path) -> None:
    # Under a remote sandbox default, even a non-virtual local route is on local
    # disk and unreachable from the sandbox shell -> no mapping.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    comp = CompositeBackend(
        default=_RemoteSandbox(store=InMemoryStore(), namespace=lambda _rt: ("default",)),
        routes={"/common/": route},
    )

    prompt = _route_host_path_prompt(comp)

    assert " -> " not in prompt
    assert _NO_HOST_HEADING in prompt
    assert "`/common/`" in prompt


def test_mix_of_host_and_non_host_routes(tmp_path: Path) -> None:
    fs = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(
        default=_local_shell(),
        routes={"/common/": fs, "/memories/": _store()},
    )

    prompt = _route_host_path_prompt(comp)

    assert f"- `/common/` -> `{fs.cwd}/`" in prompt
    assert _NO_HOST_HEADING in prompt
    assert "`/memories/`" in prompt


def test_remote_sandbox_default_suppresses_host_mappings(tmp_path: Path) -> None:
    # The same virtual-mode FilesystemBackend route that maps under a local-shell
    # default must NOT get a host mapping under a remote/sandbox default: its files
    # are on local disk, unreachable from the sandbox shell.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(
        default=_RemoteSandbox(store=InMemoryStore(), namespace=lambda _rt: ("default",)),
        routes={"/common/": route},
    )

    prompt = _route_host_path_prompt(comp)

    assert " -> " not in prompt  # no host mapping emitted
    assert _NO_HOST_HEADING in prompt
    assert "`/common/`" in prompt
