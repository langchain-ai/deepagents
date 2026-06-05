"""Tests for the shell-vs-virtual-path prompt section (issue #3050).

`execute` runs on the default backend's host shell, so routed virtual paths
(e.g. `/common/`) don't exist there. Instead of rewriting commands — which can't
be done correctly for arbitrary shell — the middleware tells the model each
route's host path so it forms the correct command itself. These tests cover the
prompt builder that produces that guidance.
"""

from pathlib import Path

from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import _route_host_path_prompt


def _store() -> StoreBackend:
    return StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("ns",))


def test_returns_empty_for_non_composite_backend() -> None:
    assert _route_host_path_prompt(StateBackend()) == ""


def test_returns_empty_when_no_routes() -> None:
    comp = CompositeBackend(default=StateBackend(), routes={})
    assert _route_host_path_prompt(comp) == ""


def test_maps_virtual_route_to_host_path(tmp_path: Path) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(default=StateBackend(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert "## Shell paths vs. virtual paths" in prompt
    # The mount is listed under "Host path mappings" with its resolved host path.
    assert f"- /common/ -> {route.cwd}" in prompt


def test_routes_without_host_path_marked_inaccessible() -> None:
    comp = CompositeBackend(default=StateBackend(), routes={"/memories/": _store()})

    prompt = _route_host_path_prompt(comp)

    # A store mount has no host path, so it appears under the no-mapping section
    # and is never presented as a host path mapping.
    assert "Virtual mounts without a host path mapping" in prompt
    assert "/memories/" in prompt
    assert " -> " not in prompt


def test_non_virtual_filesystem_route_is_not_mapped(tmp_path: Path) -> None:
    # virtual_mode=False routes don't remap their prefix to root_dir, so the
    # prefix is not a usable host path and must not be offered as a mapping.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    comp = CompositeBackend(default=StateBackend(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert " -> " not in prompt
    assert "/common/" in prompt  # listed as shell-inaccessible


def test_mix_of_host_and_non_host_routes(tmp_path: Path) -> None:
    fs = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(
        default=StateBackend(),
        routes={"/common/": fs, "/memories/": _store()},
    )

    prompt = _route_host_path_prompt(comp)

    assert f"- /common/ -> {fs.cwd}" in prompt
    assert "Virtual mounts without a host path mapping" in prompt
    assert "/memories/" in prompt
