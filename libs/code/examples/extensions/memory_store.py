"""Reference extension: shared data through a virtual storage route."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends import StoreBackend

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    """Make LangGraph store data available under `/memories/`.

    Args:
        d: The dcode extension API.
    """
    d.register_backend_route(
        "/memories/",
        StoreBackend(namespace=lambda _runtime: ("filesystem",)),
    )
