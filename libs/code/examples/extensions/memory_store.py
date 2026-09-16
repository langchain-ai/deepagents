"""Reference extension: shared data through a virtual storage route."""

from deepagents.backends import StoreBackend

from deepagents_code.extensions import ExtensionAPI


async def extension(d: ExtensionAPI) -> None:  # noqa: RUF029  # Public extension factories are async by contract.
    """Make LangGraph store data available under `/memories/`.

    Args:
        d: The dcode extension API.
    """
    d.register_backend_route(
        "/memories/",
        StoreBackend(namespace=lambda _runtime: ("filesystem",)),
    )
