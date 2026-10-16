"""The `/btw` side-exchange shape shared by the client and server.

Kept free of storage and model imports so the client can validate server
responses without loading the server's persistence layer.
"""

from __future__ import annotations

type Exchange = tuple[str, str]
"""A completed side question and the answer returned for it."""


def parse_exchanges(raw: object) -> list[Exchange] | None:
    """Validate decoded side history.

    Args:
        raw: Decoded JSON that should hold question/answer string pairs.

    Returns:
        Ordered exchanges, or `None` if `raw` has any other shape.
    """
    if not isinstance(raw, list):
        return None
    exchanges: list[Exchange] = []
    for pair in raw:
        match pair:
            case [str(question), str(answer)]:
                exchanges.append((question, answer))
            case _:
                return None
    return exchanges
