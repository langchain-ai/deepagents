"""Link wallet access through a user-owned MDA connection.

The caller's Link credential is resolved at run time with
``connections.get("link-user", {"type": "user"})``. When the caller has not yet
granted access, that call raises a ``credential_authorization_required``
interrupt carrying the Link consent URL, which Slack and LangSmith Studio render
for the caller. The run resumes with that person's token once they authorize.

Nothing here reads ``LINK_ACCESS_TOKEN``: the credential belongs to the caller,
not the deployment, so it never enters ``.env`` or the deployment secrets.
"""

from __future__ import annotations

import httpx
from langchain.tools import tool
from managed_deepagents import connections

#: Link REST API. The Link CLI wraps these same endpoints.
LINK_API = "https://api.link.com"

#: Connection slug created with `mda connections create link-user --oauth ...`.
LINK_CONNECTION = "link-user"


async def _link(method: str, path: str, **kwargs: object) -> dict:
    """Call the Link API as the current caller."""
    access_token = await connections.get(LINK_CONNECTION, {"type": "user"})
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.request(
            method,
            f"{LINK_API}{path}",
            headers={"Authorization": f"Bearer {access_token}"},
            **kwargs,
        )
        response.raise_for_status()
        return response.json()


@tool
async def link_user_info() -> str:
    """Check which Link account the caller has connected.

    Use this to confirm the caller has authorized their Link wallet before
    attempting any purchase. If they have not, this pauses the run and asks
    them to connect.
    """
    info = await _link("GET", "/userinfo")
    return str(info)
