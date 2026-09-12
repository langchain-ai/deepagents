"""Sandbox for the agentic shopper.

Gives the agent an isolated shell (the `execute` tool) so it can run the Stripe
Link CLI installed by `setup.sh`. One sandbox per durable thread.

Link credentials via the auth proxy
-----------------------------------
The real Link token never enters the box. A sandbox auth-proxy rule matches
outbound requests to `api.link.com` (the CLI's `DEFAULT_API_BASE_URL`) and
injects `Authorization: Bearer <token>` on the wire. The header is declared
`opaque`, so it is encrypted at rest and never returned by the LangSmith API.

`link-cli` refuses to run without a credential and sets its own `Authorization`
header from whatever it finds, so the rule also sets a *placeholder*
`LINK_ACCESS_TOKEN` — enough for the CLI to start, useless if it leaks. The
proxy replaces the header it sends. Rule `env_vars` are plaintext and readable
through the API, so nothing secret goes in them.

`LINK_NO_REFRESH=1` because the refresh grant is deliberately withheld: a
refresh is a POST to `login.link.com` with the token in the body, which a
header-injecting proxy cannot supply. Link access tokens last ~1 hour; when one
expires, mint a new one and redeploy. Moving refresh into a host-side authored
tool that resolves an MDA connection is the tracked follow-up (see README).
"""

from __future__ import annotations

import os

from managed_deepagents import define_sandbox

#: Host the Link CLI talks to (`DEFAULT_API_BASE_URL` in its bundle).
_LINK_API_HOST = "api.link.com"

#: Non-secret stand-in so `link-cli` starts; the proxy supplies the real token.
_TOKEN_PLACEHOLDER = "proxy-injected"


def _proxy_config() -> dict[str, object] | None:
    """Auth-proxy rule that injects the Link bearer token, or None if unset."""
    token = os.environ.get("LINK_ACCESS_TOKEN")
    if not token:
        return None
    return {
        "rules": [
            {
                "name": "link-api",
                "match_hosts": [_LINK_API_HOST],
                "headers": [
                    {
                        "name": "Authorization",
                        "type": "opaque",
                        "value": f"Bearer {token}",
                    }
                ],
                "env_vars": {
                    "LINK_ACCESS_TOKEN": _TOKEN_PLACEHOLDER,
                    "LINK_NO_REFRESH": "1",
                },
            }
        ]
    }


sandbox = define_sandbox(
    # Payment approval waits on a human tapping the Link app, so allow long
    # commands and don't reclaim the box between steps too eagerly.
    idle_ttl_seconds=1800,
    default_timeout=900,
    proxy_config=_proxy_config(),
)
