"""Deny-by-default network policy with credential brokering for Vercel Sandbox.

Running agent- or user-generated code means treating the sandbox as hostile.
Two ideas make that safe in production:

1. **Deny by default.** Block every private/metadata subnet outright, then
   allow only the exact public hosts the workload needs. A compromised model
   can't reach the cloud metadata endpoint (``169.254.169.254``) or pivot into
   your VPC, and it can't exfiltrate to an arbitrary host.

2. **Broker credentials at the firewall, not in the environment.** The sandbox
   process never holds your API token. Instead, the Vercel network policy
   carries a ``transform`` rule that injects ``Authorization: Bearer <token>``
   on egress to your own API host. Even with full RCE inside the sandbox,
   ``print(os.environ)`` reveals nothing — there is no token to steal.

The builder below is intentionally generic. Adapt the allow-lists to your
workload and pass the result to ``Sandbox.create(network_policy=...)``.
"""

from __future__ import annotations

from vercel.sandbox import (
    NetworkPolicyCustom,
    NetworkPolicyRule,
    NetworkPolicySubnets,
    NetworkTransformer,
)

# Subnets the sandbox may never reach, regardless of the allow-list. Denied at
# connect time, so a DNS-rebind pointing a public hostname at a private IP is
# still blocked. This list is the high-value part of the policy.
DENIED_SUBNETS: tuple[str, ...] = (
    "10.0.0.0/8",  # RFC1918 private
    "172.16.0.0/12",  # RFC1918 private
    "192.168.0.0/16",  # RFC1918 private
    "169.254.0.0/16",  # link-local (cloud metadata endpoint lives here)
    "127.0.0.0/8",  # loopback
    "0.0.0.0/8",  # this network
    "100.64.0.0/10",  # carrier-grade NAT
)

# PyPI, so the sandbox can `pip install` at runtime when you are not using a
# pre-baked snapshot.
PYPI_HOSTS: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
)

# LLM provider hosts a code-writing agent might call from inside the sandbox.
# Trim to the providers you actually use.
LLM_HOSTS: tuple[str, ...] = (
    "api.anthropic.com",
    "api.openai.com",
)


def build_network_policy(
    *,
    extra_allow_hosts: tuple[str, ...] = (),
    brokered_hosts: dict[str, str] | None = None,
) -> NetworkPolicyCustom:
    """Build a deny-by-default policy with optional credential brokering.

    Args:
        extra_allow_hosts: Additional public hosts to plain-allow (no auth
            injected) on top of PyPI and the LLM providers — e.g. a public
            data API the agent reads from.
        brokered_hosts: Mapping of ``host -> bearer token``. Each host is
            allow-listed with a ``transform`` rule that injects
            ``Authorization: Bearer <token>`` on egress, so the sandbox can
            call the host authenticated without ever holding the token. Use
            this for your own API.

    Returns:
        A ``NetworkPolicyCustom`` ready to pass to ``Sandbox.create``.
    """
    brokered_hosts = brokered_hosts or {}

    # Plain-allow (no auth injected).
    allow: dict[str, list[NetworkPolicyRule]] = {}
    for host in (*PYPI_HOSTS, *LLM_HOSTS, *extra_allow_hosts):
        allow[host] = []

    # Brokered: allow-list with a bearer-injecting transform. The token lives in
    # the policy (host-side), never in the sandbox process.
    for host, token in brokered_hosts.items():
        allow[host] = [
            NetworkPolicyRule(
                transform=[
                    NetworkTransformer(
                        headers={"Authorization": f"Bearer {token}"},
                    ),
                ],
            ),
        ]

    return NetworkPolicyCustom(
        allow=allow,
        subnets=NetworkPolicySubnets(deny=list(DENIED_SUBNETS)),
    )
