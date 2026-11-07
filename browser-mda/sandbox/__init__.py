"""Managed LangSmith sandbox: the box where Chromium and the `browse` CLI live.

`setup.sh` runs once at deploy time and is baked into a snapshot that every
thread clones, so browsing starts fast and installs nothing at run time. Each
durable thread gets its own sandbox, and therefore its own browser profile,
cookies, and `/workspace`.
"""

from managed_deepagents import define_sandbox

# --- Outbound network policy -------------------------------------------------
# The sandbox reaches the open internet by default. That is what a general
# browsing agent needs, but it is also the agent's whole SSRF and exfiltration
# surface: a prompt-injected page can try to steer the browser somewhere else.
# Instructions discourage that; the proxy is the only hard control.
#
# If this agent only ever needs a known set of sites, switch to an allow list --
# the platform then refuses every other destination:
#
#     "access_control": {"allow_list": ["docs.langchain.com", "api.github.com"]}
#
# Do that for anything running unattended, on a schedule, or on input you do not
# control. Until then, deny the destinations that are never a legitimate browsing
# target: the cloud metadata endpoints and loopback. These match by host, so they
# close the obvious doors rather than every private address -- an allow list is
# what actually closes the rest.
sandbox = define_sandbox(
    # Browsing has long think-pauses between commands; don't reclaim mid-task.
    idle_ttl_seconds=900,
    # Per-command budget. Page loads are slow, but a hung `browse` call should
    # fail fast enough for the agent to retry rather than burn the whole run.
    default_timeout=180,
    proxy_config={
        "access_control": {
            "deny_list": [
                "169.254.169.254",
                "metadata.google.internal",
                "metadata",
                "localhost",
                "127.0.0.1",
                "[::1]",
            ],
        },
    },
)
