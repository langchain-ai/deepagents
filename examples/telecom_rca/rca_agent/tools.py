"""RCA Agent Tools.

This module provides search, reflection, and event classification utilities
for the telecom RCA agent.
"""

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from typing import TypedDict
from typing_extensions import Annotated, Literal

tavily_client = TavilyClient()


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )

    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = fetch_webpage_content(url)
        result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
        result_texts.append(result_text)

    response = f"""🔍 Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""

    return response


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on RCA progress and decision-making.

    Use this tool after each search or analysis step to reason about findings
    and plan next investigation steps systematically.

    When to use:
    - After classifying an event: Is the classification correct? What domains are affected?
    - After receiving search results: What root causes are supported by evidence?
    - Before delegating to sub-agents: What specific aspects need deeper research?
    - Before writing the final report: Do I have enough evidence for confident conclusions?

    Reflection should address:
    1. Analysis of current findings - What evidence have I gathered so far?
    2. Gap assessment - Which probable causes still lack supporting evidence?
    3. Confidence evaluation - How confident am I in each root cause hypothesis?
    4. Strategic decision - Should I research more or proceed to synthesis?

    Args:
        reflection: Your detailed reflection on investigation progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


class EventClassification(TypedDict):
    """Structured classification of a telecom network event."""

    domain: str
    event_type: str
    severity: str
    affected_kpis: list[str]
    search_keywords: list[str]


@tool(parse_docstring=True)
def classify_event(event_description: str) -> EventClassification:
    """Classify a raw telecom network event into structured categories.

    Performs deterministic keyword-based classification of network events
    against a telecom event taxonomy. No external API call is made.

    Args:
        event_description: Free-text description of the network event or alarm

    Returns:
        Structured classification with domain, event_type, severity, affected_kpis,
        and search_keywords
    """
    text = event_description.lower()

    # --- Domain detection ---
    ran_keywords = [
        "cell", "secteur", "sector", "rsrp", "rsrq", "sinr", "handover",
        "ho", "rlf", "radio", "enb", "gnb", "lte", "5g", "nr", "mimo",
        "prb", "cdr", "call drop", "coverage", "pilot", "antenna", "beam",
        "pci", "earfcn", "arfcn", "rrc", "erab", "drb", "bearer",
        "outage cellulaire", "cell outage",
    ]
    core_keywords = [
        "mme", "amf", "smf", "upf", "pgw", "sgw", "hss", "pcrf", "ims",
        "sip", "diameter", "s11", "s1", "n2", "n3", "n4", "n6",
        "attach", "detach", "pdn", "apn", "eps", "5gc", "core",
        "authentication", "authorisation", "registration",
    ]
    transport_keywords = [
        "link", "lien", "backhaul", "fronthaul", "midhaul", "mpls",
        "ip", "latency", "latence", "jitter", "packet loss", "perte",
        "bandwidth", "bande passante", "vlan", "ospf", "bgp", "isis",
        "fiber", "fibre", "microwave", "faisceau", "transmission",
    ]

    ran_score = sum(1 for kw in ran_keywords if kw in text)
    core_score = sum(1 for kw in core_keywords if kw in text)
    transport_score = sum(1 for kw in transport_keywords if kw in text)

    if ran_score >= core_score and ran_score >= transport_score:
        domain = "RAN"
    elif core_score >= transport_score:
        domain = "Core"
    else:
        domain = "Transport"

    # --- Event type detection ---
    if any(kw in text for kw in ["outage", "down", "panne", "indisponible", "hors service"]):
        event_type = "Cell Outage" if domain == "RAN" else ("Core Node Outage" if domain == "Core" else "Link Failure")
    elif any(kw in text for kw in ["call drop", "cdr", "chute", "drop rate"]):
        event_type = "High Call Drop Rate"
    elif any(kw in text for kw in ["handover", "ho failure", "handover success", "hosr"]):
        event_type = "Handover Failure"
    elif any(kw in text for kw in ["coverage", "couverture", "rsrp", "signal low", "signal faible"]):
        event_type = "Coverage Degradation"
    elif any(kw in text for kw in ["congestion", "prb utilization", "high load", "overload"]):
        event_type = "Congestion / High Load"
    elif any(kw in text for kw in ["latency", "latence", "delay", "délai"]):
        event_type = "High Latency"
    elif any(kw in text for kw in ["packet loss", "perte paquet", "link failure", "lien"]):
        event_type = "Link Failure / Packet Loss"
    elif any(kw in text for kw in ["attach fail", "registration fail", "authentication fail"]):
        event_type = "Attach / Registration Failure"
    elif any(kw in text for kw in ["interference", "interférence", "pci conflict", "pilot pollution"]):
        event_type = "Interference / Pilot Pollution"
    else:
        event_type = "Unknown / Generic Alarm"

    # --- Severity detection ---
    if any(kw in text for kw in ["critical", "critique", "emergency", "urgence", "outage", "panne", "down", "indisponible"]):
        severity = "Critical"
    elif any(kw in text for kw in ["major", "majeur", "high", "élevé", "degradation", "dégradation", "drop"]):
        severity = "Major"
    elif any(kw in text for kw in ["minor", "mineur", "warning", "avertissement", "low"]):
        severity = "Minor"
    else:
        severity = "Warning"

    # --- Affected KPIs per domain/event_type ---
    kpi_map = {
        "RAN": {
            "Cell Outage": ["Cell Availability (%)", "RSRP (dBm)", "Active Users", "Traffic Volume (Gbps)"],
            "High Call Drop Rate": ["CDR (%)", "RSRP (dBm)", "RSRQ (dB)", "SINR (dB)", "RLF Rate (%)"],
            "Handover Failure": ["HOSR (%)", "HO Preparation Success Rate (%)", "HO Execution Success Rate (%)", "RSRP (dBm)"],
            "Coverage Degradation": ["RSRP (dBm)", "RSRQ (dB)", "SINR (dB)", "DL Throughput (Mbps)"],
            "Congestion / High Load": ["PRB Utilization (%)", "DL Throughput (Mbps)", "Active Users", "Scheduler Wait Time (ms)"],
            "Interference / Pilot Pollution": ["SINR (dB)", "RSRQ (dB)", "Interference Level (dBm)", "CDR (%)"],
        },
        "Core": {
            "Core Node Outage": ["Node Availability (%)", "Signaling Success Rate (%)", "Active Sessions", "PDP Context Success Rate (%)"],
            "Attach / Registration Failure": ["Attach Success Rate (%)", "Authentication Success Rate (%)", "Registration Success Rate (%)"],
            "High Latency": ["S11 Latency (ms)", "Diameter RTT (ms)", "N2/N4 Response Time (ms)"],
        },
        "Transport": {
            "Link Failure / Packet Loss": ["Link Availability (%)", "Packet Loss Rate (%)", "BER", "Traffic Volume (Gbps)"],
            "High Latency": ["RTT (ms)", "Jitter (ms)", "Packet Loss Rate (%)"],
        },
    }

    default_kpis = {
        "RAN": ["RSRP (dBm)", "RSRQ (dB)", "SINR (dB)", "CDR (%)", "HOSR (%)"],
        "Core": ["Attach Success Rate (%)", "Signaling Success Rate (%)", "Active Sessions"],
        "Transport": ["Link Availability (%)", "Packet Loss Rate (%)", "RTT (ms)"],
    }

    affected_kpis = (
        kpi_map.get(domain, {}).get(event_type)
        or default_kpis.get(domain, ["Availability (%)", "Throughput (Mbps)"])
    )

    # --- Search keywords ---
    keyword_map = {
        ("RAN", "Cell Outage"): [
            "LTE 5G cell outage root cause", "eNodeB gNodeB cell down troubleshooting",
            "Ericsson Nokia Huawei cell outage alarm",
        ],
        ("RAN", "High Call Drop Rate"): [
            "LTE high call drop rate root cause", "CDR degradation RLF troubleshooting",
            "RSRP SINR drop call failure 5G NR",
        ],
        ("RAN", "Handover Failure"): [
            "LTE 5G handover failure root cause", "HOSR degradation X2 S1 handover",
            "intra-frequency inter-frequency handover failure",
        ],
        ("RAN", "Coverage Degradation"): [
            "RSRP degradation root cause", "LTE coverage hole troubleshooting",
            "antenna tilt power degradation coverage",
        ],
        ("RAN", "Congestion / High Load"): [
            "LTE PRB utilization high root cause", "5G NR cell congestion throughput drop",
            "scheduler overload radio resource management",
        ],
        ("RAN", "Interference / Pilot Pollution"): [
            "LTE pilot pollution interference root cause", "PCI conflict interference SINR degradation",
            "inter-cell interference coordination ICIC",
        ],
        ("Core", "Core Node Outage"): [
            "MME AMF outage root cause telecom", "core network node failure troubleshooting",
            "EPC 5GC signaling failure",
        ],
        ("Core", "Attach / Registration Failure"): [
            "LTE attach failure root cause", "5G registration failure AMF SMF",
            "EPS bearer setup failure troubleshooting",
        ],
        ("Transport", "Link Failure / Packet Loss"): [
            "backhaul link failure root cause", "microwave fiber transmission packet loss",
            "MPLS IP backhaul outage troubleshooting",
        ],
    }

    search_keywords = keyword_map.get(
        (domain, event_type),
        [
            f"{domain} {event_type} root cause analysis",
            f"telecom {event_type.lower()} troubleshooting",
            f"3GPP {domain} alarm {event_type.lower()}",
        ],
    )

    return EventClassification(
        domain=domain,
        event_type=event_type,
        severity=severity,
        affected_kpis=list(affected_kpis),
        search_keywords=search_keywords,
    )
