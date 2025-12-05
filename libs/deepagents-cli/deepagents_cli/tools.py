"""Custom tools for the CLI agent."""

import json
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import requests
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain.tools import tool

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None


# --- AviationBot helpers and tools -----------------------------------------------------------

AVIATIONBOT_BASE_URL = os.getenv("AVIATION_BOT_BASE_URL", "https://beta.aviation.bot/api/v1")
# AVIATIONBOT_BASE_URL = "https://localhost:8000/api/v1"

AVIATIONBOT_DEFAULT_TIMEOUT = int(os.getenv("AVIATION_BOT_TIMEOUT", "60"))
AVIATIONBOT_API_KEY = os.getenv("AVIATION_BOT_API_KEY", "")


def _build_auth_headers() -> dict[str, str]:
    """Return Authorization header using AVIATION_BOT_API_KEY env if present."""
    api_key = os.getenv("AVIATION_BOT_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _aviationbot_request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json: Any = None,
    timeout: int | None = None,
):
    """Internal shared request helper for AviationBot endpoints."""
    url = f"{AVIATIONBOT_BASE_URL}{path}"
    effective_timeout = timeout or AVIATIONBOT_DEFAULT_TIMEOUT
    try:
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers={"Accept": "application/json", **_build_auth_headers()},
            timeout=effective_timeout,
        )
        response.raise_for_status()
        # Prefer JSON, fallback to text
        try:
            return {"success": True, "status_code": response.status_code, "data": response.json()}
        except ValueError:
            return {"success": True, "status_code": response.status_code, "data": response.text}
    except requests.HTTPError as exc:  # includes status >=400
        payload: dict[str, Any] = {"success": False, "status_code": exc.response.status_code}
        try:
            payload["error"] = exc.response.json()
        except Exception:
            payload["error"] = exc.response.text
        return payload
    except requests.RequestException as exc:  # network/timeout etc
        return {"success": False, "status_code": 0, "error": str(exc)}


@tool(
    "easa_document_retrieval",
    description=(
        "Primary tool to answer questions about EASA Easy Access Rules. "
        "Use this AFTER you have identified the relevant Easy Access Rules document(s) and their file_id "
        "values (typically via easa_doc_finder or other mapping tools). "
        "REQUIRED: provide one or more file_ids from those results; do NOT keep calling easa_doc_finder "
        "once you already have suitable file_ids for the current question."
    ),
)
def easa_document_retrieval(
    query: str,
    file_ids: Sequence[str],
) -> Any:
    """Fetch EASA aviation regulations matching a natural-language query.

    Args:
        query: Search phrase or question.
        file_ids: Required list of Easy Access Rules file IDs to restrict results. Use multiple when the query spans several domains.
    Returns:
        Dict with success flag, status_code, and data/error from AviationBot.
    """

    if not file_ids:
        return {
            "success": False,
            "status_code": 0,
            "error": "file_ids is required; provide one or more file identifiers.",
        }

    params: dict[str, Any] = {"query": query}
    params["file_ids"] = list(file_ids)
    return _aviationbot_request(
        "GET",
        "/tool/EASA/document-retrieval",
        params=params,
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


@tool(
    "get_easa_regulatory_metamodel",
    description="Retrieve EASA regulatory metamodel info (AviationBot /tool/EASA/meta-model)",
)
def get_easa_regulatory_metamodel(query: str) -> Any:
    """Get structured information about the EASA regulatory framework."""

    return _aviationbot_request(
        "GET",
        "/tool/EASA/meta-model",
        params={"query": query},
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


@tool(
    "easa_doc_finder",
    description=(
        "FIRST STEP for EASA questions that require Easy Access Rules. "
        "Given a natural-language description of the topic or user background, returns candidate EASA Easy Access Rules "
        "documents including their file.file_id values. Use this to select 1–3 relevant file_ids, then immediately call "
        "easa_document_retrieval with the user's question and those file_ids. Avoid calling this tool more than 1–2 times "
        "for the same user question unless the topic or scope changes significantly."
    ),
)
def easa_doc_finder(query: str) -> Any:

    return _aviationbot_request(
        "GET",
        "/tool/EASA/doc-finder",
        # return_file_ids is True so the agent can see the file_ids per EASA EAR
        params={"query": query, "return_file_ids": True},
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


# --- EASA local JSON helpers --------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_regulations() -> dict:
    """Load EASA regulations JSON from uploaded_files."""
    path = _REPO_ROOT / "uploaded_files" / "EASA" / "regulations" / "regulations.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_regulations_map() -> list:
    """Load EASA regulations map JSON from uploaded_files."""
    path = _REPO_ROOT / "uploaded_files" / "EASA" / "regulations_map" / "regulations_map.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_certification_specifications() -> dict:
    """Load EASA certification specifications JSON from uploaded_files."""
    path = _REPO_ROOT / "uploaded_files" / "EASA" / "certification-specifications" / "certification-specifications.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_domain_mapping() -> dict:
    """Create a case-insensitive domain mapping (lowercase -> proper case)."""
    regulations_data = _load_regulations()
    domains = regulations_data.get("metadata", {}).get("domains", [])
    return {domain.lower(): domain for domain in domains}


def _normalize_domain(domain: str, domain_mapping: dict) -> str:
    """Normalize a domain name to its proper case."""
    return domain_mapping.get(domain.lower(), domain)


def _should_remove_fields(json_str: str) -> bool:
    """Check if JSON size exceeds 10000 characters."""
    return len(json_str) > 10_000


def _remove_verbose_fields(data: dict) -> dict:
    """Remove verbose fields to reduce JSON size."""
    fields_to_remove = {"url", "entity_label", "extended_title", "eurlex"}

    def clean_item(item):
        if isinstance(item, dict):
            return {k: v for k, v in item.items() if k not in fields_to_remove}
        return item

    if "regulations" in data:
        cleaned_regulations = {}
        for category, regulations in data["regulations"].items():
            if isinstance(regulations, list):
                cleaned_regulations[category] = [clean_item(reg) for reg in regulations]
            else:
                cleaned_regulations[category] = regulations
        data["regulations"] = cleaned_regulations

    return data


class FilterRegulationsByDomainInput(BaseModel):
    domain: str = Field(
        description=(
            "The EASA regulation domain to filter by. Supported domains include: Air Traffic Management; Aircraft & products; "
            "Aircrew & Medical; Air Operations; Aerodromes; Drones & Air Mobility; Cybersecurity; Environment; General Aviation; "
            "International cooperation; Research & Innovation; Rotorcraft & VTOL; Safety Management & Promotion; Third Country Operators."
        )
    )


@tool(args_schema=FilterRegulationsByDomainInput)
def filter_easa_regulations_by_domain(domain: str) -> str:
    """
    Filters EASA regulations by a specified domain and returns the filtered JSON.

    The tool:
    1. Loads the complete EASA regulations database
    2. Filters regulations to include only those with the specified domain (case-insensitive)
    3. If the result exceeds 10,000 characters, removes verbose fields (url, entity_label, extended_title, eurlex) to reduce size
    4. Returns the filtered and optionally reduced JSON as a string
    """
    try:
        regulations_data = _load_regulations()
        domain_mapping = _get_domain_mapping()
        normalized_domain = _normalize_domain(domain, domain_mapping)

        filtered_data = {"metadata": regulations_data.get("metadata", {}), "regulations": {}}

        for category, regulations_list in regulations_data.get("regulations", {}).items():
            filtered_regulations = []
            if isinstance(regulations_list, list):
                for regulation in regulations_list:
                    regulation_domains = regulation.get("domains", [])
                    if any(d.lower() == normalized_domain.lower() for d in regulation_domains):
                        filtered_regulations.append(regulation)
            if filtered_regulations:
                filtered_data["regulations"][category] = filtered_regulations

        json_str = json.dumps(filtered_data, indent=2, ensure_ascii=False)
        if _should_remove_fields(json_str):
            filtered_data = _remove_verbose_fields(filtered_data)
            json_str = json.dumps(filtered_data, indent=2, ensure_ascii=False)
        return json_str
    except FileNotFoundError:
        return json.dumps({"error": "regulations.json file not found in uploaded_files/EASA/regulations"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to parse regulations.json - invalid JSON"})
    except Exception as e:
        return json.dumps({"error": f"Failed to filter regulations: {str(e)}"})


class FetchNestedRulesInput(BaseModel):
    short_name: str | None = Field(
        default=None,
        description="Optional short name of the regulation (typically left null for EAR queries)",
    )
    parent_title_path: list[str] = Field(description="The path of parent titles as a list, e.g., ['Part-1', 'Subpart A']")
    include_all_descendants: bool = Field(
        default=True, description="Whether to include all descendant rules"
    )
    metadata: list[str] | None = Field(default=None, description="Optional metadata fields to include")
    truncate_metadata_chars_max: int | None = Field(
        default=500, description="Maximum characters for metadata truncation"
    )


@tool(args_schema=FetchNestedRulesInput)
def fetch_easa_nested_rules(
    parent_title_path: list[str],
    short_name: str | None = None,
    include_all_descendants: bool = True,
    metadata: list[str] | None = None,
    truncate_metadata_chars_max: int | None = 500,
) -> dict:
    """Fetch nested rules/elements from EASA regulations by traversing a parent title path."""

    if not AVIATIONBOT_API_KEY:
        return {"error": "AVIATION_BOT_API_KEY environment variable not set"}

    payload = {
        "short_name": short_name,
        "parent_title_path": parent_title_path,
        "include_all_descendants": include_all_descendants,
        "metadata": metadata or ["markdown_with_html_table"],
        "truncate_metadata_chars_max": truncate_metadata_chars_max,
    }

    return _aviationbot_request(
        "POST",
        "/tool/EASA/easa-ear-child-elements",
        json=payload,
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


class FetchParentTitlePathInput(BaseModel):
    references: list[str] = Field(
        description="List of regulatory references to resolve, e.g., ['AMC 25.201']"
    )
    include_base_without_parentheses: bool = Field(
        default=True, description="Include base references without parentheses"
    )
    include_without_amc_gm_prefix: bool = Field(
        default=True, description="Include references without AMC/GM prefix"
    )


@tool(args_schema=FetchParentTitlePathInput)
def fetch_easa_parent_title_path(
    references: list[str],
    include_base_without_parentheses: bool = True,
    include_without_amc_gm_prefix: bool = True,
) -> dict:
    """Fetch EAR hierarchy information for given regulatory references."""

    if not AVIATIONBOT_API_KEY:
        return {"error": "AVIATION_BOT_API_KEY environment variable not set"}

    payload = {
        "references": references,
        "include_base_without_parentheses": include_base_without_parentheses,
        "include_without_amc_gm_prefix": include_without_amc_gm_prefix,
    }

    return _aviationbot_request(
        "POST",
        "/tool/EASA/easa-reference-to-ear-hierarchy",
        json=payload,
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


class FetchRulesDocumentInput(BaseModel):
    perma_ids: list[str] = Field(
        description="List of perma_id's to retrieve specific documents"
    )
    metadata: list[str] = Field(
        description="List of metadata fields to include, e.g., ['markdown_with_html_table', 'parent_perma_ids', 'child_perma_ids', 'parent_titles', 'children_titles']"
    )


@tool(args_schema=FetchRulesDocumentInput)
def fetch_easa_rules_document(perma_ids: list[str], metadata: list[str]) -> dict:
    """Fetch one or multiple regulatory rules documents with their content and metadata."""

    if not AVIATIONBOT_API_KEY:
        return {"error": "AVIATION_BOT_API_KEY environment variable not set"}

    payload = {"perma_ids": perma_ids, "metadata": metadata}

    return _aviationbot_request(
        "POST",
        "/document/",
        json=payload,
        timeout=AVIATIONBOT_DEFAULT_TIMEOUT,
    )


@tool
def get_easa_regulations_map() -> str:
    """Retrieve the EASA regulations map as raw JSON for LLM processing."""
    try:
        regulations_map = _load_regulations_map()
        return json.dumps(regulations_map, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"error": "regulations_map.json file not found in uploaded_files/EASA/regulations_map"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to parse regulations_map.json - invalid JSON"})
    except Exception as e:
        return json.dumps({"error": f"Failed to load regulations map: {str(e)}"})


@tool
def get_easa_certification_specifications() -> str:
    """Retrieve EASA certification specifications as raw JSON for LLM processing."""
    try:
        cert_specs = _load_certification_specifications()
        return json.dumps(cert_specs, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"error": "certification-specifications.json file not found in uploaded_files/EASA/certification-specifications"})
    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to parse certification-specifications.json - invalid JSON"})
    except Exception as e:
        return json.dumps({"error": f"Failed to load certification specifications: {str(e)}"})


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except Exception as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}
