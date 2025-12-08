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
from langchain_google_genai import ChatGoogleGenerativeAI

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
        description="Optional list of metadata fields to include, e.g., ['markdown_with_html_table', 'parent_perma_ids', 'child_perma_ids', 'parent_titles', 'children_titles']"
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


class FileToLaunchingToolInput(BaseModel):
    """Input schema for the file_to_launching_tool."""

    file_path: str = Field(
        description="Absolute path to the file on the local filesystem to read."
    )
    start_line: int | None = Field(
        default=None,
        description="Optional 1-indexed start line number. If provided, only content from this line onwards will be included.",
    )
    end_line: int | None = Field(
        default=None,
        description="Optional 1-indexed end line number (inclusive). If provided, only content up to this line will be included.",
    )


@tool(
    "file_to_launching_tool",
    args_schema=FileToLaunchingToolInput,
    description=(
        "Read a file from the local filesystem to prepare it for use with launching tools. "
        "This tool bridges the gap between local files and external API-connected tools. "
        "Optionally specify start_line and end_line to read only a portion of the file. "
        "Currently returns the first 2 lines as a test to verify the tool works correctly."
    ),
)
def file_to_launching_tool(
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a file from the filesystem with optional line range selection.

    This tool reads file content from the local filesystem and prepares it for use
    with launching tools that connect to external APIs. It supports reading entire
    files or specific line ranges.

    Args:
        file_path: Absolute path to the file to read.
        start_line: Optional 1-indexed start line (inclusive). Defaults to beginning of file.
        end_line: Optional 1-indexed end line (inclusive). Defaults to end of file.

    Returns:
        Dictionary containing:
        - success: Whether the file was read successfully
        - file_path: The path that was read
        - content: The file content (or selected lines)
        - total_lines: Total number of lines in the original file
        - lines_read: Number of lines in the returned content
        - line_range: The actual line range that was read (start, end)
        - preview: First 2 lines as a test preview
    """
    try:
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_path": file_path,
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "file_path": file_path,
            }

        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback for binary-like files
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to decode file content: {str(e)}",
                    "file_path": file_path,
                }

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Validate and adjust line ranges
        actual_start = 1
        actual_end = total_lines

        if start_line is not None:
            if start_line < 1:
                return {
                    "success": False,
                    "error": f"start_line must be >= 1, got {start_line}",
                    "file_path": file_path,
                }
            actual_start = min(start_line, total_lines)

        if end_line is not None:
            if end_line < 1:
                return {
                    "success": False,
                    "error": f"end_line must be >= 1, got {end_line}",
                    "file_path": file_path,
                }
            actual_end = min(end_line, total_lines)

        if actual_start > actual_end:
            return {
                "success": False,
                "error": f"start_line ({actual_start}) cannot be greater than end_line ({actual_end})",
                "file_path": file_path,
            }

        # Extract the requested line range (convert to 0-indexed for slicing)
        selected_lines = lines[actual_start - 1 : actual_end]
        selected_content = "".join(selected_lines)

        # TEST: For now, return first 2 lines as a preview to verify the tool works
        preview_lines = selected_lines[:2]
        preview_content = "".join(preview_lines).rstrip("\n")

        return {
            "success": True,
            "file_path": file_path,
            "content": selected_content,
            "total_lines": total_lines,
            "lines_read": len(selected_lines),
            "line_range": {"start": actual_start, "end": actual_end},
            "preview": preview_content,  # TEST: First 2 lines to verify tool works
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied reading file: {file_path}",
            "file_path": file_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error reading file: {str(e)}",
            "file_path": file_path,
        }


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


# --- Google Custom Search Engine (CSE) helpers and tools -------------------------------------------

# API key from environment (set in .env as GOOGLE_CLOUD_CONSOLE_CUSTOM_SEARCH_API)
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CLOUD_CONSOLE_CUSTOM_SEARCH_API", "")


def _google_cse_search(
    query: str,
    cse_id: str,
    api_key: str | None = None,
    num_results: int = 10,
    start_index: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Perform a Google Custom Search Engine query.
    
    This is a reusable helper function for making CSE API requests.
    
    Args:
        query: The search query string.
        cse_id: The Custom Search Engine ID (cx parameter).
        api_key: Google API key. If None, uses GOOGLE_CLOUD_CONSOLE_CUSTOM_SEARCH_API env var.
        num_results: Number of results to return (max 10 per request).
        start_index: Starting index for results (1-indexed, for pagination).
        **kwargs: Additional parameters to pass to the CSE API.
    
    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - query: The original query
        - total_results: Total number of matching results
        - items: List of search results, each with title, link, snippet
        - error: Error message if the request failed
    """
    effective_api_key = api_key or GOOGLE_CSE_API_KEY
    
    if not effective_api_key:
        return {
            "success": False,
            "error": "Google CSE API key not configured. Set GOOGLE_CLOUD_CONSOLE_CUSTOM_SEARCH_API environment variable.",
            "query": query,
        }
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": effective_api_key,
        "cx": cse_id,
        "num": min(num_results, 10),  # API max is 10 per request
        "start": start_index,
    }
    params.update(kwargs)
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract and format results
        items = []
        if "items" in data:
            for item in data["items"]:
                items.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_link": item.get("displayLink", ""),
                })
        
        # Get total results count
        search_info = data.get("searchInformation", {})
        total_results = search_info.get("totalResults", "0")
        
        return {
            "success": True,
            "query": query,
            "total_results": total_results,
            "items": items,
            "result_count": len(items),
        }
        
    except requests.HTTPError as exc:
        error_msg = f"HTTP error {exc.response.status_code}"
        try:
            error_data = exc.response.json()
            if "error" in error_data:
                error_msg = error_data["error"].get("message", error_msg)
        except Exception:
            pass
        return {
            "success": False,
            "error": error_msg,
            "query": query,
        }
    except requests.RequestException as exc:
        return {
            "success": False,
            "error": f"Request error: {str(exc)}",
            "query": query,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Unexpected error: {str(exc)}",
            "query": query,
        }


def create_google_cse_tool(
    cse_id: str,
    tool_name: str,
    tool_description: str,
    cse_name: str | None = None,
):
    """Factory function to create a LangChain tool for a specific Google CSE.
    
    This allows easily creating multiple CSE-based search tools with different
    search engine IDs while sharing the common search logic.
    
    Args:
        cse_id: The Google Custom Search Engine ID.
        tool_name: Name for the LangChain tool (should be snake_case).
        tool_description: Description for the tool that helps the LLM understand when to use it.
        cse_name: Human-readable name for the CSE (used in return data). Defaults to tool_name.
    
    Returns:
        A LangChain tool function that can be added to an agent's toolset.
    
    Example:
        >>> my_cse_tool = create_google_cse_tool(
        ...     cse_id="your_cse_id_here",
        ...     tool_name="search_my_domain",
        ...     tool_description="Search for information in my custom knowledge base.",
        ...     cse_name="My Custom Search"
        ... )
    """
    effective_cse_name = cse_name or tool_name
    
    @tool(tool_name, description=tool_description)
    def cse_search_tool(
        query: str,
        num_results: int = 10,
    ) -> dict[str, Any]:
        """Search using a Google Custom Search Engine."""
        result = _google_cse_search(
            query=query,
            cse_id=cse_id,
            num_results=num_results,
        )
        result["cse_name"] = effective_cse_name
        result["cse_id"] = cse_id
        return result
    
    return cse_search_tool


# --- Pre-configured CSE Tools ----------------------------------------------------------------

# Aviation.bot EASA UAS Custom Search Engine
# Public URL: https://cse.google.com/cse?cx=67b0166f411444635
EASA_UAS_CSE_ID = "67b0166f411444635"


class SearchEasaUasInput(BaseModel):
    """Input schema for the EASA UAS search tool."""
    
    query: str = Field(
        description="Search query for EASA UAS (Unmanned Aircraft Systems) regulations, drone rules, and related aviation documentation."
    )
    num_results: int = Field(
        default=10,
        description="Number of search results to return (max 10).",
        ge=1,
        le=10,
    )


@tool(
    "search_easa_uas_regulations",
    args_schema=SearchEasaUasInput,
    description=(
        "Search the EASA UAS (Unmanned Aircraft Systems) regulatory corpus using Google Custom Search. "
        "This tool searches official EASA documentation specifically focused on drone regulations, "
        "UAS operations, specific category requirements, open category rules, and certified category "
        "operations. Use this for questions about European drone regulations, SORA methodology, "
        "remote pilot requirements, and UAS operational authorizations."
    ),
)
def search_easa_uas_regulations(
    query: str,
    num_results: int = 10,
) -> dict[str, Any]:
    """Search EASA UAS regulations using Aviation.bot's custom search engine.
    
    This tool queries the EASA UAS (Unmanned Aircraft Systems) documentation corpus
    to find relevant regulatory information about drone operations in Europe.
    
    Args:
        query: The search query for UAS/drone regulations.
        num_results: Number of results to return (1-10, default 10).
    
    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - query: The original search query
        - total_results: Total number of matching documents
        - items: List of results with title, link, snippet, and display_link
        - result_count: Number of results returned
        - cse_name: Name of the custom search engine used
        - cse_id: The CSE identifier
        - error: Error message if the request failed
    """
    result = _google_cse_search(
        query=query,
        cse_id=EASA_UAS_CSE_ID,
        num_results=num_results,
    )
    result["cse_name"] = "Aviation.bot EASA UAS"
    result["cse_id"] = EASA_UAS_CSE_ID
    return result


# --- URL Document Query Tool -------------------------------------------------------------------

# Google AI Studio API key (used for underlying LLM)
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")


class QueryUrlsInput(BaseModel):
    """Input schema for the URL document query tool."""
    
    query: str = Field(
        description=(
            "The question to answer based on the content of the external documents at the URLs. "
            "Be specific about what information you need. The query will be answered using the "
            "actual full content of the documents, not just snippets."
        )
    )
    urls: list[str] = Field(
        description=(
            "List of URLs pointing to external documents (web pages or PDFs) to analyze. "
            "These are the source documents that will be read to answer your query. "
            "Maximum recommended: 1 URLs per request, unless a comparison of multiple documents is required. Make seperate tool calls per url."
        ),
        min_length=1,
    )
    additional_context: str | None = Field(
        default=None,
        description=(
            "Optional additional instructions to guide the extraction. "
            "For example: 'Focus on section 3.2' or 'Extract the table on page 5'."
        ),
    )


@tool(
    "query_urls",
    args_schema=QueryUrlsInput,
    description=(
        "Query external documents at specified URLs to extract information and answer questions. "
        "This tool reads the FULL CONTENT of web pages or PDF documents and answers your query "
        "based on what it finds. Use this when you need to: "
        "1) Read full documents from URLs returned by search tools (which only give snippets), "
        "2) Extract specific sections, tables, or requirements from regulatory documents, "
        "3) Answer detailed questions that require reading the actual document content. "
        "The response prioritizes literal/verbatim text from the documents rather than rewrites. "
        "If documents are large, use follow-up queries on the same URLs to explore other sections."
    ),
)
def query_urls(
    query: str,
    urls: list[str],
    additional_context: str | None = None,
) -> dict[str, Any]:
    """Query external documents at URLs to extract information.
    
    This tool loads the full content of documents at the specified URLs and
    answers your query based on what it finds. It's designed for:
    - Following up on search results to read full documents (not just snippets)
    - Extracting specific regulatory text, tables, or requirements
    - Getting verbatim quotes and literal content from source documents
    
    The tool prioritizes returning actual text from the documents rather than
    paraphrased summaries. For large documents, it will indicate what other
    sections are available for follow-up queries.
    
    Args:
        query: The question to answer based on the document content.
        urls: List of URLs to external documents (web pages or PDFs).
        additional_context: Optional extra instructions for the extraction.
    
    Returns:
        Dictionary containing:
        - success: Whether the query succeeded
        - query: The original query
        - urls: The URLs that were analyzed
        - answer: The extracted information/answer from the documents
        - error: Error message if the request failed
    """
    if not GOOGLE_AI_STUDIO_API_KEY:
        return {
            "success": False,
            "error": "API key not configured. Set GOOGLE_AI_STUDIO_API_KEY environment variable.",
            "query": query,
            "urls": urls,
        }
    
    if not urls:
        return {
            "success": False,
            "error": "At least one URL must be provided.",
            "query": query,
            "urls": urls,
        }

    # TODO whitelist domains to prevent malicious urls

    
    try:
        # Initialize the LLM with URL reading capability
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            # model="gemini-3-flash-preview",

            temperature=0,
            google_api_key=GOOGLE_AI_STUDIO_API_KEY,
        )
        
        # Bind the url_context tool for native URL reading
        llm_with_url_context = llm.bind(
            tools=[{"url_context": {}}]
        )
        
        # Build the prompt with URLs and query
        urls_formatted = "\n".join(f"- {url}" for url in urls)
        
        prompt_parts = [
            f"Question: {query}", # Repeat in begin and end
            f"Read the following external documents:\n{urls_formatted}",
            "",
            f"Question: {query}", # Repeat in begin and end
        ]
        
        if additional_context:
            prompt_parts.append(f"\nAdditional instructions: {additional_context}")
        
        # Key instruction: prioritize literal text over rewrites
        prompt_parts.append(
            "\n\n## Response Instructions\n"
            "1. PRIORITIZE LITERAL TEXT: Quote or reproduce the actual text from the documents "
            "rather than paraphrasing or rewriting. Use verbatim excerpts where relevant.\n"
            "2. CITE LOCATIONS: Indicate where in the document(s) the information comes from "
            "(e.g., section numbers, page references, headings).\n"
            "3. TRUNCATION ALLOWED: If the relevant content is extensive, you may truncate and "
            "indicate what additional sections or content exist. Mention that follow-up queries "
            "using the same URLs can retrieve more detail from specific sections.\n"
            "4. STRUCTURE FOR DENSITY: Present information in a dense, structured format "
            "(bullet points, tables, or numbered lists) rather than prose when appropriate.\n"
            "5. NOT FOUND: If the information is not available in the provided documents, "
            "clearly state that and list what topics/sections ARE covered that might be related."
        )
        
        prompt = "\n".join(prompt_parts)
        
        # Invoke the model with URL context
        response = llm_with_url_context.invoke(prompt)
        
        # Extract the response content
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "success": True,
            "query": query,
            "urls": urls,
            "answer": answer,
        }
        
    except Exception as exc:
        error_message = str(exc)
        
        return {
            "success": False,
            "error": f"Failed to query documents: {error_message}",
            "query": query,
            "urls": urls,
        }
