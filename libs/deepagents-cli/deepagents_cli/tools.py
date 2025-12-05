"""Custom tools for the CLI agent."""

from pathlib import Path
from typing import Any, Literal

import requests
from markdownify import markdownify
from tavily import TavilyClient

from deepagents_cli.config import settings
from deepagents_cli.rag import CodeRAG

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None

# Global RAG instance cache (per workspace)
_rag_instances: dict[str, CodeRAG] = {}


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


def semantic_search(
    query: str,
    workspace_root: str,
    max_results: int = 5,
    file_type: str | None = None,
    force_reindex: bool = False,
    use_hybrid: bool = True,
) -> dict[str, Any]:
    """Search the codebase using hybrid search (semantic + lexical) with RAG.

    This tool performs hybrid search across any codebase folder to find code, functions, classes,
    or documentation that matches your query. It combines:
    - Semantic search: Uses embeddings to understand the meaning of your query
    - Lexical search: Uses keyword matching (like grep) for exact matches
    
    This hybrid approach (similar to Cursor IDE) provides comprehensive results by leveraging
    both meaning-based and keyword-based matching.

    Args:
        query: The search query (describe what you're looking for or use keywords)
        workspace_root: REQUIRED - The folder/directory to search in. Can be any absolute or relative path.
                        Examples: "/path/to/project", "./src", "../other-project"
        max_results: Maximum number of results to return (default: 5)
        file_type: Optional file type filter (e.g., ".py", ".js", ".md")
        force_reindex: Force reindexing of the codebase (default: False)
        use_hybrid: Enable hybrid search combining semantic and lexical (default: True)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - content: The code chunk content
            - file_path: Full path to the file
            - relative_path: Relative path from workspace root
            - file_name: Name of the file
            - score: Similarity score (normalized, higher is better)
            - match_type: "semantic" or "lexical" indicating match type
        - query: The original search query
        - indexed: Whether the codebase was indexed

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Use the file_path to read the full file if needed
    3. The results show code chunks - you may need to read the full file for context
    4. Higher normalized scores indicate better matches
    5. Results include both semantic (meaning-based) and lexical (keyword-based) matches
    6. Use this tool when you need to find code by meaning, functionality, or exact keywords
    """
    try:
        # Resolve and validate workspace root
        workspace_path = Path(workspace_root).resolve()
        
        if not workspace_path.exists():
            return {
                "error": f"Workspace root does not exist: {workspace_root}",
                "query": query,
                "suggestion": "Provide a valid folder path that exists",
            }
        
        if not workspace_path.is_dir():
            return {
                "error": f"Workspace root is not a directory: {workspace_root}",
                "query": query,
                "suggestion": "Provide a folder path, not a file path",
            }
        
        workspace_root = str(workspace_path)

        # Get or create RAG instance for this workspace
        if workspace_root not in _rag_instances:
            _rag_instances[workspace_root] = CodeRAG(workspace_root=workspace_root)

        rag = _rag_instances[workspace_root]

        # Index if needed
        try:
            if force_reindex:
                rag.index(force=True)
            else:
                rag.index(force=False)
        except ValueError as e:
            # If indexing fails due to missing API key, return helpful error
            if "API key" in str(e):
                return {
                    "error": str(e),
                    "query": query,
                    "suggestion": "Set OPENAI_API_KEY environment variable to enable semantic search",
                }
            raise

        # Prepare metadata filter
        filter_metadata = None
        if file_type:
            filter_metadata = {"file_type": file_type}

        # Perform hybrid search (semantic + lexical)
        results = rag.search(
            query, 
            k=max_results, 
            filter_metadata=filter_metadata,
            use_hybrid=use_hybrid,
        )

        return {
            "results": results,
            "query": query,
            "workspace_root": workspace_root,
            "indexed": True,
        }

    except Exception as e:
        return {
            "error": f"Semantic search error: {e!s}",
            "query": query,
        }
