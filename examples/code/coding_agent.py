import os
import subprocess
import tempfile
import platform
import requests
from typing import List, Dict, Any, Optional, Union, Literal
from pathlib import Path

from tavily import TavilyClient
from deepagents import create_deep_agent, SubAgent
from utils import validate_command_safety
from subagents import code_reviewer_agent, test_generator_agent
from langgraph.types import Command
from coding_instructions import get_coding_instructions

# LangSmith tracing imports
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langchain_core.tracers.langchain import LangChainTracer
import dotenv

dotenv.load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def execute_bash(command: str, timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
    """
    Execute bash/shell commands safely with prompt injection detection and human approval.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        cwd: Working directory for command execution

    Returns:
        Dictionary with execution results including stdout, stderr, and success status
    """
    try:
        # First, validate command safety (focusing on prompt injection)
        safety_validation = validate_command_safety(command)
        
        # If command is not safe, return error without executing
        if not safety_validation.is_safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command blocked - safety validation failed:\nThreat Type: {safety_validation.threat_type}\nReasoning: {safety_validation.reasoning}\nDetected Patterns: {', '.join(safety_validation.detected_patterns)}",
                "return_code": -1,
                "safety_validation": safety_validation.model_dump()
            }
        
        if platform.system() == "Windows":
            shell_cmd = ["cmd", "/c", command]
        else:
            shell_cmd = ["bash", "-c", command]

        # Execute the command
        result = subprocess.run(
            shell_cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "safety_validation": safety_validation.model_dump()
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "return_code": -1,
        }


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Union[str, Dict] = None,
    params: Dict[str, str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP requests to APIs and web services.

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
        # Prepare request parameters
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

        # Make the request
        response = requests.request(**kwargs)

        # Try to parse JSON response, fallback to text
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
            "content": f"Request error: {str(e)}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {str(e)}",
            "url": url,
        }

def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for programming-related information."""
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query
        }
    
    try:
        search_docs = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return search_docs
    except Exception as e:
        return {
            "error": f"Web search error: {str(e)}",
            "query": query
        }

# Get coding instructions from separate file
coding_instructions = get_coding_instructions()

# Create the coding agent with interrupt handling and LangSmith tracing
config = {"recursion_limit": 1000}

agent = create_deep_agent(
    [execute_bash, http_request, web_search],
    coding_instructions,
    subagents=[code_reviewer_agent, test_generator_agent],
    local_filesystem=True,
).with_config(config)
