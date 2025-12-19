"""Tool interceptors for MCP tools."""

import logging
import asyncio
from typing import Callable, Awaitable, Dict, Any

import httpx
from langchain_core.tools import ToolException
from langchain_mcp_adapters.tools import MCPToolCallRequest, MCPToolCallResult

logger = logging.getLogger(__name__)


class RetryAndSanitizeInterceptor:
    """Interceptor that sanitizes None arguments and retries on transient errors.
    
    This interceptor:
    1. Removes None-valued arguments to prevent validation errors
    2. Retries on HTTP 5xx errors and timeouts (transient errors)
    3. Does NOT retry on ToolException (validation errors that won't succeed on retry)
    4. Logs all errors for debugging
    """
    
    def __init__(self, max_attempts: int = 3, backoff_base: float = 1.0):
        """Initialize the interceptor.
        
        Args:
            max_attempts: Maximum number of retry attempts
            backoff_base: Base time in seconds for exponential backoff
        """
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
    
    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None-valued arguments.
        
        Args:
            args: Original arguments dictionary
            
        Returns:
            Dictionary with None values removed
        """
        return {k: v for k, v in args.items() if v is not None}
    
    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        """Intercept tool call to sanitize arguments and add retry logic.
        
        Args:
            request: Tool call request with name, args, and context
            handler: Async function that executes the actual tool call
            
        Returns:
            Result from the tool execution
            
        Raises:
            ToolException: If tool is called with invalid arguments (not retried)
            httpx.HTTPStatusError: If HTTP error persists after retries
            httpx.TimeoutException: If timeout persists after retries
        """
        # Sanitize None arguments before first attempt
        sanitized_args = self._sanitize_args(request.args)
        sanitized_request = MCPToolCallRequest(
            name=request.name,
            args=sanitized_args,
            server_name=request.server_name,
            headers=request.headers,
            runtime=request.runtime,
        )
        
        logger.debug(
            "Tool %s called with args=%s (sanitized from %s)", 
            request.name, 
            sanitized_args, 
            request.args
        )
        
        # Retry loop for transient errors
        for attempt in range(1, self.max_attempts + 1):
            logger.debug("Tool %s attempt %d/%d", request.name, attempt, self.max_attempts)
            
            try:
                result = await handler(sanitized_request)
                logger.debug("Tool %s succeeded on attempt %d", request.name, attempt)
                return result
                
            except ToolException as e:
                # ToolException indicates wrong arguments or validation errors
                # These should NOT be retried as they won't succeed on retry
                logger.warning(
                    "Tool %s called with invalid arguments: %s", 
                    request.name, 
                    e
                )
                raise
                
            except BaseExceptionGroup as eg:
                # Handle ExceptionGroup from anyio task groups (Python 3.11+)
                # Check for ToolException first (non-retryable)
                tool_exceptions = [e for e in eg.exceptions if isinstance(e, ToolException)]
                if tool_exceptions:
                    e = tool_exceptions[0]
                    logger.warning(
                        "Tool %s called with invalid arguments (in ExceptionGroup): %s",
                        request.name,
                        e
                    )
                    raise e
                
                # Check for HTTP errors
                http_errors = [e for e in eg.exceptions if isinstance(e, httpx.HTTPStatusError)]
                if http_errors:
                    e = http_errors[0]
                    status = getattr(e.response, "status_code", None)
                    if status is not None and 500 <= status < 600 and attempt < self.max_attempts:
                        sleep_time = self.backoff_base * (2 ** (attempt - 1))
                        logger.warning(
                            "Tool %s got HTTP %s (in ExceptionGroup) on attempt %d/%d, retrying after %.1fs",
                            request.name, status, attempt, self.max_attempts, sleep_time
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    # Non-retryable or last attempt
                    logger.error(
                        "Tool %s failed with HTTP error (in ExceptionGroup): %s",
                        request.name, e, exc_info=True
                    )
                    raise e
                
                # Check for timeout errors
                timeout_errors = [
                    e for e in eg.exceptions 
                    if isinstance(e, (httpx.TimeoutException, httpx.ReadTimeout, 
                                     httpx.ConnectTimeout, asyncio.TimeoutError))
                ]
                if timeout_errors and attempt < self.max_attempts:
                    sleep_time = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Tool %s timed out (in ExceptionGroup) on attempt %d/%d, retrying after %.1fs",
                        request.name, attempt, self.max_attempts, sleep_time
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                
                # Not a retryable error or last attempt
                logger.error("Tool %s raised ExceptionGroup: %s", request.name, eg, exc_info=True)
                raise
                
            except httpx.HTTPStatusError as e:
                status = getattr(e.response, "status_code", None)
                # Retry on 5xx errors (including 504 Gateway Timeout)
                if status is not None and 500 <= status < 600 and attempt < self.max_attempts:
                    sleep_time = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Tool %s got HTTP %s on attempt %d/%d, retrying after %.1fs",
                        request.name, status, attempt, self.max_attempts, sleep_time
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                # Non-retryable or last attempt
                logger.error("Tool %s failed with HTTP error: %s", request.name, e, exc_info=True)
                raise
                
            except (httpx.TimeoutException, httpx.ReadTimeout, 
                   httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                # Retry on timeout errors
                if attempt < self.max_attempts:
                    sleep_time = self.backoff_base * (2 ** (attempt - 1))
                    logger.warning(
                        "Tool %s timed out on attempt %d/%d, retrying after %.1fs",
                        request.name, attempt, self.max_attempts, sleep_time
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                logger.error(
                    "Tool %s failed with timeout (final attempt): %s",
                    request.name, e, exc_info=True
                )
                raise
                
            except Exception as e:
                # Unexpected error - log and raise
                logger.error("Tool %s raised unexpected exception: %s", request.name, e, exc_info=True)
                raise
        
        # Should not reach here
        raise RuntimeError(f"Tool {request.name}: Maximum retries ({self.max_attempts}) exceeded")
