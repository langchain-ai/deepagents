"""MCP HTTP Transport Implementation with Security.

This module implements streamable HTTP transport for MCP connections
with integrated transport security as required by the MCP 2025-06-18 specification.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from urllib.parse import urlparse
import uuid

try:
    from aiohttp import web, hdrs
    from aiohttp.web import Request, Response, StreamResponse
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    Request = None
    Response = None
    StreamResponse = None

from .transport_security import (
    TransportSecurityManager,
    TransportSecurityConfig,
    TransportSecurityError,
    SecurityPolicy,
    create_development_config,
    create_production_config
)
from .initialization import (
    MCPInitializationManager,
    ImplementationInfo,
    ServerCapabilities,
    create_default_initialization_manager
)

logger = logging.getLogger(__name__)


class MCPHTTPTransport:
    """MCP HTTP Transport with integrated security features."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8000,
                 security_config: Optional[TransportSecurityConfig] = None,
                 message_handler: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
                 initialization_manager: Optional[MCPInitializationManager] = None):
        """Initialize MCP HTTP transport.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            security_config: Transport security configuration
            message_handler: Async function to handle MCP messages
            initialization_manager: MCP initialization manager for lifecycle handling
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for HTTP transport: pip install aiohttp")
        
        self.host = host
        self.port = port
        self.security_manager = TransportSecurityManager(security_config)
        self.message_handler = message_handler or self._default_message_handler
        
        # Initialize MCP initialization manager
        self.initialization_manager = initialization_manager or create_default_initialization_manager(
            "deepagents-mcp",
            "1.0.0"
        )
        
        # Session tracking for MCP initialization
        self.active_sessions: Dict[str, str] = {}  # connection_id -> session_id
        
        # Validate server binding before starting
        self.security_manager.validate_server_binding(host, port)
        
        # Create aiohttp application
        self.app = web.Application()
        self._setup_routes()
        
        # Active SSE connections for streaming
        self.sse_connections: Dict[str, StreamResponse] = {}
        
        logger.info(f"MCP HTTP transport initialized on {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Setup HTTP routes for MCP transport."""
        # Main MCP endpoint supporting POST and GET
        self.app.router.add_route('POST', '/mcp', self._handle_post_request)
        self.app.router.add_route('GET', '/mcp', self._handle_get_request)
        
        # Health check endpoint
        self.app.router.add_route('GET', '/health', self._handle_health_check)
        
        # Add security middleware
        self.app.middlewares.append(self._security_middleware)
    
    async def _security_middleware(self, request: Request, handler: Callable) -> Response:
        """Security middleware for transport-level validation.
        
        Args:
            request: HTTP request
            handler: Request handler
            
        Returns:
            HTTP response with security headers
        """
        try:
            # Extract request headers and URL
            request_headers = dict(request.headers)
            request_url = str(request.url)
            
            # Validate transport security
            validation_result = self.security_manager.validate_request(request_headers, request_url)
            
            # Process request if validation passed
            response = await handler(request)
            
            # Add security headers to response
            for header_name, header_value in validation_result["security_headers"].items():
                response.headers[header_name] = header_value
                
            return response
            
        except TransportSecurityError as e:
            logger.error(f"Transport security validation failed: {e}")
            return web.Response(
                status=403,
                text=f"Transport security violation: {e}",
                headers={"Content-Type": "text/plain"}
            )
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return web.Response(
                status=500,
                text="Internal security error",
                headers={"Content-Type": "text/plain"}
            )
    
    async def _handle_post_request(self, request: Request) -> Response:
        """Handle POST requests for MCP messages.
        
        Args:
            request: HTTP POST request
            
        Returns:
            JSON response with MCP message result
        """
        try:
            # Generate or extract session ID for MCP initialization tracking
            session_id = request.headers.get('X-MCP-Session-ID') or str(uuid.uuid4())
            
            # Validate MCP-Protocol-Version header if present
            protocol_version = request.headers.get('MCP-Protocol-Version')
            if protocol_version and not self._validate_protocol_version(protocol_version):
                return web.Response(
                    status=400,
                    text=f"Unsupported protocol version: {protocol_version}",
                    headers={"Content-Type": "text/plain"}
                )
            
            # Parse JSON-RPC message
            try:
                message_data = await request.json()
            except Exception as e:
                logger.error(f"Invalid JSON in request: {e}")
                return web.Response(
                    status=400,
                    text="Invalid JSON format",
                    headers={"Content-Type": "application/json"}
                )
            
            # Validate JSON-RPC format
            if not self._validate_jsonrpc_message(message_data):
                return web.Response(
                    status=400,
                    text="Invalid JSON-RPC 2.0 format",
                    headers={"Content-Type": "application/json"}
                )
            
            # Handle MCP initialization lifecycle messages
            method = message_data.get("method")
            if method == "initialize":
                response_data = await self.initialization_manager.handle_initialize_request(message_data, session_id)
                # Track active session
                self.active_sessions[request.remote] = session_id
            elif method == "notifications/initialized":
                await self.initialization_manager.handle_initialized_notification(message_data, session_id)
                response_data = None  # Notifications don't have responses
            else:
                # Check if session is initialized for other methods
                if method not in ["ping"]:  # Allow ping without initialization
                    if not self.initialization_manager.is_session_initialized(session_id):
                        return web.Response(
                            text=json.dumps({
                                "jsonrpc": "2.0",
                                "id": message_data.get("id"),
                                "error": {
                                    "code": -32002,  # Server error
                                    "message": "MCP session not initialized. Send initialize request first."
                                }
                            }),
                            status=400,
                            headers={"Content-Type": "application/json"}
                        )
                
                # Process through regular message handler
                response_data = await self.message_handler(message_data)
            
            # Return response (if any)
            if response_data is not None:
                response_headers = {"Content-Type": "application/json"}
                
                # Add MCP-Protocol-Version header if session is initialized
                if self.initialization_manager.is_session_initialized(session_id):
                    negotiated_version = self.initialization_manager.get_negotiated_version(session_id)
                    if negotiated_version:
                        response_headers["MCP-Protocol-Version"] = negotiated_version
                
                return web.Response(
                    text=json.dumps(response_data),
                    headers=response_headers
                )
            else:
                # No response for notifications
                response_headers = {}
                
                # Add MCP-Protocol-Version header even for notifications if session is initialized
                if self.initialization_manager.is_session_initialized(session_id):
                    negotiated_version = self.initialization_manager.get_negotiated_version(session_id)
                    if negotiated_version:
                        response_headers["MCP-Protocol-Version"] = negotiated_version
                
                return web.Response(status=204, headers=response_headers)  # No Content
            
        except Exception as e:
            logger.error(f"Error handling POST request: {e}")
            
            # Return JSON-RPC error response
            error_response = {
                "jsonrpc": "2.0",
                "id": message_data.get("id") if 'message_data' in locals() else None,
                "error": {
                    "code": -32603,  # Internal error
                    "message": "Internal server error",
                    "data": str(e)
                }
            }
            
            return web.Response(
                text=json.dumps(error_response),
                status=500,
                headers={"Content-Type": "application/json"}
            )
    
    async def _handle_get_request(self, request: Request) -> StreamResponse:
        """Handle GET requests for Server-Sent Events streaming.
        
        Args:
            request: HTTP GET request
            
        Returns:
            SSE stream response
        """
        try:
            # Create SSE response
            response = web.StreamResponse(
                status=200,
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
            await response.prepare(request)
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            self.sse_connections[connection_id] = response
            
            logger.info(f"SSE connection established: {connection_id}")
            
            try:
                # Keep connection alive
                while True:
                    # Send keep-alive ping
                    await response.write(b"event: ping\ndata: ping\n\n")
                    await asyncio.sleep(30)  # Ping every 30 seconds
                    
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"SSE connection error: {e}")
            finally:
                # Clean up connection and associated session
                if connection_id in self.sse_connections:
                    del self.sse_connections[connection_id]
                
                # Clean up session if associated with this connection
                if connection_id in self.active_sessions:
                    session_id = self.active_sessions[connection_id]
                    self.initialization_manager.shutdown_session(session_id)
                    del self.active_sessions[connection_id]
                
                logger.info(f"SSE connection closed: {connection_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling GET request: {e}")
            return web.Response(
                status=500,
                text="Internal server error",
                headers={"Content-Type": "text/plain"}
            )
    
    async def _handle_health_check(self, request: Request) -> Response:
        """Handle health check requests.
        
        Args:
            request: HTTP request
            
        Returns:
            Health status response
        """
        health_status = {
            "status": "healthy",
            "transport": "http",
            "security_policy": self.security_manager.config.security_policy.value,
            "active_connections": len(self.sse_connections),
            "version": "2025-06-18"
        }
        
        return web.Response(
            text=json.dumps(health_status),
            headers={"Content-Type": "application/json"}
        )
    
    def _validate_protocol_version(self, version: str) -> bool:
        """Validate MCP protocol version.
        
        Args:
            version: Protocol version string
            
        Returns:
            True if version is supported
        """
        supported_versions = ["2025-06-18", "2025-03-26"]  # Add supported versions
        return version in supported_versions
    
    def _validate_jsonrpc_message(self, message: Dict[str, Any]) -> bool:
        """Validate JSON-RPC 2.0 message format.
        
        Args:
            message: Message to validate
            
        Returns:
            True if message is valid JSON-RPC 2.0
        """
        # Check required fields
        if not isinstance(message, dict):
            return False
        
        # Must have jsonrpc field with value "2.0"
        if message.get("jsonrpc") != "2.0":
            return False
        
        # Must have method for requests/notifications
        if "method" in message:
            if not isinstance(message["method"], str):
                return False
            
            # Requests must have ID (notifications must not)
            if "id" in message:
                # Request: ID must be string, number, or null (but not null for MCP)
                return message["id"] is not None
            else:
                # Notification: must not have ID
                return True
        
        # Responses must have ID and either result or error
        elif "id" in message:
            return ("result" in message) != ("error" in message)  # XOR: exactly one
        
        return False
    
    async def _default_message_handler(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Default message handler for MCP messages.
        
        Args:
            message: MCP message to handle
            
        Returns:
            Response message
        """
        method = message.get("method")
        logger.warning(f"No message handler configured, received method: {method}")
        
        # Handle basic ping method
        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {}
            }
        
        # Return method not found error for other methods
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,  # Method not found
                "message": f"Method not found: {method}. No message handler configured."
            }
        }
    
    async def send_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification to all SSE connections.
        
        Args:
            notification: MCP notification to send
        """
        if not self.sse_connections:
            return
        
        event_data = f"data: {json.dumps(notification)}\n\n"
        
        # Send to all active connections
        disconnected_connections = []
        for connection_id, response in self.sse_connections.items():
            try:
                await response.write(event_data.encode())
            except Exception as e:
                logger.warning(f"Failed to send notification to {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            if connection_id in self.sse_connections:
                del self.sse_connections[connection_id]
    
    def set_message_handler(self, handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        """Set message handler for MCP messages.
        
        Args:
            handler: Async function to handle MCP messages
        """
        self.message_handler = handler
        logger.info("Message handler updated")
    
    async def start(self) -> None:
        """Start the HTTP transport server."""
        # Get SSL context if HTTPS is required
        ssl_context = self.security_manager.get_ssl_context()
        
        # Create and start server
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(
            runner, 
            self.host, 
            self.port,
            ssl_context=ssl_context
        )
        
        await site.start()
        
        protocol = "https" if ssl_context else "http"
        logger.info(f"MCP HTTP transport started on {protocol}://{self.host}:{self.port}/mcp")
    
    async def stop(self) -> None:
        """Stop the HTTP transport server."""
        # Close all SSE connections
        for connection_id, response in list(self.sse_connections.items()):
            try:
                await response.write_eof()
            except Exception:
                pass
        
        self.sse_connections.clear()
        logger.info("MCP HTTP transport stopped")


def create_secure_http_transport(host: str = "localhost",
                                port: int = 8000,
                                allowed_origins: Optional[list] = None,
                                production_mode: bool = False) -> MCPHTTPTransport:
    """Create MCP HTTP transport with security configuration.
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        allowed_origins: List of allowed origins for CORS
        production_mode: Whether to use production security settings
        
    Returns:
        Configured MCP HTTP transport
    """
    if production_mode:
        if not allowed_origins:
            raise ValueError("allowed_origins required for production mode")
        
        security_config = create_production_config(
            allowed_origins=allowed_origins,
            allowed_hosts=[host]
        )
    else:
        security_config = create_development_config()
    
    return MCPHTTPTransport(
        host=host,
        port=port,
        security_config=security_config
    )


# Example usage with proper initialization


async def main():
    """Example server startup with proper initialization."""
    # Create custom initialization manager
    server_capabilities = ServerCapabilities(
        tools={"listChanged": True},
        resources={"subscribe": True, "listChanged": True},
        logging={}
    )
    
    server_info = ImplementationInfo(
        name="example-mcp-server",
        version="1.0.0",
        description="Example MCP HTTP server with proper initialization",
        homepage="https://github.com/example/mcp-server"
    )
    
    initialization_manager = MCPInitializationManager(
        server_info=server_info,
        server_capabilities=server_capabilities
    )
    
    # Create transport with security and initialization
    transport = MCPHTTPTransport(
        host="localhost",
        port=8000,
        security_config=create_development_config(),
        initialization_manager=initialization_manager
    )
    
    # Set up custom message handler for non-initialization methods
    async def custom_message_handler(message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom MCP methods after initialization."""
        method = message.get("method")
        
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "example_tool",
                            "description": "An example tool for demonstration",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"}
                                }
                            }
                        }
                    ]
                }
            }
        elif method == "tools/call":
            params = message.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "example_tool":
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Hello! You sent: {arguments.get('message', 'No message')}"
                            }
                        ]
                    }
                }
        
        # Fallback to default handler
        return await transport._default_message_handler(message)
    
    transport.set_message_handler(custom_message_handler)
    
    # Start server
    await transport.start()
    print(f"🚀 Example MCP server running on http://localhost:8000/mcp")
    print("📋 Health check: http://localhost:8000/health")
    print("💬 Send POST requests to /mcp endpoint")
    print("📡 SSE streaming available via GET to /mcp")
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
    finally:
        await transport.stop()


if __name__ == "__main__":
    asyncio.run(main())