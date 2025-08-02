"""MCP Initialization and Capability Negotiation.

This module implements the MCP 2025-06-18 specification requirements for
initialization lifecycle, version negotiation, and capability negotiation.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import re

from .jsonrpc_validation import JSONRPCValidator, ValidationResult, create_default_validator

logger = logging.getLogger(__name__)


class InitializationState(Enum):
    """MCP initialization states."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized" 
    SHUTDOWN = "shutdown"


class ProtocolVersion(Enum):
    """Supported MCP protocol versions."""
    V2025_06_18 = "2025-06-18"
    V2025_03_26 = "2025-03-26"


@dataclass
class ServerCapabilities:
    """MCP server capabilities."""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    completion: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default capability structures."""
        if self.tools is None:
            self.tools = {}
        if self.resources is None:
            self.resources = {}
        if self.prompts is None:
            self.prompts = {}
        if self.completion is None:
            self.completion = {}
        if self.logging is None:
            self.logging = {}


@dataclass 
class ClientCapabilities:
    """MCP client capabilities."""
    roots: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    elicitation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default capability structures."""
        if self.roots is None:
            self.roots = {}
        if self.sampling is None:
            self.sampling = {}
        if self.elicitation is None:
            self.elicitation = {}


@dataclass
class ImplementationInfo:
    """Implementation information for clients and servers."""
    name: str
    version: str
    description: Optional[str] = None
    homepage: Optional[str] = None
    license: Optional[str] = None


@dataclass
class InitializationSession:
    """Tracks initialization session state."""
    session_id: str
    state: InitializationState = InitializationState.NOT_INITIALIZED
    client_version: Optional[str] = None
    server_version: Optional[str] = None
    negotiated_version: Optional[str] = None
    client_capabilities: Optional[ClientCapabilities] = None
    server_capabilities: Optional[ServerCapabilities] = None
    client_info: Optional[ImplementationInfo] = None
    server_info: Optional[ImplementationInfo] = None
    created_at: float = field(default_factory=time.time)
    initialized_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPInitializationManager:
    """Manages MCP initialization lifecycle and capability negotiation."""
    
    def __init__(self, 
                 server_info: ImplementationInfo,
                 server_capabilities: ServerCapabilities,
                 supported_versions: Optional[List[str]] = None,
                 strict_validation: bool = True):
        """Initialize MCP initialization manager.
        
        Args:
            server_info: Server implementation information
            server_capabilities: Server capabilities
            supported_versions: List of supported protocol versions
            strict_validation: Whether to enforce strict validation
        """
        self.server_info = server_info
        self.server_capabilities = server_capabilities
        self.supported_versions = supported_versions or [
            ProtocolVersion.V2025_06_18.value,
            ProtocolVersion.V2025_03_26.value
        ]
        self.strict_validation = strict_validation
        
        # Active initialization sessions
        self.sessions: Dict[str, InitializationSession] = {}
        
        # JSON-RPC validator
        self.validator = create_default_validator()
        
        # Lifecycle handlers
        self.initialized_handler: Optional[Callable[[InitializationSession], Awaitable[None]]] = None
        
        logger.info(f"MCP initialization manager created for {server_info.name} v{server_info.version}")
    
    def set_initialized_handler(self, handler: Callable[[InitializationSession], Awaitable[None]]) -> None:
        """Set handler called when initialization completes.
        
        Args:
            handler: Async function called with session when initialized
        """
        self.initialized_handler = handler
    
    async def handle_initialize_request(self, message: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle initialize request from client.
        
        Args:
            message: JSON-RPC initialize request
            session_id: Session identifier
            
        Returns:
            JSON-RPC response message
            
        Raises:
            ValueError: If request is invalid
        """
        # Validate JSON-RPC format
        validation_result = self.validator.validate_message(message)
        if not validation_result.valid:
            return self._create_error_response(
                message.get("id"),
                -32600,  # Invalid Request
                f"Invalid JSON-RPC format: {'; '.join(validation_result.errors)}"
            )
        
        # Check if session already exists
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.state != InitializationState.NOT_INITIALIZED:
                return self._create_error_response(
                    message.get("id"),
                    -32603,  # Internal error
                    f"Session already in state: {session.state.value}"
                )
        
        # Create or get session
        session = self.sessions.get(session_id, InitializationSession(session_id=session_id))
        session.state = InitializationState.INITIALIZING
        self.sessions[session_id] = session
        
        try:
            # Validate initialize request structure
            params = message.get("params", {})
            if not isinstance(params, dict):
                return self._create_error_response(
                    message.get("id"),
                    -32602,  # Invalid params
                    "Initialize params must be an object"
                )
            
            # Extract and validate required fields
            validation_errors = []
            
            # Protocol version
            client_version = params.get("protocolVersion")
            if not client_version:
                validation_errors.append("Missing required field: protocolVersion")
            elif not isinstance(client_version, str):
                validation_errors.append("protocolVersion must be a string")
            
            # Client capabilities
            client_caps_data = params.get("capabilities")
            if not client_caps_data:
                validation_errors.append("Missing required field: capabilities")
            elif not isinstance(client_caps_data, dict):
                validation_errors.append("capabilities must be an object")
            
            # Client info
            client_info_data = params.get("clientInfo")
            if not client_info_data:
                validation_errors.append("Missing required field: clientInfo")
            elif not isinstance(client_info_data, dict):
                validation_errors.append("clientInfo must be an object")
            elif "name" not in client_info_data:
                validation_errors.append("clientInfo missing required field: name")
            
            if validation_errors:
                return self._create_error_response(
                    message.get("id"),
                    -32602,  # Invalid params
                    f"Invalid initialize request: {'; '.join(validation_errors)}"
                )
            
            # Version negotiation
            negotiated_version = self._negotiate_version(client_version)
            if not negotiated_version:
                return self._create_error_response(
                    message.get("id"),
                    -32603,  # Internal error
                    f"Unsupported protocol version: {client_version}. Supported: {', '.join(self.supported_versions)}"
                )
            
            # Parse client capabilities
            try:
                client_capabilities = self._parse_client_capabilities(client_caps_data)
            except ValueError as e:
                return self._create_error_response(
                    message.get("id"),
                    -32602,  # Invalid params
                    f"Invalid client capabilities: {e}"
                )
            
            # Parse client info
            try:
                client_info = self._parse_implementation_info(client_info_data)
            except ValueError as e:
                return self._create_error_response(
                    message.get("id"),
                    -32602,  # Invalid params
                    f"Invalid client info: {e}"
                )
            
            # Update session
            session.client_version = client_version
            session.negotiated_version = negotiated_version
            session.server_version = negotiated_version
            session.client_capabilities = client_capabilities
            session.server_capabilities = self.server_capabilities
            session.client_info = client_info
            session.server_info = self.server_info
            
            # Create successful response
            response = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": negotiated_version,
                    "capabilities": self._serialize_server_capabilities(),
                    "serverInfo": self._serialize_implementation_info(self.server_info)
                }
            }
            
            logger.info(f"Initialize successful for session {session_id}: {client_info.name} v{client_info.version} -> {negotiated_version}")
            return response
            
        except Exception as e:
            logger.error(f"Initialize request error for session {session_id}: {e}")
            session.state = InitializationState.NOT_INITIALIZED
            return self._create_error_response(
                message.get("id"),
                -32603,  # Internal error
                f"Initialization failed: {e}"
            )
    
    async def handle_initialized_notification(self, message: Dict[str, Any], session_id: str) -> Optional[Dict[str, Any]]:
        """Handle initialized notification from client.
        
        Args:
            message: JSON-RPC initialized notification
            session_id: Session identifier
            
        Returns:
            None (notifications don't have responses)
        """
        # Validate JSON-RPC format
        validation_result = self.validator.validate_message(message)
        if not validation_result.valid:
            logger.error(f"Invalid initialized notification from session {session_id}: {validation_result.errors}")
            return None
        
        # Check session state
        if session_id not in self.sessions:
            logger.error(f"Initialized notification for unknown session: {session_id}")
            return None
        
        session = self.sessions[session_id]
        if session.state != InitializationState.INITIALIZING:
            logger.error(f"Initialized notification for session {session_id} in wrong state: {session.state.value}")
            return None
        
        # Complete initialization
        session.state = InitializationState.INITIALIZED
        session.initialized_at = time.time()
        
        logger.info(f"Session {session_id} fully initialized")
        
        # Call initialization handler if configured
        if self.initialized_handler:
            try:
                await self.initialized_handler(session)
            except Exception as e:
                logger.error(f"Initialization handler error for session {session_id}: {e}")
        
        return None
    
    def _negotiate_version(self, client_version: str) -> Optional[str]:
        """Negotiate protocol version with client.
        
        Args:
            client_version: Version requested by client
            
        Returns:
            Negotiated version or None if incompatible
        """
        # Exact match preferred
        if client_version in self.supported_versions:
            return client_version
        
        # Version compatibility logic
        # For now, only support exact matches
        # Future: implement semantic version compatibility
        
        # Return highest supported version as fallback
        if self.supported_versions:
            return self.supported_versions[0]
        
        return None
    
    def _parse_client_capabilities(self, caps_data: Dict[str, Any]) -> ClientCapabilities:
        """Parse client capabilities from request data.
        
        Args:
            caps_data: Capabilities data from request
            
        Returns:
            Parsed client capabilities
            
        Raises:
            ValueError: If capabilities are invalid
        """
        capabilities = ClientCapabilities()
        
        # Parse roots capability
        if "roots" in caps_data:
            roots_cap = caps_data["roots"]
            if not isinstance(roots_cap, dict):
                raise ValueError("roots capability must be an object")
            capabilities.roots = roots_cap
        
        # Parse sampling capability
        if "sampling" in caps_data:
            sampling_cap = caps_data["sampling"]
            if not isinstance(sampling_cap, dict):
                raise ValueError("sampling capability must be an object")
            capabilities.sampling = sampling_cap
        
        # Parse elicitation capability
        if "elicitation" in caps_data:
            elicitation_cap = caps_data["elicitation"]
            if not isinstance(elicitation_cap, dict):
                raise ValueError("elicitation capability must be an object")
            capabilities.elicitation = elicitation_cap
        
        return capabilities
    
    def _parse_implementation_info(self, info_data: Dict[str, Any]) -> ImplementationInfo:
        """Parse implementation info from request data.
        
        Args:
            info_data: Implementation info data
            
        Returns:
            Parsed implementation info
            
        Raises:
            ValueError: If info is invalid
        """
        if "name" not in info_data:
            raise ValueError("Missing required field: name")
        
        name = info_data["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        
        version = info_data.get("version", "unknown")
        if not isinstance(version, str):
            raise ValueError("version must be a string")
        
        description = info_data.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("description must be a string")
        
        homepage = info_data.get("homepage")
        if homepage is not None and not isinstance(homepage, str):
            raise ValueError("homepage must be a string")
        
        license_info = info_data.get("license")
        if license_info is not None and not isinstance(license_info, str):
            raise ValueError("license must be a string")
        
        return ImplementationInfo(
            name=name.strip(),
            version=version,
            description=description,
            homepage=homepage,
            license=license_info
        )
    
    def _serialize_server_capabilities(self) -> Dict[str, Any]:
        """Serialize server capabilities for response.
        
        Returns:
            Serialized capabilities dictionary
        """
        capabilities = {}
        
        if self.server_capabilities.tools:
            capabilities["tools"] = self.server_capabilities.tools
        
        if self.server_capabilities.resources:
            capabilities["resources"] = self.server_capabilities.resources
        
        if self.server_capabilities.prompts:
            capabilities["prompts"] = self.server_capabilities.prompts
        
        if self.server_capabilities.completion:
            capabilities["completion"] = self.server_capabilities.completion
        
        if self.server_capabilities.logging:
            capabilities["logging"] = self.server_capabilities.logging
        
        return capabilities
    
    def _serialize_implementation_info(self, info: ImplementationInfo) -> Dict[str, Any]:
        """Serialize implementation info for response.
        
        Args:
            info: Implementation info to serialize
            
        Returns:
            Serialized info dictionary
        """
        result = {
            "name": info.name,
            "version": info.version
        }
        
        if info.description:
            result["description"] = info.description
        
        if info.homepage:
            result["homepage"] = info.homepage
        
        if info.license:
            result["license"] = info.license
        
        return result
    
    def _create_error_response(self, request_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create JSON-RPC error response.
        
        Args:
            request_id: Request ID (can be None for parse errors)
            code: Error code
            message: Error message
            data: Optional error data
            
        Returns:
            JSON-RPC error response
        """
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        
        if data is not None:
            error_response["error"]["data"] = data
        
        return error_response
    
    def get_session(self, session_id: str) -> Optional[InitializationSession]:
        """Get initialization session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def is_session_initialized(self, session_id: str) -> bool:
        """Check if session is fully initialized.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session is initialized
        """
        session = self.sessions.get(session_id)
        return session is not None and session.state == InitializationState.INITIALIZED
    
    def get_negotiated_version(self, session_id: str) -> Optional[str]:
        """Get negotiated protocol version for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Negotiated version or None if not negotiated
        """
        session = self.sessions.get(session_id)
        return session.negotiated_version if session else None
    
    def shutdown_session(self, session_id: str) -> bool:
        """Shutdown initialization session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was shutdown
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.state = InitializationState.SHUTDOWN
            logger.info(f"Session {session_id} shutdown")
            return True
        return False
    
    def cleanup_expired_sessions(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired sessions.
        
        Args:
            max_age_seconds: Maximum session age in seconds
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.created_at > max_age_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        return len(expired_sessions)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get initialization session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        state_counts = {}
        for session in self.sessions.values():
            state = session.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            "total_sessions": len(self.sessions),
            "state_counts": state_counts,
            "supported_versions": self.supported_versions,
            "server_info": self._serialize_implementation_info(self.server_info)
        }


def create_default_initialization_manager(server_name: str, 
                                        server_version: str,
                                        tools_enabled: bool = True,
                                        resources_enabled: bool = True,
                                        prompts_enabled: bool = False) -> MCPInitializationManager:
    """Create default MCP initialization manager.
    
    Args:
        server_name: Server name
        server_version: Server version
        tools_enabled: Whether to enable tools capability
        resources_enabled: Whether to enable resources capability
        prompts_enabled: Whether to enable prompts capability
        
    Returns:
        Configured initialization manager
    """
    server_info = ImplementationInfo(
        name=server_name,
        version=server_version,
        description=f"{server_name} MCP Server",
        homepage="https://github.com/yourusername/your-mcp-server"
    )
    
    server_capabilities = ServerCapabilities()
    
    if tools_enabled:
        server_capabilities.tools = {"listChanged": True}
    
    if resources_enabled:
        server_capabilities.resources = {
            "subscribe": True,
            "listChanged": True
        }
    
    if prompts_enabled:
        server_capabilities.prompts = {"listChanged": True}
    
    # Always enable logging
    server_capabilities.logging = {}
    
    return MCPInitializationManager(
        server_info=server_info,
        server_capabilities=server_capabilities
    )


# Example usage for integration testing
if __name__ == "__main__":
    import asyncio
    
    async def test_initialization():
        """Test initialization flow."""
        manager = create_default_initialization_manager(
            "test-server",
            "1.0.0"
        )
        
        # Test initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        response = await manager.handle_initialize_request(initialize_request, "test-session")
        print("Initialize response:", json.dumps(response, indent=2))
        
        # Test initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        await manager.handle_initialized_notification(initialized_notification, "test-session")
        
        # Check session state
        session = manager.get_session("test-session")
        print(f"Session state: {session.state.value}")
        print(f"Negotiated version: {session.negotiated_version}")
        
        # Get stats
        stats = manager.get_session_stats()
        print("Manager stats:", json.dumps(stats, indent=2))
    
    asyncio.run(test_initialization())