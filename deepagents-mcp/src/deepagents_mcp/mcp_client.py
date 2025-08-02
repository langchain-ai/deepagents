"""MCP client integration for DeepAgents.

This module allows DeepAgents to use MCP servers as additional tools,
leveraging the langchain-mcp-adapters library with OAuth 2.1 security.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
import logging

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MultiServerMCPClient = None

# Import security framework
try:
    from .security import (
        OAuth21ResourceServer,
        ResourceServerConfig,
        TokenClaims,
        AuthorizationScope,
        SessionManager,
        ResourceIndicatorsValidator,
        extract_bearer_token,
        InvalidTokenError,
        ConfusedDeputyError
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Import consent framework
try:
    from .consent import (
        ConsentManager,
        ConsentRequest,
        ConsentDecision,
        ConsentScope,
        RiskLevel,
        get_consent_manager,
        format_consent_request_for_display,
        create_consent_prompt
    )
    CONSENT_AVAILABLE = True
except ImportError:
    CONSENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """Provider for loading tools from MCP servers with OAuth 2.1 security."""
    
    def __init__(self, connections: Optional[Dict[str, Any]] = None, 
                 security_config: Optional[ResourceServerConfig] = None,
                 enable_security: bool = False,
                 enable_consent: bool = True,
                 consent_manager: Optional[ConsentManager] = None):
        """Initialize MCP tool provider.
        
        Args:
            connections: Dictionary mapping server names to connection configs.
                Example:
                {
                    "calculator": {
                        "command": "python",
                        "args": ["/path/to/calculator_server.py"],
                        "transport": "stdio",
                        "auth": {
                            "type": "oauth2.1",
                            "token": "access_token_here"
                        }
                    },
                    "weather": {
                        "url": "http://localhost:8000/mcp",
                        "transport": "streamable_http",
                        "auth": {
                            "type": "oauth2.1",
                            "authorization_header": "Bearer token_here"
                        }
                    }
                }
            security_config: OAuth 2.1 resource server configuration
            enable_security: Whether to enable OAuth 2.1 security features
            enable_consent: Whether to require user consent for tool execution
            consent_manager: Optional consent manager instance
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "langchain-mcp-adapters not available. "
                "Install with: pip install langchain-mcp-adapters"
            )
        
        self.connections = connections or {}
        self.client = MultiServerMCPClient(self.connections) if self.connections else None
        self._cached_tools: Optional[List[BaseTool]] = None
        
        # Security configuration
        self.enable_security = enable_security and SECURITY_AVAILABLE
        if self.enable_security:
            if not security_config:
                from .security import create_default_resource_server_config
                security_config = create_default_resource_server_config()
            self.oauth_server = OAuth21ResourceServer(security_config)
            self.session_manager = SessionManager()
            logger.info("OAuth 2.1 security enabled for MCP client")
        else:
            self.oauth_server = None
            self.session_manager = None
            if enable_security and not SECURITY_AVAILABLE:
                logger.warning("Security requested but security dependencies not available")
    
    def add_server(self, name: str, connection: Dict[str, Any]) -> None:
        """Add a new MCP server connection.
        
        Args:
            name: Name to identify the server
            connection: Connection configuration dict
        """
        self.connections[name] = connection
        if self.client is None:
            self.client = MultiServerMCPClient(self.connections)
        else:
            self.client.connections[name] = connection
        
        # Clear cache to force reload
        self._cached_tools = None
    
    async def get_tools(self, server_name: Optional[str] = None, force_reload: bool = False,
                       authorization_header: Optional[str] = None) -> List[BaseTool]:
        """Get tools from MCP servers with optional authorization.
        
        Args:
            server_name: Optional specific server to get tools from
            force_reload: Force reload tools even if cached
            authorization_header: Optional Bearer token for authorization
            
        Returns:
            List of LangChain tools from MCP servers
        """
        if not self.client:
            return []
        
        # Validate authorization if security is enabled
        if self.enable_security and authorization_header:
            try:
                token = extract_bearer_token(authorization_header)
                if token:
                    claims = await self.oauth_server.validate_token(token)
                    # Check if token has permission to read tools
                    if not self.oauth_server.check_scope(claims, AuthorizationScope.TOOLS_READ):
                        logger.warning("Token lacks required scope for tools access")
                        return []
                    logger.info(f"Authorized tools access for user: {claims.sub}")
            except (InvalidTokenError, ConfusedDeputyError) as e:
                logger.error(f"Authorization failed: {e}")
                return []
        elif self.enable_security:
            logger.info("Security enabled but no authorization header provided - allowing unauthenticated access")
        
        if not force_reload and self._cached_tools is not None and server_name is None:
            return self._cached_tools
        
        try:
            tools = await self.client.get_tools(server_name=server_name)
            
            # Wrap tools with authorization checks if security enabled
            if self.enable_security:
                tools = [self._wrap_tool_with_auth(tool, authorization_header) for tool in tools]
            
            # Cache if getting all tools
            if server_name is None:
                self._cached_tools = tools
            
            return tools
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            return []
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Optional[Dict[str, Any]] = None):
        """Get a prompt from an MCP server."""
        if not self.client:
            return None
        
        try:
            return await self.client.get_prompt(server_name, prompt_name, arguments=arguments)
        except Exception as e:
            print(f"Warning: Failed to get prompt '{prompt_name}' from '{server_name}': {e}")
            return None
    
    async def get_resources(self, server_name: str, uris: Optional[List[str]] = None):
        """Get resources from an MCP server."""
        if not self.client:
            return []
        
        try:
            return await self.client.get_resources(server_name, uris=uris)
        except Exception as e:
            print(f"Warning: Failed to get resources from '{server_name}': {e}")
            return []
    
    def list_servers(self) -> List[str]:
        """List configured MCP server names."""
        return list(self.connections.keys())
    
    def _wrap_tool_with_auth(self, tool: BaseTool, authorization_header: Optional[str]) -> BaseTool:
        """Wrap a tool with authorization checks.
        
        Args:
            tool: Original LangChain tool
            authorization_header: Authorization header for token validation
            
        Returns:
            Tool wrapped with authorization checks
        """
        from langchain_core.tools import tool as create_tool
        from functools import wraps
        import asyncio
        
        original_run = tool._run if hasattr(tool, '_run') else None
        original_arun = tool._arun if hasattr(tool, '_arun') else None
        
        @wraps(original_run if original_run else original_arun)
        async def authorized_run(*args, **kwargs):
            """Run tool with authorization and consent checks."""
            user_id = "anonymous"
            client_id = "unknown"
            
            if self.enable_security:
                try:
                    if authorization_header:
                        token = extract_bearer_token(authorization_header)
                        if token:
                            claims = await self.oauth_server.validate_token(token)
                            # Check if token has permission to execute tools
                            if not self.oauth_server.check_scope(claims, AuthorizationScope.TOOLS_EXECUTE):
                                return f"Error: Insufficient permissions to execute tool '{tool.name}'"
                            logger.info(f"Authorized tool execution '{tool.name}' for user: {claims.sub}")
                            user_id = claims.sub
                            client_id = claims.client_id or "unknown"
                        else:
                            return f"Error: Invalid authorization header for tool '{tool.name}'"
                    else:
                        logger.warning(f"Tool '{tool.name}' executed without authorization")
                except (InvalidTokenError, ConfusedDeputyError) as e:
                    return f"Error: Authorization failed for tool '{tool.name}': {e}"
            
            # Check user consent if consent framework is enabled
            if self.enable_consent and CONSENT_AVAILABLE:
                try:
                    consent_manager = self.consent_manager or get_consent_manager()
                    
                    # Create consent request
                    consent_request = await consent_manager.request_consent(
                        tool_name=tool.name,
                        tool_description=tool.description or f"Execute {tool.name} tool",
                        parameters=kwargs,
                        user_id=user_id,
                        client_id=client_id,
                        justification=f"User requested execution of {tool.name} tool",
                        predicted_effects=[f"Execute {tool.name} with provided parameters"],
                        data_access_description="Tool may access data as specified in its parameters"
                    )
                    
                    # If consent is not automatically approved, request user interaction
                    if consent_request.decision != ConsentDecision.APPROVED:
                        # In a real implementation, this would trigger UI or API callback
                        # For now, we'll simulate immediate approval for low-risk tools
                        if consent_request.risk_level == RiskLevel.LOW:
                            await consent_manager.provide_consent(
                                consent_request.request_id,
                                ConsentDecision.APPROVED,
                                ConsentScope.SESSION
                            )
                            logger.info(f"Auto-approved low-risk tool '{tool.name}' for user '{user_id}'")
                        else:
                            # For higher risk tools, deny by default and require explicit user approval
                            logger.warning(f"Tool '{tool.name}' requires explicit user consent (risk: {consent_request.risk_level.value})")
                            return f"Error: Tool '{tool.name}' requires user consent. Risk level: {consent_request.risk_level.value}. Please approve this tool execution."
                    
                    # Check final consent decision
                    if consent_request.decision != ConsentDecision.APPROVED:
                        logger.info(f"Tool execution '{tool.name}' denied by user consent")
                        return f"Error: User consent denied for tool '{tool.name}'"
                        
                    logger.info(f"User consent approved for tool '{tool.name}' execution")
                    
                except Exception as e:
                    logger.error(f"Consent check failed for tool '{tool.name}': {e}")
                    return f"Error: Consent validation failed for tool '{tool.name}': {e}"
            
            # Execute original tool
            if original_arun:
                return await original_arun(*args, **kwargs)
            elif original_run:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: original_run(*args, **kwargs))
            else:
                return f"Error: Tool '{tool.name}' has no execution method"
        
        # Create new tool with authorization wrapper
        return create_tool(
            func=authorized_run,
            name=tool.name,
            description=f"[AUTH REQUIRED] {tool.description}",
            args_schema=tool.args_schema
        )
    
    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session ID if security is enabled.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if valid, None otherwise
        """
        if not self.enable_security or not self.session_manager:
            return {"valid": True}  # Allow all sessions if security disabled
        
        return self.session_manager.validate_session(session_id)
    
    async def create_session(self, user_id: str, client_id: str) -> str:
        """Create secure session if security is enabled.
        
        Args:
            user_id: User identifier
            client_id: Client identifier
            
        Returns:
            Session ID
        """
        if not self.enable_security or not self.session_manager:
            return f"unsecured-session-{user_id}"  # Generate placeholder session
            
        return self.session_manager.create_session(user_id, client_id)
    
    def generate_auth_challenge(self, error: str = "invalid_token") -> str:
        """Generate WWW-Authenticate header for 401 responses.
        
        Args:
            error: OAuth 2.1 error code
            
        Returns:
            WWW-Authenticate header value
        """
        if not self.enable_security or not self.oauth_server:
            return 'Bearer realm="mcp-client"'
            
        return self.oauth_server.generate_www_authenticate_header(error)
    
    def create_authorization_url(self, client_id: str, redirect_uri: str, 
                                scope: str = "mcp:tools:read mcp:tools:execute", 
                                state: Optional[str] = None) -> Optional[str]:
        """Create Resource Indicators-compliant authorization URL.
        
        Args:
            client_id: OAuth client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes (default: basic MCP scopes)
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL with resource parameter (RFC 8707) or None if security disabled
        """
        if not self.enable_security or not self.oauth_server:
            logger.warning("Cannot create authorization URL - security not enabled")
            return None
            
        if not state:
            import secrets
            state = secrets.token_urlsafe(32)
            
        return self.oauth_server.create_authorization_url(client_id, redirect_uri, scope, state)
    
    def create_token_request_params(self, code: str, client_id: str, 
                                   redirect_uri: str, code_verifier: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Create Resource Indicators-compliant token request parameters.
        
        Args:
            code: Authorization code
            client_id: OAuth client ID
            redirect_uri: Redirect URI used in authorization
            code_verifier: PKCE code verifier
            
        Returns:
            Token request parameters with resource parameter (RFC 8707) or None if security disabled
        """
        if not self.enable_security or not self.oauth_server:
            logger.warning("Cannot create token request - security not enabled")
            return None
            
        return self.oauth_server.create_token_request_params(code, client_id, redirect_uri, code_verifier)
    
    def validate_token_response(self, token_response: Dict[str, Any]) -> bool:
        """Validate token response includes proper resource binding.
        
        Args:
            token_response: Token response from authorization server
            
        Returns:
            True if token is properly bound to this resource, or True if security disabled
        """
        if not self.enable_security or not self.oauth_server:
            return True  # Allow all if security disabled
            
        return self.oauth_server.validate_token_response(token_response)


async def load_mcp_tools(connections: Optional[Dict[str, Any]] = None,
                        security_config: Optional[ResourceServerConfig] = None,
                        enable_security: bool = False,
                        authorization_header: Optional[str] = None) -> List[BaseTool]:
    """Convenience function to load MCP tools with optional security.
    
    Args:
        connections: MCP server connections configuration
        security_config: OAuth 2.1 resource server configuration
        enable_security: Whether to enable OAuth 2.1 security features
        authorization_header: Optional Bearer token for authorization
        
    Returns:
        List of tools from all configured MCP servers
    """
    if not connections:
        return []
    
    provider = MCPToolProvider(connections, security_config, enable_security)
    return await provider.get_tools(authorization_header=authorization_header)


def create_mcp_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load MCP configuration from a file.
    
    Args:
        config_path: Path to JSON or YAML config file
        
    Returns:
        Configuration dictionary for MCP servers
    """
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"MCP config file not found: {config_path}")
    
    with open(config_file) as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")


# Default MCP configurations for common servers
DEFAULT_MCP_CONFIGS = {
    "example_stdio": {
        "command": "python",
        "args": ["-m", "mcp_server"],  # Placeholder
        "transport": "stdio"
    },
    "example_http": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http"
    },
    "secure_http": {
        "url": "https://secure-mcp-server.example.com/mcp",
        "transport": "streamable_http",
        "auth": {
            "type": "oauth2.1",
            "authorization_server": "https://auth.example.com",
            "client_id": "your_client_id",
            "scopes": ["mcp:tools:read", "mcp:tools:execute"]
        }
    }
}


class SecureMCPClient:
    """High-level MCP client with built-in security features.
    
    This class provides a simplified interface for creating MCP clients
    with OAuth 2.1 security enabled by default.
    """
    
    def __init__(self, connections: Dict[str, Any], 
                 authorization_server_url: str,
                 server_identifier: str = "https://deepagents.local/mcp"):
        """Initialize secure MCP client.
        
        Args:
            connections: MCP server connections
            authorization_server_url: OAuth 2.1 authorization server URL
            server_identifier: This resource server's identifier
        """
        if not SECURITY_AVAILABLE:
            raise ImportError("Security dependencies required for SecureMCPClient")
            
        from .security import ResourceServerConfig
        
        # Create security configuration
        security_config = ResourceServerConfig(
            server_identifier=server_identifier,
            authorization_server_url=authorization_server_url,
            issuer=authorization_server_url,
            required_scopes=["mcp:tools:read"]
        )
        
        # Initialize provider with security enabled
        self.provider = MCPToolProvider(
            connections=connections,
            security_config=security_config,
            enable_security=True
        )
    
    async def get_tools(self, authorization_header: str) -> List[BaseTool]:
        """Get tools with authorization.
        
        Args:
            authorization_header: Bearer token authorization header
            
        Returns:
            List of authorized tools
        """
        return await self.provider.get_tools(authorization_header=authorization_header)
    
    async def create_session(self, user_id: str, client_id: str) -> str:
        """Create secure session.
        
        Args:
            user_id: User identifier
            client_id: Client identifier
            
        Returns:
            Session ID
        """
        return await self.provider.create_session(user_id, client_id)
    
    def get_auth_challenge(self, error: str = "invalid_token") -> str:
        """Get WWW-Authenticate challenge header.
        
        Args:
            error: OAuth 2.1 error code
            
        Returns:
            WWW-Authenticate header value
        """
        return self.provider.generate_auth_challenge(error)
    
    def create_authorization_url(self, client_id: str, redirect_uri: str, 
                                scope: str = "mcp:tools:read mcp:tools:execute", 
                                state: Optional[str] = None) -> str:
        """Create Resource Indicators-compliant authorization URL.
        
        Args:
            client_id: OAuth client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes (default: basic MCP scopes)
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL with resource parameter (RFC 8707)
        """
        url = self.provider.create_authorization_url(client_id, redirect_uri, scope, state)
        if not url:
            raise RuntimeError("Failed to create authorization URL - security not properly configured")
        return url
    
    def create_token_request_params(self, code: str, client_id: str, 
                                   redirect_uri: str, code_verifier: Optional[str] = None) -> Dict[str, str]:
        """Create Resource Indicators-compliant token request parameters.
        
        Args:
            code: Authorization code
            client_id: OAuth client ID
            redirect_uri: Redirect URI used in authorization
            code_verifier: PKCE code verifier
            
        Returns:
            Token request parameters with resource parameter (RFC 8707)
        """
        params = self.provider.create_token_request_params(code, client_id, redirect_uri, code_verifier)
        if not params:
            raise RuntimeError("Failed to create token request - security not properly configured")
        return params
    
    def validate_token_response(self, token_response: Dict[str, Any]) -> bool:
        """Validate token response includes proper resource binding.
        
        Args:
            token_response: Token response from authorization server
            
        Returns:
            True if token is properly bound to this resource
        """
        return self.provider.validate_token_response(token_response)