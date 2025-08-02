"""MCP Security Framework - OAuth 2.1 Resource Server Implementation

This module implements OAuth 2.1 compliance and security features for MCP integration
as required by the 2025-06-18 MCP specification.
"""

import asyncio
import json
import secrets
import time
from typing import Dict, Any, Optional, List, Union
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass
from enum import Enum
import hashlib
import base64
import logging
from pathlib import Path

try:
    import jwt
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenError(Exception):
    """Base exception for token-related errors."""
    pass


class InvalidTokenError(TokenError):
    """Raised when token validation fails."""
    pass


class ConfusedDeputyError(TokenError):
    """Raised when token audience validation fails (confused deputy attack prevention)."""
    pass


class AuthorizationScope(Enum):
    """OAuth 2.1 authorization scopes for MCP operations."""
    TOOLS_READ = "mcp:tools:read"
    TOOLS_EXECUTE = "mcp:tools:execute"
    RESOURCES_READ = "mcp:resources:read"
    PROMPTS_READ = "mcp:prompts:read"
    ADMIN = "mcp:admin"


@dataclass
class TokenClaims:
    """OAuth 2.1 token claims structure."""
    sub: str  # Subject (user ID)
    aud: str  # Audience (resource server identifier)
    iss: str  # Issuer (authorization server)
    exp: int  # Expiration time
    iat: int  # Issued at time
    scope: List[str]  # Authorized scopes
    resource: Optional[str] = None  # Resource Indicators (RFC 8707)
    client_id: Optional[str] = None  # Client identifier


@dataclass
class ResourceServerConfig:
    """Configuration for OAuth 2.1 resource server."""
    server_identifier: str  # Unique identifier for this resource server
    authorization_server_url: str  # OAuth 2.1 authorization server URL  
    issuer: str  # Expected token issuer
    jwks_uri: Optional[str] = None  # JSON Web Key Set URI
    public_key: Optional[str] = None  # Public key for token verification
    required_scopes: List[str] = None  # Required scopes for access
    token_max_age: int = 3600  # Maximum token age in seconds


class OAuth21ResourceServer:
    """OAuth 2.1 Resource Server implementation for MCP compliance."""
    
    def __init__(self, config: ResourceServerConfig):
        """Initialize OAuth 2.1 resource server.
        
        Args:
            config: Resource server configuration
        """
        if not JWT_AVAILABLE:
            raise ImportError("JWT dependencies required: pip install pyjwt cryptography")
        
        self.config = config
        self.jwks_cache: Dict[str, Any] = {}
        self.jwks_cache_time: float = 0
        self.jwks_cache_ttl: int = 3600  # 1 hour
        
        # Initialize Resource Indicators validator (RFC 8707)
        self.resource_validator = ResourceIndicatorsValidator(config.server_identifier)
        
    async def validate_token(self, token: str) -> TokenClaims:
        """Validate OAuth 2.1 access token.
        
        Args:
            token: Bearer access token
            
        Returns:
            Validated token claims
            
        Raises:
            InvalidTokenError: If token is invalid
            ConfusedDeputyError: If audience validation fails
        """
        try:
            # Decode token header to get key ID
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')
            
            # Get public key for verification
            public_key = await self._get_public_key(kid)
            
            # Verify and decode token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=['RS256', 'ES256'],
                audience=self.config.server_identifier,
                issuer=self.config.issuer,
                options={"verify_exp": True, "verify_aud": True}
            )
            
            # Validate token claims
            claims = self._validate_claims(payload)
            
            # Check for confused deputy attack (audience validation)
            if claims.aud != self.config.server_identifier:
                raise ConfusedDeputyError(
                    f"Token audience mismatch: expected {self.config.server_identifier}, "
                    f"got {claims.aud}"
                )
                
            # Validate Resource Indicators (RFC 8707) - critical for MCP compliance
            if not self.resource_validator.validate_resource_binding(claims):
                raise ConfusedDeputyError(
                    "Token failed Resource Indicators validation - confused deputy attack prevented"
                )
            
            # Check token age
            current_time = int(time.time())
            token_age = current_time - claims.iat
            if token_age > self.config.token_max_age:
                raise InvalidTokenError(f"Token too old: {token_age}s > {self.config.token_max_age}s")
            
            return claims
            
        except jwt.ExpiredSignatureError:
            raise InvalidTokenError("Token has expired")
        except jwt.InvalidAudienceError:
            raise ConfusedDeputyError("Invalid token audience")
        except jwt.InvalidIssuerError:
            raise InvalidTokenError("Invalid token issuer")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Token validation failed: {e}")
    
    async def _get_public_key(self, kid: Optional[str] = None) -> str:
        """Get public key for token verification.
        
        Args:
            kid: Key ID from token header
            
        Returns:
            Public key for verification
        """
        if self.config.public_key:
            return self.config.public_key
            
        if self.config.jwks_uri:
            return await self._fetch_jwks_key(kid)
            
        raise InvalidTokenError("No public key configured for token verification")
    
    async def _fetch_jwks_key(self, kid: Optional[str]) -> str:
        """Fetch public key from JWKS endpoint.
        
        Args:
            kid: Key ID to fetch
            
        Returns:
            Public key for verification
        """
        current_time = time.time()
        
        # Check cache
        if (current_time - self.jwks_cache_time < self.jwks_cache_ttl and 
            kid in self.jwks_cache):
            return self.jwks_cache[kid]
        
        # Fetch JWKS (in production, use proper HTTP client)
        # This is a placeholder - implement proper JWKS fetching
        logger.warning("JWKS fetching not implemented - using placeholder")
        
        # Placeholder key for development
        self.jwks_cache[kid or 'default'] = "placeholder_key"
        self.jwks_cache_time = current_time
        
        return self.jwks_cache[kid or 'default']
    
    def _validate_claims(self, payload: Dict[str, Any]) -> TokenClaims:
        """Validate token claims structure.
        
        Args:
            payload: Decoded JWT payload
            
        Returns:
            Validated token claims
        """
        required_fields = ['sub', 'aud', 'iss', 'exp', 'iat']
        for field in required_fields:
            if field not in payload:
                raise InvalidTokenError(f"Missing required claim: {field}")
        
        # Parse scopes
        scope_str = payload.get('scope', '')
        scopes = scope_str.split() if isinstance(scope_str, str) else []
        
        return TokenClaims(
            sub=payload['sub'],
            aud=payload['aud'],
            iss=payload['iss'],
            exp=payload['exp'],
            iat=payload['iat'],
            scope=scopes,
            resource=payload.get('resource'),
            client_id=payload.get('client_id')
        )
    
    def check_scope(self, claims: TokenClaims, required_scope: AuthorizationScope) -> bool:
        """Check if token has required scope.
        
        Args:
            claims: Validated token claims
            required_scope: Required authorization scope
            
        Returns:
            True if token has required scope
        """
        return required_scope.value in claims.scope
    
    def generate_www_authenticate_header(self, error: str = "invalid_token") -> str:
        """Generate WWW-Authenticate header for 401 responses.
        
        Args:
            error: OAuth 2.1 error code
            
        Returns:
            WWW-Authenticate header value
        """
        return (
            f'Bearer realm="{self.config.server_identifier}", '
            f'error="{error}", '
            f'error_description="The access token is invalid"'
        )
    
    def create_authorization_url(self, client_id: str, redirect_uri: str, 
                                scope: str, state: str) -> str:
        """Create Resource Indicators-compliant authorization URL.
        
        Args:
            client_id: OAuth client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL with resource parameter (RFC 8707)
        """
        return self.resource_validator.create_authorization_url(
            self.config.authorization_server_url,
            client_id,
            redirect_uri,
            scope,
            state
        )
    
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
        return self.resource_validator.create_token_request_params(
            code, client_id, redirect_uri, code_verifier
        )
    
    def validate_token_response(self, token_response: Dict[str, Any]) -> bool:
        """Validate token response includes proper resource binding.
        
        Args:
            token_response: Token response from authorization server
            
        Returns:
            True if token is properly bound to this resource
        """
        return self.resource_validator.validate_token_response(token_response)


class ResourceIndicatorsValidator:
    """RFC 8707 Resource Indicators implementation for MCP compliance."""
    
    def __init__(self, server_identifier: str):
        """Initialize Resource Indicators validator.
        
        Args:
            server_identifier: This resource server's identifier
        """
        self.server_identifier = server_identifier
        self.allowed_resource_identifiers = {server_identifier}
    
    def add_allowed_resource(self, resource_identifier: str) -> None:
        """Add an allowed resource identifier.
        
        Args:
            resource_identifier: Resource identifier to allow
        """
        self.allowed_resource_identifiers.add(resource_identifier)
    
    def validate_authorization_request(self, request_params: Dict[str, Any]) -> bool:
        """Validate authorization request includes resource parameter.
        
        Per RFC 8707, MCP clients MUST include resource parameter to prevent
        confused deputy attacks.
        
        Args:
            request_params: Authorization request parameters
            
        Returns:
            True if request includes valid resource parameter
        """
        resource = request_params.get('resource')
        if not resource:
            logger.error("Authorization request missing resource parameter (RFC 8707 violation)")
            return False
            
        # Resource can be single string or list of strings
        resources = resource if isinstance(resource, list) else [resource]
        
        # Validate each resource identifier
        for res in resources:
            if res not in self.allowed_resource_identifiers:
                logger.error(f"Invalid resource parameter: {res} not in allowed resources")
                return False
                
        logger.info(f"Resource Indicators validation passed for: {resources}")
        return True
    
    def create_authorization_url(self, auth_server_url: str, client_id: str, 
                                redirect_uri: str, scope: str, state: str) -> str:
        """Create authorization URL with Resource Indicators.
        
        Args:
            auth_server_url: Authorization server URL
            client_id: OAuth client ID
            redirect_uri: Redirect URI after authorization
            scope: Requested scopes
            state: State parameter for CSRF protection
            
        Returns:
            Authorization URL with resource parameter
        """
        from urllib.parse import urlencode, urljoin
        
        params = {
            'response_type': 'code',
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'state': state,
            'resource': self.server_identifier  # RFC 8707 requirement
        }
        
        query_string = urlencode(params)
        return f"{auth_server_url}/authorize?{query_string}"
    
    def create_token_request_params(self, code: str, client_id: str, 
                                   redirect_uri: str, code_verifier: Optional[str] = None) -> Dict[str, str]:
        """Create token request parameters with Resource Indicators.
        
        Args:
            code: Authorization code
            client_id: OAuth client ID
            redirect_uri: Redirect URI used in authorization
            code_verifier: PKCE code verifier
            
        Returns:
            Token request parameters with resource parameter
        """
        params = {
            'grant_type': 'authorization_code',
            'code': code,
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'resource': self.server_identifier  # RFC 8707 requirement
        }
        
        if code_verifier:
            params['code_verifier'] = code_verifier
            
        return params
    
    def validate_token_response(self, token_response: Dict[str, Any]) -> bool:
        """Validate token response includes proper resource binding.
        
        Args:
            token_response: Token response from authorization server
            
        Returns:
            True if token is properly bound to resource
        """
        # Check if token includes resource claim
        access_token = token_response.get('access_token')
        if not access_token:
            logger.error("Token response missing access_token")
            return False
            
        try:
            # Decode token header to check if it's a JWT
            header = jwt.get_unverified_header(access_token)
            if header:
                # If it's a JWT, decode claims to validate resource binding
                payload = jwt.decode(access_token, options={"verify_signature": False})
                resource_claim = payload.get('resource')
                
                if resource_claim and resource_claim != self.server_identifier:
                    logger.error(f"Token resource claim mismatch: {resource_claim} != {self.server_identifier}")
                    return False
                    
                logger.info("Token resource binding validated")
                return True
        except Exception as e:
            logger.warning(f"Could not decode token for resource validation: {e}")
            # For opaque tokens, we'll validate during introspection
            return True
            
        return True
    
    def extract_resource_from_token(self, claims: TokenClaims) -> Optional[str]:
        """Extract resource claim from token.
        
        Args:
            claims: Token claims
            
        Returns:
            Resource claim value if present
        """
        return claims.resource
    
    def validate_resource_binding(self, claims: TokenClaims) -> bool:
        """Validate token is bound to correct resource.
        
        Args:
            claims: Token claims to validate
            
        Returns:
            True if token is properly bound to this resource
        """
        if not claims.resource:
            logger.warning("Token missing resource claim - potentially vulnerable to confused deputy attacks")
            return False
            
        if claims.resource != self.server_identifier:
            logger.error(f"Token resource binding violation: {claims.resource} != {self.server_identifier}")
            return False
            
        return True


class SessionManager:
    """Secure session management for MCP connections."""
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout: int = 3600  # 1 hour
    
    def create_session(self, user_id: str, client_id: str) -> str:
        """Create secure session ID.
        
        Args:
            user_id: User identifier
            client_id: Client identifier
            
        Returns:
            Secure session ID
        """
        # Generate cryptographically secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Bind session to user information
        self.sessions[session_id] = {
            'user_id': user_id,
            'client_id': client_id,
            'created_at': time.time(),
            'last_access': time.time()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and update last access time.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data if valid, None otherwise
        """
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['last_access'] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # Update last access time
        session['last_access'] = current_time
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was revoked
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['last_access'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Extract Bearer token from Authorization header.
    
    Args:
        authorization_header: Authorization header value
        
    Returns:
        Bearer token if present and valid format
    """
    if not authorization_header:
        return None
    
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]


def create_default_resource_server_config() -> ResourceServerConfig:
    """Create default resource server configuration for development.
    
    Returns:
        Default configuration
    """
    return ResourceServerConfig(
        server_identifier="https://deepagents.local/mcp",
        authorization_server_url="https://auth.deepagents.local",
        issuer="https://auth.deepagents.local",
        required_scopes=[AuthorizationScope.TOOLS_READ.value]
    )