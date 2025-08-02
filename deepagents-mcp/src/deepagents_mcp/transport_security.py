"""Transport security for MCP HTTP connections.

This module implements HTTPS enforcement, Origin validation, and transport-level
security measures as required by the MCP 2025-06-18 specification.
"""

import re
import ssl
import urllib.parse
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TransportSecurityError(Exception):
    """Base exception for transport security violations."""
    pass


class OriginValidationError(TransportSecurityError):
    """Raised when Origin header validation fails."""
    pass


class HTTPSRequiredError(TransportSecurityError):
    """Raised when HTTPS is required but not used."""
    pass


class HostBindingError(TransportSecurityError):
    """Raised when server binding violates security policy."""
    pass


class SecurityPolicy(Enum):
    """Transport security policy levels."""
    DEVELOPMENT = "development"    # Relaxed for local development
    STAGING = "staging"           # Moderate security for testing
    PRODUCTION = "production"     # Strict security for production
    PARANOID = "paranoid"        # Maximum security


@dataclass
class TransportSecurityConfig:
    """Configuration for transport security enforcement."""
    security_policy: SecurityPolicy = SecurityPolicy.PRODUCTION
    require_https: bool = True
    allowed_origins: Set[str] = None
    allowed_hosts: Set[str] = None
    require_origin_header: bool = True
    allow_localhost_in_production: bool = False
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    ssl_verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    ssl_minimum_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    hsts_max_age: int = 31536000  # 1 year
    enable_hsts: bool = True
    enable_csp: bool = True
    csp_policy: str = "default-src 'none'; script-src 'none'; object-src 'none';"


class OriginValidator:
    """Validates Origin headers to prevent DNS rebinding and CSRF attacks."""
    
    def __init__(self, config: TransportSecurityConfig):
        """Initialize Origin validator.
        
        Args:
            config: Transport security configuration
        """
        self.config = config
        self.allowed_origins = config.allowed_origins or set()
        
        # Add default safe origins based on policy
        if config.security_policy == SecurityPolicy.DEVELOPMENT:
            self.allowed_origins.update([
                "http://localhost",
                "https://localhost",
                "http://127.0.0.1",
                "https://127.0.0.1"
            ])
        elif config.security_policy == SecurityPolicy.STAGING:
            self.allowed_origins.update([
                "https://staging.example.com",
                "https://test.example.com"
            ])
        
        logger.info(f"Origin validator initialized with {len(self.allowed_origins)} allowed origins")
    
    def add_allowed_origin(self, origin: str) -> None:
        """Add an allowed origin.
        
        Args:
            origin: Origin URL to allow (e.g., "https://example.com")
        """
        if not self._is_valid_origin_format(origin):
            raise ValueError(f"Invalid origin format: {origin}")
        
        # Normalize origin
        normalized = self._normalize_origin(origin)
        self.allowed_origins.add(normalized)
        logger.info(f"Added allowed origin: {normalized}")
    
    def _is_valid_origin_format(self, origin: str) -> bool:
        """Validate origin format.
        
        Args:
            origin: Origin to validate
            
        Returns:
            True if origin format is valid
        """
        try:
            parsed = urllib.parse.urlparse(origin)
            return (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc and
                not parsed.path and  # Origin should not have path
                not parsed.query and
                not parsed.fragment
            )
        except Exception:
            return False
    
    def _normalize_origin(self, origin: str) -> str:
        """Normalize origin URL.
        
        Args:
            origin: Origin to normalize
            
        Returns:
            Normalized origin
        """
        parsed = urllib.parse.urlparse(origin)
        
        # Remove default ports
        if parsed.port:
            if (parsed.scheme == 'http' and parsed.port == 80) or \
               (parsed.scheme == 'https' and parsed.port == 443):
                netloc = parsed.hostname
            else:
                netloc = parsed.netloc
        else:
            netloc = parsed.netloc
        
        return f"{parsed.scheme}://{netloc}"
    
    def validate_origin(self, origin_header: Optional[str], request_host: Optional[str] = None) -> bool:
        """Validate Origin header against allowed origins.
        
        Args:
            origin_header: Origin header value from request
            request_host: Host header value for additional validation
            
        Returns:
            True if origin is valid
            
        Raises:
            OriginValidationError: If origin validation fails
        """
        # Check if Origin header is required
        if self.config.require_origin_header and not origin_header:
            raise OriginValidationError("Origin header is required but missing")
        
        # Allow requests without Origin header in development mode
        if not origin_header:
            if self.config.security_policy == SecurityPolicy.DEVELOPMENT:
                logger.warning("Origin header missing - allowed in development mode")
                return True
            else:
                raise OriginValidationError("Origin header is required for security")
        
        # Validate origin format
        if not self._is_valid_origin_format(origin_header):
            raise OriginValidationError(f"Invalid origin format: {origin_header}")
        
        # Normalize for comparison
        normalized_origin = self._normalize_origin(origin_header)
        
        # Check against allowed origins
        if normalized_origin not in self.allowed_origins:
            # Additional validation for localhost/development
            if self._is_localhost_origin(normalized_origin):
                if self.config.security_policy == SecurityPolicy.DEVELOPMENT:
                    logger.warning(f"Localhost origin allowed in development: {normalized_origin}")
                    return True
                elif self.config.allow_localhost_in_production:
                    logger.warning(f"Localhost origin allowed by configuration: {normalized_origin}")
                    return True
                else:
                    raise OriginValidationError(f"Localhost origin not allowed in {self.config.security_policy.value} mode")
            
            raise OriginValidationError(f"Origin not allowed: {normalized_origin}")
        
        # Additional Host header validation
        if request_host and self.config.security_policy in [SecurityPolicy.PRODUCTION, SecurityPolicy.PARANOID]:
            self._validate_host_origin_consistency(normalized_origin, request_host)
        
        logger.debug(f"Origin validation passed: {normalized_origin}")
        return True
    
    def _is_localhost_origin(self, origin: str) -> bool:
        """Check if origin is localhost.
        
        Args:
            origin: Normalized origin
            
        Returns:
            True if origin is localhost
        """
        localhost_patterns = [
            r'https?://localhost',
            r'https?://127\.0\.0\.1',
            r'https?://\[::1\]',  # IPv6 localhost
        ]
        
        return any(re.match(pattern, origin) for pattern in localhost_patterns)
    
    def _validate_host_origin_consistency(self, origin: str, host: str) -> None:
        """Validate consistency between Origin and Host headers.
        
        Args:
            origin: Normalized origin
            host: Host header value
            
        Raises:
            OriginValidationError: If headers are inconsistent
        """
        try:
            origin_parsed = urllib.parse.urlparse(origin)
            origin_host = origin_parsed.netloc
            
            # Remove port from host if it's default
            if ':' in host:
                host_name, port = host.split(':', 1)
                if (origin_parsed.scheme == 'https' and port == '443') or \
                   (origin_parsed.scheme == 'http' and port == '80'):
                    host = host_name
            
            if origin_host != host:
                raise OriginValidationError(
                    f"Origin host '{origin_host}' does not match Host header '{host}'"
                )
        except Exception as e:
            if isinstance(e, OriginValidationError):
                raise
            raise OriginValidationError(f"Failed to validate Origin/Host consistency: {e}")


class HTTPSEnforcer:
    """Enforces HTTPS requirements and SSL/TLS security."""
    
    def __init__(self, config: TransportSecurityConfig):
        """Initialize HTTPS enforcer.
        
        Args:
            config: Transport security configuration
        """
        self.config = config
        
    def validate_https_required(self, request_url: str, is_local: bool = False) -> bool:
        """Validate HTTPS requirement.
        
        Args:
            request_url: URL of the request
            is_local: Whether this is a local request
            
        Returns:
            True if HTTPS requirement is satisfied
            
        Raises:
            HTTPSRequiredError: If HTTPS is required but not used
        """
        if not self.config.require_https:
            return True
        
        parsed = urllib.parse.urlparse(request_url)
        
        # Allow HTTP for localhost in development
        if (self.config.security_policy == SecurityPolicy.DEVELOPMENT and 
            is_local and parsed.scheme == 'http'):
            logger.warning("HTTP allowed for localhost in development mode")
            return True
        
        if parsed.scheme != 'https':
            raise HTTPSRequiredError(
                f"HTTPS required but request uses {parsed.scheme}: {request_url}"
            )
        
        return True
    
    def get_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with security requirements.
        
        Returns:
            Configured SSL context
        """
        context = ssl.create_default_context()
        
        # Set minimum TLS version
        context.minimum_version = self.config.ssl_minimum_version
        
        # Set verification mode
        context.verify_mode = self.config.ssl_verify_mode
        
        # Disable weak ciphers in production/paranoid mode
        if self.config.security_policy in [SecurityPolicy.PRODUCTION, SecurityPolicy.PARANOID]:
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses.
        
        Returns:
            Dictionary of security headers
        """
        headers = {}
        
        # HTTP Strict Transport Security
        if self.config.enable_hsts and self.config.require_https:
            headers['Strict-Transport-Security'] = f'max-age={self.config.hsts_max_age}; includeSubDomains'
        
        # Content Security Policy
        if self.config.enable_csp:
            headers['Content-Security-Policy'] = self.config.csp_policy
        
        # Additional security headers
        headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        })
        
        return headers


class HostBindingValidator:
    """Validates server host binding for security."""
    
    def __init__(self, config: TransportSecurityConfig):
        """Initialize host binding validator.
        
        Args:
            config: Transport security configuration
        """
        self.config = config
        self.allowed_hosts = config.allowed_hosts or set()
        
        # Default safe hosts based on policy
        if config.security_policy == SecurityPolicy.DEVELOPMENT:
            self.allowed_hosts.update(['localhost', '127.0.0.1', '::1'])
        elif config.security_policy in [SecurityPolicy.PRODUCTION, SecurityPolicy.PARANOID]:
            # Production should bind to specific interfaces only
            pass
        
    def validate_bind_address(self, bind_host: str, bind_port: int) -> bool:
        """Validate server bind address.
        
        Args:
            bind_host: Host address to bind to
            bind_port: Port to bind to
            
        Returns:
            True if binding is allowed
            
        Raises:
            HostBindingError: If binding violates security policy
        """
        # Check for dangerous bindings
        if bind_host in ['0.0.0.0', '::']:
            if self.config.security_policy in [SecurityPolicy.PRODUCTION, SecurityPolicy.PARANOID]:
                if not self.config.allow_localhost_in_production:
                    raise HostBindingError(
                        f"Binding to {bind_host} not allowed in {self.config.security_policy.value} mode"
                    )
            logger.warning(f"Server binding to all interfaces: {bind_host}:{bind_port}")
        
        # Validate against allowed hosts
        if self.allowed_hosts and bind_host not in self.allowed_hosts:
            raise HostBindingError(f"Host binding not allowed: {bind_host}")
        
        # Port validation
        if bind_port < 1024 and self.config.security_policy == SecurityPolicy.PARANOID:
            logger.warning(f"Binding to privileged port {bind_port} - ensure proper permissions")
        
        return True


class TransportSecurityManager:
    """Main transport security manager."""
    
    def __init__(self, config: Optional[TransportSecurityConfig] = None):
        """Initialize transport security manager.
        
        Args:
            config: Transport security configuration
        """
        self.config = config or TransportSecurityConfig()
        self.origin_validator = OriginValidator(self.config)
        self.https_enforcer = HTTPSEnforcer(self.config)
        self.host_validator = HostBindingValidator(self.config)
        
        logger.info(f"Transport security initialized with policy: {self.config.security_policy.value}")
    
    def validate_request(self, request_headers: Dict[str, str], request_url: str) -> Dict[str, Any]:
        """Validate incoming request for transport security.
        
        Args:
            request_headers: HTTP request headers
            request_url: Request URL
            
        Returns:
            Validation result with security metadata
            
        Raises:
            TransportSecurityError: If validation fails
        """
        result = {
            "valid": False,
            "origin_valid": False,
            "https_valid": False,
            "security_headers": {},
            "warnings": []
        }
        
        try:
            # Validate HTTPS requirement
            is_local = self._is_local_request(request_url)
            result["https_valid"] = self.https_enforcer.validate_https_required(request_url, is_local)
            
            # Validate Origin header
            origin_header = request_headers.get('Origin')
            host_header = request_headers.get('Host')
            result["origin_valid"] = self.origin_validator.validate_origin(origin_header, host_header)
            
            # Validate request size
            content_length = request_headers.get('Content-Length')
            if content_length:
                try:
                    size = int(content_length)
                    if size > self.config.max_request_size:
                        raise TransportSecurityError(
                            f"Request size {size} exceeds maximum {self.config.max_request_size}"
                        )
                except ValueError:
                    raise TransportSecurityError("Invalid Content-Length header")
            
            # Get security headers for response
            result["security_headers"] = self.https_enforcer.get_security_headers()
            
            result["valid"] = True
            logger.debug(f"Transport security validation passed for: {request_url}")
            
        except TransportSecurityError as e:
            logger.error(f"Transport security validation failed: {e}")
            raise
        
        return result
    
    def _is_local_request(self, request_url: str) -> bool:
        """Check if request is from localhost.
        
        Args:
            request_url: Request URL
            
        Returns:
            True if request is local
        """
        try:
            parsed = urllib.parse.urlparse(request_url)
            hostname = parsed.hostname
            return hostname in ['localhost', '127.0.0.1', '::1']
        except Exception:
            return False
    
    def validate_server_binding(self, bind_host: str, bind_port: int) -> bool:
        """Validate server binding configuration.
        
        Args:
            bind_host: Host to bind to
            bind_port: Port to bind to
            
        Returns:
            True if binding is valid
        """
        return self.host_validator.validate_bind_address(bind_host, bind_port)
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get SSL context for HTTPS server.
        
        Returns:
            SSL context if HTTPS is required, None otherwise
        """
        if self.config.require_https:
            return self.https_enforcer.get_ssl_context()
        return None


def create_development_config() -> TransportSecurityConfig:
    """Create development transport security configuration.
    
    Returns:
        Development security configuration
    """
    return TransportSecurityConfig(
        security_policy=SecurityPolicy.DEVELOPMENT,
        require_https=False,
        require_origin_header=False,
        allow_localhost_in_production=True,
        ssl_minimum_version=ssl.TLSVersion.TLSv1_2,
        enable_hsts=False
    )


def create_production_config(allowed_origins: List[str], allowed_hosts: Optional[List[str]] = None) -> TransportSecurityConfig:
    """Create production transport security configuration.
    
    Args:
        allowed_origins: List of allowed origins
        allowed_hosts: List of allowed host bindings
        
    Returns:
        Production security configuration
    """
    return TransportSecurityConfig(
        security_policy=SecurityPolicy.PRODUCTION,
        require_https=True,
        allowed_origins=set(allowed_origins),
        allowed_hosts=set(allowed_hosts) if allowed_hosts else None,
        require_origin_header=True,
        allow_localhost_in_production=False,
        ssl_minimum_version=ssl.TLSVersion.TLSv1_2,
        enable_hsts=True,
        enable_csp=True
    )