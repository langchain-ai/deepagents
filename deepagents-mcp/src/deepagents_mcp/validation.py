"""Input validation and sanitization for MCP integration.

This module provides comprehensive validation for MCP protocol messages,
tool inputs, and user data to prevent security vulnerabilities.
"""

import re
import json
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class SecurityViolationError(ValidationError):
    """Raised when input poses security risk."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when input doesn't match expected schema."""
    pass


@dataclass
class ValidationRule:
    """Represents a validation rule for input sanitization."""
    name: str
    pattern: Optional[str] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    disallowed_patterns: List[str] = field(default_factory=list)
    sanitize_html: bool = False
    sanitize_urls: bool = False
    required: bool = True


class SecurityLevel(Enum):
    """Security validation levels."""
    PERMISSIVE = "permissive"  # Basic validation
    STANDARD = "standard"     # Standard security checks  
    STRICT = "strict"         # Strict security validation
    PARANOID = "paranoid"     # Maximum security validation


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """Initialize input sanitizer.
        
        Args:
            security_level: Level of security validation to apply
        """
        self.security_level = security_level
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup dangerous patterns based on security level."""
        # Common dangerous patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # JavaScript
            r'javascript:',               # JavaScript URLs
            r'vbscript:',                # VBScript URLs
            r'on\w+\s*=',               # Event handlers
            r'data:text/html',          # Data URLs with HTML
            r'eval\s*\(',               # eval() calls
            r'exec\s*\(',               # exec() calls
            r'__import__\s*\(',         # Python imports
            r'subprocess\.',            # subprocess calls
            r'os\.system',              # os.system calls
            r'open\s*\(',               # File operations
        ]
        
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            self.dangerous_patterns.extend([
                r'\.\./',                # Directory traversal
                r'\.\.\\',               # Windows directory traversal
                r'/etc/',                # System directories
                r'/proc/',               # Process directories
                r'file://',              # File protocol
                r'ftp://',               # FTP protocol
                r'[;&|]',                # Command injection
                r'`[^`]*`',              # Command substitution
                r'\$\([^)]*\)',          # Command substitution
            ])
        
        if self.security_level == SecurityLevel.PARANOID:
            self.dangerous_patterns.extend([
                r'http://',              # Unencrypted HTTP (paranoid level)
                r'\.exe\b',              # Executables
                r'\.bat\b',              # Batch files
                r'\.sh\b',               # Shell scripts
                r'rm\s+',                # Delete commands
                r'del\s+',               # Windows delete
                r'format\s+',            # Format commands
            ])
    
    def sanitize_string(self, value: str, rule: ValidationRule) -> str:
        """Sanitize string input according to validation rule.
        
        Args:
            value: Input string to sanitize
            rule: Validation rule to apply
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityViolationError: If input poses security risk
            ValidationError: If input fails validation
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string for {rule.name}, got {type(value)}")
        
        original_value = value
        
        # Length validation
        if rule.max_length and len(value) > rule.max_length:
            raise ValidationError(f"{rule.name} exceeds maximum length {rule.max_length}")
        
        if rule.min_length and len(value) < rule.min_length:
            raise ValidationError(f"{rule.name} below minimum length {rule.min_length}")
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, value):
            raise ValidationError(f"{rule.name} doesn't match required pattern")
        
        # Character restrictions
        if rule.allowed_chars:
            allowed_set = set(rule.allowed_chars)
            if not all(c in allowed_set for c in value):
                raise ValidationError(f"{rule.name} contains disallowed characters")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityViolationError(f"{rule.name} contains dangerous pattern: {pattern}")
        
        # Check custom disallowed patterns
        for pattern in rule.disallowed_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityViolationError(f"{rule.name} contains disallowed pattern: {pattern}")
        
        # HTML sanitization
        if rule.sanitize_html:
            value = html.escape(value)
        
        # URL sanitization
        if rule.sanitize_urls:
            value = self._sanitize_urls(value)
        
        # Log sanitization if value changed
        if value != original_value:
            logger.info(f"Sanitized {rule.name}: '{original_value}' -> '{value}'")
        
        return value
    
    def _sanitize_urls(self, value: str) -> str:
        """Sanitize URLs in string value.
        
        Args:
            value: String that may contain URLs
            
        Returns:
            String with sanitized URLs
        """
        # Find URLs and validate them
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        
        def validate_url(match):
            url = match.group(0)
            try:
                parsed = urllib.parse.urlparse(url)
                
                # Only allow HTTP/HTTPS
                if parsed.scheme not in ['http', 'https']:
                    return '[REMOVED_INVALID_URL]'
                
                # Block dangerous TLDs if paranoid
                if self.security_level == SecurityLevel.PARANOID:
                    dangerous_tlds = ['.tk', '.ml', '.ga', '.cf']
                    if any(parsed.netloc.endswith(tld) for tld in dangerous_tlds):
                        return '[REMOVED_SUSPICIOUS_URL]'
                
                # Require HTTPS for strict/paranoid levels
                if (self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID] 
                    and parsed.scheme != 'https'):
                    return '[REMOVED_INSECURE_URL]'
                
                return url
            except Exception:
                return '[REMOVED_MALFORMED_URL]'
        
        return re.sub(url_pattern, validate_url, value)
    
    def validate_dict(self, data: Dict[str, Any], schema: Dict[str, ValidationRule]) -> Dict[str, Any]:
        """Validate and sanitize dictionary data.
        
        Args:
            data: Input dictionary to validate
            schema: Schema defining validation rules for each field
            
        Returns:
            Validated and sanitized dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dictionary, got {type(data)}")
        
        result = {}
        
        # Check for required fields
        for field_name, rule in schema.items():
            if rule.required and field_name not in data:
                raise ValidationError(f"Required field '{field_name}' is missing")
        
        # Validate each field
        for field_name, value in data.items():
            if field_name not in schema:
                if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
                    logger.warning(f"Unknown field '{field_name}' in input data")
                    continue  # Skip unknown fields in strict mode
                else:
                    result[field_name] = value  # Allow unknown fields in permissive mode
                    continue
            
            rule = schema[field_name]
            
            if isinstance(value, str):
                result[field_name] = self.sanitize_string(value, rule)
            elif isinstance(value, (int, float, bool)):
                result[field_name] = value
            elif isinstance(value, list):
                result[field_name] = [self.sanitize_string(str(item), rule) for item in value]
            elif isinstance(value, dict):
                # Recursively validate nested dictionaries
                result[field_name] = self.validate_dict(value, {})
            else:
                result[field_name] = str(value)
        
        return result
    
    def validate_json_rpc_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON-RPC 2.0 message format.
        
        Args:
            message: JSON-RPC message to validate
            
        Returns:
            Validated message
            
        Raises:
            ValidationError: If message format is invalid
        """
        if not isinstance(message, dict):
            raise ValidationError("JSON-RPC message must be a dictionary")
        
        # Define JSON-RPC validation schema
        schema = {
            "jsonrpc": ValidationRule(
                name="jsonrpc",
                pattern=r"^2\.0$",
                required=True
            ),
            "method": ValidationRule(
                name="method",
                pattern=r"^[a-zA-Z_][a-zA-Z0-9_/]*$",
                max_length=100,
                required=False  # Not required for responses
            ),
            "id": ValidationRule(
                name="id",
                max_length=100,
                required=False  # Not required for notifications
            ),
            "params": ValidationRule(
                name="params",
                required=False
            ),
            "result": ValidationRule(
                name="result",
                required=False
            ),
            "error": ValidationRule(
                name="error",
                required=False
            )
        }
        
        # Check if it's a valid JSON-RPC message structure
        if "jsonrpc" not in message or message["jsonrpc"] != "2.0":
            raise ValidationError("Invalid JSON-RPC version")
        
        # Validate message type
        if "method" in message:
            # Request or notification
            if not isinstance(message["method"], str):
                raise ValidationError("Method must be a string")
        elif "result" in message or "error" in message:
            # Response
            if "id" not in message:
                raise ValidationError("Response must include id field")
        else:
            raise ValidationError("Invalid JSON-RPC message structure")
        
        return self.validate_dict(message, schema)
    
    def validate_tool_input(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool input parameters.
        
        Args:
            tool_name: Name of the tool being called
            input_data: Input parameters for the tool
            
        Returns:
            Validated input parameters
            
        Raises:
            ValidationError: If input validation fails
        """
        # Define common tool input validation rules
        common_rules = {
            "path": ValidationRule(
                name="path",
                max_length=4096,
                disallowed_patterns=[r'\.\./', r'\.\.\\', r'/etc/', r'/proc/', r'~'],
                sanitize_urls=True
            ),
            "url": ValidationRule(
                name="url",
                max_length=2048,
                pattern=r'^https?://.+',
                sanitize_urls=True
            ),
            "filename": ValidationRule(
                name="filename",
                max_length=255,
                pattern=r'^[a-zA-Z0-9._-]+$',
                disallowed_patterns=[r'\.\.', r'/']
            ),
            "content": ValidationRule(
                name="content",
                max_length=1048576,  # 1MB max
                sanitize_html=True
            ),
            "query": ValidationRule(
                name="query",
                max_length=1000,
                sanitize_html=True
            )
        }
        
        # Apply validation rules based on parameter names
        validated_data = {}
        for param_name, param_value in input_data.items():
            rule = common_rules.get(param_name, ValidationRule(
                name=param_name,
                max_length=10000,
                sanitize_html=True
            ))
            
            if isinstance(param_value, str):
                validated_data[param_name] = self.sanitize_string(param_value, rule)
            else:
                validated_data[param_name] = param_value
        
        logger.info(f"Validated input for tool '{tool_name}': {len(validated_data)} parameters")
        return validated_data


class MCPMessageValidator:
    """Specialized validator for MCP protocol messages."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """Initialize MCP message validator.
        
        Args:
            security_level: Security validation level
        """
        self.sanitizer = InputSanitizer(security_level)
        self.security_level = security_level
    
    def validate_initialize_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate initialize request message.
        
        Args:
            message: Initialize request message
            
        Returns:
            Validated message
        """
        schema = {
            "protocolVersion": ValidationRule(
                name="protocolVersion",
                pattern=r"^\d{4}-\d{2}-\d{2}$",
                required=True
            ),
            "capabilities": ValidationRule(
                name="capabilities",
                required=True
            ),
            "clientInfo": ValidationRule(
                name="clientInfo",
                required=True
            )
        }
        
        validated = self.sanitizer.validate_json_rpc_message(message)
        
        if "params" in validated:
            validated["params"] = self.sanitizer.validate_dict(validated["params"], schema)
        
        return validated
    
    def validate_tool_call_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool call request message.
        
        Args:
            message: Tool call request message
            
        Returns:
            Validated message
        """
        validated = self.sanitizer.validate_json_rpc_message(message)
        
        if "params" in validated and "name" in validated["params"]:
            tool_name = validated["params"]["name"]
            if "arguments" in validated["params"]:
                validated["params"]["arguments"] = self.sanitizer.validate_tool_input(
                    tool_name, validated["params"]["arguments"]
                )
        
        return validated
    
    def validate_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response message.
        
        Args:
            message: Response message
            
        Returns:
            Validated message
        """
        validated = self.sanitizer.validate_json_rpc_message(message)
        
        # Sanitize result content if present
        if "result" in validated and isinstance(validated["result"], dict):
            if "content" in validated["result"]:
                content_rule = ValidationRule(
                    name="result_content",
                    max_length=1048576,  # 1MB max response
                    sanitize_html=True
                )
                if isinstance(validated["result"]["content"], str):
                    validated["result"]["content"] = self.sanitizer.sanitize_string(
                        validated["result"]["content"], content_rule
                    )
        
        return validated


def create_validation_schema(tool_name: str, input_schema: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationRule]:
    """Create validation schema for a specific tool.
    
    Args:
        tool_name: Name of the tool
        input_schema: Optional JSON schema for the tool's input
        
    Returns:
        Dictionary of validation rules
    """
    # Default rules based on tool name patterns
    if "file" in tool_name.lower():
        return {
            "path": ValidationRule(
                name="path",
                max_length=4096,
                disallowed_patterns=[r'\.\./', r'/etc/', r'/proc/'],
                required=True
            ),
            "content": ValidationRule(
                name="content",
                max_length=1048576,
                sanitize_html=True
            )
        }
    elif "search" in tool_name.lower():
        return {
            "query": ValidationRule(
                name="query",
                max_length=1000,
                sanitize_html=True,
                required=True
            ),
            "limit": ValidationRule(
                name="limit",
                pattern=r"^\d+$",
                max_length=10
            )
        }
    elif "http" in tool_name.lower() or "url" in tool_name.lower():
        return {
            "url": ValidationRule(
                name="url",
                max_length=2048,
                pattern=r'^https://.+',  # Require HTTPS
                sanitize_urls=True,
                required=True
            )
        }
    else:
        # Generic validation rules
        return {
            param: ValidationRule(
                name=param,
                max_length=10000,
                sanitize_html=True
            ) for param in ["query", "input", "text", "data", "content"]
        }


# Pre-configured validators for different security levels
PERMISSIVE_VALIDATOR = MCPMessageValidator(SecurityLevel.PERMISSIVE)
STANDARD_VALIDATOR = MCPMessageValidator(SecurityLevel.STANDARD)  
STRICT_VALIDATOR = MCPMessageValidator(SecurityLevel.STRICT)
PARANOID_VALIDATOR = MCPMessageValidator(SecurityLevel.PARANOID)