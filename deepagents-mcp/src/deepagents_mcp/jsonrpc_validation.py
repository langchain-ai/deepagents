"""JSON-RPC 2.0 Message Format Validation Framework.

This module provides comprehensive validation for JSON-RPC 2.0 messages
as required by the MCP 2025-06-18 specification.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class JSONRPCError(Exception):
    """Base exception for JSON-RPC validation errors."""
    pass


class InvalidRequestError(JSONRPCError):
    """Raised when request format is invalid."""
    pass


class InvalidResponseError(JSONRPCError):
    """Raised when response format is invalid."""
    pass


class InvalidNotificationError(JSONRPCError):
    """Raised when notification format is invalid."""
    pass


class MessageType(Enum):
    """JSON-RPC message types."""
    REQUEST = "request"
    RESPONSE = "response" 
    ERROR_RESPONSE = "error_response"
    NOTIFICATION = "notification"


@dataclass
class ValidationResult:
    """Result of JSON-RPC message validation."""
    valid: bool
    message_type: Optional[MessageType] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class JSONRPCValidator:
    """Comprehensive JSON-RPC 2.0 message validator."""
    
    def __init__(self, strict_mode: bool = True, mcp_mode: bool = True):
        """Initialize JSON-RPC validator.
        
        Args:
            strict_mode: Whether to enforce strict JSON-RPC 2.0 compliance
            mcp_mode: Whether to apply MCP-specific validation rules
        """
        self.strict_mode = strict_mode
        self.mcp_mode = mcp_mode
        
        # MCP-specific method patterns
        self.mcp_methods = {
            # Core protocol methods
            'initialize', 'initialized', 'ping', 'progress', 'cancelled',
            
            # Server capabilities
            'tools/list', 'tools/call',
            'resources/list', 'resources/read', 'resources/templates/list',
            'prompts/list', 'prompts/get',
            'completion/complete',
            
            # Client capabilities  
            'roots/list', 'sampling/createMessage', 'elicitation/create',
            
            # Notifications
            'notifications/tools/list_changed',
            'notifications/resources/updated', 'notifications/resources/list_changed',
            'notifications/prompts/list_changed',
            'notifications/roots/list_changed',
            'notifications/progress',
            'notifications/cancelled',
            'notifications/message',
            'notifications/log'
        }
        
        # Reserved method prefixes
        self.reserved_prefixes = {'rpc.', 'notifications/'}
        
        logger.info(f"JSON-RPC validator initialized (strict={strict_mode}, mcp={mcp_mode})")
    
    def validate_message(self, message: Any) -> ValidationResult:
        """Validate a JSON-RPC 2.0 message.
        
        Args:
            message: Message to validate (parsed JSON)
            
        Returns:
            Validation result
        """
        result = ValidationResult(valid=False)
        
        try:
            # Basic type validation
            if not isinstance(message, dict):
                result.errors.append(f"Message must be an object, got {type(message).__name__}")
                return result
            
            # Validate jsonrpc field
            if not self._validate_jsonrpc_field(message, result):
                return result
            
            # Determine message type and validate accordingly
            message_type = self._determine_message_type(message)
            result.message_type = message_type
            
            if message_type == MessageType.REQUEST:
                self._validate_request(message, result)
            elif message_type == MessageType.RESPONSE:
                self._validate_response(message, result)
            elif message_type == MessageType.ERROR_RESPONSE:
                self._validate_error_response(message, result)
            elif message_type == MessageType.NOTIFICATION:
                self._validate_notification(message, result)
            else:
                result.errors.append("Unable to determine message type")
                return result
            
            # Additional validations
            self._validate_reserved_fields(message, result)
            
            if self.mcp_mode:
                self._validate_mcp_specific(message, result)
            
            # Set final validation status
            result.valid = len(result.errors) == 0
            
            if result.valid:
                logger.debug(f"JSON-RPC message validation passed: {message_type.value}")
            else:
                logger.warning(f"JSON-RPC message validation failed: {result.errors}")
            
        except Exception as e:
            result.errors.append(f"Validation error: {e}")
            logger.error(f"JSON-RPC validation exception: {e}")
        
        return result
    
    def _validate_jsonrpc_field(self, message: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate jsonrpc field."""
        if 'jsonrpc' not in message:
            result.errors.append("Missing required 'jsonrpc' field")
            return False
        
        if message['jsonrpc'] != '2.0':
            result.errors.append(f"Invalid jsonrpc version: expected '2.0', got '{message['jsonrpc']}'")
            return False
        
        return True
    
    def _determine_message_type(self, message: Dict[str, Any]) -> Optional[MessageType]:
        """Determine JSON-RPC message type."""
        has_method = 'method' in message
        has_id = 'id' in message
        has_result = 'result' in message
        has_error = 'error' in message
        
        if has_method:
            if has_id:
                return MessageType.REQUEST
            else:
                return MessageType.NOTIFICATION
        elif has_id:
            if has_result:
                return MessageType.RESPONSE
            elif has_error:
                return MessageType.ERROR_RESPONSE
        
        return None
    
    def _validate_request(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate JSON-RPC request message."""
        # Validate method field
        if not self._validate_method_field(message, result):
            return
        
        # Validate id field
        if not self._validate_id_field(message, result, required=True):
            return
        
        # Validate params field (optional)
        if 'params' in message:
            self._validate_params_field(message, result)
        
        # MCP-specific request validation
        if self.mcp_mode:
            self._validate_mcp_request(message, result)
    
    def _validate_response(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate JSON-RPC response message."""
        # Validate id field
        if not self._validate_id_field(message, result, required=True):
            return
        
        # Must have result field
        if 'result' not in message:
            result.errors.append("Response missing required 'result' field")
            return
        
        # Must not have error field
        if 'error' in message:
            result.errors.append("Response cannot have both 'result' and 'error' fields")
            return
        
        # Must not have method field
        if 'method' in message:
            result.errors.append("Response cannot have 'method' field")
        
        # MCP-specific response validation
        if self.mcp_mode:
            self._validate_mcp_response(message, result)
    
    def _validate_error_response(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate JSON-RPC error response message."""
        # Validate id field (can be null for parse errors)
        if not self._validate_id_field(message, result, required=True, allow_null=True):
            return
        
        # Must have error field
        if 'error' not in message:
            result.errors.append("Error response missing required 'error' field")
            return
        
        # Must not have result field
        if 'result' in message:
            result.errors.append("Error response cannot have both 'result' and 'error' fields")
            return
        
        # Must not have method field
        if 'method' in message:
            result.errors.append("Error response cannot have 'method' field")
        
        # Validate error object structure
        self._validate_error_object(message['error'], result)
    
    def _validate_notification(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate JSON-RPC notification message."""
        # Validate method field
        if not self._validate_method_field(message, result):
            return
        
        # Must not have id field
        if 'id' in message:
            result.errors.append("Notification cannot have 'id' field")
            return
        
        # Validate params field (optional)
        if 'params' in message:
            self._validate_params_field(message, result)
        
        # MCP-specific notification validation
        if self.mcp_mode:
            self._validate_mcp_notification(message, result)
    
    def _validate_method_field(self, message: Dict[str, Any], result: ValidationResult) -> bool:
        """Validate method field."""
        if 'method' not in message:
            result.errors.append("Missing required 'method' field")
            return False
        
        method = message['method']
        if not isinstance(method, str):
            result.errors.append(f"Method must be a string, got {type(method).__name__}")
            return False
        
        if not method:
            result.errors.append("Method cannot be empty string")
            return False
        
        # Validate method name format
        if self.strict_mode:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_./]*[a-zA-Z0-9_]$|^[a-zA-Z]$', method):
                result.warnings.append(f"Method name '{method}' may not follow naming conventions")
        
        # Check for reserved method names
        if method.startswith('rpc.'):
            if method not in ['rpc.discover']:  # Allow some extensions
                result.errors.append(f"Method name '{method}' uses reserved 'rpc.' prefix")
                return False
        
        return True
    
    def _validate_id_field(self, message: Dict[str, Any], result: ValidationResult, 
                          required: bool = True, allow_null: bool = False) -> bool:
        """Validate id field."""
        if 'id' not in message:
            if required:
                result.errors.append("Missing required 'id' field")
                return False
            return True
        
        id_value = message['id']
        
        # Check allowed types
        if id_value is None:
            if not allow_null:
                result.errors.append("ID cannot be null")
                return False
        elif not isinstance(id_value, (str, int, float)):
            result.errors.append(f"ID must be string, number, or null, got {type(id_value).__name__}")
            return False
        
        # Additional MCP validation
        if self.mcp_mode and id_value is None and not allow_null:
            result.errors.append("MCP requires non-null IDs for requests and responses")
            return False
        
        return True
    
    def _validate_params_field(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate params field."""
        params = message['params']
        
        # Params must be structured (object or array)
        if not isinstance(params, (dict, list)):
            result.errors.append(f"Params must be object or array, got {type(params).__name__}")
            return
        
        # Additional validation for structured params
        if isinstance(params, dict):
            # Check for reserved parameter names
            reserved_params = {'jsonrpc', 'method', 'id', 'result', 'error'}
            for param_name in params:
                if param_name in reserved_params:
                    result.errors.append(f"Parameter name '{param_name}' is reserved")
    
    def _validate_error_object(self, error: Any, result: ValidationResult) -> None:
        """Validate JSON-RPC error object."""
        if not isinstance(error, dict):
            result.errors.append(f"Error must be an object, got {type(error).__name__}")
            return
        
        # Validate required fields
        if 'code' not in error:
            result.errors.append("Error object missing required 'code' field")
        else:
            code = error['code']
            if not isinstance(code, int):
                result.errors.append(f"Error code must be integer, got {type(code).__name__}")
            else:
                # Validate error code ranges
                if -32768 <= code <= -32000:
                    # Reserved for JSON-RPC
                    known_codes = {
                        -32700: "Parse error",
                        -32600: "Invalid Request", 
                        -32601: "Method not found",
                        -32602: "Invalid params",
                        -32603: "Internal error"
                    }
                    if code not in known_codes and code != -32099:  # -32099 to -32000 are reserved
                        result.warnings.append(f"Error code {code} is in reserved range but not standard")
        
        if 'message' not in error:
            result.errors.append("Error object missing required 'message' field")
        else:
            message = error['message']
            if not isinstance(message, str):
                result.errors.append(f"Error message must be string, got {type(message).__name__}")
        
        # Data field is optional
        if 'data' in error:
            # Data can be any type, no specific validation needed
            pass
    
    def _validate_reserved_fields(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate that reserved fields are not misused."""
        reserved_fields = {'jsonrpc'}
        
        # Check for unexpected reserved fields
        for field in message:
            if field.startswith('_') and field != '_meta':
                result.warnings.append(f"Field '{field}' starts with underscore (reserved)")
    
    def _validate_mcp_specific(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Apply MCP-specific validation rules."""
        # Check _meta field if present
        if '_meta' in message:
            self._validate_meta_field(message['_meta'], result)
    
    def _validate_mcp_request(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate MCP-specific request rules."""
        method = message.get('method', '')
        
        # Validate known MCP methods
        if method in self.mcp_methods:
            # Method-specific validation
            if method == 'initialize':
                self._validate_initialize_request(message, result)
            elif method == 'tools/call':
                self._validate_tools_call_request(message, result)
            elif method.startswith('resources/'):
                self._validate_resource_request(message, result)
        else:
            # Check if method follows MCP patterns
            if not any(method.startswith(prefix) for prefix in ['tools/', 'resources/', 'prompts/', 'completion/', 'roots/', 'sampling/', 'elicitation/']):
                result.warnings.append(f"Method '{method}' does not follow MCP naming patterns")
    
    def _validate_mcp_response(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate MCP-specific response rules."""
        # General MCP response validation
        result_data = message.get('result')
        if isinstance(result_data, dict):
            # Check for common MCP result fields
            if 'protocolVersion' in result_data:
                version = result_data['protocolVersion']
                if not isinstance(version, str) or not re.match(r'\d{4}-\d{2}-\d{2}', version):
                    result.warnings.append(f"Protocol version '{version}' may not follow YYYY-MM-DD format")
    
    def _validate_mcp_notification(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate MCP-specific notification rules."""
        method = message.get('method', '')
        
        # Validate notification method patterns
        if not method.startswith('notifications/'):
            result.warnings.append(f"MCP notification method '{method}' should start with 'notifications/'")
        
        # Validate known notification types
        known_notifications = {
            'notifications/initialized',
            'notifications/progress', 
            'notifications/cancelled',
            'notifications/message',
            'notifications/log',
            'notifications/tools/list_changed',
            'notifications/resources/updated',
            'notifications/resources/list_changed', 
            'notifications/prompts/list_changed',
            'notifications/roots/list_changed'
        }
        
        if method not in known_notifications:
            result.warnings.append(f"Unknown MCP notification type: {method}")
    
    def _validate_initialize_request(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate initialize request structure."""
        params = message.get('params', {})
        if not isinstance(params, dict):
            result.errors.append("Initialize request params must be an object")
            return
        
        # Required fields
        required_fields = ['protocolVersion', 'capabilities', 'clientInfo']
        for field in required_fields:
            if field not in params:
                result.errors.append(f"Initialize request missing required field: {field}")
        
        # Validate clientInfo structure
        if 'clientInfo' in params:
            client_info = params['clientInfo']
            if not isinstance(client_info, dict):
                result.errors.append("clientInfo must be an object")
            elif 'name' not in client_info:
                result.errors.append("clientInfo missing required 'name' field")
    
    def _validate_tools_call_request(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate tools/call request structure."""
        params = message.get('params', {})
        if not isinstance(params, dict):
            result.errors.append("tools/call request params must be an object")
            return
        
        # Required fields
        if 'name' not in params:
            result.errors.append("tools/call request missing required 'name' field")
        
        # Optional arguments field
        if 'arguments' in params and not isinstance(params['arguments'], dict):
            result.errors.append("tools/call arguments must be an object")
    
    def _validate_resource_request(self, message: Dict[str, Any], result: ValidationResult) -> None:
        """Validate resource-related request structure."""
        method = message.get('method', '')
        params = message.get('params', {})
        
        if method == 'resources/read':
            if not isinstance(params, dict) or 'uri' not in params:
                result.errors.append("resources/read request must include 'uri' parameter")
    
    def _validate_meta_field(self, meta: Any, result: ValidationResult) -> None:
        """Validate _meta field structure."""
        if not isinstance(meta, dict):
            result.errors.append("_meta field must be an object")
            return
        
        # Validate meta field keys
        for key in meta:
            if not isinstance(key, str):
                result.errors.append(f"_meta keys must be strings, got {type(key).__name__}")
            elif key.startswith('modelcontextprotocol') or key.startswith('mcp'):
                # Reserved for MCP use
                result.metadata['has_mcp_meta'] = True
    
    def validate_batch(self, messages: List[Any]) -> List[ValidationResult]:
        """Validate a batch of JSON-RPC messages.
        
        Note: MCP 2025-06-18 does not support batching, so this will generate warnings.
        
        Args:
            messages: List of messages to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        if self.mcp_mode:
            # Create a single result indicating batching is not supported
            result = ValidationResult(valid=False)
            result.errors.append("JSON-RPC batching is not supported in MCP 2025-06-18")
            return [result]
        
        for i, message in enumerate(messages):
            try:
                result = self.validate_message(message)
                result.metadata['batch_index'] = i
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(valid=False)
                error_result.errors.append(f"Batch validation error at index {i}: {e}")
                error_result.metadata['batch_index'] = i
                results.append(error_result)
        
        return results


def create_default_validator() -> JSONRPCValidator:
    """Create default JSON-RPC validator for MCP compliance.
    
    Returns:
        Configured validator instance
    """
    return JSONRPCValidator(strict_mode=True, mcp_mode=True)


def create_lenient_validator() -> JSONRPCValidator:
    """Create lenient JSON-RPC validator for development.
    
    Returns:
        Configured validator instance
    """
    return JSONRPCValidator(strict_mode=False, mcp_mode=True)


def validate_json_string(json_string: str, validator: Optional[JSONRPCValidator] = None) -> ValidationResult:
    """Validate JSON-RPC message from string.
    
    Args:
        json_string: JSON string to validate
        validator: Optional validator instance
        
    Returns:
        Validation result
    """
    if validator is None:
        validator = create_default_validator()
    
    try:
        message = json.loads(json_string)
        return validator.validate_message(message)
    except json.JSONDecodeError as e:
        result = ValidationResult(valid=False)
        result.errors.append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result = ValidationResult(valid=False)
        result.errors.append(f"Validation error: {e}")
        return result


# Example usage and testing functions
def example_valid_messages() -> Dict[str, Dict[str, Any]]:
    """Get examples of valid JSON-RPC messages for testing."""
    return {
        "initialize_request": {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        },
        "initialize_response": {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True}
                },
                "serverInfo": {
                    "name": "test-server",
                    "version": "1.0.0"
                }
            }
        },
        "notification": {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        },
        "error_response": {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32601,
                "message": "Method not found",
                "data": {"method": "unknown_method"}
            }
        }
    }


if __name__ == "__main__":
    # Example usage
    validator = create_default_validator()
    
    examples = example_valid_messages()
    for name, message in examples.items():
        result = validator.validate_message(message)
        print(f"{name}: {'VALID' if result.valid else 'INVALID'}")
        if not result.valid:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")