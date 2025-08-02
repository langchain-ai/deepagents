"""User consent management for MCP tool execution.

This module implements the MCP specification requirement that tools require
explicit user consent before invocation. It provides consent tracking, UI
patterns, and audit logging for all tool executions.
"""

import asyncio
import time
import json
import secrets
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConsentDecision(Enum):
    """User consent decisions for tool execution."""
    APPROVED = "approved"
    DENIED = "denied"
    DEFERRED = "deferred"  # User asked to review later
    EXPIRED = "expired"    # Consent request timed out


class ConsentScope(Enum):
    """Scope of consent for tool operations."""
    SINGLE_USE = "single_use"      # One-time consent
    SESSION = "session"            # Valid for current session
    PERSISTENT = "persistent"      # Remembered across sessions
    BULK_APPROVE = "bulk_approve"  # Approve all similar operations


class RiskLevel(Enum):
    """Risk assessment levels for tool operations."""
    LOW = "low"          # Read-only operations, safe tools
    MEDIUM = "medium"    # Write operations, external API calls
    HIGH = "high"        # System operations, file modifications
    CRITICAL = "critical" # Destructive operations, security-sensitive


@dataclass
class ConsentRequest:
    """Represents a user consent request for tool execution."""
    request_id: str
    tool_name: str
    tool_description: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    user_id: str
    client_id: str
    timestamp: float
    expires_at: float
    scope: ConsentScope = ConsentScope.SINGLE_USE
    justification: Optional[str] = None
    predicted_effects: List[str] = field(default_factory=list)
    data_access_description: Optional[str] = None
    decision: Optional[ConsentDecision] = None
    decision_timestamp: Optional[float] = None
    audit_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentHistory:
    """Historical record of user consent decisions."""
    user_id: str
    tool_name: str
    decision: ConsentDecision
    timestamp: float
    parameters_hash: str  # Hash of parameters for similarity matching
    risk_level: RiskLevel
    session_id: Optional[str] = None
    client_id: Optional[str] = None


class ConsentManager:
    """Manages user consent for MCP tool execution."""
    
    def __init__(self, consent_timeout: int = 300, audit_log_path: Optional[Path] = None):
        """Initialize consent manager.
        
        Args:
            consent_timeout: Timeout in seconds for consent requests
            audit_log_path: Path to audit log file
        """
        self.consent_timeout = consent_timeout
        self.audit_log_path = audit_log_path
        
        # Active consent requests awaiting user decision
        self.pending_requests: Dict[str, ConsentRequest] = {}
        
        # Historical consent decisions for learning user preferences
        self.consent_history: List[ConsentHistory] = []
        
        # Session-based consent cache
        self.session_consents: Dict[str, Dict[str, ConsentDecision]] = {}
        
        # Persistent consent preferences (across sessions)
        self.persistent_consents: Dict[str, Dict[str, ConsentDecision]] = {}
        
        # Risk assessment rules
        self.risk_rules = self._initialize_risk_rules()
    
    def _initialize_risk_rules(self) -> Dict[str, RiskLevel]:
        """Initialize risk assessment rules for different tool types."""
        return {
            # Low risk - read-only operations
            "get": RiskLevel.LOW,
            "read": RiskLevel.LOW,
            "list": RiskLevel.LOW,
            "search": RiskLevel.LOW,
            "query": RiskLevel.LOW,
            "info": RiskLevel.LOW,
            "status": RiskLevel.LOW,
            
            # Medium risk - write operations
            "create": RiskLevel.MEDIUM,
            "update": RiskLevel.MEDIUM,
            "modify": RiskLevel.MEDIUM,
            "send": RiskLevel.MEDIUM,
            "post": RiskLevel.MEDIUM,
            "put": RiskLevel.MEDIUM,
            
            # High risk - system operations
            "delete": RiskLevel.HIGH,
            "remove": RiskLevel.HIGH,
            "execute": RiskLevel.HIGH,
            "run": RiskLevel.HIGH,
            "install": RiskLevel.HIGH,
            "download": RiskLevel.HIGH,
            
            # Critical risk - destructive operations
            "format": RiskLevel.CRITICAL,
            "wipe": RiskLevel.CRITICAL,
            "destroy": RiskLevel.CRITICAL,
            "admin": RiskLevel.CRITICAL,
            "root": RiskLevel.CRITICAL,
            "sudo": RiskLevel.CRITICAL,
        }
    
    def assess_risk_level(self, tool_name: str, parameters: Dict[str, Any]) -> RiskLevel:
        """Assess risk level for a tool execution request.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool execution parameters
            
        Returns:
            Assessed risk level
        """
        tool_lower = tool_name.lower()
        
        # Check for exact matches first
        if tool_lower in self.risk_rules:
            return self.risk_rules[tool_lower]
        
        # Check for keyword matches in tool name
        for keyword, risk_level in self.risk_rules.items():
            if keyword in tool_lower:
                return risk_level
        
        # Analyze parameters for risk indicators
        param_text = json.dumps(parameters, default=str).lower()
        
        # Critical risk indicators in parameters
        critical_indicators = ["rm -rf", "delete", "format", "drop table", "truncate", "destroy"]
        if any(indicator in param_text for indicator in critical_indicators):
            return RiskLevel.CRITICAL
        
        # High risk indicators
        high_indicators = ["sudo", "admin", "root", "execute", "system", "shell"]
        if any(indicator in param_text for indicator in high_indicators):
            return RiskLevel.HIGH
        
        # Check for file paths that might indicate system access
        import re
        if re.search(r'/etc/|/proc/|/sys/|C:\\Windows\\|C:\\Program Files\\', param_text):
            return RiskLevel.HIGH
        
        # Default to medium risk for unknown tools
        return RiskLevel.MEDIUM
    
    async def request_consent(self, 
                            tool_name: str,
                            tool_description: str,
                            parameters: Dict[str, Any],
                            user_id: str,
                            client_id: str,
                            session_id: Optional[str] = None,
                            justification: Optional[str] = None,
                            predicted_effects: Optional[List[str]] = None,
                            data_access_description: Optional[str] = None) -> ConsentRequest:
        """Request user consent for tool execution.
        
        Args:
            tool_name: Name of the tool to execute
            tool_description: Human-readable description of the tool
            parameters: Tool execution parameters
            user_id: User identifier
            client_id: Client identifier
            session_id: Optional session identifier
            justification: Reason why this tool needs to be executed
            predicted_effects: List of predicted effects of tool execution
            data_access_description: Description of what data will be accessed
            
        Returns:
            Consent request object
        """
        request_id = secrets.token_urlsafe(16)
        risk_level = self.assess_risk_level(tool_name, parameters)
        current_time = time.time()
        
        # Create consent request
        consent_request = ConsentRequest(
            request_id=request_id,
            tool_name=tool_name,
            tool_description=tool_description,
            parameters=parameters,
            risk_level=risk_level,
            user_id=user_id,
            client_id=client_id,
            timestamp=current_time,
            expires_at=current_time + self.consent_timeout,
            justification=justification,
            predicted_effects=predicted_effects or [],
            data_access_description=data_access_description,
            audit_metadata={
                "session_id": session_id,
                "request_origin": "mcp_tool_execution",
                "risk_assessment_version": "1.0"
            }
        )
        
        # Check for existing consent based on user preferences
        existing_consent = self._check_existing_consent(
            user_id, tool_name, parameters, session_id, risk_level
        )
        
        if existing_consent:
            consent_request.decision = existing_consent
            consent_request.decision_timestamp = current_time
            logger.info(f"Using existing consent {existing_consent.value} for tool '{tool_name}' user '{user_id}'")
        else:
            # Store pending request
            self.pending_requests[request_id] = consent_request
            logger.info(f"Created consent request {request_id} for tool '{tool_name}' user '{user_id}'")
        
        return consent_request
    
    def _check_existing_consent(self, 
                               user_id: str, 
                               tool_name: str, 
                               parameters: Dict[str, Any],
                               session_id: Optional[str],
                               risk_level: RiskLevel) -> Optional[ConsentDecision]:
        """Check for existing consent decisions that might apply.
        
        Args:
            user_id: User identifier
            tool_name: Tool name
            parameters: Tool parameters
            session_id: Session identifier
            risk_level: Assessed risk level
            
        Returns:
            Existing consent decision if applicable
        """
        # For critical risk operations, always require fresh consent
        if risk_level == RiskLevel.CRITICAL:
            return None
        
        # Check session-based consent
        if session_id and session_id in self.session_consents:
            session_consent = self.session_consents[session_id].get(tool_name)
            if session_consent in [ConsentDecision.APPROVED, ConsentDecision.DENIED]:
                return session_consent
        
        # Check persistent consent preferences
        user_key = f"{user_id}:{tool_name}"
        if user_key in self.persistent_consents:
            persistent_consent = self.persistent_consents[user_key].get("default")
            if persistent_consent in [ConsentDecision.APPROVED, ConsentDecision.DENIED]:
                return persistent_consent
        
        return None
    
    async def provide_consent(self, 
                            request_id: str, 
                            decision: ConsentDecision,
                            scope: ConsentScope = ConsentScope.SINGLE_USE) -> bool:
        """Provide user consent decision for a pending request.
        
        Args:
            request_id: Request identifier
            decision: User's consent decision
            scope: Scope of the consent decision
            
        Returns:
            True if consent was successfully recorded
        """
        if request_id not in self.pending_requests:
            logger.warning(f"Consent request {request_id} not found")
            return False
        
        request = self.pending_requests[request_id]
        current_time = time.time()
        
        # Check if request has expired
        if current_time > request.expires_at:
            request.decision = ConsentDecision.EXPIRED
            logger.warning(f"Consent request {request_id} has expired")
            return False
        
        # Record decision
        request.decision = decision
        request.decision_timestamp = current_time
        request.scope = scope
        
        # Store consent based on scope
        if scope == ConsentScope.SESSION and request.audit_metadata.get("session_id"):
            session_id = request.audit_metadata["session_id"]
            if session_id not in self.session_consents:
                self.session_consents[session_id] = {}
            self.session_consents[session_id][request.tool_name] = decision
        
        elif scope == ConsentScope.PERSISTENT:
            user_key = f"{request.user_id}:{request.tool_name}"
            if user_key not in self.persistent_consents:
                self.persistent_consents[user_key] = {}
            self.persistent_consents[user_key]["default"] = decision
        
        # Add to consent history
        params_hash = self._hash_parameters(request.parameters)
        history_entry = ConsentHistory(
            user_id=request.user_id,
            tool_name=request.tool_name,
            decision=decision,
            timestamp=current_time,
            parameters_hash=params_hash,
            risk_level=request.risk_level,
            session_id=request.audit_metadata.get("session_id"),
            client_id=request.client_id
        )
        self.consent_history.append(history_entry)
        
        # Audit logging
        await self._audit_log_consent(request)
        
        # Remove from pending requests
        del self.pending_requests[request_id]
        
        logger.info(f"Recorded consent decision {decision.value} for request {request_id}")
        return True
    
    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """Create hash of parameters for similarity matching."""
        import hashlib
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    async def _audit_log_consent(self, request: ConsentRequest) -> None:
        """Log consent decision to audit log.
        
        Args:
            request: Consent request with decision
        """
        if not self.audit_log_path:
            return
        
        try:
            audit_entry = {
                "timestamp": request.decision_timestamp,
                "event_type": "tool_consent",
                "request_id": request.request_id,
                "user_id": request.user_id,
                "client_id": request.client_id,
                "tool_name": request.tool_name,
                "decision": request.decision.value if request.decision else None,
                "risk_level": request.risk_level.value,
                "scope": request.scope.value,
                "parameters_hash": self._hash_parameters(request.parameters),
                "justification": request.justification,
                "predicted_effects": request.predicted_effects,
                "audit_metadata": request.audit_metadata
            }
            
            # Append to audit log file
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_pending_requests(self, user_id: Optional[str] = None) -> List[ConsentRequest]:
        """Get pending consent requests for a user.
        
        Args:
            user_id: Optional user ID to filter requests
            
        Returns:
            List of pending consent requests
        """
        current_time = time.time()
        pending = []
        
        # Clean up expired requests
        expired_requests = []
        for request_id, request in self.pending_requests.items():
            if current_time > request.expires_at:
                request.decision = ConsentDecision.EXPIRED
                expired_requests.append(request_id)
            elif not user_id or request.user_id == user_id:
                pending.append(request)
        
        # Remove expired requests
        for request_id in expired_requests:
            del self.pending_requests[request_id]
        
        return pending
    
    def get_consent_history(self, 
                          user_id: str, 
                          tool_name: Optional[str] = None,
                          limit: int = 100) -> List[ConsentHistory]:
        """Get consent history for a user.
        
        Args:
            user_id: User identifier
            tool_name: Optional tool name filter
            limit: Maximum number of entries to return
            
        Returns:
            List of consent history entries
        """
        filtered_history = [
            entry for entry in self.consent_history
            if entry.user_id == user_id and (not tool_name or entry.tool_name == tool_name)
        ]
        
        # Sort by timestamp descending and limit
        filtered_history.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_history[:limit]
    
    def revoke_session_consents(self, session_id: str) -> int:
        """Revoke all consents for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of consents revoked
        """
        if session_id in self.session_consents:
            count = len(self.session_consents[session_id])
            del self.session_consents[session_id]
            logger.info(f"Revoked {count} session consents for session {session_id}")
            return count
        return 0
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user consent preferences and statistics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of user preferences and statistics
        """
        history = self.get_consent_history(user_id)
        
        # Calculate statistics
        total_requests = len(history)
        approved_count = sum(1 for entry in history if entry.decision == ConsentDecision.APPROVED)
        denied_count = sum(1 for entry in history if entry.decision == ConsentDecision.DENIED)
        
        # Get most frequently used tools
        tool_usage = {}
        for entry in history:
            tool_usage[entry.tool_name] = tool_usage.get(entry.tool_name, 0) + 1
        
        frequently_used = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_consent_requests": total_requests,
            "approved_requests": approved_count,
            "denied_requests": denied_count,
            "approval_rate": approved_count / total_requests if total_requests > 0 else 0,
            "frequently_used_tools": frequently_used,
            "persistent_consents": list(self.persistent_consents.keys()),
            "last_activity": history[0].timestamp if history else None
        }


# UI Helper functions for consent presentation

def format_consent_request_for_display(request: ConsentRequest) -> Dict[str, Any]:
    """Format consent request for user display.
    
    Args:
        request: Consent request to format
        
    Returns:
        Dictionary formatted for UI display
    """
    return {
        "request_id": request.request_id,
        "tool_name": request.tool_name,
        "description": request.tool_description,
        "risk_level": request.risk_level.value,
        "risk_color": {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow", 
            RiskLevel.HIGH: "orange",
            RiskLevel.CRITICAL: "red"
        }[request.risk_level],
        "justification": request.justification,
        "predicted_effects": request.predicted_effects,
        "data_access": request.data_access_description,
        "parameters": request.parameters,
        "expires_in_seconds": max(0, int(request.expires_at - time.time())),
        "timestamp": request.timestamp
    }


def create_consent_prompt(request: ConsentRequest) -> str:
    """Create a text prompt for console-based consent requests.
    
    Args:
        request: Consent request
        
    Returns:
        Formatted consent prompt string
    """
    risk_indicators = {
        RiskLevel.LOW: "🟢 LOW",
        RiskLevel.MEDIUM: "🟡 MEDIUM", 
        RiskLevel.HIGH: "🟠 HIGH",
        RiskLevel.CRITICAL: "🔴 CRITICAL"
    }
    
    prompt = f"""
🔐 TOOL EXECUTION CONSENT REQUEST

Tool: {request.tool_name}
Description: {request.tool_description}
Risk Level: {risk_indicators[request.risk_level]}

"""
    
    if request.justification:
        prompt += f"Justification: {request.justification}\n"
    
    if request.predicted_effects:
        prompt += f"Predicted Effects:\n"
        for effect in request.predicted_effects:
            prompt += f"  • {effect}\n"
    
    if request.data_access_description:
        prompt += f"Data Access: {request.data_access_description}\n"
    
    if request.parameters:
        prompt += f"Parameters: {json.dumps(request.parameters, indent=2)}\n"
    
    expires_in = max(0, int(request.expires_at - time.time()))
    prompt += f"\nThis request expires in {expires_in} seconds.\n"
    prompt += "\nDo you approve this tool execution? (y/n/s=session/p=persistent): "
    
    return prompt


# Global consent manager instance (can be configured)
_global_consent_manager: Optional[ConsentManager] = None


def get_consent_manager() -> ConsentManager:
    """Get or create global consent manager instance."""
    global _global_consent_manager
    if _global_consent_manager is None:
        audit_path = Path.home() / ".deepagents" / "audit" / "consent.log"
        _global_consent_manager = ConsentManager(audit_log_path=audit_path)
    return _global_consent_manager


def set_consent_manager(manager: ConsentManager) -> None:
    """Set global consent manager instance."""
    global _global_consent_manager
    _global_consent_manager = manager