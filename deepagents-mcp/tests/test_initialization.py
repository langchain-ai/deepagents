"""Comprehensive tests for MCP initialization lifecycle."""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from deepagents_mcp.initialization import (
    MCPInitializationManager,
    InitializationState,
    ServerCapabilities,
    ClientCapabilities,
    ImplementationInfo,
    InitializationSession,
    create_default_initialization_manager
)
from deepagents_mcp.jsonrpc_validation import create_default_validator


class TestMCPInitializationManager:
    """Test MCP initialization manager functionality."""
    
    @pytest.fixture
    def server_info(self):
        """Create test server info."""
        return ImplementationInfo(
            name="test-server",
            version="1.0.0",
            description="Test MCP server",
            homepage="https://example.com"
        )
    
    @pytest.fixture
    def server_capabilities(self):
        """Create test server capabilities."""
        return ServerCapabilities(
            tools={"listChanged": True},
            resources={"subscribe": True, "listChanged": True},
            logging={}
        )
    
    @pytest.fixture
    def manager(self, server_info, server_capabilities):
        """Create test initialization manager."""
        return MCPInitializationManager(
            server_info=server_info,
            server_capabilities=server_capabilities
        )
    
    @pytest.fixture
    def valid_initialize_request(self):
        """Create valid initialize request."""
        return {
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
                    "version": "1.0.0",
                    "description": "Test MCP client"
                }
            }
        }
    
    @pytest.fixture
    def valid_initialized_notification(self):
        """Create valid initialized notification."""
        return {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
    
    async def test_successful_initialization_flow(self, manager, valid_initialize_request, valid_initialized_notification):
        """Test complete successful initialization flow."""
        session_id = "test-session"
        
        # Initial state should be empty
        assert len(manager.sessions) == 0
        assert not manager.is_session_initialized(session_id)
        
        # Handle initialize request
        response = await manager.handle_initialize_request(valid_initialize_request, session_id)
        
        # Verify response structure
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        
        result = response["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert "capabilities" in result
        assert "serverInfo" in result
        
        # Verify session state
        session = manager.get_session(session_id)
        assert session is not None
        assert session.state == InitializationState.INITIALIZING
        assert session.client_version == "2025-06-18"
        assert session.negotiated_version == "2025-06-18"
        assert session.client_info.name == "test-client"
        
        # Not yet fully initialized
        assert not manager.is_session_initialized(session_id)
        
        # Handle initialized notification
        notify_response = await manager.handle_initialized_notification(valid_initialized_notification, session_id)
        
        # Notifications don't have responses
        assert notify_response is None
        
        # Now should be fully initialized
        assert manager.is_session_initialized(session_id)
        session = manager.get_session(session_id)
        assert session.state == InitializationState.INITIALIZED
        assert session.initialized_at is not None
    
    async def test_initialize_request_validation(self, manager):
        """Test validation of initialize requests."""
        session_id = "test-session"
        
        # Test missing jsonrpc
        invalid_request = {
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid Request
        
        # Test missing params
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert response["error"]["code"] == -32602  # Invalid params
        
        # Test missing protocolVersion
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {},
                "clientInfo": {"name": "test"}
            }
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert "protocolVersion" in response["error"]["message"]
        
        # Test missing capabilities
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "clientInfo": {"name": "test"}
            }
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert "capabilities" in response["error"]["message"]
        
        # Test missing clientInfo
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {}
            }
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert "clientInfo" in response["error"]["message"]
        
        # Test clientInfo missing name
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"version": "1.0.0"}
            }
        }
        response = await manager.handle_initialize_request(invalid_request, session_id)
        assert "error" in response
        assert "name" in response["error"]["message"]
    
    async def test_version_negotiation(self, manager, valid_initialize_request):
        """Test protocol version negotiation."""
        session_id = "test-session"
        
        # Test supported version
        response = await manager.handle_initialize_request(valid_initialize_request, session_id)
        assert response["result"]["protocolVersion"] == "2025-06-18"
        
        # Test unsupported version
        unsupported_request = valid_initialize_request.copy()
        unsupported_request["params"]["protocolVersion"] = "2020-01-01"
        
        response = await manager.handle_initialize_request(unsupported_request, "test-session-2")
        assert "error" in response
        assert "Unsupported protocol version" in response["error"]["message"]
        
        # Test fallback to older supported version
        older_request = valid_initialize_request.copy()
        older_request["params"]["protocolVersion"] = "2025-03-26"
        
        response = await manager.handle_initialize_request(older_request, "test-session-3")
        assert response["result"]["protocolVersion"] == "2025-03-26"
    
    async def test_capability_parsing(self, manager, valid_initialize_request):
        """Test client capability parsing."""
        session_id = "test-session"
        
        # Test with complex capabilities
        complex_request = valid_initialize_request.copy()
        complex_request["params"]["capabilities"] = {
            "roots": {
                "listChanged": True,
                "subscribe": True
            },
            "sampling": {
                "maxTokens": 1000
            },
            "elicitation": {
                "experimental": True
            }
        }
        
        response = await manager.handle_initialize_request(complex_request, session_id)
        assert "result" in response
        
        session = manager.get_session(session_id)
        assert session.client_capabilities.roots["listChanged"] is True
        assert session.client_capabilities.sampling["maxTokens"] == 1000
        assert session.client_capabilities.elicitation["experimental"] is True
    
    async def test_duplicate_initialization(self, manager, valid_initialize_request):
        """Test handling of duplicate initialization attempts."""
        session_id = "test-session"
        
        # First initialization
        response1 = await manager.handle_initialize_request(valid_initialize_request, session_id)
        assert "result" in response1
        
        # Second initialization attempt should fail
        response2 = await manager.handle_initialize_request(valid_initialize_request, session_id)
        assert "error" in response2
        assert "already in state" in response2["error"]["message"]
    
    async def test_initialized_notification_validation(self, manager, valid_initialize_request, valid_initialized_notification):
        """Test initialized notification validation."""
        session_id = "test-session"
        
        # Initialize first
        await manager.handle_initialize_request(valid_initialize_request, session_id)
        
        # Test invalid JSON-RPC notification
        invalid_notification = {
            "method": "notifications/initialized"
            # Missing jsonrpc field
        }
        response = await manager.handle_initialized_notification(invalid_notification, session_id)
        assert response is None  # Invalid notifications are ignored
        
        # Session should still be in INITIALIZING state
        session = manager.get_session(session_id)
        assert session.state == InitializationState.INITIALIZING
        
        # Test notification for unknown session
        response = await manager.handle_initialized_notification(valid_initialized_notification, "unknown-session")
        assert response is None
        
        # Test notification in wrong state
        manager.sessions[session_id].state = InitializationState.INITIALIZED
        response = await manager.handle_initialized_notification(valid_initialized_notification, session_id)
        assert response is None
        
        # Reset to INITIALIZING and test valid notification
        manager.sessions[session_id].state = InitializationState.INITIALIZING
        response = await manager.handle_initialized_notification(valid_initialized_notification, session_id)
        assert response is None
        
        # Should now be initialized
        session = manager.get_session(session_id)
        assert session.state == InitializationState.INITIALIZED
    
    async def test_initialization_handler(self, manager, valid_initialize_request, valid_initialized_notification):
        """Test initialization completion handler."""
        session_id = "test-session"
        handler_called = False
        session_data = None
        
        async def test_handler(session: InitializationSession):
            nonlocal handler_called, session_data
            handler_called = True
            session_data = session
        
        manager.set_initialized_handler(test_handler)
        
        # Complete initialization flow
        await manager.handle_initialize_request(valid_initialize_request, session_id)
        await manager.handle_initialized_notification(valid_initialized_notification, session_id)
        
        # Handler should have been called
        assert handler_called
        assert session_data is not None
        assert session_data.session_id == session_id
        assert session_data.state == InitializationState.INITIALIZED
    
    def test_session_management(self, manager):
        """Test session management functions."""
        session_id = "test-session"
        
        # Initially no sessions
        assert len(manager.sessions) == 0
        assert manager.get_session(session_id) is None
        assert not manager.is_session_initialized(session_id)
        assert manager.get_negotiated_version(session_id) is None
        
        # Create a session
        session = InitializationSession(session_id=session_id)
        session.negotiated_version = "2025-06-18"
        manager.sessions[session_id] = session
        
        # Test session retrieval
        retrieved = manager.get_session(session_id)
        assert retrieved is not None
        assert retrieved.session_id == session_id
        
        # Test version retrieval
        version = manager.get_negotiated_version(session_id)
        assert version == "2025-06-18"
        
        # Test initialization check (not initialized yet)
        assert not manager.is_session_initialized(session_id)
        
        # Mark as initialized
        session.state = InitializationState.INITIALIZED
        assert manager.is_session_initialized(session_id)
        
        # Test shutdown
        assert manager.shutdown_session(session_id)
        assert manager.sessions[session_id].state == InitializationState.SHUTDOWN
        
        # Test shutdown of non-existent session
        assert not manager.shutdown_session("non-existent")
    
    def test_session_cleanup(self, manager):
        """Test expired session cleanup."""
        # Create old session
        old_session = InitializationSession(session_id="old-session")
        old_session.created_at = time.time() - 7200  # 2 hours ago
        manager.sessions["old-session"] = old_session
        
        # Create recent session
        recent_session = InitializationSession(session_id="recent-session")
        recent_session.created_at = time.time() - 30  # 30 seconds ago
        manager.sessions["recent-session"] = recent_session
        
        # Cleanup with 1 hour max age
        cleaned_up = manager.cleanup_expired_sessions(max_age_seconds=3600)
        
        # Should have cleaned up 1 session
        assert cleaned_up == 1
        assert "old-session" not in manager.sessions
        assert "recent-session" in manager.sessions
    
    def test_session_stats(self, manager):
        """Test session statistics."""
        # Create sessions in different states
        session1 = InitializationSession(session_id="session1")
        session1.state = InitializationState.NOT_INITIALIZED
        manager.sessions["session1"] = session1
        
        session2 = InitializationSession(session_id="session2")
        session2.state = InitializationState.INITIALIZING
        manager.sessions["session2"] = session2
        
        session3 = InitializationSession(session_id="session3")
        session3.state = InitializationState.INITIALIZED
        manager.sessions["session3"] = session3
        
        stats = manager.get_session_stats()
        
        assert stats["total_sessions"] == 3
        assert stats["state_counts"]["not_initialized"] == 1
        assert stats["state_counts"]["initializing"] == 1
        assert stats["state_counts"]["initialized"] == 1
        assert "supported_versions" in stats
        assert "server_info" in stats


class TestInitializationIntegration:
    """Test initialization integration with other components."""
    
    def test_default_manager_creation(self):
        """Test creation of default initialization manager."""
        manager = create_default_initialization_manager(
            "test-server",
            "1.0.0",
            tools_enabled=True,
            resources_enabled=True,
            prompts_enabled=False
        )
        
        assert manager.server_info.name == "test-server"
        assert manager.server_info.version == "1.0.0"
        assert manager.server_capabilities.tools is not None
        assert manager.server_capabilities.resources is not None
        assert manager.server_capabilities.prompts == {}
        assert manager.server_capabilities.logging == {}
    
    async def test_jsonrpc_validator_integration(self, manager, valid_initialize_request):
        """Test integration with JSON-RPC validator."""
        # The manager uses the validator internally
        validator = create_default_validator()
        
        # Test that the manager validates messages properly
        session_id = "test-session"
        
        # Valid message should work
        response = await manager.handle_initialize_request(valid_initialize_request, session_id)
        assert "result" in response
        
        # Invalid JSON-RPC should fail
        invalid_request = valid_initialize_request.copy()
        invalid_request["jsonrpc"] = "1.0"  # Wrong version
        
        response = await manager.handle_initialize_request(invalid_request, "test-session-2")
        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid Request


class TestInitializationErrorHandling:
    """Test error handling in initialization."""
    
    @pytest.fixture
    def manager(self):
        """Create test manager."""
        server_info = ImplementationInfo(name="test", version="1.0.0")
        server_capabilities = ServerCapabilities()
        return MCPInitializationManager(server_info, server_capabilities)
    
    async def test_exception_handling(self, manager):
        """Test handling of unexpected exceptions."""
        session_id = "test-session"
        
        # Test with completely invalid message structure
        invalid_message = "not a dict"
        
        response = await manager.handle_initialize_request(invalid_message, session_id)
        assert "error" in response
        assert response["error"]["code"] == -32600  # Invalid Request
        
        # Test with message that causes parsing error
        problematic_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": "not an object",  # Should be dict
                "clientInfo": {"name": "test"}
            }
        }
        
        response = await manager.handle_initialize_request(problematic_message, session_id)
        assert "error" in response
        assert response["error"]["code"] == -32602  # Invalid params


if __name__ == "__main__":
    # Run basic tests if executed directly
    import sys
    
    async def run_basic_tests():
        """Run basic functionality tests."""
        print("Running basic initialization tests...")
        
        # Create manager
        server_info = ImplementationInfo(
            name="test-server",
            version="1.0.0"
        )
        server_capabilities = ServerCapabilities(
            tools={"listChanged": True},
            logging={}
        )
        manager = MCPInitializationManager(server_info, server_capabilities)
        
        # Test initialization flow
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "roots": {"listChanged": True}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        session_id = "test-session"
        
        # Initialize
        response = await manager.handle_initialize_request(initialize_request, session_id)
        if "error" in response:
            print(f"Initialize failed: {response['error']}")
            return False
        
        print(f"Initialize successful: {response['result']['protocolVersion']}")
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        await manager.handle_initialized_notification(initialized_notification, session_id)
        
        # Check final state
        if manager.is_session_initialized(session_id):
            print("Session fully initialized ✓")
            return True
        else:
            print("Session not properly initialized ✗")
            return False
    
    # Run the test
    if asyncio.run(run_basic_tests()):
        print("Basic tests passed!")
        sys.exit(0) 
    else:
        print("Basic tests failed!")
        sys.exit(1)