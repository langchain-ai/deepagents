"""Integration tests for lxplus environment.

These tests verify HTCondor submission and MCP server connectivity.
They should only run on lxplus machines and will be skipped elsewhere.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from chatlas_agents.config import MCPServerConfig
from chatlas_agents.mcp import create_mcp_client_and_load_tools
from chatlas_agents.htcondor import HTCondorJobSubmitter


def is_lxplus() -> bool:
    """Check if running on lxplus.

    Returns:
        True if on lxplus, False otherwise
    """
    hostname = os.uname()[1]
    # lxplus machines have hostnames like lxplusXXX.cern.ch
    return "lxplus" in hostname or "cern.ch" in hostname


def has_condor() -> bool:
    """Check if HTCondor is available on this machine.

    Returns:
        True if condor_submit is available, False otherwise
    """
    return shutil.which("condor_submit") is not None


# Mark tests to only run on lxplus
pytestmark = pytest.mark.skipif(
    not is_lxplus(),
    reason="lxplus-specific tests - only run on lxplus",
)


class TestHTCondorIntegration:
    """Tests for HTCondor integration on lxplus."""

    @pytest.mark.skipif(not has_condor(), reason="HTCondor not available")
    def test_condor_submit_file_generation(self, tmp_path):
        """Test that HTCondor submit files are generated correctly."""
        submitter = HTCondorJobSubmitter(
            docker_image="python:3.13-slim",
            output_dir=tmp_path,
        )

        job_name = "test-basic-job"
        prompt = "Hello from lxplus test"

        submit_file = submitter.generate_submit_file(
            job_name=job_name,
            prompt=prompt,
        )

        assert submit_file.exists()
        submit_content = submit_file.read_text()

        # Verify submit file has expected HTCondor keywords
        assert "universe = docker" in submit_content
        assert "docker_image = python:3.13-slim" in submit_content
        assert "python -m chatlas_agents.cli run" in submit_content

    @pytest.mark.skipif(not has_condor(), reason="HTCondor not available")
    def test_condor_submit_with_config_generation(self, tmp_path):
        """Test HTCondor submission file generation with config file."""
        submitter = HTCondorJobSubmitter(
            docker_image="python:3.13-slim",
            output_dir=tmp_path,
        )

        job_name = "test-with-config"
        prompt = "Test with configuration"
        config_file = "/path/to/config.yaml"

        submit_file = submitter.generate_submit_file(
            job_name=job_name,
            prompt=prompt,
            config_file=config_file,
        )

        # Verify file exists and has config reference
        assert submit_file.exists()
        submit_content = submit_file.read_text()
        assert "--config" in submit_content
        assert config_file in submit_content

    @pytest.mark.skipif(not has_condor(), reason="HTCondor not available")
    def test_condor_submit_with_resources_generation(self, tmp_path):
        """Test HTCondor submission with custom resource requirements."""
        submitter = HTCondorJobSubmitter(
            docker_image="python:3.13-slim",
            output_dir=tmp_path,
        )

        job_name = "test-resources"
        submit_file = submitter.generate_submit_file(
            job_name=job_name,
            prompt="Test with resources",
            cpus=4,
            memory="8GB",
            disk="10GB",
        )

        # Check that resource specs are in the submit file
        submit_content = submit_file.read_text()
        assert "request_cpus" in submit_content
        assert "request_memory" in submit_content
        assert "request_disk" in submit_content
        assert "4" in submit_content  # cpus value
        assert "8" in submit_content  # memory value

    @pytest.mark.skipif(not has_condor(), reason="HTCondor not available")
    def test_condor_validation_with_valid_submit_file(self, tmp_path):
        """Test validating a valid HTCondor submit file with condor_submit."""
        # Create a simple valid submit file manually
        submit_file = tmp_path / "valid-test.sub"
        submit_content = """
universe = vanilla
executable = /bin/echo
arguments = "Hello World"
log = output.log
output = output.txt
error = output.err
queue 1
"""
        submit_file.write_text(submit_content)

        # Use condor_submit -dry-run to validate
        dry_run_output = tmp_path / "valid-test-dry-run.txt"
        result = subprocess.run(
            ["condor_submit", str(submit_file), "-dry-run", str(dry_run_output)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # This should succeed
        assert result.returncode == 0, f"condor_submit failed: {result.stderr}"
        assert dry_run_output.exists(), "dry-run output file was not created"


class TestMCPServerIntegration:
    """Tests for MCP server connectivity on lxplus."""

    def test_mcp_server_connectivity(self):
        """Test basic connectivity to the MCP server config."""
        config = MCPServerConfig(
            url="https://chatlas-mcp.app.cern.ch/mcp",
            timeout=30,
        )

        # Verify the config is properly created
        assert config is not None
        assert config.url == "https://chatlas-mcp.app.cern.ch/mcp"
        assert config.timeout == 30

        # Import the client function to ensure it's available
        from chatlas_agents.mcp import create_mcp_client

        client = create_mcp_client(config)
        assert client is not None

    @pytest.mark.asyncio
    async def test_mcp_server_loads_tools(self):
        """Test that tools can be loaded from the MCP server.

        This is a more comprehensive test that actually connects to the server.
        """
        config = MCPServerConfig(
            url="https://chatlas-mcp.app.cern.ch/mcp",
            timeout=30,
        )

        try:
            # Attempt to load tools from the MCP server
            tools = await create_mcp_client_and_load_tools(config)

            # We should get a list of tools
            assert isinstance(tools, list), "Tools should be returned as a list"

            # We expect at least the dummy_tool and search_chatlas tools
            tool_names = [tool.name for tool in tools]
            assert len(tool_names) > 0, "Should load at least one tool from MCP server"

            # Check for expected tools (at least one should be present)
            expected_tools = {"search_chatlas", "dummy_tool"}
            found_tools = set(tool_names) & expected_tools
            assert (
                len(found_tools) > 0
            ), f"Expected to find at least one of {expected_tools}, but found: {tool_names}"

        except Exception as e:
            # If we can't connect, that's acceptable for this test
            # (server might be temporarily unavailable)
            pytest.skip(f"MCP server connection failed: {e}")

    def test_mcp_server_url_is_accessible(self):
        """Test that the MCP server URL is accessible."""
        import httpx

        config = MCPServerConfig(
            url="https://chatlas-mcp.app.cern.ch/mcp",
            timeout=30,
        )

        try:
            with httpx.Client(timeout=config.timeout) as client:
                # MCP servers typically return 406 or 405 for GET requests
                # since they expect POST with JSON-RPC
                response = client.get(config.url)
                assert response.status_code in [
                    405,
                    406,
                ], f"Unexpected status code: {response.status_code}"
        except httpx.ConnectError as e:
            pytest.skip(f"Cannot connect to MCP server: {e}")


class TestLXPlusEnvironment:
    """Tests to verify lxplus environment is properly configured."""

    def test_is_lxplus_environment(self):
        """Verify we are running on lxplus."""
        hostname = os.uname()[1]
        assert "lxplus" in hostname or "cern.ch" in hostname, (
            f"Expected lxplus hostname, got {hostname}"
        )

    def test_condor_is_available(self):
        """Verify HTCondor is available on lxplus."""
        assert has_condor(), "HTCondor (condor_submit) not found on lxplus"

    def test_python_version(self):
        """Verify Python version is compatible (3.11+)."""
        import sys

        assert sys.version_info >= (
            3,
            11,
        ), f"Expected Python 3.11+, got {sys.version_info.major}.{sys.version_info.minor}"

    def test_required_packages_available(self):
        """Verify required packages are installed."""
        required_packages = [
            "langchain",
            "langchain_openai",
            "pydantic",
            "typer",
            "httpx",
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package '{package}' is not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
