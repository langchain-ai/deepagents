"""MCP server process management.

This module provides a manager for MCP server processes, allowing them to be
started, stopped, and monitored.
"""

import os
import subprocess
import threading
import time
from typing import Dict, Optional, Any


class McpProcessManager:
    """Manages MCP server processes.

    This class tracks running MCP server processes and provides methods
    to start, stop, and restart them.
    """

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.keep_alive_threads: Dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()

    def register_config(self, server_name: str, config: Dict[str, Any]) -> None:
        """Register an MCP server configuration.

        Args:
            server_name: Name of the MCP server.
            config: MCP server configuration dictionary.
        """
        self.configs[server_name] = config

    def get_server_config(self, server_name: str) -> Dict[str, Any] | None:
        """Get the configuration for an MCP server.

        Args:
            server_name: Name of the MCP server.

        Returns:
            Configuration dictionary, or None if not registered.
        """
        return self.configs.get(server_name)

    def _keep_alive(self, server_name: str, process: subprocess.Popen) -> None:
        """Background thread to keep MCP server alive by periodically writing to stdin.

        Args:
            server_name: Name of the MCP server.
            process: The subprocess.Popen object.
        """
        try:
            while True:
                if process.poll() is not None:
                    # Process has exited
                    break
                if process.stdin and not process.stdin.closed:
                    try:
                        # Write a newline to keep stdin active
                        process.stdin.write('\n')
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        # Pipe broken, process likely exited
                        break
                time.sleep(1)
        except Exception:
            pass
        finally:
            # Clean up thread reference
            with self._thread_lock:
                if server_name in self.keep_alive_threads:
                    del self.keep_alive_threads[server_name]

    def start_server(self, server_name: str) -> Optional[subprocess.Popen]:
        """Start an MCP server process.

        Args:
            server_name: Name of the MCP server to start.

        Returns:
            The Popen process object if started successfully, None otherwise.
        """
        if server_name not in self.configs:
            raise ValueError(f"No configuration found for MCP server '{server_name}'")

        # Check if server is already running
        if self.is_running(server_name):
            # Return existing process
            return self.processes[server_name]

        config = self.configs[server_name]
        server_type = config.get("type", "stdio")

        if server_type != "stdio":
            raise ValueError(f"Unsupported MCP server type: {server_type}")

        command = config.get("command")
        args = config.get("args", [])
        env = config.get("env", {})

        if not command:
            raise ValueError(f"No command specified for MCP server '{server_name}'")

        # Prepare environment
        full_env = {**os.environ, **env}

        try:
            process = subprocess.Popen(
                [command] + args,
                env=full_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            self.processes[server_name] = process

            # Start keep-alive thread
            keep_alive_thread = threading.Thread(
                target=self._keep_alive,
                args=(server_name, process),
                daemon=True,
            )
            keep_alive_thread.start()
            with self._thread_lock:
                self.keep_alive_threads[server_name] = keep_alive_thread

            # Give the process a moment to start up
            time.sleep(0.1)

            # Check if process is still running
            if process.poll() is not None:
                # Process exited immediately
                returncode = process.poll()
                # Clean up thread
                with self._thread_lock:
                    if server_name in self.keep_alive_threads:
                        del self.keep_alive_threads[server_name]
                raise RuntimeError(f"MCP server failed to start with return code: {returncode}")

            return process
        except Exception:
            # Clean up if process was created but failed
            if server_name in self.processes:
                del self.processes[server_name]
            # Clean up thread if created
            with self._thread_lock:
                if server_name in self.keep_alive_threads:
                    del self.keep_alive_threads[server_name]
            raise

    def stop_server(self, server_name: str, force: bool = False) -> bool:
        """Stop an MCP server process.

        Args:
            server_name: Name of the MCP server to stop.
            force: If True, force kill the process. Otherwise, send SIGTERM.

        Returns:
            True if the process was stopped, False if it wasn't running.
        """
        if server_name not in self.processes:
            return False

        process = self.processes[server_name]
        if process.poll() is None:
            # Process is still running
            if force:
                process.kill()
            else:
                process.terminate()

            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Clean up keep-alive thread
        with self._thread_lock:
            if server_name in self.keep_alive_threads:
                del self.keep_alive_threads[server_name]

        # Remove from tracking
        del self.processes[server_name]
        return True

    def restart_server(self, server_name: str) -> Optional[subprocess.Popen]:
        """Restart an MCP server process.

        Args:
            server_name: Name of the MCP server to restart.

        Returns:
            The new Popen process object if started successfully, None otherwise.
        """
        self.stop_server(server_name)
        return self.start_server(server_name)

    def is_running(self, server_name: str) -> bool:
        """Check if an MCP server process is running.

        Args:
            server_name: Name of the MCP server to check.

        Returns:
            True if the process is running, False otherwise.
        """
        if server_name not in self.processes:
            return False

        process = self.processes[server_name]
        return process.poll() is None

    def get_process_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an MCP server process.

        Args:
            server_name: Name of the MCP server.

        Returns:
            Dictionary with process information, or None if not running.
        """
        if server_name not in self.processes:
            return None

        process = self.processes[server_name]
        return {
            "pid": process.pid,
            "returncode": process.poll(),
            "running": process.poll() is None,
        }

    def list_servers(self) -> Dict[str, Dict[str, Any]]:
        """List all registered MCP servers and their status.

        Returns:
            Dictionary mapping server names to their status information.
        """
        result = {}
        for server_name in self.configs:
            process_info = self.get_process_info(server_name)
            result[server_name] = {
                "configured": True,
                "running": process_info["running"] if process_info else False,
                "pid": process_info["pid"] if process_info else None,
                "type": self.configs[server_name].get("type", "unknown"),
            }
        return result

    def cleanup(self) -> None:
        """Stop all running MCP server processes."""
        for server_name in list(self.processes.keys()):
            try:
                self.stop_server(server_name, force=True)
            except Exception:
                pass


# Global MCP process manager instance
_mcp_manager: Optional[McpProcessManager] = None


def get_mcp_manager() -> McpProcessManager:
    """Get or create the global MCP process manager instance.

    Returns:
        McpProcessManager instance.
    """
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = McpProcessManager()
    return _mcp_manager