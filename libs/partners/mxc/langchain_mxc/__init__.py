"""Microsoft Execution Containers (MXC) sandbox integration for Deep Agents.

The package exposes one backend, `MXCSandbox`. Shell commands issued through a
Deep Agent's `execute` tool run inside an MXC process sandbox enforced by the
operating system (process container on Windows, Seatbelt on macOS, bubblewrap
on Linux). File tools keep running in the host process, limited to the
project folder by Deep Agents' own path checks.
"""

from langchain_mxc._executor import MXCExecutorNotFoundError, MXCUnsupportedPlatformError
from langchain_mxc._version import __version__
from langchain_mxc.sandbox import MXCSandbox

__all__ = [
    "MXCExecutorNotFoundError",
    "MXCSandbox",
    "MXCUnsupportedPlatformError",
    "__version__",
]
