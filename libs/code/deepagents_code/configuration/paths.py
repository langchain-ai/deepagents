"""Fixed operating-system paths for managed configuration."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def managed_config_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Return the fixed managed-config path for the current operating system."""
    active_platform = sys.platform if platform is None else platform
    active_environ = os.environ if environ is None else environ
    if active_platform == "darwin":
        return Path("/Library/Application Support/dcode/managed_config.toml")
    if active_platform == "win32":
        program_data = active_environ.get("ProgramData") or active_environ.get(
            "PROGRAMDATA"
        )
        root = Path(program_data) if program_data else Path("C:/ProgramData")
        return root / "dcode" / "managed_config.toml"
    return Path("/etc/dcode/managed_config.toml")
