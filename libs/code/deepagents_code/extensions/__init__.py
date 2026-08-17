"""Python extension system for dcode.

One pipeline discovers, trust-gates, loads, and hosts every kind of unit an
extension can contribute. P0 ships middleware, tools, and commands; a new unit
kind is a new verb on `ExtensionAPI`, not a new loader.

Extension authors only need `ExtensionAPI` and `CommandContext`:

```python
from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    d.register_tool(my_tool)
```

See `EXTENSIONS.md` in this package for the author guide.
"""

from deepagents_code.extensions.api import ExtensionAPI
from deepagents_code.extensions.models import (
    CommandContext,
    CommandHandler,
    ExtensionError,
    LoadedExtension,
    SourceInfo,
    UnitOrigin,
    UnitScope,
    UnitSource,
)
from deepagents_code.extensions.registry import ExtensionRegistry
from deepagents_code.extensions.runtime import (
    ExtensionLoadResult,
    load_extensions,
    load_extensions_blocking,
    project_extensions_trusted,
    shutdown_extensions,
)
from deepagents_code.extensions.settings import (
    ExtensionSettings,
    TrustPolicy,
    load_extension_settings,
)

__all__ = [
    "CommandContext",
    "CommandHandler",
    "ExtensionAPI",
    "ExtensionError",
    "ExtensionLoadResult",
    "ExtensionRegistry",
    "ExtensionSettings",
    "LoadedExtension",
    "SourceInfo",
    "TrustPolicy",
    "UnitOrigin",
    "UnitScope",
    "UnitSource",
    "load_extension_settings",
    "load_extensions",
    "load_extensions_blocking",
    "project_extensions_trusted",
    "shutdown_extensions",
]
