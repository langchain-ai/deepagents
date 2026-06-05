"""langchain-quickjs: persistent JS REPL middleware for agents."""

from quickjs_rs import ModuleScope

from langchain_quickjs._extensions import (
    ExtensionContext,
    ExtensionError,
    HostFunction,
    InterpreterExtension,
)
from langchain_quickjs._ptc import PTCOption
from langchain_quickjs._swarm import (
    SwarmExtension,
    SwarmSubAgent,
    SwarmTaskMode,
    VariantCache,
    create_swarm_task_tool,
    swarm,
)
from langchain_quickjs.middleware import CodeInterpreterMiddleware

__all__ = [
    "CodeInterpreterMiddleware",
    "ExtensionContext",
    "ExtensionError",
    "HostFunction",
    "InterpreterExtension",
    "ModuleScope",
    "PTCOption",
    "SwarmExtension",
    "SwarmSubAgent",
    "SwarmTaskMode",
    "VariantCache",
    "create_swarm_task_tool",
    "swarm",
]
