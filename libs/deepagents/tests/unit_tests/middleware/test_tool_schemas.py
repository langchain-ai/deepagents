"""Unit tests for tool schema validation."""

from langchain_core.tools import StructuredTool

from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware


class TestFilesystemToolSchemas:
    """Test that filesystem tool JSON schemas have types and descriptions for all args."""

    def test_all_filesystem_tools_have_arg_descriptions(self) -> None:
        """Verify all filesystem tool args have type and description in JSON schema.

        Uses tool_call_schema.model_json_schema() which is the schema passed to the LLM.
        """
        # Create the middleware to get the tools
        backend = StateBackend(None)  # type: ignore[arg-type]
        middleware = FilesystemMiddleware(backend=backend)
        tools = middleware.tools

        # Expected tools and their user-facing args (excludes `runtime` which is internal)
        expected_tools = {
            "ls": ["path"],
            "read_file": ["file_path", "offset", "limit"],
            "write_file": ["file_path", "content"],
            "edit_file": ["file_path", "old_string", "new_string", "replace_all"],
            "glob": ["pattern", "path"],
            "grep": ["pattern", "path", "glob", "output_mode"],
            "execute": ["command"],
        }

        tool_map = {tool.name: tool for tool in tools}

        for tool_name, expected_args in expected_tools.items():
            assert tool_name in tool_map, f"Tool '{tool_name}' not found in filesystem tools"
            tool = tool_map[tool_name]

            # Get the JSON schema that's passed to the LLM
            schema = tool.tool_call_schema.model_json_schema()
            properties = schema.get("properties", {})

            for arg_name in expected_args:
                assert arg_name in properties, f"Arg '{arg_name}' not found in schema for tool '{tool_name}'"
                arg_schema = properties[arg_name]

                # Check type is present
                has_type = "type" in arg_schema or "anyOf" in arg_schema or "$ref" in arg_schema
                assert has_type, f"Arg '{arg_name}' in tool '{tool_name}' is missing type in JSON schema"

                # Check description is present
                assert "description" in arg_schema, (
                    f"Arg '{arg_name}' in tool '{tool_name}' is missing description in JSON schema. "
                    f"Add an Annotated type hint with a description string."
                )

    def test_sync_async_schema_parity(self) -> None:
        """Verify sync and async functions produce identical JSON schemas.

        This ensures that the sync_* and async_* function pairs would generate
        the same tool schema if used independently.
        """
        backend = StateBackend(None)  # type: ignore[arg-type]
        middleware = FilesystemMiddleware(backend=backend)
        tools = middleware.tools

        for tool in tools:
            # Create temporary tools from sync and async functions independently
            sync_tool = StructuredTool.from_function(func=tool.func, name=f"{tool.name}_sync")
            async_tool = StructuredTool.from_function(coroutine=tool.coroutine, name=f"{tool.name}_async")

            sync_schema = sync_tool.tool_call_schema.model_json_schema()
            async_schema = async_tool.tool_call_schema.model_json_schema()

            # Remove fields that differ by design (title from name, description from docstring)
            for schema in (sync_schema, async_schema):
                schema.pop("title", None)
                schema.pop("description", None)

            assert sync_schema == async_schema, (
                f"Tool '{tool.name}' has mismatched JSON schemas between sync and async functions.\n"
                f"Sync schema: {sync_schema}\n"
                f"Async schema: {async_schema}"
            )
