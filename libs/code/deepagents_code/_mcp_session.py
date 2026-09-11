"""Keep rejected MCP results actionable without accepting invalid output."""

from mcp import ClientSession
from mcp.types import CallToolResult, TextContent


class _MCPClientSession(ClientSession):
    """Report output-schema violations through the normal tool-error path."""

    async def validate_tool_result(self, name: str, result: CallToolResult) -> None:
        """Validate output, preserving rejected content as failed tool output.

        Args:
            name: Tool whose declared output schema applies.
            result: Response to validate, or mark as an error on rejection.

        Raises:
            RuntimeError: A failure unrelated to output-schema validation.
        """
        try:
            await super().validate_tool_result(name, result)
        except RuntimeError as exc:
            # The SDK currently exposes validation failures as plain RuntimeError.
            # Keep transport/discovery failures out of this compatibility path.
            if not str(exc).startswith(
                (
                    (
                        f"Tool {name} has an output schema "
                        "but did not return structured content"
                    ),
                    f"Invalid structured content returned by tool {name}:",
                    f"Invalid schema for tool {name}:",
                )
            ):
                raise
            result.is_error = True
            result.structured_content = None
            result.content.insert(
                0,
                TextContent(
                    type="text",
                    text=(
                        "MCP response failed output-schema validation. "
                        "The following content is failed tool output. "
                        "The operation may have run; do not repeat writes without "
                        "checking their outcome."
                    ),
                ),
            )
