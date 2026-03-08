"""
Middleware to handle Gemini's MALFORMED_FUNCTION_CALL errors.

This middleware intercepts model responses and handles the case where Gemini 2.5 Flash
generates malformed function calls (often Python code like `print(...)` instead of
proper JSON tool calls).

Solution approaches:
1. Detect MALFORMED_FUNCTION_CALL in response_metadata
2. Retry the model call with a modified prompt emphasizing JSON format
3. Clean message history to remove dangling MALFORMED responses

Based on GitHub issue: https://github.com/langchain-ai/langchain-google/issues/725
"""

import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

logger = logging.getLogger(__name__)


class FixMalformedGeminiMiddleware(AgentMiddleware):
    """Middleware to handle and retry MALFORMED_FUNCTION_CALL errors from Gemini models."""

    def __init__(self, max_retries: int = 2, fallback_to_text: bool = True):
        """
        Initialize the middleware.

        Args:
            max_retries: Maximum number of retries when MALFORMED_FUNCTION_CALL occurs (default: 2)
            fallback_to_text: If True, fall back to text response after max retries (default: True)
        """
        self.max_retries = max_retries
        self.fallback_to_text = fallback_to_text
        super().__init__()

    @property
    def name(self) -> str:
        """Return the middleware name."""
        return "FixMalformedGeminiMiddleware"

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """
        Clean up any existing MALFORMED messages from the message history.

        This prevents accumulation of malformed responses that could confuse the model.
        """
        logger.info("=" * 80)
        logger.info("🔍 FixMalformedGeminiMiddleware.before_agent() CALLED")
        logger.info("=" * 80)

        messages = state["messages"]
        logger.debug(f"   Total messages in state: {len(messages) if messages else 0}")

        if not messages or len(messages) == 0:
            logger.debug("   No messages to clean, returning None")
            return None

        cleaned_messages = []
        removed_count = 0

        for idx, msg in enumerate(messages):
            # Check if this is an AI message with MALFORMED finish reason
            if hasattr(msg, 'type') and msg.type == 'ai':
                finish_reason = msg.response_metadata.get('finish_reason') if hasattr(msg, 'response_metadata') else None
                logger.debug(f"   Message {idx}: type={msg.type}, finish_reason={finish_reason}")

                if finish_reason == 'MALFORMED_FUNCTION_CALL':
                    # Skip this message - don't add to cleaned list
                    removed_count += 1
                    logger.warning(
                        f"🧹 Removed MALFORMED_FUNCTION_CALL message from history "
                        f"(id: {msg.id if hasattr(msg, 'id') else 'unknown'})"
                    )
                    continue

            # Keep non-malformed messages
            cleaned_messages.append(msg)

        if removed_count > 0:
            logger.info(f"🧹 Cleaned {removed_count} MALFORMED messages from history")
            return {"messages": Overwrite(cleaned_messages)}

        logger.debug("   No MALFORMED messages found, returning None")
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """
        Intercept model calls to detect and handle MALFORMED_FUNCTION_CALL errors.

        Strategy:
        1. Call the model
        2. Check if response has MALFORMED_FUNCTION_CALL finish reason
        3. If yes:
           a. Try adding explicit instructions about JSON format
           b. Retry up to max_retries times
           c. If still failing, fall back to text response (no tool calls)
        """
        logger.info("=" * 80)
        logger.info("🔍 FixMalformedGeminiMiddleware.awrap_model_call() CALLED")
        logger.info("=" * 80)

        retry_count = 0

        while retry_count <= self.max_retries:
            logger.info(f"📞 Calling model (attempt {retry_count + 1}/{self.max_retries + 1})")

            # Call the model
            response = await handler(request)
            logger.debug(f"   Response received, type: {type(response)}")

            # Check if we got a valid response
            if not response.result or len(response.result) == 0:
                logger.warning("⚠️  Empty response from model")
                return response

            # Get the AI message
            ai_msg = response.result[0] if isinstance(response.result, list) else response.result
            logger.debug(f"   AI message type: {type(ai_msg)}")
            logger.debug(f"   AI message has response_metadata: {hasattr(ai_msg, 'response_metadata')}")

            # Check for MALFORMED_FUNCTION_CALL
            if hasattr(ai_msg, 'response_metadata'):
                finish_reason = ai_msg.response_metadata.get('finish_reason')
                logger.info(f"   ✅ Finish reason: {finish_reason}")
                logger.debug(f"   Full response_metadata: {ai_msg.response_metadata}")

                if finish_reason == 'MALFORMED_FUNCTION_CALL':
                    retry_count += 1
                    logger.error("=" * 80)
                    logger.error(f"🔴 MALFORMED_FUNCTION_CALL DETECTED!")
                    logger.error(f"   Attempt: {retry_count}/{self.max_retries + 1}")
                    logger.error("=" * 80)

                    # Log metadata for debugging
                    if hasattr(ai_msg, 'additional_kwargs'):
                        logger.debug(f"   Additional kwargs: {ai_msg.additional_kwargs}")
                    if hasattr(ai_msg, 'content'):
                        content_preview = str(ai_msg.content)[:200] if ai_msg.content else "None"
                        logger.error(f"   Content preview: {content_preview}")

                    logger.debug(f"   Response metadata keys: {list(ai_msg.response_metadata.keys())}")
                    logger.debug(f"   Model name: {ai_msg.response_metadata.get('model_name')}")
                    logger.debug(f"   Usage metadata: {ai_msg.response_metadata.get('usage_metadata')}")

                    # If we've hit max retries
                    if retry_count > self.max_retries:
                        if self.fallback_to_text:
                            logger.warning("=" * 80)
                            logger.warning("🔄 MAX RETRIES REACHED - Falling back to text-only response")
                            logger.warning("=" * 80)
                            # Create a helpful text response instead of malformed tool call
                            fallback_msg = AIMessage(
                                content=(
                                    "I encountered a technical issue while trying to use my tools. "
                                    "Let me provide a text-based response instead. "
                                    "Could you please rephrase your question or break it into smaller parts?"
                                ),
                                response_metadata={"finish_reason": "STOP", "fallback": True}
                            )
                            return ModelResponse(
                                result=[fallback_msg],
                                structured_response=response.structured_response
                            )
                        else:
                            logger.error("=" * 80)
                            logger.error("❌ MAX RETRIES REACHED - Returning malformed response")
                            logger.error("=" * 80)
                            return response

                    # Retry: Add explicit instruction to use proper JSON for tool calls
                    logger.warning("=" * 80)
                    logger.warning(f"♻️  RETRYING WITH ENHANCED INSTRUCTIONS")
                    logger.warning(f"   Next attempt: {retry_count + 1}/{self.max_retries + 1}")
                    logger.warning("=" * 80)

                    # Modify the request messages to add clarifying instruction
                    # Add a human message emphasizing JSON format
                    retry_instruction = HumanMessage(
                        content=(
                            "IMPORTANT INSTRUCTION: When using tools, you MUST use valid JSON format for arguments. "
                            "Do NOT write Python code like print(...) or any other programming syntax. "
                            "Use ONLY the proper tool calling format with valid JSON arguments."
                        )
                    )

                    # Add instruction to messages
                    modified_messages = list(request.messages) + [retry_instruction]

                    # Create modified request with the new messages
                    modified_request = replace(
                        request,
                        messages=modified_messages
                    )

                    logger.debug(f"   Added retry instruction, new message count: {len(modified_messages)}")

                    # Update request for next iteration
                    request = modified_request

                    # Retry with modified request
                    continue

            # Success - no MALFORMED error
            if retry_count > 0:
                logger.info("=" * 80)
                logger.info(f"✅ RETRY SUCCESSFUL after {retry_count} attempts!")
                logger.info("=" * 80)
            else:
                logger.info("=" * 80)
                logger.info("✅ Model call SUCCESSFUL on first attempt")
                logger.info("=" * 80)

            return response

        # Should never reach here, but just in case
        logger.error("⚠️  Exited retry loop unexpectedly, returning last response")
        return response

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        Synchronous version - not implemented as we use async astream.

        This middleware is designed for async usage.
        """
        msg = (
            "Synchronous wrap_model_call not implemented for FixMalformedGeminiMiddleware. "
            "This middleware is designed for async agent execution (astream, ainvoke). "
            "Use the async version or invoke the agent asynchronously."
        )
        raise NotImplementedError(msg)
