"""Codex provider integration for deepagents-cli."""

from __future__ import annotations

import subprocess
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class CodexChatModel(BaseChatModel):
    """Chat model wrapper that delegates generation to `codex exec`."""

    model: str = Field(default="o4-mini")
    timeout_seconds: int = Field(default=120)

    @property
    def _llm_type(self) -> str:
        """Return the provider identifier for LangChain diagnostics."""
        return "codex"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Generate a response by invoking `codex exec` with the prompt text."""
        prompt = "\n".join(
            content
            for message in messages
            if (content := self._stringify_message_content(message)).strip()
        )
        if not prompt:
            prompt = " "

        try:
            result = subprocess.run(
                ["codex", "exec", "--model", self.model, prompt],  # noqa: S603, S607
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            msg = "Codex CLI is not installed or not on PATH"
            raise RuntimeError(msg) from exc
        except subprocess.TimeoutExpired as exc:
            msg = "Codex CLI timed out while generating a response"
            raise RuntimeError(msg) from exc
        except OSError as exc:
            msg = f"Failed to execute Codex CLI: {exc}"
            raise RuntimeError(msg) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip() or "unknown error"
            msg = f"Codex CLI failed: {stderr}"
            raise RuntimeError(msg)

        content = result.stdout.strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @staticmethod
    def _stringify_message_content(message: BaseMessage) -> str:
        """Flatten LangChain message content to plain text."""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return " ".join(chunks)
        return str(content)
