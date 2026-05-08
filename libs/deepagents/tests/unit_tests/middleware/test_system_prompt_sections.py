"""Unit tests for the named system-prompt sections API."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage

from deepagents.middleware._system_prompt import SystemPromptAssemblerMiddleware
from deepagents.middleware._utils import (
    BUILTIN_SECTION_ORDER,
    SystemSection,
    _BASE_SYSTEM_MESSAGE_KEY,
    _SYSTEM_SECTIONS_KEY,
    append_to_system_message,
    assemble_system_message,
    get_system_section,
    set_system_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> MagicMock:
    model = MagicMock()
    model.__class__ = MagicMock  # not ChatAnthropic
    return model


def _make_request(
    *,
    system_message: SystemMessage | None = None,
    model_settings: dict[str, Any] | None = None,
) -> ModelRequest[None]:
    return ModelRequest(
        model=_make_model(),
        messages=[],
        system_message=system_message,
        tools=[],
        runtime=None,  # type: ignore[arg-type]
        state={"messages": []},
        model_settings=model_settings or {},
    )


def _fake_handler(captured: list[ModelRequest[Any]]) -> Callable[[ModelRequest[Any]], ModelResponse[Any]]:
    def handler(req: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.append(req)
        return ModelResponse(result=[AIMessage(content="ok")])

    return handler


# ---------------------------------------------------------------------------
# set_system_section / get_system_section
# ---------------------------------------------------------------------------


class TestSetGetSystemSection:
    def test_set_stores_section(self) -> None:
        req = _make_request()
        req2 = set_system_section(req, "memory", "My memory content")

        sections: dict[str, SystemSection] = req2.model_settings[_SYSTEM_SECTIONS_KEY]
        assert "memory" in sections
        assert sections["memory"].content == "My memory content"

    def test_set_does_not_mutate_original(self) -> None:
        req = _make_request()
        set_system_section(req, "memory", "content")
        assert _SYSTEM_SECTIONS_KEY not in req.model_settings

    def test_set_replaces_existing_section(self) -> None:
        req = _make_request()
        req = set_system_section(req, "skills", "old content")
        req = set_system_section(req, "skills", "new content")

        sections: dict[str, SystemSection] = req.model_settings[_SYSTEM_SECTIONS_KEY]
        assert sections["skills"].content == "new content"

    def test_set_preserves_other_sections(self) -> None:
        req = _make_request()
        req = set_system_section(req, "filesystem", "fs content")
        req = set_system_section(req, "memory", "memory content")

        sections: dict[str, SystemSection] = req.model_settings[_SYSTEM_SECTIONS_KEY]
        assert "filesystem" in sections
        assert "memory" in sections

    def test_set_preserves_other_model_settings(self) -> None:
        req = _make_request(model_settings={"existing_key": 42})
        req = set_system_section(req, "memory", "content")

        assert req.model_settings["existing_key"] == 42

    def test_builtin_section_order_applied(self) -> None:
        req = _make_request()
        req = set_system_section(req, "memory", "content")
        sections: dict[str, SystemSection] = req.model_settings[_SYSTEM_SECTIONS_KEY]
        assert sections["memory"].order == BUILTIN_SECTION_ORDER["memory"]

    def test_unknown_key_defaults_order_100(self) -> None:
        req = _make_request()
        req = set_system_section(req, "custom_section", "content")
        sections: dict[str, SystemSection] = req.model_settings[_SYSTEM_SECTIONS_KEY]
        assert sections["custom_section"].order == 100

    def test_explicit_order_overrides_builtin(self) -> None:
        req = _make_request()
        req = set_system_section(req, "memory", "content", order=5)
        sections: dict[str, SystemSection] = req.model_settings[_SYSTEM_SECTIONS_KEY]
        assert sections["memory"].order == 5

    def test_get_returns_content(self) -> None:
        req = _make_request()
        req = set_system_section(req, "skills", "Skills text")
        assert get_system_section(req, "skills") == "Skills text"

    def test_get_returns_none_when_absent(self) -> None:
        req = _make_request()
        assert get_system_section(req, "memory") is None


# ---------------------------------------------------------------------------
# assemble_system_message
# ---------------------------------------------------------------------------


class TestAssembleSystemMessage:
    def test_returns_none_when_nothing(self) -> None:
        result = assemble_system_message(base=None, sections={})
        assert result is None

    def test_base_only(self) -> None:
        base = SystemMessage(content="Base prompt")
        result = assemble_system_message(base=base, sections={})
        assert result is not None
        texts = [b["text"] for b in result.content_blocks if isinstance(b, dict)]  # type: ignore[index]
        assert any("Base prompt" in t for t in texts)

    def test_section_only(self) -> None:
        sections = {"memory": SystemSection(key="memory", content="Memory stuff", order=40)}
        result = assemble_system_message(base=None, sections=sections)
        assert result is not None
        texts = [b["text"] for b in result.content_blocks if isinstance(b, dict)]  # type: ignore[index]
        assert any("Memory stuff" in t for t in texts)

    def test_sections_sorted_by_order(self) -> None:
        sections = {
            "memory": SystemSection(key="memory", content="MEMORY", order=40),
            "filesystem": SystemSection(key="filesystem", content="FILESYSTEM", order=10),
            "skills": SystemSection(key="skills", content="SKILLS", order=30),
        }
        result = assemble_system_message(base=None, sections=sections)
        assert result is not None
        texts = [b["text"] for b in result.content_blocks if isinstance(b, dict)]  # type: ignore[index]
        combined = "\n".join(texts)
        assert combined.index("FILESYSTEM") < combined.index("SKILLS") < combined.index("MEMORY")

    def test_sections_after_base(self) -> None:
        base = SystemMessage(content="BASE")
        sections = {"memory": SystemSection(key="memory", content="MEMORY", order=40)}
        result = assemble_system_message(base=base, sections=sections)
        assert result is not None
        texts = [b["text"] for b in result.content_blocks if isinstance(b, dict)]  # type: ignore[index]
        combined = "\n".join(texts)
        assert combined.index("BASE") < combined.index("MEMORY")

    def test_separator_between_blocks(self) -> None:
        base = SystemMessage(content="BASE")
        sections = {"memory": SystemSection(key="memory", content="MEMORY", order=40)}
        result = assemble_system_message(base=base, sections=sections)
        assert result is not None
        section_block = result.content_blocks[-1]
        assert isinstance(section_block, dict)
        assert section_block["text"].startswith("\n\n")  # type: ignore[index]

    def test_empty_section_skipped(self) -> None:
        sections = {
            "memory": SystemSection(key="memory", content="", order=40),
            "skills": SystemSection(key="skills", content="SKILLS", order=30),
        }
        result = assemble_system_message(base=None, sections=sections)
        assert result is not None
        assert len(result.content_blocks) == 1

    def test_cache_control_not_added_for_non_anthropic(self) -> None:
        sections = {"memory": SystemSection(key="memory", content="MEMORY", cache_control=True, order=40)}
        result = assemble_system_message(base=None, sections=sections, model=_make_model())
        assert result is not None
        block = result.content_blocks[-1]
        assert isinstance(block, dict)
        assert "cache_control" not in block

    def test_no_cache_control_when_flag_false(self) -> None:
        sections = {"memory": SystemSection(key="memory", content="MEMORY", cache_control=False, order=40)}
        result = assemble_system_message(base=None, sections=sections, model=None)
        assert result is not None
        block = result.content_blocks[-1]
        assert isinstance(block, dict)
        assert "cache_control" not in block


# ---------------------------------------------------------------------------
# SystemPromptAssemblerMiddleware
# ---------------------------------------------------------------------------


class TestSystemPromptAssemblerMiddleware:
    def _middleware(self) -> SystemPromptAssemblerMiddleware:
        return SystemPromptAssemblerMiddleware()

    def test_passthrough_when_no_sections(self) -> None:
        mw = self._middleware()
        req = _make_request(system_message=SystemMessage(content="Base"))
        captured: list[ModelRequest[Any]] = []
        mw.wrap_model_call(req, _fake_handler(captured))

        passed = captured[0]
        # system_message unchanged when no sections are present
        assert passed.system_message is not None
        assert _SYSTEM_SECTIONS_KEY not in passed.model_settings
        assert _BASE_SYSTEM_MESSAGE_KEY not in passed.model_settings

    def test_assembles_sections_into_system_message(self) -> None:
        mw = self._middleware()
        req = _make_request()
        req = set_system_section(req, "filesystem", "FS content")
        req = set_system_section(req, "memory", "Memory content")

        captured: list[ModelRequest[Any]] = []
        mw.wrap_model_call(req, _fake_handler(captured))

        passed = captured[0]
        assert passed.system_message is not None
        full_text = " ".join(
            b["text"]  # type: ignore[index]
            for b in passed.system_message.content_blocks
            if isinstance(b, dict)
        )
        assert "FS content" in full_text
        assert "Memory content" in full_text

    def test_uses_saved_base_for_ordering(self) -> None:
        """Assembler rebuilds from original base, not the direct-appended system_message."""
        mw = self._middleware()
        req = _make_request()
        # Set memory first (order=40), then filesystem (order=10) — registration order reversed.
        req = set_system_section(req, "memory", "MEMORY")
        req = set_system_section(req, "filesystem", "FILESYSTEM")

        captured: list[ModelRequest[Any]] = []
        mw.wrap_model_call(req, _fake_handler(captured))

        passed = captured[0]
        texts = [b["text"] for b in passed.system_message.content_blocks if isinstance(b, dict)]  # type: ignore[index, union-attr]
        combined = "\n".join(texts)
        # filesystem (order=10) should come before memory (order=40)
        assert combined.index("FILESYSTEM") < combined.index("MEMORY")

    def test_strips_sections_key_from_model_settings(self) -> None:
        mw = self._middleware()
        req = _make_request(model_settings={"other": "value"})
        req = set_system_section(req, "skills", "Skills text")

        captured: list[ModelRequest[Any]] = []
        mw.wrap_model_call(req, _fake_handler(captured))

        passed = captured[0]
        assert _SYSTEM_SECTIONS_KEY not in passed.model_settings
        assert passed.model_settings.get("other") == "value"

    def test_preserves_base_system_message(self) -> None:
        mw = self._middleware()
        base = SystemMessage(content="User base prompt")
        req = _make_request(system_message=base)
        req = set_system_section(req, "memory", "Memory")

        captured: list[ModelRequest[Any]] = []
        mw.wrap_model_call(req, _fake_handler(captured))

        passed = captured[0]
        full_text = " ".join(
            b["text"]  # type: ignore[index]
            for b in passed.system_message.content_blocks  # type: ignore[union-attr]
            if isinstance(b, dict)
        )
        assert "User base prompt" in full_text
        assert "Memory" in full_text

    @pytest.mark.asyncio
    async def test_async_assembles_sections(self) -> None:
        mw = self._middleware()
        req = _make_request()
        req = set_system_section(req, "filesystem", "FS async content")

        captured: list[ModelRequest[Any]] = []

        async def async_handler(r: ModelRequest[Any]) -> ModelResponse[Any]:
            captured.append(r)
            return ModelResponse(result=[AIMessage(content="ok")])

        await mw.awrap_model_call(req, async_handler)

        passed = captured[0]
        assert _SYSTEM_SECTIONS_KEY not in passed.model_settings
        assert passed.system_message is not None


# ---------------------------------------------------------------------------
# Backward compatibility: append_to_system_message deprecation
# ---------------------------------------------------------------------------


class TestAppendToSystemMessageDeprecation:
    def test_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = append_to_system_message(None, "some text")
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert result.content_blocks[-1]["text"] == "some text"  # type: ignore[index]

    def test_still_works_with_existing_content(self) -> None:
        existing = SystemMessage(content="Existing")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = append_to_system_message(existing, "Appended")
        texts = [b["text"] for b in result.content_blocks if isinstance(b, dict)]  # type: ignore[index]
        assert any("Existing" in t for t in texts)
        assert any("Appended" in t for t in texts)
