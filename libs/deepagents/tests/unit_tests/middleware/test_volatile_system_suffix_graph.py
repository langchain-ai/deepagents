"""End-to-end tests for `create_deep_agent(volatile_system_suffix=...)`.

These verify the assembled middleware stack so the prompt-cache breakpoint
lands where intended:

1. With a suffix set, the breakpoint is on the stable block, NOT the suffix.
2. The suffix is the last system content block.
3. Backward compat: with the param unset, breakpoint placement is unchanged.
4. The `AGENTS.md`/memory breakpoint is preserved when memory is configured.
5. Non-Anthropic models add no `cache_control`; the suffix still trails.

Plus that the middleware is wired into all three agent stacks (main agent and
both subagent builders).
"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware import volatile_system_suffix as vss_module
from tests.unit_tests.chat_model import GenericFakeChatModel

_VOLATILE = "Current date: 2026-06-01. You are speaking with Ada."


def _make_capturing_anthropic(captured: list[SystemMessage], responses: list[AIMessage] | None = None) -> ChatAnthropic:
    """A `ChatAnthropic` whose `_generate` records system messages and returns scripted replies."""
    reply_iter = iter(responses or [AIMessage(content="ok")])

    class _CapturingAnthropic(ChatAnthropic):
        def _generate(self, messages: list, *args: object, **kwargs: object) -> ChatResult:
            captured.extend(m for m in messages if isinstance(m, SystemMessage))
            try:
                message = next(reply_iter)
            except StopIteration:
                message = AIMessage(content="ok")
            return ChatResult(generations=[ChatGeneration(message=message)])

    return _CapturingAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake")  # type: ignore[call-arg]


# --- 1 & 2: breakpoint on stable block, suffix is last ----------------------


def test_breakpoint_on_stable_block_not_volatile_suffix() -> None:
    """With a suffix, the cache breakpoint stays on the base prompt; the suffix trails uncached."""
    captured: list[SystemMessage] = []
    agent = create_deep_agent(model=_make_capturing_anthropic(captured), volatile_system_suffix=_VOLATILE)
    agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert captured, "Model never received a SystemMessage"
    blocks = captured[0].content_blocks

    # Suffix is the final block...
    assert _VOLATILE in blocks[-1].get("text", "")
    assert "cache_control" not in blocks[-1]
    # ...and the breakpoint is on the stable block before it.
    assert blocks[-2].get("cache_control", {}).get("type") == "ephemeral"


# --- 3: backward compatibility ----------------------------------------------


def test_without_volatile_suffix_breakpoint_unchanged() -> None:
    """Unset param leaves the breakpoint on the last (stable) block, as before."""
    captured: list[SystemMessage] = []
    agent = create_deep_agent(model=_make_capturing_anthropic(captured))
    agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert captured, "Model never received a SystemMessage"
    blocks = captured[0].content_blocks
    assert blocks[-1].get("cache_control", {}).get("type") == "ephemeral"


# --- 4: memory breakpoint preserved -----------------------------------------


def test_memory_breakpoint_preserved_with_volatile_suffix(tmp_path: Path) -> None:
    """With memory configured, both the base and AGENTS.md breakpoints survive; suffix trails."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    memory_path = str(tmp_path / "AGENTS.md")
    backend.upload_files([(memory_path, b"# Memory\nBe concise.")])

    captured: list[SystemMessage] = []
    agent = create_deep_agent(
        backend=backend,
        memory=[memory_path],
        model=_make_capturing_anthropic(captured),
        volatile_system_suffix=_VOLATILE,
    )
    agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert captured, "Model never received a SystemMessage"
    blocks = captured[0].content_blocks

    # Volatile suffix is last and uncached.
    assert _VOLATILE in blocks[-1].get("text", "")
    assert "cache_control" not in blocks[-1]
    # The AGENTS.md memory block keeps its breakpoint (Memory middleware)...
    assert "<agent_memory>" in blocks[-2].get("text", "")
    assert blocks[-2].get("cache_control", {}).get("type") == "ephemeral"
    # ...and the stable prefix keeps its breakpoint (AnthropicPromptCachingMiddleware
    # tags the last stable block, which sits just before the memory block).
    assert blocks[-3].get("cache_control", {}).get("type") == "ephemeral"


# --- 5: non-Anthropic provider ----------------------------------------------


def test_non_anthropic_no_cache_control_suffix_trails() -> None:
    """Non-Anthropic models get the suffix as the trailing block and no cache_control anywhere."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    agent = create_deep_agent(model=model, volatile_system_suffix=_VOLATILE)
    agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert model.call_history, "Model was never called"
    system_messages = [m for m in model.call_history[0]["messages"] if isinstance(m, SystemMessage)]
    assert system_messages, "Model never received a SystemMessage"
    blocks = system_messages[0].content_blocks

    assert _VOLATILE in blocks[-1].get("text", "")
    assert all("cache_control" not in block for block in blocks)


# --- wiring into all three stacks -------------------------------------------


def test_wired_into_main_and_both_subagent_stacks(monkeypatch) -> None:
    """The middleware is constructed once for the main agent and once per subagent builder."""
    suffixes: list[object] = []
    original_init = vss_module.VolatileSystemSuffixMiddleware.__init__

    def spy_init(self: object, suffix: object) -> None:
        suffixes.append(suffix)
        original_init(self, suffix)  # type: ignore[arg-type]

    monkeypatch.setattr(vss_module.VolatileSystemSuffixMiddleware, "__init__", spy_init)

    create_deep_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        volatile_system_suffix=_VOLATILE,
        subagents=[
            {
                "name": "researcher",
                "description": "Researches things.",
                "system_prompt": "You are a researcher.",
            }
        ],
    )

    # Main agent + auto-added general-purpose subagent + the declared inline subagent.
    assert len(suffixes) == 3
    assert all(s == _VOLATILE for s in suffixes)


def test_not_wired_when_suffix_unset(monkeypatch) -> None:
    """No middleware is constructed when the param is unset."""
    suffixes: list[object] = []
    original_init = vss_module.VolatileSystemSuffixMiddleware.__init__

    def spy_init(self: object, suffix: object) -> None:
        suffixes.append(suffix)
        original_init(self, suffix)  # type: ignore[arg-type]

    monkeypatch.setattr(vss_module.VolatileSystemSuffixMiddleware, "__init__", spy_init)

    create_deep_agent(model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])))

    assert suffixes == []


def test_suffix_reaches_inline_subagent_at_runtime() -> None:
    """A declarative subagent's system message ends with the volatile suffix, uncached."""
    subagent_captured: list[SystemMessage] = []
    subagent_model = _make_capturing_anthropic(subagent_captured)

    parent_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": "Look it up", "subagent_type": "researcher"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    agent = create_deep_agent(
        model=parent_model,
        volatile_system_suffix=_VOLATILE,
        subagents=[
            {
                "name": "researcher",
                "description": "Researches things.",
                "system_prompt": "You are a researcher.",
                "model": subagent_model,
            }
        ],
    )
    agent.invoke({"messages": [HumanMessage(content="research X")]})

    assert subagent_captured, "Subagent model never received a SystemMessage"
    blocks = subagent_captured[0].content_blocks
    assert _VOLATILE in blocks[-1].get("text", "")
    assert "cache_control" not in blocks[-1]
    assert "You are a researcher." in blocks[0].get("text", "")
