from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langsmith.schemas import AgentContext, FileEntry
from langsmith.utils import LangSmithNotFoundError
from pydantic import Field

from agent import CONTEXT_HUB_REPO, build_agent
from seed_context_hub import load_seed_files

if TYPE_CHECKING:
    from langchain_core.outputs import ChatResult
    from langgraph.graph.state import CompiledStateGraph

_ids = count()


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that accepts tools and records each system prompt it sees."""

    system_prompts: list[str] = Field(default_factory=list)

    def bind_tools(self, tools: object, **kwargs: Any) -> ScriptedModel:  # noqa: ARG002
        return self

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        self.system_prompts.append(messages[0].text)
        return super()._generate(messages, **kwargs)


class FakeContextHub:
    """In-memory stand-in for the LangSmith Context Hub client."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.commits = 1 if files else 0

    def pull_agent(self, identifier: str) -> AgentContext:
        assert identifier == CONTEXT_HUB_REPO
        if not self.commits:
            raise LangSmithNotFoundError(identifier)
        entries = {path: FileEntry(content=content) for path, content in self.files.items()}
        return AgentContext(commit_id=uuid4(), commit_hash=f"{self.commits:08x}", files=entries)

    def push_agent(self, identifier: str, *, files: dict[str, FileEntry | None], **kwargs: Any) -> str:  # noqa: ARG002
        for path, entry in files.items():
            if entry is None:
                self.files.pop(path, None)
            else:
                self.files[path] = entry.content
        self.commits += 1
        return f"https://smith.langchain.com/context/langsmith-memory-agent/{self.commits:08x}"


def tool_call(name: str, **args: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def chat(agent: CompiledStateGraph, user_id: str, thread_id: str, text: str) -> list[BaseMessage]:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context={"user_id": user_id},
    )
    return result["messages"]


def tool_results(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages if isinstance(message, ToolMessage)]


def test_user_memory_is_isolated_while_agent_memory_is_shared() -> None:
    hub = FakeContextHub({"AGENTS.md": "# Agent memory\n"})
    store = InMemoryStore()
    alice_model = ScriptedModel(
        messages=iter(
            [
                tool_call("write_file", file_path="/memories/user/profile.md", content="- Name: Alice"),
                tool_call(
                    "edit_file",
                    file_path="/memories/agent/AGENTS.md",
                    old_string="# Agent memory\n",
                    new_string="# Agent memory\n- Incident summaries end with Follow-ups.\n",
                ),
                AIMessage("Saved."),
            ]
        )
    )
    chat(build_agent(model=alice_model, hub_client=hub, store=store), "alice", "t1", "I'm Alice. Team rule: end with Follow-ups.")

    bob_model = ScriptedModel(messages=iter([AIMessage("Hi Bob.")]))
    chat(build_agent(model=bob_model, hub_client=hub, store=store), "bob", "t2", "What do you know about me?")

    assert store.get(("memory-agent", "alice", "files"), "/profile.md").value["content"] == "- Name: Alice"
    assert store.get(("memory-agent", "bob", "files"), "/profile.md") is None
    assert hub.commits == 2
    assert "Follow-ups" in bob_model.system_prompts[0]
    assert "Name: Alice" not in bob_model.system_prompts[0]


def test_context_hub_skills_load_but_are_read_only() -> None:
    seed = {path: entry.content for path, entry in load_seed_files().items()}
    hub = FakeContextHub(seed)
    model = ScriptedModel(
        messages=iter(
            [
                tool_call("write_file", file_path="/memories/agent/skills/incident-summary/SKILL.md", content="tampered"),
                AIMessage("I can't change curated skills."),
            ]
        )
    )
    messages = chat(build_agent(model=model, hub_client=hub, store=InMemoryStore()), "alice", "t1", "Rewrite the skill.")

    assert "incident-summary" in model.system_prompts[0]
    assert "permission denied" in tool_results(messages)[0].lower()
    assert hub.files == seed
    assert hub.commits == 1


def test_past_conversations_are_searchable_per_user() -> None:
    store = InMemoryStore()
    model = ScriptedModel(
        messages=iter(
            [
                AIMessage("Here is your on-call handoff checklist."),
                tool_call("search_past_conversations", query="handoff"),
                AIMessage("Last time we drafted a handoff checklist."),
                tool_call("search_past_conversations", query="handoff"),
                AIMessage("We haven't talked before."),
            ]
        )
    )
    agent = build_agent(model=model, hub_client=FakeContextHub(), store=store, checkpointer=InMemorySaver())

    chat(agent, "alice", "t1", "Draft an on-call handoff checklist.")
    alice_later = tool_results(chat(agent, "alice", "t2", "What did we work on last time?"))
    bob = tool_results(chat(agent, "bob", "t3", "What did we work on last time?"))

    assert "on-call handoff checklist" in alice_later[0]
    assert bob == ["No earlier conversations found."]
