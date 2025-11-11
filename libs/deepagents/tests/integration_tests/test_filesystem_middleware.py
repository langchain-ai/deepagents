import uuid

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import (
    WRITE_FILE_TOOL_DESCRIPTION,
    FileData,
    FilesystemMiddleware,
)

from ..utils import ResearchMiddleware, get_la_liga_standings, get_nba_standings, get_nfl_standings, get_premier_league_standings


def build_composite_state_backend(runtime, *, routes):
    built_routes = {}
    for prefix, backend_or_factory in routes.items():
        if callable(backend_or_factory):
            built_routes[prefix] = backend_or_factory(runtime)
        else:
            built_routes[prefix] = backend_or_factory
    default_state = StateBackend(runtime)
    return CompositeBackend(default=default_state, routes=built_routes)


@pytest.mark.requires("langchain_anthropic")
class TestFilesystem:
    def test_filesystem_system_prompt_override(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                    system_prompt="In every single response, you must say the word 'pokemon'! You love it!",
                )
            ],
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pokemon" in response["messages"][1].text.lower()

    def test_filesystem_system_prompt_override_with_composite_backend(self):
        backend = lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=backend,
                    system_prompt="In every single response, you must say the word 'pizza'! You love it!",
                )
            ],
            store=InMemoryStore(),
        )
        response = agent.invoke({"messages": [HumanMessage(content="What do you like?")]})
        assert "pizza" in response["messages"][1].text.lower()

    def test_filesystem_tool_prompt_override(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert tools["write_file"].description == WRITE_FILE_TOOL_DESCRIPTION
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"

    def test_filesystem_tool_prompt_override_with_longterm_memory(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                    custom_tool_descriptions={
                        "ls": "Charmander",
                        "read_file": "Bulbasaur",
                        "edit_file": "Squirtle",
                    },
                )
            ],
            store=InMemoryStore(),
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "ls" in tools
        assert tools["ls"].description == "Charmander"
        assert "read_file" in tools
        assert tools["read_file"].description == "Bulbasaur"
        assert "write_file" in tools
        assert tools["write_file"].description == WRITE_FILE_TOOL_DESCRIPTION
        assert "edit_file" in tools
        assert tools["edit_file"].description == "Squirtle"

    def test_ls_longterm_without_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="List your files in root")],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
        assert "/pizza.txt" in ls_message.text
        assert "/pokemon/squirtle.txt" not in ls_message.text
        assert "/memories/test.txt" not in ls_message.text
        assert "/memories/pokemon/charmander.txt" not in ls_message.text
        # Verify directories are listed with trailing /
        assert "/pokemon/" in ls_message.text
        assert "/memories/" in ls_message.text

    def test_ls_longterm_with_path(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="List all of your files in the /pokemon directory")],
                "files": {
                    "/pizza.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/pokemon/squirtle.txt": FileData(
                        content=["Splash"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
        assert "/pokemon/squirtle.txt" in ls_message.text
        assert "/memories/pokemon/charmander.txt" not in ls_message.text

    def test_read_file_longterm_local_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from the local filesystem")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
        assert read_file_message is not None
        assert "Goodbye world" in read_file_message.content

    def test_read_file_longterm_store_file(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read test.txt from the memories directory")],
                "files": {
                    "/test.txt": FileData(
                        content=["Goodbye world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
        assert read_file_message is not None
        assert "Hello world" in read_file_message.content

    def test_read_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/test.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Ember"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=(lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Read the contents of the file about charmander from the memories directory.")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        ai_msg_w_toolcall = next(
            message
            for message in messages
            if message.type == "ai"
            and any(tc["name"] == "read_file" and tc["args"]["file_path"] == "/memories/pokemon/charmander.txt" for tc in message.tool_calls)
        )
        assert ai_msg_w_toolcall is not None

    def test_write_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Write a haiku about Charmander to the memories directory in /charmander.txt, use the word 'fiery'")
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(message for message in messages if message.type == "tool" and message.name == "write_file")
        assert write_file_message is not None
        file_item = store.get(("filesystem",), "/charmander.txt")
        assert file_item is not None
        assert any("fiery" in c for c in file_item.value["content"]) or any("Fiery" in c for c in file_item.value["content"])

    def test_write_file_fail_already_exists_in_store(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/charmander.txt",
            {
                "content": ["Hello world"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Write a haiku about Charmander to /memories/charmander.txt, use the word 'fiery'")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(message for message in messages if message.type == "tool" and message.name == "write_file")
        assert write_file_message is not None
        assert "Cannot write" in write_file_message.content

    def test_write_file_fail_already_exists_in_local(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Write a haiku about Charmander to /charmander.txt, use the word 'fiery'")],
                "files": {
                    "/charmander.txt": FileData(
                        content=["Hello world"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    )
                },
            },
            config=config,
        )
        messages = response["messages"]
        write_file_message = next(message for message in messages if message.type == "tool" and message.name == "write_file")
        assert write_file_message is not None
        assert "Cannot write" in write_file_message.content

    def test_edit_file_longterm(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/charmander.txt",
            {
                "content": ["The fire burns brightly. The fire burns hot."],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Edit the file about charmander in the memories directory, to replace all instances of the word 'fire' with 'embers'"
                    )
                ],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        edit_file_message = next(message for message in messages if message.type == "tool" and message.name == "edit_file")
        assert edit_file_message is not None
        assert store.get(("filesystem",), "/charmander.txt").value["content"] == ["The embers burns brightly. The embers burns hot."]

    def test_longterm_memory_multiple_tools(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        assert_longterm_mem_tools(agent, store)

    def test_longterm_memory_multiple_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        backend = lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})
        agent = create_deep_agent(backend=backend, checkpointer=checkpointer, store=store)
        assert_longterm_mem_tools(agent, store)

    def test_shortterm_memory_multiple_tools_deepagent(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        agent = create_deep_agent(backend=lambda rt: StateBackend(rt), checkpointer=checkpointer, store=store)
        assert_shortterm_mem_tools(agent)

    def test_tool_call_with_tokens_exceeding_limit(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            tools=[get_nba_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the NBA standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 10000
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"].keys())

    def test_tool_call_with_tokens_exceeding_custom_limit(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            tools=[get_nfl_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                    tool_token_limit_before_evict=1000,
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the NFL standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"].keys())

    def test_command_with_tool_call(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            tools=[get_la_liga_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                    tool_token_limit_before_evict=1000,
                )
            ],
        )
        response = agent.invoke(
            {"messages": [HumanMessage(content="Get the la liga standings using your tool. If the tool returns bad results, tell the user.")]}
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 1
        assert any("large_tool_results" in key for key in response["files"].keys())

    def test_command_with_tool_call_existing_state(self):
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            tools=[get_premier_league_standings],
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                    tool_token_limit_before_evict=1000,
                ),
                ResearchMiddleware(),
            ],
        )
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Get the premier league standings using your tool. If the tool returns bad results, tell the user.")
                ],
            }
        )
        assert response["messages"][2].type == "tool"
        assert len(response["messages"][2].content) < 1500
        assert len(response["files"].keys()) == 2
        assert any("large_tool_results" in key for key in response["files"].keys())
        assert "/test.txt" in response["files"].keys()
        assert "research" in response

    def test_glob_search_shortterm_only(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                )
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files")],
                "files": {
                    "/test.py": FileData(
                        content=["import os"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/readme.txt": FileData(
                        content=["Documentation"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/test.py" in glob_message.content
        assert "/main.py" in glob_message.content
        assert "/readme.txt" not in glob_message.content

    def test_glob_search_longterm_only(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/config.py",
            {
                "content": ["DEBUG = True"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/settings.py",
            {
                "content": ["SECRET_KEY = 'abc'"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/notes.txt",
            {
                "content": ["Important notes"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files in /memories")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/memories/config.py" in glob_message.content
        assert "/memories/settings.py" in glob_message.content
        assert "/memories/notes.txt" not in glob_message.content

    def test_glob_search_mixed_memory(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/longterm.py",
            {
                "content": ["# Longterm file"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/longterm.txt",
            {
                "content": ["Text file"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use glob to find all Python files")],
                "files": {
                    "/shortterm.py": FileData(
                        content=["# Shortterm file"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/shortterm.txt": FileData(
                        content=["Another text file"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        glob_message = next(message for message in messages if message.type == "tool" and message.name == "glob")
        assert "/shortterm.py" in glob_message.content
        assert "/memories/longterm.py" in glob_message.content
        assert "/shortterm.txt" not in glob_message.content
        assert "/memories/longterm.txt" not in glob_message.content

    def test_grep_search_shortterm_only(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: StateBackend(rt),
                )
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files containing the word 'import'")],
                "files": {
                    "/test.py": FileData(
                        content=["import os", "import sys"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/helper.py": FileData(
                        content=["import json"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        assert "/test.py" in grep_message.content
        assert "/helper.py" in grep_message.content
        assert "/main.py" not in grep_message.content

    def test_grep_search_longterm_only(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/pokemon/charmander.txt",
            {
                "content": ["Charmander is a fire type", "It evolves into Charmeleon"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/squirtle.txt",
            {
                "content": ["Squirtle is a water type", "It evolves into Wartortle"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/pokemon/bulbasaur.txt",
            {
                "content": ["Bulbasaur is a grass type"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files in the memories directory containing the word 'fire'")],
                "files": {},
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        assert "/memories/pokemon/charmander.txt" in grep_message.content
        assert "/memories/pokemon/squirtle.txt" not in grep_message.content
        assert "/memories/pokemon/bulbasaur.txt" not in grep_message.content

    def test_grep_search_mixed_memory(self):
        checkpointer = MemorySaver()
        store = InMemoryStore()
        store.put(
            ("filesystem",),
            "/longterm_config.py",
            {
                "content": ["DEBUG = True", "TESTING = False"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        store.put(
            ("filesystem",),
            "/longterm_settings.py",
            {
                "content": ["SECRET_KEY = 'abc'"],
                "created_at": "2021-01-01",
                "modified_at": "2021-01-01",
            },
        )
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(
                    backend=lambda rt: build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))}),
                )
            ],
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        response = agent.invoke(
            {
                "messages": [HumanMessage(content="Use grep to find all files containing 'DEBUG'")],
                "files": {
                    "/shortterm_config.py": FileData(
                        content=["DEBUG = False", "VERBOSE = True"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                    "/shortterm_main.py": FileData(
                        content=["def main(): pass"],
                        created_at="2021-01-01",
                        modified_at="2021-01-01",
                    ),
                },
            },
            config=config,
        )
        messages = response["messages"]
        grep_message = next(message for message in messages if message.type == "tool" and message.name == "grep")
        print(grep_message.content)
        assert "/shortterm_config.py" in grep_message.content
        assert "/memories/longterm_config.py" in grep_message.content
        assert "/shortterm_main.py" not in grep_message.content
        assert "/memories/longterm_settings.py" not in grep_message.content

    def test_default_backend_fallback(self):
        checkpointer = MemorySaver()
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware()  # No backend specified
            ],
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": uuid.uuid4()}}

        response = agent.invoke(
            {"messages": [HumanMessage(content="Write 'Hello World' to /test.txt")]},
            config=config,
        )

        assert "/test.txt" in response["files"]
        assert any("Hello World" in line for line in response["files"]["/test.txt"]["content"])

        response = agent.invoke(
            {"messages": [HumanMessage(content="Read /test.txt")]},
            config=config,
        )
        messages = response["messages"]
        read_message = next(msg for msg in messages if msg.type == "tool" and msg.name == "read_file")
        assert "Hello World" in read_message.content

    def test_execute_tool_filtered_for_non_sandbox_backend(self):
        """Verify execute tool is filtered out when backend doesn't support it."""
        from langchain.agents import AgentMiddleware

        from deepagents.backends.protocol import ExecuteResponse

        # Track what tools are passed to the model
        captured_tools = []

        class CapturingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                # Capture tool names
                captured_tools.clear()
                captured_tools.extend([tool.name if hasattr(tool, "name") else tool.get("name") for tool in request.tools])
                return handler(request)

        # Test with StateBackend (no execution support)
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
                CapturingMiddleware(),
            ],
        )

        agent.invoke({"messages": [HumanMessage(content="List files")]})

        # Execute tool should NOT be in the tools passed to model
        assert "execute" not in captured_tools
        assert "read_file" in captured_tools
        assert "write_file" in captured_tools

        # Test with sandbox backend (has execution support)
        class MockSandboxBackend(StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(output="test", exit_code=0, truncated=False)

        agent_with_sandbox = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(backend=lambda rt: MockSandboxBackend(rt)),
                CapturingMiddleware(),
            ],
        )

        captured_tools.clear()
        agent_with_sandbox.invoke({"messages": [HumanMessage(content="List files")]})

        # Execute tool SHOULD be in the tools passed to model
        assert "execute" in captured_tools
        assert "read_file" in captured_tools

    def test_system_prompt_includes_execute_instructions_only_when_supported(self):
        """Verify EXECUTION_SYSTEM_PROMPT is only added when backend supports execution."""
        from langchain.agents import AgentMiddleware

        from deepagents.backends.protocol import ExecuteResponse

        # Track system prompts passed to the model
        captured_prompts = []

        class CapturingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                captured_prompts.clear()
                if request.system_prompt:
                    captured_prompts.append(request.system_prompt)
                return handler(request)

        # Test with StateBackend (no execution support)
        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
                CapturingMiddleware(),
            ],
        )

        agent.invoke({"messages": [HumanMessage(content="List files")]})

        # System prompt should NOT include execute instructions
        assert len(captured_prompts) > 0
        prompt = captured_prompts[0]
        assert "execute" not in prompt.lower() or "Execute Tool" not in prompt

        # Test with sandbox backend (has execution support)
        class MockSandboxBackend(StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(output="test", exit_code=0, truncated=False)

        agent_with_sandbox = create_agent(
            model=ChatAnthropic(model="claude-sonnet-4-20250514"),
            middleware=[
                FilesystemMiddleware(backend=lambda rt: MockSandboxBackend(rt)),
                CapturingMiddleware(),
            ],
        )

        captured_prompts.clear()
        agent_with_sandbox.invoke({"messages": [HumanMessage(content="List files")]})

        # System prompt SHOULD include execute instructions
        assert len(captured_prompts) > 0
        prompt = captured_prompts[0]
        assert "Execute Tool" in prompt or "execute" in prompt

    def test_composite_backend_execution_support_detection(self):
        """Verify _supports_execution correctly detects CompositeBackend capabilities."""
        from deepagents.backends.protocol import ExecuteResponse
        from deepagents.middleware.filesystem import _supports_execution

        # Mock sandbox backend
        class MockSandboxBackend(StateBackend):
            def execute(self, command: str) -> ExecuteResponse:
                return ExecuteResponse(output="test", exit_code=0, truncated=False)

        # Create runtimes
        state = {"messages": [], "files": {}}
        rt = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )

        # Test CompositeBackend with sandbox default
        comp_with_sandbox = CompositeBackend(
            default=MockSandboxBackend(rt),
            routes={"/memories/": StoreBackend(rt)},
        )
        assert _supports_execution(comp_with_sandbox)

        # Test CompositeBackend with non-sandbox default
        comp_without_sandbox = CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": StoreBackend(rt)},
        )
        assert not _supports_execution(comp_without_sandbox)


# Take actions on multiple threads to test longterm memory
def assert_longterm_mem_tools(agent, store):
    # Write a longterm memory file
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    agent.invoke(
        {"messages": [HumanMessage(content="Write a haiku about Charmander to /memories/charmander.txt, use the word 'fiery'")]},
        config=config,
    )
    namespaces = store.list_namespaces()
    assert len(namespaces) == 1
    assert namespaces[0] == ("filesystem",)
    file_item = store.get(("filesystem",), "/charmander.txt")
    assert file_item is not None
    assert file_item.key == "/charmander.txt"

    # Read the longterm memory file
    config2 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander from /memories/charmander.txt")]},
        config=config2,
    )
    messages = response["messages"]
    read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
    assert "fiery" in read_file_message.content or "Fiery" in read_file_message.content

    # List all of the files in longterm memory
    config3 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in the memories directory at /memories")]},
        config=config3,
    )
    messages = response["messages"]
    ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
    assert "/memories/charmander.txt" in ls_message.content

    # Edit the longterm memory file
    config4 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="Edit the haiku about Charmander at /memories/charmander.txt to use the word 'ember'")]},
        config=config4,
    )
    file_item = store.get(("filesystem",), "/charmander.txt")
    assert file_item is not None
    assert file_item.key == "/charmander.txt"
    assert any("ember" in c for c in file_item.value["content"]) or any("Ember" in c for c in file_item.value["content"])

    # Read the longterm memory file
    config5 = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander at /memories/charmander.txt")]},
        config=config5,
    )
    messages = response["messages"]
    read_file_message = next(message for message in messages if message.type == "tool" and message.name == "read_file")
    assert "ember" in read_file_message.content or "Ember" in read_file_message.content


def assert_shortterm_mem_tools(agent):
    # Write a shortterm memory file
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    response = agent.invoke(
        {"messages": [HumanMessage(content="Write a haiku about Charmander to /charmander.txt, use the word 'fiery'")]},
        config=config,
    )
    files = response["files"]
    assert "/charmander.txt" in files

    # Read the shortterm memory file
    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander from /charmander.txt")]},
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(message for message in reversed(messages) if message.type == "tool" and message.name == "read_file")
    assert "fiery" in read_file_message.content or "Fiery" in read_file_message.content

    # List all of the files in shortterm memory
    response = agent.invoke(
        {"messages": [HumanMessage(content="List all of the files in your filesystem")]},
        config=config,
    )
    messages = response["messages"]
    ls_message = next(message for message in messages if message.type == "tool" and message.name == "ls")
    assert "/charmander.txt" in ls_message.content

    # Edit the shortterm memory file
    response = agent.invoke(
        {"messages": [HumanMessage(content="Edit the haiku about Charmander to use the word 'ember'")]},
        config=config,
    )
    files = response["files"]
    assert "/charmander.txt" in files
    assert any("ember" in c for c in files["/charmander.txt"]["content"]) or any("Ember" in c for c in files["/charmander.txt"]["content"])

    # Read the shortterm memory file
    response = agent.invoke(
        {"messages": [HumanMessage(content="Read the haiku about Charmander at /charmander.txt")]},
        config=config,
    )
    messages = response["messages"]
    read_file_message = next(message for message in reversed(messages) if message.type == "tool" and message.name == "read_file")
    assert "ember" in read_file_message.content or "Ember" in read_file_message.content
