"""Personal assistant with persistent memory across sessions.

Demonstrates:
- StoreBackend: files under /memories/ persist across all conversations
- CompositeBackend: routes /memories/ to StoreBackend, everything else to StateBackend
- MemoryMiddleware: loads /memories/profile.md into the system prompt automatically
- Multi-user isolation: each user_id gets its own private namespace in the store

The key insight: StoreBackend keeps files in LangGraph's BaseStore, which lives
outside the conversation thread. Even when you start a new conversation (new
thread_id), the agent's memory file is still there.

Three store backends are supported (swap via --store):
- memory  : InMemoryStore — resets when the process exits (default, good for demos)
- sqlite  : SqliteStore   — persists to a local .db file across process runs
                            requires: pip install langgraph-checkpoint-sqlite
- postgres: PostgresStore — persists to Postgres, suitable for production
                            requires: pip install langgraph-checkpoint-postgres
                            set DATABASE_URL env var before running

Usage:
    # Run the built-in demo (two sessions, same user, in-memory store)
    python assistant.py --demo

    # Interactive with in-memory store (resets on exit)
    python assistant.py --user alice "I prefer Python and I'm working on a FastAPI backend"
    python assistant.py --user alice "What do you know about me?"

    # Persist across terminal runs with SQLite
    python assistant.py --store sqlite --user alice "I prefer Python"
    python assistant.py --store sqlite --user alice "What do you know about me?"

    # Persist with Postgres (set DATABASE_URL first)
    python assistant.py --store postgres --user alice "I prefer Python"

    # Show that different users have isolated memories
    python assistant.py --user bob "My name is Bob and I use TypeScript"
    python assistant.py --user alice "What do you know about me?"  # Alice's memory, not Bob's
"""

import argparse
import os
import sqlite3
import uuid

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend

SYSTEM_PROMPT = """You are a personal assistant with persistent memory.

## Your memory file
You have a persistent memory file at /memories/profile.md that survives across
all conversations. Use it to remember important facts about the user.

**When to update memory:**
- User shares their name, role, or background
- User mentions ongoing projects or goals
- User states a preference (language, style, format, tools)
- User corrects you or gives feedback on your work

**How to update memory:**
- First conversation: use write_file to create /memories/profile.md
- Later conversations: use edit_file to update it (the file already exists)

Keep the profile file concise and factual — bullet points work well.
Never store API keys, passwords, or other credentials.

## At the start of each conversation
Your memory (if any) is already loaded into your context above. Use it.
If the user asks what you remember about them, tell them directly.
"""


def create_assistant(user_id: str, store: InMemoryStore) -> object:
    """Create a personal assistant agent for a given user.

    Uses CompositeBackend to split storage:
    - /memories/ → StoreBackend (persistent, survives across sessions)
    - everything else → StateBackend (ephemeral, cleared each session)

    Args:
        user_id: Unique identifier for the user. Used to namespace their memory
                 so different users don't share data.
        store: The LangGraph store to use. In production, swap InMemoryStore for
               a persistent store (e.g. PostgresStore, SqliteStore).
    """
    # Each user gets their own namespace in the store.
    # All their memory files are stored under ("user:<user_id>", "memories").
    store_backend = StoreBackend(
        store=store,
        namespace=lambda rt: (f"user:{user_id}", "memories"),
    )

    # Route /memories/ to the persistent StoreBackend.
    # All other paths (scratch files, working files) go to ephemeral StateBackend.
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": store_backend},
    )

    return create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt=SYSTEM_PROMPT,
        # MemoryMiddleware automatically loads this file into the system prompt
        # at the start of each conversation. If it doesn't exist yet, the agent
        # sees "(No memory loaded)" and knows to create it.
        memory=["/memories/profile.md"],
        backend=backend,
        # Checkpointer enables multi-turn conversations within a session.
        # We use InMemorySaver here; in production use SqliteSaver or PostgresSaver.
        checkpointer=InMemorySaver(),
    )


def chat(agent: object, thread_id: str, message: str) -> str:
    """Send a message to the agent and return its response.

    Args:
        agent: The compiled deep agent.
        thread_id: Conversation thread ID. Use the same ID to continue a
                   conversation, or a new UUID for a fresh session.
        message: The user's message.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def build_store(backend: str) -> BaseStore:
    """Return a configured store for the given backend name.

    Args:
        backend: One of "memory", "sqlite", or "postgres".

    "memory"  — InMemoryStore. Resets when the process exits. Good for demos
                and tests.

    "sqlite"  — SqliteStore backed by a local file (assistant_memory.db).
                Survives across separate process runs. Requires:
                    pip install langgraph-checkpoint-sqlite
                Uses a direct sqlite3 connection with isolation_level=None
                (autocommit) as required by SqliteStore — same as from_conn_string.

    "postgres" — PostgresStore. Requires:
                    pip install langgraph-checkpoint-postgres
                Set DATABASE_URL to a valid Postgres connection string before
                running, e.g.:
                    export DATABASE_URL=postgresql://user:pass@localhost/mydb
    """
    if backend == "memory":
        return InMemoryStore()

    if backend == "sqlite":
        try:
            from langgraph.store.sqlite import SqliteStore
        except ImportError:
            raise ImportError(
                "SqliteStore requires langgraph-checkpoint-sqlite.\n"
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from None
        conn = sqlite3.connect("assistant_memory.db", isolation_level=None)
        store = SqliteStore(conn)
        store.setup()
        return store

    if backend == "postgres":
        try:
            from langgraph.store.postgres import PostgresStore
        except ImportError:
            raise ImportError(
                "PostgresStore requires langgraph-checkpoint-postgres.\n"
                "Install with: pip install langgraph-checkpoint-postgres"
            ) from None
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise ValueError(
                "Set DATABASE_URL before using the postgres store.\n"
                "Example: export DATABASE_URL=postgresql://user:pass@localhost/mydb"
            )
        with PostgresStore.from_conn_string(db_url) as store:
            store.setup()
            return store

    raise ValueError(f"Unknown store backend {backend!r}. Choose from: memory, sqlite, postgres")


def demo() -> None:
    """Run a two-session demo showing memory persistence.

    Session 1: Alice introduces herself and shares her preferences.
    Session 2: New thread (simulates a new conversation), but the agent
               still remembers Alice because /memories/profile.md persisted.
    """
    print("=" * 60)
    print("Persistent Memory Assistant Demo")
    print("=" * 60)

    # A single store shared across all sessions for this demo.
    # In a real app this would be backed by a database.
    store = InMemoryStore()
    agent = create_assistant(user_id="alice", store=store)

    # --- Session 1 ---
    session_1 = str(uuid.uuid4())
    print(f"\n[Session 1 | thread: {session_1[:8]}...]")

    msg1 = "Hi! I'm Alice. I'm a senior Python developer building a FastAPI backend. I strongly prefer type hints and async/await patterns."
    print(f"User: {msg1}")
    response1 = chat(agent, session_1, msg1)
    print(f"Assistant: {response1}\n")

    msg2 = "One more thing — I like concise, bullet-point answers rather than long prose."
    print(f"User: {msg2}")
    response2 = chat(agent, session_1, msg2)
    print(f"Assistant: {response2}\n")

    # --- Session 2 (new thread = new conversation) ---
    session_2 = str(uuid.uuid4())
    print(f"[Session 2 | thread: {session_2[:8]}...] (new conversation)")

    msg3 = "Hey, what do you remember about me?"
    print(f"User: {msg3}")
    response3 = chat(agent, session_2, msg3)
    print(f"Assistant: {response3}\n")

    msg4 = "Can you show me a quick example of the kind of code I'd write?"
    print(f"User: {msg4}")
    response4 = chat(agent, session_2, msg4)
    print(f"Assistant: {response4}\n")

    print("=" * 60)
    print("Notice: Session 2 knew Alice's name, stack, and preferences")
    print("even though it started a brand new conversation thread.")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personal assistant with persistent cross-session memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python assistant.py --demo
  python assistant.py --user alice "I prefer Python and FastAPI"
  python assistant.py --user alice "What do you know about me?"
  python assistant.py --store sqlite --user alice "I prefer Python"
  python assistant.py --store sqlite --user alice "What do you know about me?"
  python assistant.py --user bob "I work on data pipelines in Spark"
  python assistant.py --user alice "What do you know about me?"  # Alice's memory, not Bob's
        """,
    )
    parser.add_argument("--demo", action="store_true", help="Run the built-in two-session demo")
    parser.add_argument("--user", default="default", help="User ID (default: 'default')")
    parser.add_argument(
        "--store",
        default="memory",
        choices=["memory", "sqlite", "postgres"],
        help="Storage backend. 'memory' resets on exit; 'sqlite' and 'postgres' persist across runs. (default: memory)",
    )
    parser.add_argument("message", nargs="?", help="Message to send")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if not args.message:
        parser.error("Provide a message or use --demo")

    store = build_store(args.store)
    agent = create_assistant(user_id=args.user, store=store)
    thread_id = str(uuid.uuid4())

    print(f"[user={args.user} | store={args.store} | thread={thread_id[:8]}...]")
    print(f"User: {args.message}")
    response = chat(agent, thread_id, args.message)
    print(f"Assistant: {response}")


if __name__ == "__main__":
    main()