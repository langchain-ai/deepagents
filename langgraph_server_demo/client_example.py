"""Example Python client for the LangGraph Server.

This shows how to interact with your deepagent via the HTTP API.
"""

import httpx
import json


BASE_URL = "http://localhost:2024"


def create_thread():
    """Create a new conversation thread."""
    response = httpx.post(f"{BASE_URL}/threads")
    thread = response.json()
    print(f"Created thread: {thread['thread_id']}")
    return thread["thread_id"]


def send_message(thread_id: str, message: str):
    """Send a message to the agent and get response."""
    payload = {
        "assistant_id": "deepagent",
        "input": {
            "messages": [{"role": "user", "content": message}]
        }
    }

    response = httpx.post(
        f"{BASE_URL}/threads/{thread_id}/runs",
        json=payload,
        timeout=60.0,
    )

    result = response.json()
    print(f"\nAgent response:")
    print(json.dumps(result, indent=2))
    return result


def stream_message(thread_id: str, message: str):
    """Stream a message response from the agent."""
    payload = {
        "assistant_id": "deepagent",
        "input": {
            "messages": [{"role": "user", "content": message}]
        },
        "stream_mode": "messages",
    }

    with httpx.stream(
        "POST",
        f"{BASE_URL}/threads/{thread_id}/runs/stream",
        json=payload,
        timeout=60.0,
    ) as response:
        print("\nStreaming response:")
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                if data and data != "[DONE]":
                    try:
                        event = json.loads(data)
                        print(event)
                    except json.JSONDecodeError:
                        pass


def get_thread_state(thread_id: str):
    """Get the current state of a thread."""
    response = httpx.get(f"{BASE_URL}/threads/{thread_id}/state")
    state = response.json()
    print(f"\nThread state:")
    print(json.dumps(state, indent=2))
    return state


def main():
    """Run example interactions."""
    # Create a thread
    thread_id = create_thread()

    # Send a simple message
    print("\n=== Sending message ===")
    send_message(thread_id, "Hello! What can you help me with?")

    # Stream a response
    print("\n=== Streaming message ===")
    stream_message(thread_id, "List the files in the current directory")

    # Get thread state
    print("\n=== Getting thread state ===")
    get_thread_state(thread_id)


if __name__ == "__main__":
    main()
