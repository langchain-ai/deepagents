import json
import os
from typing import List, Literal, Dict, Any, Annotated
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedToolCallId

# Define the Todo type
class Todo:
    def __init__(self, content: str, status: Literal["pending", "in_progress", "completed"]):
        self.content = content
        self.status = status
    
    def to_dict(self):
        return {"content": self.content, "status": self.status}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["content"], data["status"])

# File path for storing todos
TODO_FILE = "todos.json"

def load_todos() -> List[Todo]:
    """Load todos from the JSON file"""
    if not os.path.exists(TODO_FILE):
        return []
    
    try:
        with open(TODO_FILE, 'r') as f:
            data = json.load(f)
            return [Todo.from_dict(todo_data) for todo_data in data]
    except (json.JSONDecodeError, KeyError):
        return []

def save_todos(todos: List[Todo]):
    """Save todos to the JSON file"""
    with open(TODO_FILE, 'w') as f:
        json.dump([todo.to_dict() for todo in todos], f, indent=2)

@tool(description="""Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

#### When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

#### When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

#### Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.

Parameters:
- todos: The updated todo list with all tasks and their current status (each todo should have 'content' and 'status' fields)""")
def write_todos(todos: List[dict], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Write todos to a local JSON file and return a command to update the state"""
    # Convert dict todos to Todo objects
    todo_objects = []
    for todo_dict in todos:
        if isinstance(todo_dict, dict) and "content" in todo_dict and "status" in todo_dict:
            todo_objects.append(Todo(todo_dict["content"], todo_dict["status"]))
        else:
            # Handle case where todo_dict might be a string or other format
            continue
    
    # Save to file
    save_todos(todo_objects)
    
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list with {len(todos)} items and saved to {TODO_FILE}", tool_call_id=tool_call_id)
            ],
        }
    )

@tool(description="Get the current todo list from the local file")
def get_todos(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Load and return the current todos from the local file"""
    todo_objects = load_todos()
    todos_dict = [todo.to_dict() for todo in todo_objects]
    
    return Command(
        update={
            "todos": todos_dict,
            "messages": [
                ToolMessage(f"Loaded {len(todos_dict)} todos from {TODO_FILE}", tool_call_id=tool_call_id)
            ],
        }
    )
