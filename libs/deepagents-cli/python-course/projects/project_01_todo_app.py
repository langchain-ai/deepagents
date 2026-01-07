"""
FINAL PROJECT 1: TODO APPLICATION
==================================

Build a complete command-line todo application using OOP.

Features:
- Add tasks with priority
- Mark tasks as complete
- Delete tasks
- List all tasks
- Save/load from file
- Search tasks

This project combines:
- Classes and Objects
- File I/O
- Exception Handling
- Collections (lists, dicts)
- User input/output
"""

import json
import os
from datetime import datetime


class Task:
    """Represents a single task."""
    
    def __init__(self, title, priority=1, description=""):
        self.title = title
        self.priority = priority
        self.description = description
        self.completed = False
        self.created_at = datetime.now().isoformat()
    
    def mark_complete(self):
        """Mark task as completed."""
        self.completed = True
    
    def mark_incomplete(self):
        """Mark task as incomplete."""
        self.completed = False
    
    def to_dict(self):
        """Convert task to dictionary for JSON."""
        return {
            "title": self.title,
            "priority": self.priority,
            "description": self.description,
            "completed": self.completed,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create task from dictionary."""
        task = cls(data["title"], data["priority"], data["description"])
        task.completed = data["completed"]
        task.created_at = data["created_at"]
        return task
    
    def __str__(self):
        status = "✓" if self.completed else " "
        return f"[{status}] {self.title} (Priority: {self.priority})"


class TodoApp:
    """Main todo application."""
    
    def __init__(self, filename="todos.json"):
        self.filename = filename
        self.tasks = []
        self.load_tasks()
    
    def add_task(self, title, priority=1, description=""):
        """Add a new task."""
        task = Task(title, priority, description)
        self.tasks.append(task)
        print(f"✓ Task added: {title}")
        return task
    
    def remove_task(self, index):
        """Remove task by index."""
        try:
            task = self.tasks[index]
            self.tasks.pop(index)
            print(f"✓ Removed: {task.title}")
        except IndexError:
            print("Error: Invalid task number")
    
    def mark_complete(self, index):
        """Mark task as complete."""
        try:
            self.tasks[index].mark_complete()
            print(f"✓ Task completed: {self.tasks[index].title}")
        except IndexError:
            print("Error: Invalid task number")
    
    def list_tasks(self):
        """Display all tasks."""
        if not self.tasks:
            print("No tasks. Add one to get started!")
            return
        
        print("\n" + "=" * 60)
        print("TODO LIST")
        print("=" * 60)
        for i, task in enumerate(self.tasks, 1):
            status = "✓" if task.completed else " "
            priority_star = "★" * task.priority
            print(f"{i}. [{status}] {task.title} {priority_star}")
            if task.description:
                print(f"   Description: {task.description}")
        print("=" * 60 + "\n")
    
    def search(self, keyword):
        """Search for tasks by keyword."""
        results = [t for t in self.tasks if keyword.lower() in t.title.lower()]
        if results:
            print(f"\nFound {len(results)} task(s):")
            for task in results:
                print(f"  {task}")
        else:
            print(f"No tasks found containing '{keyword}'")
    
    def save_tasks(self):
        """Save tasks to JSON file."""
        try:
            data = [task.to_dict() for task in self.tasks]
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(self.tasks)} task(s)")
        except IOError as e:
            print(f"Error saving tasks: {e}")
    
    def load_tasks(self):
        """Load tasks from JSON file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.tasks = [Task.from_dict(t) for t in data]
                print(f"✓ Loaded {len(self.tasks)} task(s)")
            except IOError as e:
                print(f"Error loading tasks: {e}")
    
    def get_stats(self):
        """Get task statistics."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.completed)
        pending = total - completed
        
        return {
            "total": total,
            "completed": completed,
            "pending": pending
        }


def main():
    """Main application loop."""
    app = TodoApp()
    
    print("\n" + "=" * 60)
    print("WELCOME TO TODO APP")
    print("=" * 60)
    print("Commands: add, list, complete, remove, search, stats, save, quit\n")
    
    while True:
        try:
            command = input("> ").strip().lower()
            
            if command == "quit":
                app.save_tasks()
                print("Goodbye!")
                break
            
            elif command == "add":
                title = input("Task title: ")
                priority = input("Priority (1-3, default 1): ") or "1"
                desc = input("Description (optional): ")
                app.add_task(title, int(priority), desc)
            
            elif command == "list":
                app.list_tasks()
            
            elif command == "complete":
                app.list_tasks()
                index = int(input("Task number to complete: ")) - 1
                app.mark_complete(index)
            
            elif command == "remove":
                app.list_tasks()
                index = int(input("Task number to remove: ")) - 1
                app.remove_task(index)
            
            elif command == "search":
                keyword = input("Search keyword: ")
                app.search(keyword)
            
            elif command == "stats":
                stats = app.get_stats()
                print(f"\nStats: {stats['total']} total, "
                      f"{stats['completed']} completed, "
                      f"{stats['pending']} pending")
            
            elif command == "save":
                app.save_tasks()
            
            else:
                print("Unknown command. Try: add, list, complete, remove, search, stats, save, quit")
        
        except ValueError:
            print("Error: Invalid input")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Demonstration mode (no interactive input)
    print("\n" + "=" * 60)
    print("PROJECT 1: TODO APPLICATION DEMO")
    print("=" * 60 + "\n")
    
    app = TodoApp("/tmp/demo_todos.json")
    
    # Add sample tasks
    print("Adding sample tasks...")
    app.add_task("Learn Python", 1, "Complete Module 1-5")
    app.add_task("Build project", 1, "Create a real application")
    app.add_task("Practice exercises", 2, "Do all module exercises")
    app.add_task("Read documentation", 3, "Learn more about Python")
    
    # Show tasks
    app.list_tasks()
    
    # Mark as complete
    print("Marking first task as complete...")
    app.mark_complete(0)
    
    # Search
    print("Searching for 'Python'...")
    app.search("Python")
    
    # Stats
    stats = app.get_stats()
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Pending: {stats['pending']}")
    
    # Save
    app.save_tasks()
    
    print("\n" + "=" * 60)
    print("To use interactive mode, uncomment main() call at bottom")
    print("=" * 60)
    # Uncomment to run interactive:
    # main()
