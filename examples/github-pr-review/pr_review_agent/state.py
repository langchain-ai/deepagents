"""Repository-specific state and memory management.

Each repository gets its own directory for:
- Conversation checkpoints (for resumable agent sessions)
- Repository memories (learned patterns, preferences, codebase knowledge)
- PR-specific state (review history, feedback tracking)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver


# Base directory for all state - configurable via environment
STATE_BASE_DIR = Path(os.environ.get("STATE_DIR", "./data"))


def get_repo_dir(owner: str, repo: str) -> Path:
    """Get the directory for a specific repository's state.
    
    Creates the directory if it doesn't exist.
    """
    repo_dir = STATE_BASE_DIR / owner / repo
    repo_dir.mkdir(parents=True, exist_ok=True)
    return repo_dir


def get_pr_dir(owner: str, repo: str, pr_number: int) -> Path:
    """Get the directory for a specific PR's state."""
    pr_dir = get_repo_dir(owner, repo) / "prs" / str(pr_number)
    pr_dir.mkdir(parents=True, exist_ok=True)
    return pr_dir


class RepoMemory:
    """Manages learned knowledge about a repository.
    
    Stores things like:
    - Code style preferences discovered during reviews
    - Common patterns in the codebase
    - Team preferences and conventions
    - Past review feedback patterns
    """
    
    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        self.memory_file = get_repo_dir(owner, repo) / "memory.json"
        self._data: dict[str, Any] = self._load()
    
    def _load(self) -> dict[str, Any]:
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "style_preferences": {},
            "common_patterns": [],
            "team_conventions": {},
            "user_memories": [],  # User-provided memories via /remember
            "review_history": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _save(self) -> None:
        """Save memory to disk."""
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.memory_file.write_text(json.dumps(self._data, indent=2))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in memory."""
        self._data[key] = value
        self._save()
    
    def add_style_preference(self, category: str, preference: str) -> None:
        """Record a discovered style preference."""
        if category not in self._data["style_preferences"]:
            self._data["style_preferences"][category] = []
        if preference not in self._data["style_preferences"][category]:
            self._data["style_preferences"][category].append(preference)
            self._save()
    
    def add_review_summary(self, pr_number: int, summary: dict) -> None:
        """Record a summary of a completed review."""
        self._data["review_history"].append({
            "pr_number": pr_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **summary,
        })
        # Keep only last 100 reviews
        self._data["review_history"] = self._data["review_history"][-100:]
        self._save()
    
    def add_user_memory(self, memory: str, added_by: str, pr_number: int | None = None) -> None:
        """Add a user-provided memory via /remember command.
        
        Args:
            memory: The memory/convention to remember
            added_by: GitHub username who added it
            pr_number: PR number where it was added (for context)
        """
        if "user_memories" not in self._data:
            self._data["user_memories"] = []
        
        self._data["user_memories"].append({
            "content": memory,
            "added_by": added_by,
            "pr_number": pr_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
    
    def get_user_memories(self) -> list[dict]:
        """Get all user-provided memories."""
        return self._data.get("user_memories", [])
    
    def get_context_for_prompt(self) -> str:
        """Generate context string for inclusion in prompts."""
        lines = []
        
        # User-provided memories (highest priority - explicit instructions)
        user_memories = self._data.get("user_memories", [])
        if user_memories:
            lines.append("## Repository Conventions (from maintainers)")
            lines.append("These are explicit instructions from the repository maintainers. Follow them carefully:")
            lines.append("")
            for mem in user_memories:
                lines.append(f"- {mem['content']}")
            lines.append("")
        
        if self._data["style_preferences"]:
            lines.append("## Repository Style Preferences (learned from past reviews)")
            for category, prefs in self._data["style_preferences"].items():
                lines.append(f"### {category}")
                for pref in prefs:
                    lines.append(f"- {pref}")
            lines.append("")
        
        if self._data["team_conventions"]:
            lines.append("## Team Conventions")
            for key, value in self._data["team_conventions"].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        return "\n".join(lines) if lines else ""


class PRState:
    """Manages state for a specific PR.
    
    Tracks:
    - Review iterations
    - Feedback received and addressed
    - Changes made by the bot
    """
    
    def __init__(self, owner: str, repo: str, pr_number: int):
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number
        self.state_file = get_pr_dir(owner, repo, pr_number) / "state.json"
        self._data: dict[str, Any] = self._load()
    
    def _load(self) -> dict[str, Any]:
        """Load state from disk."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "reviews": [],
            "feedback_addressed": [],
            "commits_made": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _save(self) -> None:
        """Save state to disk."""
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state_file.write_text(json.dumps(self._data, indent=2))
    
    def record_review(self, command: str, summary: str) -> None:
        """Record that a review was performed."""
        self._data["reviews"].append({
            "command": command,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
    
    def record_feedback_addressed(self, feedback: str, commit_sha: str) -> None:
        """Record that feedback was addressed with a commit."""
        self._data["feedback_addressed"].append({
            "feedback": feedback,
            "commit_sha": commit_sha,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
    
    def record_commit(self, sha: str, message: str, files: list[str]) -> None:
        """Record a commit made by the bot."""
        self._data["commits_made"].append({
            "sha": sha,
            "message": message,
            "files": files,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()
    
    def get_review_count(self) -> int:
        """Get number of reviews performed on this PR."""
        return len(self._data["reviews"])
    
    def get_last_review(self) -> dict | None:
        """Get the most recent review."""
        if self._data["reviews"]:
            return self._data["reviews"][-1]
        return None


# Global checkpointer cache - one per repository
_checkpointers: dict[str, MemorySaver] = {}


def get_checkpointer(owner: str, repo: str) -> MemorySaver:
    """Get or create a checkpointer for a repository.
    
    Currently uses in-memory checkpointing. For production, consider
    using SqliteSaver or PostgresSaver for persistence.
    """
    key = f"{owner}/{repo}"
    if key not in _checkpointers:
        # For now, use in-memory. Could be upgraded to SqliteSaver:
        # from langgraph.checkpoint.sqlite import SqliteSaver
        # db_path = get_repo_dir(owner, repo) / "checkpoints.db"
        # _checkpointers[key] = SqliteSaver.from_conn_string(str(db_path))
        _checkpointers[key] = MemorySaver()
    return _checkpointers[key]


def get_thread_id(owner: str, repo: str, pr_number: int) -> str:
    """Generate a thread ID for a PR conversation."""
    return f"{owner}/{repo}/pr/{pr_number}"


class StateManager:
    """Unified interface for managing all state for a repository/PR."""
    
    def __init__(self, owner: str, repo: str, pr_number: int):
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number
        
        self.repo_memory = RepoMemory(owner, repo)
        self.pr_state = PRState(owner, repo, pr_number)
        self.checkpointer = get_checkpointer(owner, repo)
        self.thread_id = get_thread_id(owner, repo, pr_number)
    
    def get_agent_config(self) -> dict:
        """Get configuration dict for the agent."""
        return {
            "configurable": {
                "thread_id": self.thread_id,
            }
        }
    
    def get_memory_context(self) -> str:
        """Get repository memory context for prompts."""
        return self.repo_memory.get_context_for_prompt()
