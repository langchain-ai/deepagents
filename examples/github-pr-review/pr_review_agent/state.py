"""Repository-specific state and memory management.

Each repository gets its own directory for:
- Conversation checkpoints (for resumable agent sessions)
- Repository memories (learned patterns, preferences, codebase knowledge)
- PR-specific state (review history, feedback tracking)

Storage backends:
- Local filesystem (default)
- Google Cloud Storage (set GCLOUD_STORAGE_BUCKET)
- AWS S3 (set AWS_S3_BUCKET)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from .storage import get_storage, get_repo_path, get_pr_path


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
        self.storage = get_storage()
        self.storage_path = get_repo_path(owner, repo, "memory.json")
        self._data: dict[str, Any] = self._load()
    
    def _load(self) -> dict[str, Any]:
        """Load memory from storage."""
        data = self.storage.read_json(self.storage_path)
        if data:
            return data
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
        """Save memory to storage."""
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.storage.write_json(self.storage_path, self._data)
    
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
    
    def add_user_memory(
        self, 
        memory: str, 
        added_by: str, 
        permission: str = "read",
        pr_number: int | None = None,
    ) -> None:
        """Add a user-provided memory via /remember command.
        
        Args:
            memory: The memory/convention to remember
            added_by: GitHub username who added it
            permission: User's permission level (admin/write/read/none)
            pr_number: PR number where it was added (for context)
        """
        if "user_memories" not in self._data:
            self._data["user_memories"] = []
        
        self._data["user_memories"].append({
            "content": memory,
            "added_by": added_by,
            "permission": permission,
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
        
        # User-provided memories - separate by authority level
        user_memories = self._data.get("user_memories", [])
        if user_memories:
            # Split into maintainer vs contributor memories
            maintainer_memories = [m for m in user_memories if m.get("permission") in ("admin", "write")]
            contributor_memories = [m for m in user_memories if m.get("permission") not in ("admin", "write")]
            
            if maintainer_memories:
                lines.append("## Repository Conventions (from maintainers - HIGH PRIORITY)")
                lines.append("These are explicit instructions from repository owners/maintainers. Follow them carefully:")
                lines.append("")
                for mem in maintainer_memories:
                    lines.append(f"- {mem['content']} (by @{mem.get('added_by', 'unknown')})")
                lines.append("")
            
            if contributor_memories:
                lines.append("## Suggested Conventions (from contributors)")
                lines.append("These suggestions are from contributors. Consider them but defer to maintainer conventions if conflicting:")
                lines.append("")
                for mem in contributor_memories:
                    lines.append(f"- {mem['content']} (by @{mem.get('added_by', 'unknown')})")
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
        self.storage = get_storage()
        self.storage_path = get_pr_path(owner, repo, pr_number, "state.json")
        self._data: dict[str, Any] = self._load()
    
    def _load(self) -> dict[str, Any]:
        """Load state from storage."""
        data = self.storage.read_json(self.storage_path)
        if data:
            return data
        return {
            "reviews": [],
            "feedback_addressed": [],
            "commits_made": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _save(self) -> None:
        """Save state to storage."""
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.storage.write_json(self.storage_path, self._data)
    
    def record_review(self, command: str, head_sha: str, summary: str = "") -> None:
        """Record that a review was performed.
        
        Args:
            command: The command that triggered the review (e.g., 'review', 'security')
            head_sha: The HEAD commit SHA at time of review
            summary: Optional summary of the review
        """
        self._data["reviews"].append({
            "command": command,
            "head_sha": head_sha,
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
    
    def get_last_review(self, command: str | None = None) -> dict | None:
        """Get the most recent review, optionally filtered by command.
        
        Args:
            command: If provided, get the last review of this type (e.g., 'review', 'security')
        """
        reviews = self._data["reviews"]
        if command:
            reviews = [r for r in reviews if r.get("command") == command]
        if reviews:
            return reviews[-1]
        return None
    
    def has_new_commits_since_review(self, current_head_sha: str, command: str) -> bool:
        """Check if there are new commits since the last review of this type.
        
        Args:
            current_head_sha: The current HEAD SHA of the PR
            command: The review command type to check (e.g., 'review', 'security')
            
        Returns:
            True if there are new commits (or no previous review), False if unchanged
        """
        last_review = self.get_last_review(command)
        if not last_review:
            return True  # No previous review, so "new" commits exist
        
        last_sha = last_review.get("head_sha")
        if not last_sha:
            return True  # Old review format without SHA, treat as new
            
        return current_head_sha != last_sha


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
