#!/usr/bin/env python3
"""Fetch GitHub issues for triage analysis.

Uses the gh CLI to fetch issues with metadata needed for triage.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta


def fetch_issues(
    repo: str,
    days: int = 90,
    limit: int = 200,
    state: str = "open"
) -> dict:
    """Fetch issues from a GitHub repository.

    Parameters
    ----------
    repo : str
        Repository in format 'owner/repo'
    days : int
        Fetch issues from the last N days
    limit : int
        Maximum number of issues to fetch
    state : str
        Issue state: 'open', 'closed', or 'all'

    Returns
    -------
    dict
        Dictionary with issues list and metadata
    """
    # Calculate date threshold
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build gh command
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--state", state,
        "--limit", str(limit),
        "--search", f"created:>{since_date}",
        "--json", "number,title,body,state,createdAt,updatedAt,author,labels,comments,reactionGroups,url"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issues = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": f"gh CLI error: {e.stderr}", "issues": []}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "issues": []}
    except FileNotFoundError:
        return {"error": "gh CLI not found. Install from https://cli.github.com", "issues": []}

    # Process issues - extract useful fields
    processed = []
    for issue in issues:
        # Count total reactions
        reaction_count = 0
        if issue.get("reactionGroups"):
            for rg in issue["reactionGroups"]:
                reaction_count += rg.get("users", {}).get("totalCount", 0)

        # Extract label names
        labels = [l.get("name", "") for l in issue.get("labels", [])]

        # Count comments
        comment_count = len(issue.get("comments", []))

        processed.append({
            "number": issue["number"],
            "title": issue["title"],
            "body": issue.get("body", "")[:2000],  # Truncate very long bodies
            "state": issue["state"],
            "created_at": issue["createdAt"],
            "updated_at": issue["updatedAt"],
            "author": issue.get("author", {}).get("login", "unknown"),
            "labels": labels,
            "reaction_count": reaction_count,
            "comment_count": comment_count,
            "url": issue["url"]
        })

    return {
        "repo": repo,
        "fetched_at": datetime.now().isoformat(),
        "params": {
            "days": days,
            "limit": limit,
            "state": state,
            "since": since_date
        },
        "total_count": len(processed),
        "issues": processed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GitHub issues for triage analysis"
    )
    parser.add_argument(
        "repo",
        help="Repository in format 'owner/repo'"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Fetch issues from last N days (default: 90)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum issues to fetch (default: 200)"
    )
    parser.add_argument(
        "--state",
        choices=["open", "closed", "all"],
        default="open",
        help="Issue state filter (default: open)"
    )

    args = parser.parse_args()

    result = fetch_issues(
        repo=args.repo,
        days=args.days,
        limit=args.limit,
        state=args.state
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
