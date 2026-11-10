"""Seed the agent's Context Hub repo with starter memory and skills from `context_hub/`."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.schemas import FileEntry

from agent import CONTEXT_HUB_REPO

SEED_DIR = Path(__file__).parent / "context_hub"


def load_seed_files(root: Path = SEED_DIR) -> dict[str, FileEntry]:
    """Map each file under `root` to its repo-relative path."""
    return {path.relative_to(root).as_posix(): FileEntry(content=path.read_text()) for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> None:
    """Push the seed files unless the repo already holds learned memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=CONTEXT_HUB_REPO, help="Context Hub agent repo, as owner/name or -/name")
    parser.add_argument("--overwrite", action="store_true", help="Replace seeded files even if the repo already exists")
    args = parser.parse_args()

    load_dotenv()
    client = Client()
    if client.agent_exists(args.repo) and not args.overwrite:
        print(f"{args.repo} already exists; leaving its memory untouched. Pass --overwrite to reset the seeded files.")
        return
    url = client.push_agent(
        args.repo,
        files=load_seed_files(),
        description="Shared memory and skills for the LangSmith memory agent example.",
    )
    print(f"Seeded {args.repo}: {url}")


if __name__ == "__main__":
    main()
