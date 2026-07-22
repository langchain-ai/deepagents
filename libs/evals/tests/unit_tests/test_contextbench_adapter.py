"""Tests for the Context-Bench Harbor task adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harbor_adapters.contextbench.adapter import generate_task

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_task_creates_self_contained_harbor_task(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "filesystem_cloud.jsonl"
    source_jsonl.write_text(
        json.dumps(
            {
                "input": "Which resident owns the most vehicles?",
                "ground_truth": "Tammy Roberts",
                "agent_args": {
                    "extra": {
                        "question_type": "comparison_tiebreak",
                        "difficulty": "easy",
                        "required_files": [
                            "pets.txt",
                            "addresses.txt",
                            "vehicles.txt",
                            "people.txt",
                        ],
                    }
                },
            }
        )
        + "\n"
    )
    source_files_dir = tmp_path / "files"
    source_files_dir.mkdir()
    for filename in (
        "addresses.txt",
        "bank_accounts.txt",
        "people.txt",
        "pets.txt",
        "vehicles.txt",
    ):
        (source_files_dir / filename).write_text(f"{filename} source data\n")

    task_dir = generate_task(
        source_jsonl=source_jsonl,
        source_files_dir=source_files_dir,
        output_dir=tmp_path / "dataset",
        task_id="cb-cloud-1",
        line_index=0,
    )

    assert task_dir == tmp_path / "dataset" / "cb-cloud-1"
    assert (task_dir / "environment" / "files" / "bank_accounts.txt").read_text() == (
        "bank_accounts.txt source data\n"
    )
    assert (task_dir / "environment" / "Dockerfile").read_text() == (
        "FROM python:3.12-slim\n\n"
        "# Pre-install curl at build time (the build phase has network) so the\n"
        "# in-sandbox agent's runtime bootstrap skips apt; runtime egress is then\n"
        "# all-HTTPS via the task's network allowlist.\n"
        "RUN apt-get update \\\n"
        "    && apt-get install -y --no-install-recommends curl ca-certificates \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        "COPY files/ /app/files/\n"
    )
    assert (task_dir / "environment" / ".dockerignore").read_text() == (
        ".env\n.env.*\n*.pem\n*.key\n*.crt\ncredentials.json\n.git\n__pycache__/\n.venv/\n.DS_Store\n"
    )
    assert (task_dir / "instruction.md").read_text() == (
        "Which resident owns the most vehicles?\n\n"
        "Use only the files under `/app/files`. Write your final answer (and nothing else) "
        "to `/app/answer.txt`.\n"
    )
    assert (task_dir / "solution" / "solve.sh").read_text() == (
        "#!/bin/sh\nset -eu\nprintf '%s\\n' 'Tammy Roberts' > /app/answer.txt\n"
    )
    # The verifier is the faithful Letta model_judge (not string equality):
    # test.sh runs judge.py, which grades /app/answer.txt against the rubric.
    assert (task_dir / "tests" / "test.sh").read_text() == (
        "#!/bin/sh\nset -eu\n"
        "# Faithful Letta model_judge grader; writes /logs/verifier/reward.txt itself.\n"
        "python3 /tests/judge.py\n"
    )
    assert json.loads((task_dir / "tests" / "case.json").read_text()) == {
        "input": "Which resident owns the most vehicles?",
        "ground_truth": "Tammy Roberts",
    }
    assert (task_dir / "tests" / "judge.py").is_file()
    rubric = (task_dir / "tests" / "rubric.txt").read_text()
    assert "{input}" in rubric
    assert "{ground_truth}" in rubric
    assert "{submission}" in rubric
    assert (task_dir / "task.toml").read_text() == (
        'version = "1.3"\n\n'
        "[metadata]\n"
        'source = "contextbench"\n'
        'suite = "cloud"\n'
        'difficulty = "easy"\n'
        'source_difficulty = "easy"\n'
        'question_type = "comparison_tiebreak"\n\n'
        "[environment]\n"
        'network_mode = "allowlist"\n'
        'allowed_hosts = ["astral.sh", "*.astral.sh", "github.com", '
        '"*.githubusercontent.com", "pypi.org", "*.pythonhosted.org", '
        '"api.smith.langchain.com", "api.anthropic.com", "api.openai.com", '
        '"generativelanguage.googleapis.com", "openrouter.ai", "*.baseten.co", '
        '"api.fireworks.ai", "ollama.com", "api.groq.com", '
        '"integrate.api.nvidia.com", "api.x.ai"]\n'
    )
